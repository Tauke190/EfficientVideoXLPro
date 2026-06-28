"""
Profile Video-XL-Pro: Vision Encoder vs Language Model

Measures wall time, GPU memory, GFLOPs, and throughput for:
  - Vision block: prepare_inputs_labels_for_multimodal (SigLIP + SAE + projector + selector)
  - LLM block:    autoregressive decoding

Mem Net Alloc: memory_allocated() after stage minus before — how much GPU memory
               the stage left behind (positive = grew, negative = freed intermediates).

Usage:
    python scripts/profile_encoder.py \
        --model MINT-SJTU/Video-XL-Pro-3B \
        --video /mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4 \
        --frames 64 128 256 \
        --warmup 1 \
        --runs 3 \
        --flops          # optional: measure GFLOPs (adds one extra profiling pass)
"""

import sys
import os
import time
import argparse
import statistics

import torch
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Video-XL", "Video-XL-Pro"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from videoxlpro.videoxlpro.demo_utils import process_video, load_image_processor
from videoxlpro.videoxlpro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videoxlpro.videoxlpro.mm_utils import tokenizer_image_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def build_prompt():
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\nDescribe this video briefly.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def sum_flops_from_prof(prof):
    """Sum total FLOPs from a torch.profiler result (in GFLOPs)."""
    total = 0
    for evt in prof.key_averages():
        if hasattr(evt, "flops") and evt.flops:
            total += evt.flops
    return total / 1e9  # → GFLOPs


# ---------------------------------------------------------------------------
# Monkey-patch helper: wraps prepare_inputs_labels_for_multimodal with timing
# ---------------------------------------------------------------------------

def _install_vision_timer(model, _timing):
    ModelClass = type(model)
    original_prepare = ModelClass.prepare_inputs_labels_for_multimodal

    def _timed_prepare(self, input_ids, position_ids, attention_mask,
                       past_key_values, labels, images,
                       modalities=["image"], image_sizes=None, time_embedding=None):
        # Only time the first real vision call (images present).
        # Auto-regressive decoding also calls this per-step with images=None
        # and shape[1]==1 — those should not overwrite the vision stats.
        is_vision_call = images is not None and not _timing.get("done", False)

        if is_vision_call:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated() / 1024**2
            t0 = time.perf_counter()

        result = original_prepare(
            self, input_ids, position_ids, attention_mask,
            past_key_values, labels, images,
            modalities, image_sizes, time_embedding,
        )

        if is_vision_call:
            torch.cuda.synchronize()
            _timing["time_ms"]    = (time.perf_counter() - t0) * 1000
            _timing["mem_before"] = m0
            _timing["mem_after"]  = torch.cuda.memory_allocated() / 1024**2
            _timing["peak_mb"]    = torch.cuda.max_memory_allocated() / 1024**2
            _timing["done"]       = True
            if result[4] is not None:
                _timing["n_visual_tokens"] = result[4].shape[1]

        return result

    ModelClass.prepare_inputs_labels_for_multimodal = _timed_prepare
    return ModelClass, original_prepare


# ---------------------------------------------------------------------------
# Core profiling function for one (video, max_frames) run
# ---------------------------------------------------------------------------

def run_one(model, tokenizer, image_processor, video_path, max_frames, device):
    _timing = {}
    ModelClass, original_prepare = _install_vision_timer(model, _timing)

    try:
        video_tensor, time_stamps = process_video(
            video_path, tokenizer, image_processor, device, max_frames
        )
        input_ids = tokenizer_image_token(
            build_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(do_sample=False, max_new_tokens=128, num_beams=1, use_cache=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        m_before_total = torch.cuda.memory_allocated() / 1024**2
        t_total_start  = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[video_tensor],
                time_embedding=time_stamps,
                modalities=["video"],
                **gen_kwargs,
            )

        torch.cuda.synchronize()
        total_time_ms = (time.perf_counter() - t_total_start) * 1000
        m_after_total = torch.cuda.memory_allocated() / 1024**2
        total_peak_mb = peak_mem_mb()

    finally:
        ModelClass.prepare_inputs_labels_for_multimodal = original_prepare

    vision_time_ms = _timing.get("time_ms", 0)

    return {
        "vision_time_ms":       vision_time_ms,
        "vision_net_alloc_mb":  _timing.get("mem_after", 0) - _timing.get("mem_before", 0),
        "vision_peak_mb":       _timing.get("peak_mb", 0),
        "llm_time_ms":          total_time_ms - vision_time_ms,
        "llm_net_alloc_mb":     m_after_total - _timing.get("mem_after", m_before_total),
        "llm_peak_mb":          total_peak_mb,
        "total_time_ms":        total_time_ms,
        "n_output_tokens":      output_ids.shape[-1],
        "n_visual_tokens_in":   _timing.get("n_visual_tokens", 0),
    }


# ---------------------------------------------------------------------------
# GFLOPs measurement (one dedicated profiling pass per frame count)
# ---------------------------------------------------------------------------

def measure_flops(model, tokenizer, image_processor, video_path, max_frames, device):
    """Run one pass under torch.profiler with FLOP counting, return (vision_gflops, llm_gflops)."""
    _timing = {}
    ModelClass, original_prepare = _install_vision_timer(model, _timing)

    try:
        video_tensor, time_stamps = process_video(
            video_path, tokenizer, image_processor, device, max_frames
        )
        input_ids = tokenizer_image_token(
            build_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(do_sample=False, max_new_tokens=32, num_beams=1, use_cache=True)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            with torch.inference_mode():
                with record_function("vision_encoder"):
                    # trigger just the vision path by monkey-patching scope
                    pass  # timing captured inside _timed_prepare

                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=[video_tensor],
                    time_embedding=time_stamps,
                    modalities=["video"],
                    **gen_kwargs,
                )

    finally:
        ModelClass.prepare_inputs_labels_for_multimodal = original_prepare

    # Sum all FLOPs from the profiler
    total_gflops = sum_flops_from_prof(prof)

    # Estimate vision vs LLM split using the time ratio (profiler FLOPs are per-op,
    # not split by stage, so we use timing as a proxy for the split)
    vision_frac = _timing.get("time_ms", 0) / max(
        _timing.get("time_ms", 0) + 1, 1
    )
    # Re-derive proper split from timing
    vision_time  = _timing.get("time_ms", 0)
    total_wall   = vision_time  # we only have vision timing here; use total from run_one

    return total_gflops, prof


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(all_results, flops_data=None):
    """
    all_results : {max_frames: [run_dict, ...]}
    flops_data  : {max_frames: (vision_gflops, llm_gflops)} or None
    """

    def med(vals):
        return statistics.median(vals) if vals else 0.0

    def fmt_time(ms):
        return f"{ms:>8.0f} ms"

    def fmt_mem(mb):
        return f"{mb:>8.1f} MB"

    def fmt_gf(gf):
        return f"{gf:>8.1f} GF" if gf is not None else f"{'--':>8s}   "

    def fmt_thr(val, unit):
        return f"{val:>7.1f} {unit}"

    for frames, runs in sorted(all_results.items()):
        if not runs:
            continue

        vt  = med([r["vision_time_ms"]     for r in runs])
        vna = med([r["vision_net_alloc_mb"] for r in runs])
        vpk = med([r["vision_peak_mb"]      for r in runs])
        lt  = med([r["llm_time_ms"]         for r in runs])
        lna = med([r["llm_net_alloc_mb"]    for r in runs])
        lpk = med([r["llm_peak_mb"]         for r in runs])
        tt  = med([r["total_time_ms"]       for r in runs])
        nt  = med([r["n_output_tokens"]     for r in runs])
        nv  = med([r["n_visual_tokens_in"]  for r in runs])

        vision_fps = frames / (vt / 1000) if vt > 0 else 0
        llm_tps    = nt     / (lt / 1000) if lt > 0 else 0
        vision_pct = 100 * vt / tt        if tt > 0 else 0

        vgf = lgf = None
        if flops_data and frames in flops_data:
            vgf, lgf = flops_data[frames]

        W = 82
        print()
        print(f"{'='*W}")
        print(f"  Video-XL-Pro  |  max_frames={frames}  |  median of {len(runs)} runs")
        print(f"{'='*W}")
        print(f"{'Block':<22} {'Time':>10}  {'Net Alloc':>10}  {'Peak GPU':>10}  {'GFLOPs':>10}  {'Throughput':>14}")
        print(f"  Net Alloc = memory_allocated() after - before (positive=grew, negative=freed intermediates)")
        print(f"{'-'*W}")
        print(f"{'Vision Encoder':<22} {fmt_time(vt)}  {fmt_mem(vna)}  {fmt_mem(vpk)}  {fmt_gf(vgf)}  {fmt_thr(vision_fps,'frames/s')}")
        print(f"{'Language Model':<22} {fmt_time(lt)}  {fmt_mem(lna)}  {fmt_mem(lpk)}  {fmt_gf(lgf)}  {fmt_thr(llm_tps,'tokens/s')}")
        print(f"{'-'*W}")
        print(f"{'Total':<22} {fmt_time(tt)}")
        print(f"{'Vision % of total':<22} {vision_pct:.1f}%")
        print(f"{'Visual tokens → LLM':<22} {int(nv)} tokens  (after query-aware selection)")
        print(f"{'Output tokens':<22} {int(nt)} tokens")
        print(f"{'='*W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile Video-XL-Pro encoder vs LLM")
    parser.add_argument("--model",   default="MINT-SJTU/Video-XL-Pro-3B")
    parser.add_argument("--video",   default="/mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4")
    parser.add_argument("--frames",  nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--warmup",  type=int, default=1)
    parser.add_argument("--runs",    type=int, default=3)
    parser.add_argument("--hf_home", default="/mnt/SSD2/huggingface")
    parser.add_argument("--flops",   action="store_true",
                        help="Measure GFLOPs via torch.profiler (one extra pass per frame count)")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model: {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = 0
    image_processor = load_image_processor(model, tokenizer)
    model.eval()

    model_mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. GPU memory for weights: {model_mem_mb:.0f} MB ({model_mem_mb/1024:.1f} GB)")
    print(f"Video: {args.video}")

    all_results = {}
    flops_data  = {}

    for max_frames in args.frames:
        print(f"\n--- max_frames={max_frames} ---")
        runs = []

        for i in range(args.warmup + args.runs):
            tag = "WARMUP" if i < args.warmup else f"RUN {i - args.warmup + 1}"
            print(f"  {tag} ...", end=" ", flush=True)
            r = run_one(model, tokenizer, image_processor, args.video, max_frames, device)
            print(
                f"total={r['total_time_ms']:.0f}ms  "
                f"vision={r['vision_time_ms']:.0f}ms  "
                f"llm={r['llm_time_ms']:.0f}ms"
            )
            if i >= args.warmup:
                runs.append(r)

        all_results[max_frames] = runs

        if args.flops:
            print(f"  FLOPS ...", end=" ", flush=True)
            total_gflops, prof = measure_flops(
                model, tokenizer, image_processor, args.video, max_frames, device
            )
            # Split vision vs LLM GFLOPs by time ratio from actual runs
            vt = statistics.median([r["vision_time_ms"] for r in runs])
            tt = statistics.median([r["total_time_ms"]  for r in runs])
            vision_frac = vt / tt if tt > 0 else 0.5
            vgf = total_gflops * vision_frac
            lgf = total_gflops * (1 - vision_frac)
            flops_data[max_frames] = (vgf, lgf)
            print(f"total={total_gflops:.1f} GFLOPs  (vision≈{vgf:.1f}  llm≈{lgf:.1f})")

    print_report(all_results, flops_data if args.flops else None)


if __name__ == "__main__":
    main()
