"""
Profile Video-XL-Pro: Vision Encoder vs Language Model

Usage:
    python scripts/profile_encoder.py \
        --model MINT-SJTU/Video-XL-Pro-3B \
        --video /mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4 \
        --frames 64 128 256 \
        --warmup 1 \
        --runs 3
"""

import sys
import os
import time
import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ""))

from videoxlpro.videoxlpro.demo_utils import process_video, load_image_processor
from videoxlpro.videoxlpro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videoxlpro.videoxlpro.mm_utils import tokenizer_image_token


def build_prompt():
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\nDescribe this video briefly.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class _TTFTRecorder(StoppingCriteria):
    """Records time from vision-encoding-done to first token; never stops generation."""
    def __init__(self, timing, store):
        self._timing = timing  # shared with _install_vision_timer
        self._store = store

    def __call__(self, input_ids, scores, **kwargs):
        if "ttft_ms" not in self._store:
            torch.cuda.synchronize()
            t_ref = self._timing.get("t_after_vision", 0)
            if t_ref:
                self._store["ttft_ms"] = (time.perf_counter() - t_ref) * 1000
        return False


def _install_vision_timer(model, _timing):
    ModelClass = type(model)
    original_prepare = ModelClass.prepare_inputs_labels_for_multimodal

    def _timed_prepare(self, input_ids, position_ids, attention_mask,
                       past_key_values, labels, images,
                       modalities=["image"], image_sizes=None, time_embedding=None):
        is_vision_call = images is not None and not _timing.get("done", False)

        if is_vision_call:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        result = original_prepare(
            self, input_ids, position_ids, attention_mask,
            past_key_values, labels, images,
            modalities, image_sizes, time_embedding,
        )

        if is_vision_call:
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            _timing["time_ms"]      = (t_end - t0) * 1000
            _timing["t_after_vision"] = t_end
            _timing["done"] = True

        return result

    ModelClass.prepare_inputs_labels_for_multimodal = _timed_prepare
    return ModelClass, original_prepare


def run_one(model, tokenizer, image_processor, video_path, max_frames, device):
    _timing = {}
    _ttft   = {}
    ModelClass, original_prepare = _install_vision_timer(model, _timing)

    try:
        video_tensor, time_stamps = process_video(
            video_path, tokenizer, image_processor, device, max_frames
        )
        input_ids = tokenizer_image_token(
            build_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.inference_mode():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[video_tensor],
                time_embedding=time_stamps,
                modalities=["video"],
                do_sample=False,
                max_new_tokens=128,
                num_beams=1,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([_TTFTRecorder(_timing, _ttft)]),
            )

        torch.cuda.synchronize()
        total_time_ms = (time.perf_counter() - t_start) * 1000

    finally:
        ModelClass.prepare_inputs_labels_for_multimodal = original_prepare

    vision_ms = _timing.get("time_ms", 0)
    return {
        "vision_ms": vision_ms,
        "llm_ms":    total_time_ms - vision_ms,
        "ttft_ms":   _ttft.get("ttft_ms", 0),
        "total_ms":  total_time_ms,
    }


def print_report(all_results):
    def med(vals):
        return statistics.median(vals) if vals else 0.0

    header = f"{'frames':>8}  {'encoder':>10}  {'llm':>10}  {'ttft':>10}  {'total':>10}"
    print()
    print(header)
    print("-" * len(header))
    for frames, runs in sorted(all_results.items()):
        if not runs:
            continue
        vt = med([r["vision_ms"] for r in runs])
        lt = med([r["llm_ms"]    for r in runs])
        tf = med([r["ttft_ms"]   for r in runs])
        tt = med([r["total_ms"]  for r in runs])
        print(f"{frames:>8}  {vt:>8.0f}ms  {lt:>8.0f}ms  {tf:>8.0f}ms  {tt:>8.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile Video-XL-Pro encoder vs LLM")
    parser.add_argument("--model",   default="MINT-SJTU/Video-XL-Pro-3B")
    parser.add_argument("--video",   default="/mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4")
    parser.add_argument("--frames",  nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--warmup",  type=int, default=1)
    parser.add_argument("--runs",    type=int, default=3)
    parser.add_argument("--hf_home", default="/mnt/SSD2/huggingface")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
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

    all_results = {}

    for max_frames in args.frames:
        print(f"\nframes={max_frames}")
        runs = []
        for i in range(args.warmup + args.runs):
            tag = "warmup" if i < args.warmup else f"run {i - args.warmup + 1}"
            print(f"  {tag} ...", end=" ", flush=True)
            r = run_one(model, tokenizer, image_processor, args.video, max_frames, device)
            print(f"total={r['total_ms']:.0f}ms  encoder={r['vision_ms']:.0f}ms  ttft={r['ttft_ms']:.0f}ms")
            if i >= args.warmup:
                runs.append(r)
        all_results[max_frames] = runs

    print_report(all_results)


if __name__ == "__main__":
    main()
