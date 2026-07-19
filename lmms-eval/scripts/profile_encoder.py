"""
Profile Video-XL-Pro: Vision Encoder vs Language Model

Sweeps the encoder redundancy-remover variants (original / RLT / APT / APT-temporal)
crossed with the SAE (DTS temporal compression) option on/off, so the encoder cost of
each method is directly comparable at matched frame counts. All variants run in one
process by mutating model.config between them (the RLT/APT paths dispatch purely on
config flags at runtime), so the model is loaded only once.

Each variant is measured over --num_videos DISTINCT videos sampled from --video_dir
(same seeded sample for every variant). That is the axis that matters: what a remover
saves depends on the footage -- a static surveillance shot lets APT-Temporal reuse
almost everything, a moving camera lets it reuse nothing -- so repeating one clip just
measures that clip precisely. The report shows the median plus the min-max range.

Usage:
    python scripts/profile_encoder.py \
        --model MINT-SJTU/Video-XL-Pro-3B \
        --video_dir /mnt/SSD2/huggingface/mlvu_test \
        --num_videos 5 \
        --frames 64 128 256 \
        --methods original rlt apt apt_temporal \
        --sae on \
        --warmup 2
"""

import sys
import os
import time
import glob
import random
import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Import via the single `videoxlpro.*` form (same as lmms_eval) so the editable
# install's finder resolves these to THIS checkout. The double `videoxlpro.videoxlpro.*`
# form falls through to whatever is on PYTHONPATH, which can be a stale checkout
# whose process_video predates the use_sae argument.
from videoxlpro.demo_utils import process_video, load_image_processor
from videoxlpro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videoxlpro.mm_utils import tokenizer_image_token


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


def _install_encoder_timer(model, _enc):
    """Time just the SigLIP transformer stack (the 'pure encoder'), isolated from the
    projector/pool/token-scatter that prepare_inputs_labels_for_multimodal also does.

    Hooks the first/last SUBMODULES inside the encoder stack, not the encoder layers
    themselves. The layer modules look like the obvious bracket, but RLT/APT/APT-Temporal
    never call them: their _run_encoder reimplements the layer inline to inject an
    xformers attn_bias, reaching into layer.layer_norm1 / layer.self_attn.q_proj /
    layer.mlp directly and so bypassing nn.Module.__call__ on the layer -- a hook there
    fires only on the 'original' path and silently reports 0ms for every remover.
    layer_norm1 (first op of a layer) and mlp (last op) ARE invoked via __call__ by both
    HF's SiglipEncoderLayer.forward and by _run_encoder, so they bracket the stack in
    every method. Excludes post_layernorm in both paths alike. 2 syncs per forward pass;
    accumulates across frame-batches. Returns hook handles to remove afterward."""
    vt = model.get_model().get_vision_tower()
    layers = vt.vision_tower.vision_model.encoder.layers
    first, last = layers[0].layer_norm1, layers[-1].mlp

    def pre(_mod, _args):
        torch.cuda.synchronize()
        _enc["_t0"] = time.perf_counter()

    def post(_mod, _args, _out):
        if "_t0" not in _enc:
            return
        torch.cuda.synchronize()
        _enc["ms"] = _enc.get("ms", 0.0) + (time.perf_counter() - _enc["_t0"]) * 1000

    return [first.register_forward_pre_hook(pre),
            last.register_forward_hook(post)]


# Method axis: the pre-encoder redundancy remover. Mutually exclusive (the model
# asserts sum(use_rlt, use_apt, use_apt_temporal) <= 1), so each variant sets all
# three flags explicitly. "original" = no remover, i.e. the dense SigLIP encode.
METHOD_FLAGS = {
    "original":     dict(use_rlt=False, use_apt=False, use_apt_temporal=False),
    "rlt":          dict(use_rlt=True,  use_apt=False, use_apt_temporal=False),
    "apt":          dict(use_rlt=False, use_apt=True,  use_apt_temporal=False),
    "apt_temporal": dict(use_rlt=False, use_apt=False, use_apt_temporal=True),
}


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def resolve_videos(args):
    """The sampling axis is N DISTINCT videos, not N repeats of one video.

    Every number here is content-dependent: APT's partition is driven by frame entropy,
    and RLT/APT-Temporal's savings by how much actually moves. Repeating one clip
    measures that clip's quirks with tight error bars -- a static surveillance shot
    flatters the temporal methods, a hand-held pan destroys them -- so the spread across
    videos IS the quantity of interest, not noise to average away. Sorted pool + fixed
    seed, so every variant is compared on the exact same content."""
    if args.video:                      # explicit single video wins (back-compat)
        return [args.video]
    pool = sorted(p for p in glob.glob(os.path.join(args.video_dir, "*"))
                  if p.lower().endswith(VIDEO_EXTS))
    assert pool, f"no videos ({'/'.join(VIDEO_EXTS)}) under {args.video_dir}"
    n = min(args.num_videos, len(pool))
    if n < args.num_videos:
        print(f"[warn] asked for {args.num_videos} videos, {args.video_dir} only has {n}")
    return random.Random(args.video_seed).sample(pool, n)


def variant_label(method, use_sae):
    return f"{method} | sae={'on' if use_sae else 'off'}"


def apply_variant(model, method, use_sae, args):
    """Set model.config for one (method, sae) variant. Mirrors how lmms_eval's
    videoxlpro loader sets these flags, so profiled configs match eval configs."""
    cfg = model.config
    flags = METHOD_FLAGS[method]
    cfg.use_rlt          = flags["use_rlt"]
    cfg.use_apt          = flags["use_apt"]
    cfg.use_apt_temporal = flags["use_apt_temporal"]
    cfg.use_sae          = use_sae

    # Method params (harmless to set even when the method is off). APT wants one
    # threshold per non-base scale (apt_num_scales - 1); a single value broadcasts.
    # The rlt_* knobs below are shared, not RLT-only: APT-Temporal's dirty-tile check
    # runs on the same SigLIP-normalized pixel scale RLT does, so it reads the same
    # rlt_threshold / rlt_mask_mode / rlt_refresh_every / rlt_attn_mode fields (see
    # llava_arch._get_apt_temporal_module). Set them explicitly rather than leaning on
    # the getattr fallbacks in llava_arch, whose rlt_temporal_pos_scale default (1.0)
    # differs from the eval loader's (0.0) -- profiled configs must match eval configs.
    cfg.rlt_threshold          = args.rlt_threshold
    cfg.rlt_temporal_pos_scale = args.rlt_temporal_pos_scale
    cfg.rlt_attn_mode          = args.rlt_attn_mode
    cfg.rlt_mask_mode          = args.rlt_mask_mode
    cfg.rlt_refresh_every      = int(args.rlt_refresh_every)
    # mask_space and its threshold are RLT-only -- APT-Temporal tests sub-tiles of raw
    # pixels, so it ignores these and keeps reading rlt_threshold.
    cfg.rlt_mask_space         = args.rlt_mask_space
    cfg.rlt_embed_threshold    = float(args.rlt_embed_threshold)
    cfg.rlt_embed_metric       = args.rlt_embed_metric
    thr = [float(p) for p in str(args.apt_threshold).replace(",", ":").split(":") if p != ""]
    n_needed = int(args.apt_num_scales) - 1
    if len(thr) == 1:
        thr = thr * n_needed
    assert len(thr) == n_needed, f"apt needs {n_needed} thresholds, got {thr}"
    cfg.apt_thresholds = thr
    cfg.apt_num_scales = args.apt_num_scales
    cfg.apt_input_res  = args.apt_input_res


def reset_tallies(model):
    """Zero the model's running token-drop counters. Called after warmup, so kept%/miss%
    cover exactly the measured videos (they otherwise accumulate across every clip the
    model has ever seen, warmup included)."""
    for attr in ("_rlt_grand_keep", "_rlt_grand_dense",
                 "_apt_grand_keep", "_apt_grand_dense",
                 "_apt_temporal_grand_keep", "_apt_temporal_grand_dense"):
        if hasattr(model, attr):
            setattr(model, attr, 0)
    # APT-Temporal also accumulates its cell-classification diagnostics (missed_reuse
    # et al.) as a running sum + count; reset both or the rate is smeared across variants.
    if hasattr(model, "_apt_temporal_stat_sum"):
        model._apt_temporal_stat_sum = {}
        model._apt_temporal_stat_n = 0


# Per-method prefix of the model's grand-total token tallies (llava_arch sets
# _{prefix}_grand_keep / _grand_dense on every encode). "original" has no remover
# and therefore no tallies.
RETENTION_PREFIX = {"rlt": "_rlt", "apt": "_apt", "apt_temporal": "_apt_temporal"}


def read_retention(model, method):
    """% of dense tokens the remover kept during this variant (None if N/A)."""
    prefix = RETENTION_PREFIX.get(method)
    if prefix is None:
        return None
    keep = getattr(model, f"{prefix}_grand_keep", 0)
    dense = getattr(model, f"{prefix}_grand_dense", 0)
    return (100.0 * keep / dense) if dense else None


def read_missed_reuse(model, method):
    """APT-Temporal only: % of base cells that were unchanged but still had to pay for
    a fresh token because the quadtree partition wobbled between frames. This is pure
    loss and the ceiling on what TAPT can save over plain APT, so it is the number that
    explains a disappointing kept% (partition instability, not the classifier)."""
    if method != "apt_temporal":
        return None
    stat_sum = getattr(model, "_apt_temporal_stat_sum", None) or {}
    n = getattr(model, "_apt_temporal_stat_n", 0)
    if not n or "missed_reuse" not in stat_sum:
        return None
    return 100.0 * stat_sum["missed_reuse"] / n


def run_one(model, tokenizer, image_processor, video_path, max_frames, device, use_sae):
    _timing = {}
    _ttft   = {}
    _enc    = {}
    ModelClass, original_prepare = _install_vision_timer(model, _timing)
    enc_hooks = _install_encoder_timer(model, _enc)

    try:
        # use_sae must match model.config.use_sae: it sets how many <image>
        # placeholder tokens process_video emits (144/group with SAE, 4x without).
        # A mismatch makes the token-scatter in _videoxl_select_tokens fail.
        video_tensor, time_stamps = process_video(
            video_path, tokenizer, image_processor, device, max_frames, use_sae=use_sae
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
        for h in enc_hooks:
            h.remove()

    prep_ms   = _timing.get("time_ms", 0)   # full multimodal prep (was "vision_ms")
    siglip_ms = _enc.get("ms", 0.0)         # just the SigLIP transformer stack
    return {
        "siglip_ms": siglip_ms,                 # pure encoder: ~flat across sae on/off
        "prep_ms":   prep_ms,                   # siglip + SAE + projector + pool + scatter
        "post_ms":   prep_ms - siglip_ms,       # the token-count-scaling part (grows w/o SAE)
        "llm_ms":    total_time_ms - prep_ms,   # generation over the resulting sequence
        "ttft_ms":   _ttft.get("ttft_ms", 0),
        "total_ms":  total_time_ms,
    }


def _med(vals):
    return statistics.median(vals) if vals else 0.0


def print_report(all_results, frames_list, variants, videos):
    """One comparison table per frame count. Every cell is a MEDIAN over the
    (videos x runs) samples; siglip carries a min-max range because content, not
    noise, drives the spread. Columns:
      siglip - pure SigLIP transformer stack. ~Flat across sae on/off within a method
               (SAE is post-encoder); shrinks for RLT/APT (fewer tokens attended).
      range  - min-max of siglip across the sampled videos. A wide range on a temporal
               method (RLT/APT-Temporal) is the headline result, not measurement noise:
               it is the method's sensitivity to how much the footage actually moves.
      spd    - siglip speedup vs the 'original | sae=on' baseline (encoder gain).
      kept%  - fraction of dense tokens the remover kept, pooled over all videos
               (RLT/APT/APT-temporal only).
      miss%  - APT-temporal only: unchanged base cells that still cost a fresh token
               because the partition wobbled. The ceiling on TAPT's savings, so read
               it alongside kept%: a high miss% means partition instability.
      prep   - full multimodal prep = siglip + SAE + projector + pool + token scatter.
               This is the column that grows ~4x when SAE is OFF (4x more tokens
               through projector/pool/scatter), even though siglip itself is flat.
      llm    - generation over the resulting sequence (also ~4x longer w/o SAE)."""
    header = (f"{'config':<22}  {'siglip':>8}  {'range':>13}  {'spd':>5}  {'kept%':>6}  {'miss%':>6}  "
              f"{'prep':>8}  {'llm':>9}  {'ttft':>9}  {'total':>9}")
    for frames in frames_list:
        print()
        print(f"=== frames={frames}  (median over {len(videos)} video(s) x runs) ===")
        print(header)
        print("-" * len(header))
        base_key = (variant_label("original", True), frames)
        base_sig = _med([r["siglip_ms"] for r in all_results.get(base_key, [])]) or None
        for label, _method, _sae in variants:
            runs = all_results.get((label, frames), [])
            if not runs:
                continue
            sig  = _med([r["siglip_ms"] for r in runs])
            prep = _med([r["prep_ms"]   for r in runs])
            llm  = _med([r["llm_ms"]    for r in runs])
            tf   = _med([r["ttft_ms"]   for r in runs])
            tt   = _med([r["total_ms"]  for r in runs])
            spd = f"{base_sig / sig:.2f}x" if (base_sig and sig) else "-"
            kept = runs[0].get("kept_pct")
            kept_s = f"{kept:.1f}" if kept is not None else "-"
            miss = runs[0].get("missed_reuse_pct")
            miss_s = f"{miss:.1f}" if miss is not None else "-"
            sigs = [r["siglip_ms"] for r in runs]
            rng_s = f"{min(sigs):.0f}-{max(sigs):.0f}ms" if len(sigs) > 1 else "-"
            print(f"{label:<22}  {sig:>6.0f}ms  {rng_s:>13}  {spd:>5}  {kept_s:>6}  {miss_s:>6}  "
                  f"{prep:>6.0f}ms  {llm:>7.0f}ms  {tf:>7.0f}ms  {tt:>7.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile Video-XL-Pro encoder vs LLM across variants")
    parser.add_argument("--model",   default="MINT-SJTU/Video-XL-Pro-3B")
    # Sample N distinct videos (default) instead of hammering one clip: the encoder gain
    # of every remover is content-dependent, so one video measures one video.
    parser.add_argument("--video_dir", default="/mnt/SSD2/huggingface/mlvu_test",
                        help="Pool to sample --num_videos from (ignored if --video is given)")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="How many distinct videos to profile each variant on")
    parser.add_argument("--video_seed", type=int, default=0,
                        help="Seed for the video sample; fixed so every variant sees the same set")
    parser.add_argument("--video",   default=None,
                        help="Profile this single video instead of sampling --video_dir")
    parser.add_argument("--frames",  nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--methods", nargs="+",
                        default=["original", "rlt", "apt", "apt_temporal"],
                        choices=list(METHOD_FLAGS.keys()),
                        help="Pre-encoder redundancy removers to compare")
    parser.add_argument("--sae", choices=["on", "off", "both"], default="both",
                        help="SAE/DTS temporal compression: on, off, or sweep both")
    # Shared by RLT and APT-Temporal (both ask "did this patch change vs. the frame it
    # would be reused from?" on the same pixel scale), so one knob drives both methods.
    parser.add_argument("--rlt_threshold", type=float, default=0.1)
    parser.add_argument("--rlt_temporal_pos_scale", type=float, default=0.0,
                        help="RLT survivor temporal sinusoid gain. Keep 0.0 with SAE on "
                             "(SAE already carries temporal info); >0 only for the ragged path.")
    parser.add_argument("--rlt_attn_mode", choices=["reuse", "per_frame"], default="reuse",
                        help="'reuse' = survivors attend over the full frame; 'per_frame' = legacy starved attention")
    parser.add_argument("--rlt_mask_space", choices=["pixel", "embed"], default="pixel",
                        help="Where the RLT drop test compares patches: 'pixel' (paper) or "
                             "'embed' (patch embeddings; noise-robust). Uses "
                             "--rlt_embed_threshold, NOT --rlt_threshold. RLT only.")
    parser.add_argument("--rlt_embed_threshold", type=float, default=0.34,
                        help="Drop threshold for --rlt_mask_space embed. Separate scale from "
                             "--rlt_threshold; calibrate against a target keep rate.")
    parser.add_argument("--rlt_embed_metric", choices=["l2", "cosine"], default="l2",
                        help="How --rlt_mask_space embed compares patch embeddings.")
    parser.add_argument("--rlt_mask_mode", choices=["ref", "consec"], default="ref",
                        help="Dirty-check reference frame: 'ref' (bounds drift) or 'consec' (legacy)")
    parser.add_argument("--rlt_refresh_every", type=int, default=0,
                        help="ref mode only: force-refresh every Nth frame (0 disables)")
    parser.add_argument("--apt_threshold", default="4.0",
                        help="Colon-separated per-scale thresholds, e.g. 4.0:6.0 (single value broadcasts)")
    parser.add_argument("--apt_num_scales", type=int, default=3)
    parser.add_argument("--apt_input_res",  type=int, default=392)
    parser.add_argument("--warmup",  type=int, default=1,
                        help="Warmup passes (first video, excluded from timings and tallies)")
    parser.add_argument("--runs",    type=int, default=1,
                        help="Repeats PER VIDEO. The sample axis is --num_videos; raise this "
                             "only to average out timing jitter within a single clip.")
    parser.add_argument("--hf_home", default="/mnt/SSD2/huggingface")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_states = {"on": [True], "off": [False], "both": [True, False]}[args.sae]
    variants = [(variant_label(m, s), m, s) for m in args.methods for s in sae_states]

    videos = resolve_videos(args)
    print(f"Profiling {len(videos)} video(s):")
    for v in videos:
        print(f"  {v}")

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

    for label, method, use_sae in variants:
        print(f"\n########## {label} ##########")
        for max_frames in args.frames:
            apply_variant(model, method, use_sae, args)
            print(f"  frames={max_frames}")

            # Warmup on the first video only (it pays the one-time CUDA/cuDNN autotune
            # and the lazy build of the RLT/APT/TAPT module), then zero the tallies so
            # kept%/miss% describe exactly the measured videos.
            for i in range(args.warmup):
                print(f"    warmup {i + 1} ...", end=" ", flush=True)
                r = run_one(model, tokenizer, image_processor, videos[0], max_frames, device, use_sae)
                print(f"total={r['total_ms']:.0f}ms")
            reset_tallies(model)

            runs = []
            for vi, video in enumerate(videos):
                for i in range(args.runs):
                    tag = f"vid {vi + 1}/{len(videos)} {os.path.basename(video)[:28]}"
                    tag += f" run {i + 1}" if args.runs > 1 else ""
                    print(f"    {tag} ...", end=" ", flush=True)
                    r = run_one(model, tokenizer, image_processor, video, max_frames, device, use_sae)
                    print(f"total={r['total_ms']:.0f}ms  siglip={r['siglip_ms']:.0f}ms  "
                          f"prep={r['prep_ms']:.0f}ms  llm={r['llm_ms']:.0f}ms  ttft={r['ttft_ms']:.0f}ms")
                    r["video"] = video
                    runs.append(r)

            # Tallies are grand totals over every measured video, so kept%/miss% are
            # pooled rates rather than one clip's -- attach the same value to each run.
            kept = read_retention(model, method)
            miss = read_missed_reuse(model, method)
            for r in runs:
                r["kept_pct"] = kept
                r["missed_reuse_pct"] = miss
            all_results[(label, max_frames)] = runs

    print_report(all_results, args.frames, variants, videos)


if __name__ == "__main__":
    main()
