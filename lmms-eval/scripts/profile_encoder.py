"""
Profile Video-XL-Pro: Vision Encoder vs Language Model

Sweeps the encoder redundancy-remover variants (original / RLT / APT / APT-temporal)
crossed with the SAE (DTS temporal compression) option on/off, so the encoder cost of
each method is directly comparable at matched frame counts. All variants run in one
process by mutating model.config between them (the RLT/APT paths dispatch purely on
config flags at runtime), so the model is loaded only once.

Usage:
    python scripts/profile_encoder.py \
        --model MINT-SJTU/Video-XL-Pro-3B \
        --video /mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4 \
        --frames 64 128 256 \
        --methods original rlt apt \
        --sae both \
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

    Brackets the encoder layer list that ALL paths share: the original path runs it
    via SiglipVisionModel.forward, while RLT/APT iterate the same vm.encoder.layers
    directly -- so a hook on encoder.layers fires in every method (a hook on the
    vision-tower wrapper would miss RLT/APT). Pre-hook on the first layer / post-hook
    on the last layer brackets the whole stack with just 2 syncs per forward pass, and
    accumulates across frame-batches. Returns hook handles to remove afterward."""
    vt = model.get_model().get_vision_tower()
    layers = vt.vision_tower.vision_model.encoder.layers
    first, last = layers[0], layers[-1]

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
    cfg.rlt_threshold = args.rlt_threshold
    thr = [float(p) for p in str(args.apt_threshold).replace(",", ":").split(":") if p != ""]
    n_needed = int(args.apt_num_scales) - 1
    if len(thr) == 1:
        thr = thr * n_needed
    assert len(thr) == n_needed, f"apt needs {n_needed} thresholds, got {thr}"
    cfg.apt_thresholds = thr
    cfg.apt_num_scales = args.apt_num_scales
    cfg.apt_input_res  = args.apt_input_res
    cfg.apt_temporal_majority_ratio = args.apt_temporal_majority_ratio
    cfg.apt_temporal_max_frames     = args.apt_temporal_max_frames

    # Reset the model's running token-drop tallies so we can report % kept for
    # just this variant (they otherwise accumulate across every clip ever seen).
    for attr in ("_rlt_grand_keep", "_rlt_grand_dense",
                 "_apt_grand_keep", "_apt_grand_dense"):
        if hasattr(model, attr):
            setattr(model, attr, 0)


def read_retention(model, method):
    """% of dense tokens the remover kept during this variant (None if N/A)."""
    if method == "rlt":
        keep, dense = getattr(model, "_rlt_grand_keep", 0), getattr(model, "_rlt_grand_dense", 0)
    elif method == "apt":
        keep, dense = getattr(model, "_apt_grand_keep", 0), getattr(model, "_apt_grand_dense", 0)
    else:
        return None
    return (100.0 * keep / dense) if dense else None


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


def print_report(all_results, frames_list, variants):
    """One comparison table per frame count. Columns:
      siglip - pure SigLIP transformer stack. ~Flat across sae on/off within a method
               (SAE is post-encoder); shrinks for RLT/APT (fewer tokens attended).
      spd    - siglip speedup vs the 'original | sae=on' baseline (encoder gain).
      kept%  - fraction of dense tokens the remover kept (RLT/APT only).
      prep   - full multimodal prep = siglip + SAE + projector + pool + token scatter.
               This is the column that grows ~4x when SAE is OFF (4x more tokens
               through projector/pool/scatter), even though siglip itself is flat.
      llm    - generation over the resulting sequence (also ~4x longer w/o SAE)."""
    header = (f"{'config':<22}  {'siglip':>8}  {'spd':>5}  {'kept%':>6}  "
              f"{'prep':>8}  {'llm':>9}  {'ttft':>9}  {'total':>9}")
    for frames in frames_list:
        print()
        print(f"=== frames={frames} ===")
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
            print(f"{label:<22}  {sig:>6.0f}ms  {spd:>5}  {kept_s:>6}  "
                  f"{prep:>6.0f}ms  {llm:>7.0f}ms  {tf:>7.0f}ms  {tt:>7.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile Video-XL-Pro encoder vs LLM across variants")
    parser.add_argument("--model",   default="MINT-SJTU/Video-XL-Pro-3B")
    parser.add_argument("--video",   default="/mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4")
    parser.add_argument("--frames",  nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--methods", nargs="+", default=["original", "rlt", "apt"],
                        choices=list(METHOD_FLAGS.keys()),
                        help="Pre-encoder redundancy removers to compare")
    parser.add_argument("--sae", choices=["on", "off", "both"], default="both",
                        help="SAE/DTS temporal compression: on, off, or sweep both")
    parser.add_argument("--rlt_threshold", type=float, default=0.1)
    parser.add_argument("--apt_threshold", default="4.0",
                        help="Colon-separated per-scale thresholds, e.g. 4.0:6.0 (single value broadcasts)")
    parser.add_argument("--apt_num_scales", type=int, default=3)
    parser.add_argument("--apt_input_res",  type=int, default=392)
    parser.add_argument("--apt_temporal_majority_ratio", type=float, default=0.5)
    parser.add_argument("--apt_temporal_max_frames",     type=int,   default=512)
    parser.add_argument("--warmup",  type=int, default=1)
    parser.add_argument("--runs",    type=int, default=3)
    parser.add_argument("--hf_home", default="/mnt/SSD2/huggingface")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_states = {"on": [True], "off": [False], "both": [True, False]}[args.sae]
    variants = [(variant_label(m, s), m, s) for m in args.methods for s in sae_states]

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
            apply_variant(model, method, use_sae, args)  # reset retention tally per (variant, frames)
            print(f"  frames={max_frames}")
            runs = []
            for i in range(args.warmup + args.runs):
                tag = "warmup" if i < args.warmup else f"run {i - args.warmup + 1}"
                print(f"    {tag} ...", end=" ", flush=True)
                r = run_one(model, tokenizer, image_processor, args.video, max_frames, device, use_sae)
                print(f"total={r['total_ms']:.0f}ms  siglip={r['siglip_ms']:.0f}ms  "
                      f"prep={r['prep_ms']:.0f}ms  llm={r['llm_ms']:.0f}ms  ttft={r['ttft_ms']:.0f}ms")
                if i >= args.warmup:
                    runs.append(r)
            kept = read_retention(model, method)
            for r in runs:
                r["kept_pct"] = kept
            all_results[(label, max_frames)] = runs

    print_report(all_results, args.frames, variants)


if __name__ == "__main__":
    main()
