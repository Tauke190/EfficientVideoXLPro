"""
rlt_static_tokens.py
====================

Faithful port of the Run-Length Tokenization (RLT) static-token primitives.

Reference: "Don't Look Twice: Faster Video Transformers with Run-Length
Tokenization" (https://arxiv.org/pdf/2411.05222)
Original source: rlt/src/models/static_token_utils.py and the
get_sinusoid_encoding helper from rlt/src/models/tokenizer.py

These are the *architecture-agnostic* math primitives of RLT. They are pure
tensor ops (no xformers, no SigLIP weights) so they can be unit-tested in
isolation and reused by the SigLIP integration module.

Two functions:
  * get_sinusoid_encoding      -- sinusoidal table for the temporal position embed.
  * batched_find_idxs_to_keep  -- boolean keep-mask over patch tokens.

Tensor-order convention (important -- this is what keeps survivors and positions
matchable by plain indexing):

  * batched_find_idxs_to_keep takes x as (B, C, T, H, W) and returns a mask of
    shape (B, T*h*w) in **(t, h, w)** raster order (t slowest, w fastest), where
    h = H // patch_size, w = W // patch_size. The SigLIP integration gathers
    survivors with this same mask, so they stay in (t, h, w) order.
"""

import torch
import torch.nn.functional as F


def get_sinusoid_encoding(n_position: int, embed_dims: int, base: int = 10000) -> torch.Tensor:
    """Sinusoidal encoding table of shape (1, n_position, embed_dims).

    Faithful port of rlt/src/models/tokenizer.py:get_sinusoid_encoding. Used for
    both the spatial/temporal position table (base=10000) and the run-length
    table (base=1000), exactly as in the paper.
    """
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(base, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()  # dim 2i
    sinusoid_table[:, 1::2].cos_()  # dim 2i+1

    sinusoid_table = sinusoid_table.to(torch.float32)
    return sinusoid_table.unsqueeze(0)


def batched_find_idxs_to_keep(
    x: torch.Tensor,
    threshold: int = 2,
    tubelet_size: int = 2,
    patch_size: int = 16,
) -> torch.Tensor:
    """Find non-redundant (non-static) tokens in a video tensor.

    Faithful port of rlt/src/models/static_token_utils.py:batched_find_idxs_to_keep.

    A token (a tubelet_size-frame x patch_size x patch_size block) is dropped
    when it barely changes relative to the previous tubelet (mean abs intensity
    below `threshold`). The first temporal token is always kept via a dummy
    255-valued frame.

    Args:
        x: (B, C, T, H, W) video tensor.
        threshold: mean-intensity change above which a token is kept.
        tubelet_size: temporal length of a token.
        patch_size: spatial patch size.

    Returns:
        keep_idxs: (B, T'*h*w) bool, True = keep, in (t, h, w) raster order,
            where T' = T // tubelet_size, h = H // patch_size, w = W // patch_size.
    """
    assert len(x.shape) == 5, "Input must be a 5D tensor (B, C, T, H, W)"
    x = x.type(torch.float32)

    # Compare the "front" frame of one tubelet to the "back" frame of the next.
    diffs = x[:, :, (2 * tubelet_size - 1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
    diffs = torch.abs(diffs)

    # Average-pool over each patch_size x patch_size spatial block.
    avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
    # Mean over channels, keep batch + temporal + spatial-grid dims.
    avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
    # Dummy first frame (value 255) so the first temporal token is always kept.
    first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
    avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)

    keep_idxs = avg_pool_blocks.squeeze(1) > threshold   # (B, T', h, w)
    keep_idxs = keep_idxs.flatten(1)                     # (B, T'*h*w), (t,h,w) order
    return keep_idxs


@torch.no_grad()
def batched_find_idxs_to_keep_ref(
    x: torch.Tensor,
    threshold: float = 0.045,
    patch_size: int = 14,
    refresh_every: int = 0,
) -> torch.Tensor:
    """Keep-mask that diffs against the CARRIED reference, not the previous frame.

    Fixes a latent bug in the reference RLT implementation. batched_find_idxs_to_keep
    (above, faithful to the paper) tests

        |frame_t - frame_{t-1}| <= threshold   ->   drop

    but a dropped token is later reused from frame t0 = the last frame it SURVIVED,
    which may be many frames back. The condition that is tested is therefore not the
    condition that is relied on. A patch drifting slowly (say 0.01/frame) never trips
    the consecutive test, yet after 15 frames it has moved 0.15 -- far from the frame
    whose features it is actually carrying. The error is unbounded and accumulates with
    run length.

    This variant compares each patch against the reference it would REUSE, refreshing
    that reference only where a patch survives. Drift is then bounded by `threshold`
    by construction: a token is dropped only if it is still within threshold of the
    exact frame its features come from.

    Costs a modest amount of extra keep rate (the drift this catches was previously
    being silently accepted), and requires a sequential scan over frames -- cheap, since
    it is elementwise work on raw pixels, not encoder work.

    Invisible on fixed-camera datasets (Kinetics/SSv2, where the paper was validated);
    fires hard on egocentric / moving-camera video, where slow global drift is the norm.

    Args:
        x: (B, C, T, H, W) video tensor. tubelet_size is implicitly 1 (SigLIP is a 2D
           encoder, so a token is one frame deep).
        threshold: mean abs change above which a token is kept.
        patch_size: spatial patch size.
        refresh_every: if > 0, force-keep every Nth frame. Bounds how STALE a carried
            token can get (the reference test bounds how far its pixels drifted, but not
            how far its surrounding context has moved on). 0 disables.

    Returns:
        keep_idxs: (B, T*h*w) bool, True = keep, in (t, h, w) raster order.
    """
    assert len(x.shape) == 5, "Input must be a 5D tensor (B, C, T, H, W)"
    B, C, T, H, W = x.shape
    x = x.type(torch.float32)

    h, w = H // patch_size, W // patch_size
    # avg_pool discards any ragged border; crop so the reference update lines up exactly.
    x = x[:, :, :, : h * patch_size, : w * patch_size]

    keep = torch.zeros(B, T, h, w, dtype=torch.bool, device=x.device)
    keep[:, 0] = True                       # RLT always keeps all of frame 0
    ref = x[:, :, 0].clone()                # (B, C, H', W') pixels each patch carries

    for t in range(1, T):
        cur = x[:, :, t]                                            # (B, C, H', W')
        if refresh_every and t % refresh_every == 0:
            k = torch.ones(B, h, w, dtype=torch.bool, device=x.device)
        else:
            d = F.avg_pool2d((cur - ref).abs(), patch_size).mean(dim=1)   # (B, h, w)
            k = d > threshold
        keep[:, t] = k
        # Refresh the reference ONLY where the patch survived; dropped patches keep
        # carrying the pixels their features were actually computed from.
        up = k.float().repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
        up = up.unsqueeze(1)                                        # (B, 1, H', W')
        ref = ref * (1 - up) + cur * up

    return keep.flatten(1)                  # (B, T*h*w), (t,h,w) order


@torch.no_grad()
def find_idxs_to_keep_embed(
    emb: torch.Tensor,
    threshold: float = 0.34,
    metric: str = "l2",
    mask_mode: str = "ref",
    refresh_every: int = 0,
) -> torch.Tensor:
    """Keep-mask computed on PATCH EMBEDDINGS instead of raw pixels.

    Same carried-reference semantics as batched_find_idxs_to_keep_ref, but the
    redundancy test asks "did SigLIP's view of this patch change?" rather than "did
    its pixels change?". The patch-embed conv reduces 14x14x3 = 588 pixels to C
    features through learned filters, so i.i.d. sensor noise (and JPEG/codec ringing,
    and sub-threshold lighting flicker) largely averages out, while the structured
    changes the encoder is sensitive to survive. A pixel-space test cannot separate
    those: it sees only total intensity change, so grain forces a keep and a
    low-contrast but semantically real change does not.

    This is also the *right* quantity to threshold. A dropped token reuses the
    surviving token's EMBEDDING, so the error reuse actually incurs is the embedding
    distance -- which is what is measured here -- while the pixel distance is only a
    proxy for it, and a loose one wherever the patch-embed filters are not locally
    linear in pixel space.

    Costs one extra pass over the (already computed) embeddings; the patch-embed conv
    is not run twice.

    Args:
        emb: (T, P, C) patch embeddings for ONE clip, with SigLIP's spatial position
            embedding ALREADY REMOVED. Position is constant per slot p, so it cancels
            in any difference, but it does NOT cancel under the norms that `metric`
            takes -- leaving it in would make sensitivity vary by grid position, and
            would inflate cosine similarity by a shared component carrying no content.
        threshold: distance above which a patch is kept. NOT on the pixel scale of
            rlt_threshold -- see `metric` for what each scale means. The defaults are
            calibrated to hold the KEEP RATE fixed against the pixel test's own default,
            so that flipping mask_space changes the drop rule and not the amount of
            reuse (measured on 5 MLVU videos x 128 frames, mask_mode="ref"):
                pixel  thr=0.2   -> 56.9% keep   (the pixel default)
                l2     thr=0.34  -> 55.7% keep
                cosine thr=0.022 -> 57.2% keep
            Keep rate is strongly content-dependent (those 5 videos individually span
            23-74% at the pixel default), so recalibrate on your own footage before
            reading anything into a threshold transplanted from elsewhere.
        metric: how patch embeddings are compared.
            "l2" (default): euclidean distance, divided by the clip's mean patch-embed
                norm. Dimensionless, so the threshold reads as "fraction of a typical
                patch's magnitude" and transfers across SigLIP variants.
            "cosine": 1 - cos_sim, i.e. direction only. Invariant to any per-patch gain
                (exposure/contrast drift changes magnitude, not direction), but it
                amplifies noise in low-norm patches -- flat regions like sky have small
                embeddings whose direction wanders freely, so they are over-kept.
        mask_mode: "ref" (default) diffs against the embedding each patch will actually
            be REUSED from, bounding drift by `threshold` by construction. "consec"
            diffs against frame t-1 while still reusing from the last SURVIVING frame,
            reproducing the reference implementation's mismatch. Same distinction, and
            the same reasoning, as batched_find_idxs_to_keep_ref's docstring.
        refresh_every: if > 0, force-keep every Nth frame. 0 disables.

    Returns:
        keep_idxs: (T*P,) bool, True = keep, in (t, h, w) raster order -- the same
            layout batched_find_idxs_to_keep_ref returns for (B=1).
    """
    assert emb.dim() == 3, f"emb must be (T, P, C), got {tuple(emb.shape)}"
    assert metric in ("l2", "cosine"), f"unknown metric {metric!r}"
    assert mask_mode in ("ref", "consec"), f"unknown mask_mode {mask_mode!r}"
    T, P, _ = emb.shape
    dev = emb.device

    # Distances between near-identical vectors cancel most of their significant bits,
    # which fp16 does not have to spare; the whole test lives in that cancellation.
    if metric == "l2":
        # Reduce in fp32 without materializing an fp32 copy of the full (T,P,C) tensor.
        scale = torch.linalg.vector_norm(
            emb, dim=-1, dtype=torch.float32
        ).mean().clamp_min(1e-6)

    def _prep(t: int) -> torch.Tensor:
        f = emb[t].float()
        return f / scale if metric == "l2" else F.normalize(f, dim=-1)

    keep = torch.zeros(T, P, dtype=torch.bool, device=dev)
    keep[0] = True                          # RLT always keeps all of frame 0
    ref = _prep(0)                          # (P, C) embedding each patch carries

    for t in range(1, T):
        cur = _prep(t)
        if refresh_every and t % refresh_every == 0:
            k = torch.ones(P, dtype=torch.bool, device=dev)
        else:
            if metric == "l2":
                d = (cur - ref).norm(dim=-1)                # (P,)
            else:
                d = 1.0 - (cur * ref).sum(-1)               # both already unit-norm
            k = d > threshold
        keep[t] = k
        # Refresh the reference ONLY where the patch survived; dropped patches keep
        # carrying the embedding their features were actually computed from.
        ref = torch.where(k.unsqueeze(-1), cur, ref) if mask_mode == "ref" else cur

    return keep.reshape(-1)                 # (T*P,), (t,h,w) order


if __name__ == "__main__":
    # ---- self-test on a controlled synthetic clip --------------------------
    torch.manual_seed(0)
    B, C, T, H, W = 1, 3, 6, 28, 28      # 28/14 -> 2x2 grid; tubelet_size=1 here
    patch_size, tubelet_size, threshold = 14, 1, 0.5
    h = w = H // patch_size
    Tp = T // tubelet_size
    print(f"grid {h}x{w}={h*w} spatial tokens, T'={Tp}")

    # frame 0 random; 1-2 identical to 0 (redundant); top-left patch jumps at t=3;
    # 4-5 identical to 3. (B, C, T, H, W)
    base = torch.randn(B, C, 1, H, W)
    x = base.repeat(1, 1, T, 1, 1)
    x[:, :, 3:, :patch_size, :patch_size] += 255.0    # top-left patch changes at t=3

    mask = batched_find_idxs_to_keep(x, threshold=threshold,
                                     tubelet_size=tubelet_size, patch_size=patch_size)
    keep = mask.reshape(B, Tp, h, w)
    print("keep per frame [tl, tr, bl, br]:")
    for t in range(Tp):
        print(f"  t={t}: {keep[0, t].flatten().int().tolist()}")

    print(f"\nkept tokens: {int(mask.sum())} (dense={Tp*h*w})")
    # frame 0 always kept (dummy first-frame rule); top-left keeps again at t=3.
    assert keep[0, 0].all(), "first frame must always be kept"
    assert int(mask.sum()) == h * w + 1, "only t=0 (all) + the top-left jump at t=3"
    print("\nOK: faithful RLT primitives are self-consistent.")
