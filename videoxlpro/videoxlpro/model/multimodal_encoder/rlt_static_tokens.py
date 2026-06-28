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

Three functions:
  * get_sinusoid_encoding      -- sinusoidal table for position / length embeds.
  * batched_find_idxs_to_keep  -- boolean keep-mask over patch tokens.
  * batched_get_token_lengths  -- run-length of each surviving token.

Tensor-order conventions (important -- this is what keeps survivors, positions
and lengths matchable by plain indexing):

  * batched_find_idxs_to_keep takes x as (B, C, T, H, W) and returns a mask of
    shape (B, T*h*w) in **(t, h, w)** raster order (t slowest, w fastest), where
    h = H // patch_size, w = W // patch_size.

  * batched_get_token_lengths returns a flat 1-D tensor of run-lengths in
    **(h, w, t)** order (the reference's order, due to the (B,H,W,T) permute +
    boolean-index). NOTE: this differs from the (t, h, w) order of the gathered
    tokens; the SigLIP integration layer is responsible for reconciling the two
    (see siglip_rlt_embeddings.py).
"""

from typing import Tuple

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


def batched_get_token_lengths(
    token_mask: torch.Tensor,
    batch_size: int,
    input_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Run-length (duration along T) of each surviving token.

    Faithful port of rlt/src/models/static_token_utils.py:batched_get_token_lengths.

    Args:
        token_mask: (B, T'*h*w) bool keep-mask in (t, h, w) order.
        batch_size: B.
        input_shape: (T', h, w).

    Returns:
        diffs: 1-D long tensor of run-lengths, all batches concatenated, in
            **(h, w, t)** order (see module docstring on the ordering caveat).
            Values in [1, T'].
    """
    input_tensor = token_mask.reshape((batch_size,) + input_shape)   # (B, T', h, w)
    input_tensor = input_tensor.permute(0, 2, 3, 1)                  # (B, h, w, T')
    B, H, W, T = input_tensor.shape
    # Concat ones at the end so terminal tokens close with length 1.
    concat_tensor = torch.cat(
        [input_tensor, torch.ones((B, H, W, 1), dtype=torch.uint8, device=input_tensor.device)],
        dim=-1,
    )
    range_tensor = torch.arange(T + 1, device=input_tensor.device).reshape((1, 1, 1, T + 1))
    range_tensor = range_tensor.repeat(B, H, W, 1)
    diffs = range_tensor[concat_tensor.bool()].diff()
    diffs = diffs[diffs > 0]
    return diffs


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

    lengths = batched_get_token_lengths(mask, batch_size=B, input_shape=(Tp, h, w))
    print(f"\nkept tokens: {int(mask.sum())} (dense={Tp*h*w})")
    print(f"run-lengths (h,w,t order): {lengths.tolist()}")
    print(f"sum run-lengths: {int(lengths.sum())} (== Tp*h*w = {Tp*h*w})")

    assert int(mask.sum()) == lengths.numel(), "one length per surviving token"
    assert int(lengths.sum()) == Tp * h * w, "run-lengths must tile the grid-time volume"
    assert 1 <= int(lengths.min()) and int(lengths.max()) <= Tp
    print("\nOK: faithful RLT primitives are self-consistent.")
