"""
apt_static_tokens.py
====================

Faithful port of the Adaptive Patch Transformer (APT) static-token primitives.

Reference: "Adaptive Patch Transformers ..." (https://arxiv.org/pdf/2510.18091)
Original source:
  * apt/src/models/entropy_utils.py   (compute_patch_entropy_batched,
                                        select_patches_by_threshold)
  * apt/src/models/patch_tokenizer.py (PatchTokenizer.construct_masks /
                                        construct_patch_groups /
                                        compute_importance_maps)

These are the *architecture-agnostic* math primitives of APT: hierarchical
(quadtree) patchification driven by per-patch Shannon entropy, producing a
**strict spatial partition** of the base grid into patches of size
p, 2p, 4p, ... .  They are pure tensor ops (no SigLIP weights, no xformers, no
cv2/PIL/ipdb) so they can be unit-tested in isolation and reused by the SigLIP
integration module (siglip_apt_embeddings.py).

Differences from the original PatchTokenizer (all deliberate):
  * **cls-free.** SigLIP (siglip-so400m-patch14-384) has no class token
    (num_positions == num_patches), so construct_masks does NOT prepend a -1
    cls code and seqlens start at 0.  Scale codes are {1, 2, 3, ...}.
  * entropy is always (re)computed in forward (the original skipped it for
    method='entropy', expecting maps precomputed in the dataloader).
  * only the entropy importance metric is ported (the laplacian / upsample-mse
    paths are dropped to keep the dependency surface minimal).

Tensor-order conventions (this is what keeps survivors / masks / scatter-back
matchable by plain boolean indexing):

  * importance maps and per-scale masks have shape (B, G_s, G_s) where
    G_s = image_size // (base_patch_size * 2**scale_idx).
  * boolean indexing of a (B, G_s, G_s, ...) tensor by a (B, G_s, G_s) mask
    flattens in **(b, row, col)** raster order (b slowest, col fastest).  Every
    grouped tensor here (resized_patches_{ps}, full_patches_{ps}) follows that
    order, and so does the output_mask, so the embedding module and the
    scatter-back can line survivors up by scale code alone.
"""

from typing import Dict, List, Tuple, Union

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


# --------------------------------------------------------------------------- #
# Pure entropy / partition primitives (verbatim from apt/src/models/entropy_utils.py)
# --------------------------------------------------------------------------- #
def compute_patch_entropy_batched(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6):
    """Per-patch Shannon entropy maps for a batch of images, at several scales.

    Faithful port of entropy_utils.compute_patch_entropy_batched.

    Args:
        images: (B, C, H, W) tensor with values in [0, 255].
        patch_size: base patch size.
        num_scales: number of scales; patch sizes are patch_size * 2**i.
        bins: histogram bins.
        pad_value: high entropy assigned to padded (incomplete) border patches so
            they never merge into a coarse patch.

    Returns:
        dict {ps: (B, G_s, G_s)} of entropy maps, one per patch size.
    """
    batch_size, channels, H, W = images.shape
    device = images.device

    if channels == 3:
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * grayscale_weights).sum(dim=1)
    else:
        grayscale_images = images[:, 0]

    batch_entropy_maps = {}
    patch_sizes = [patch_size * (2 ** i) for i in range(num_scales)]

    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps
        num_patches_w = (W + ps - 1) // ps

        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode="constant", value=0)

        patches = padded_images.unfold(1, ps, ps).unfold(2, ps, ps)
        flat_patches = patches.reshape(batch_size, num_patches_h, num_patches_w, ps * ps)

        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins - 1)
        reshaped_patches = flat_patches_int.reshape(-1, ps * ps)

        # Per-patch histograms via scatter_add straight into (N, bins).  This is
        # numerically identical to one-hot + sum, but avoids materializing the
        # (N, ps*ps, bins) one-hot -- which for a full 128-frame clip at the base
        # scale is ~40 GiB and OOMs.  (N, bins) is ~196x smaller.
        N = reshaped_patches.size(0)
        histograms = torch.zeros(N, bins, device=device)
        histograms.scatter_add_(
            1, reshaped_patches, torch.ones_like(reshaped_patches, dtype=histograms.dtype)
        )
        histograms = histograms.reshape(batch_size, num_patches_h, num_patches_w, bins)

        probabilities = histograms.float() / (ps * ps)
        epsilon = 1e-10
        entropy_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=3)

        if pad_h > 0:
            entropy_map[:, -1, :] = pad_value
        if pad_w > 0:
            entropy_map[:, :, -1] = pad_value

        batch_entropy_maps[ps] = entropy_map

    return batch_entropy_maps


def select_patches_by_threshold(entropy_maps, thresholds, alpha=1.0):
    """Hierarchical (quadtree) patch selection from entropy maps.

    Faithful port of entropy_utils.select_patches_by_threshold.  A coarse patch
    is kept when its entropy is below the per-scale threshold; smaller scales are
    then masked out wherever a coarser patch already covers them.  The result is
    a STRICT SPATIAL PARTITION of the base grid (disjoint + exhaustive).

    Args:
        entropy_maps: dict {ps: (B, G_s, G_s)}.
        thresholds: list of length len(entropy_maps) - 1, one threshold per
            non-base scale (ordered base+1 .. coarsest).

    Returns:
        masks: dict {ps: (B, G_s, G_s)} of 0/1 keep-masks forming a partition.
    """
    patch_sizes = sorted(list(entropy_maps.keys()))

    if len(patch_sizes) == 1:
        return {patch_sizes[0]: torch.ones_like(entropy_maps[patch_sizes[0]])}

    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must be one less than "
            f"number of patch sizes ({len(patch_sizes)})"
        )

    masks = {}
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])

    for i in range(len(patch_sizes) - 1, 0, -1):
        current_size = patch_sizes[i]
        threshold = thresholds[i - 1]
        masks[current_size] = (entropy_maps[current_size] < threshold).float()

    for i in range(len(patch_sizes) - 1, 0, -1):
        current_size = patch_sizes[i]
        for j in range(i):
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size
            mask_upscaled = masks[current_size].repeat_interleave(
                scale_factor, dim=1
            ).repeat_interleave(scale_factor, dim=2)

            H_small, W_small = entropy_maps[smaller_size].shape[1:]
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]

            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)

    return masks


# --------------------------------------------------------------------------- #
# Trimmed, cls-free tokenizer (port of apt/src/models/patch_tokenizer.py)
# --------------------------------------------------------------------------- #
class APTPatchTokenizer:
    """Hierarchical patch tokenizer for SigLIP (cls-free).

    Pure-tensor helper (no learnable params); operates on whatever device the
    input frames live on.  Produces, per batch of frames:
      * masks: per-scale 0/1 partition masks {ps: (B, G_s, G_s)}.
      * output_mask: (N,) flat scale codes {1, 2, ...} for survivors, in
        frame-major then scale-major raster order.
      * seqlens / cu_seqlens / max_seqlen: per-frame survivor counts for
        block-diagonal attention.
      * resized_patches_{ps}: (n_s, C, p, p) coarse regions resized to base p.
      * full_patches_{ps}: (n_s, (ps/p)^2, C, p, p) constituent base sub-patches.
      * pos_embed_mask_{ps}: (B, G_s*G_s) bool, flattened keep-mask for gathering
        resampled position embeddings.
    """

    def __init__(
        self,
        num_scales: int,
        base_patch_size: int,
        image_size: int,
        thresholds: List[float],
        mean: List[float] = (0.5, 0.5, 0.5),
        std: List[float] = (0.5, 0.5, 0.5),
        method: str = "entropy",
    ):
        self.num_scales = num_scales
        self.base_patch_size = base_patch_size
        self.image_size = image_size
        self.thresholds = list(thresholds)
        self.method = method
        self.unnorm = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1.0 / s for s in std],
        )

    def compute_importance_maps(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Entropy maps on un-normalized [0,255] frames (SigLIP mean/std=0.5)."""
        if self.method != "entropy":
            raise ValueError(f"Only method='entropy' is supported, got {self.method!r}")
        with torch.no_grad():
            unnormalized = self.unnorm(images)
            unnormalized = torch.clamp(unnormalized * 255.0, 0, 255)
            return compute_patch_entropy_batched(
                unnormalized, patch_size=self.base_patch_size, num_scales=self.num_scales
            )

    def construct_masks(
        self, importance_maps: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[int]]:
        """Partition masks + flat scale-code output_mask + per-frame seqlens (cls-free)."""
        masks = select_patches_by_threshold(importance_maps, thresholds=self.thresholds)
        batch_size = masks[self.base_patch_size].shape[0]
        device = importance_maps[self.base_patch_size].device

        temp_masks = []
        seqlens = torch.zeros((batch_size,), device=device)
        for idx in range(self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            temp_mask = masks[cur_patch_size].flatten(1)
            seqlens += temp_mask.sum(1)
            temp_masks.append(temp_mask * (idx + 1))

        output_mask = torch.cat(temp_masks, dim=1)
        output_mask = output_mask[output_mask != 0]
        seqlens = seqlens.int().tolist()
        return masks, output_mask, seqlens

    def construct_patch_groups(
        self, images: torch.Tensor, masks: Dict[int, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pixel-level patch groups per scale (cls-free; pos handled by the embedder)."""
        output_dict = {}
        for idx in range(self.num_scales):
            cur_patch_size = self.base_patch_size * 2 ** idx
            cur_mask = masks[cur_patch_size].bool()

            scale_img = images
            if idx > 0:
                scale_img = F.interpolate(scale_img, scale_factor=0.5 ** idx, mode="bilinear")

                constituent_patches = einops.rearrange(
                    images,
                    "b c (h n1 p3) (w n2 p4) -> b h w (n1 n2) c p3 p4",
                    h=self.image_size // cur_patch_size,
                    w=self.image_size // cur_patch_size,
                    n1=cur_patch_size // self.base_patch_size,
                    n2=cur_patch_size // self.base_patch_size,
                    p3=self.base_patch_size,
                    p4=self.base_patch_size,
                )
                output_dict[f"full_patches_{cur_patch_size}"] = constituent_patches[cur_mask]

            scaled_patches = einops.rearrange(
                scale_img,
                "b c (h p1) (w p2) -> b h w c p1 p2",
                p1=self.base_patch_size,
                p2=self.base_patch_size,
            )
            output_dict[f"resized_patches_{cur_patch_size}"] = scaled_patches[cur_mask]
            output_dict[f"pos_embed_mask_{cur_patch_size}"] = masks[cur_patch_size].flatten(1).bool()
        return output_dict

    def __call__(self, images: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[int]]]:
        B, C, H, W = images.shape
        max_patches = B * H * W / (self.base_patch_size ** 2)

        importance_maps = self.compute_importance_maps(images)
        masks, output_mask, seqlens = self.construct_masks(importance_maps)
        output_dict = self.construct_patch_groups(images, masks)
        output_dict["masks"] = masks
        output_dict["output_mask"] = output_mask
        output_dict["seqlens"] = seqlens

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=images.device),
                torch.tensor(seqlens, dtype=torch.int32, device=images.device).cumsum(0),
            ]
        )
        output_dict["cu_seqlens"] = cu_seqlens
        output_dict["max_seqlen"] = max(seqlens) if seqlens else 0
        output_dict["retained_frac"] = sum(seqlens) / max_patches
        return output_dict


if __name__ == "__main__":
    # ---- self-test: hierarchical partition tiles the base grid exactly -------
    torch.manual_seed(0)
    base_p, img = 14, 392                      # 392/14 = 28 base grid; scales 14/28/56
    num_scales = 3
    G = img // base_p                          # 28
    # Half the image flat (low entropy -> merges to coarse), half detailed noise.
    x = torch.zeros(1, 3, img, img)
    x[:, :, :, img // 2:] = torch.rand(1, 3, img, img // 2)  # right half: high entropy
    # Normalize to SigLIP range so the tokenizer's un-normalize round-trips.
    x = (x - 0.5) / 0.5

    tok = APTPatchTokenizer(
        num_scales=num_scales, base_patch_size=base_p, image_size=img,
        thresholds=[5.0, 5.0],                 # below -> merge; flat region merges
    )
    out = tok(x)
    masks = out["masks"]

    # Partition check: sum of (mask area * scale_area) == base grid area.
    covered = 0
    for idx in range(num_scales):
        ps = base_p * 2 ** idx
        s = ps // base_p
        covered += int(masks[ps].sum().item()) * (s * s)
    print(f"base grid {G}x{G}={G*G}; covered base cells = {covered}")
    assert covered == G * G, f"partition must tile exactly: {covered} != {G*G}"

    # Disjointness: upsample every scale mask to the base grid; each cell hit once.
    hit = torch.zeros(1, G, G)
    for idx in range(num_scales):
        ps = base_p * 2 ** idx
        s = ps // base_p
        up = masks[ps].repeat_interleave(s, 1).repeat_interleave(s, 2)
        hit += up
    assert torch.all(hit == 1), "every base cell must be covered by exactly one scale"

    # output_mask / seqlens consistency.
    assert out["output_mask"].numel() == sum(out["seqlens"]) == sum(
        int(masks[base_p * 2 ** i].sum()) for i in range(num_scales)
    )
    n_full2 = out.get("full_patches_28")
    print(f"survivors={out['output_mask'].numel()} (dense base={G*G}), "
          f"retained_frac={out['retained_frac']:.3f}")
    print("OK: APT hierarchical partition is exhaustive, disjoint, and self-consistent.")
