#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from transformers import AutoTokenizer

from videoxlpro.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from videoxlpro.mm_utils import get_anyres_image_grid_shape
from videoxlpro.utils import rank0_print
import random
from .sae import SiglipAE
from .WindowTimeToTokenAttention import WindowTimeToTokenAttention
import numpy as np
import torch.nn.functional as F
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

            if getattr(config, "use_apt", False) or getattr(config, "use_apt_temporal", False):
                # APT's trainable patch_attn/zero_conv, owned here (not inside the
                # unregistered APT wrapper) so trained-APT checkpoints load their
                # weights and DeepSpeed Zero-3 gathers them.
                # Must match train.py's unfreeze/save condition (use_apt OR
                # use_apt_temporal): APT-Temporal's coarse tokens ARE APT's merge, so it
                # trains and saves these too. Gating construction on use_apt alone would
                # leave a TAPT checkpoint's apt_patch_attn/apt_zero_conv with no submodule
                # to bind to -- from_pretrained drops them as unexpected keys, _get_apt_module
                # then rebuilds them fresh (zero_conv re-zeroed), and the trained merge is
                # silently discarded at eval.
                self.get_apt_patch_attn()
                self.get_apt_zero_conv()

        # self.llm_tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.hidden_size=config.hidden_size
        # print(config)
        # exit(0)
        
#         self.text_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
##############################################################################
#         self.text_select_model = T5EncoderModel.from_pretrained('google-t5/t5-small')
        
        # self.text_gamma=0.75

###############################################################################
        self.text_mlp=nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.GELU(),
        )
    
        self.WindowTimeToTokenAttention=WindowTimeToTokenAttention(config.hidden_size,4)
        
        self.sae=SiglipAE()
        #self.sae.load_state_dict(torch.load('/share/LXRlxr0_0/code/videoxlturbo2.0/videoxl_adaptfps/longva/longva/model/encoder.pth'),strict=False)
        
###############################################################################
        # self.vision_select=nn.Parameter(
        #         torch.randn((4, self.config.hidden_size), dtype=self.dtype)
        # )
##############################################################################
        
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_apt_patch_attn(self):
        """APT's trainable Conv2d^i (paper arXiv 2510.18091 Eq. 2, self.patch_attn).

        Owned by the base model (not the APT wrapper, which is kept out of the
        module registry like the RLT wrapper above) so it appears in
        named_parameters() -> optimizer + checkpoint, and DeepSpeed Zero-3 gathers
        it via its forward. Default (non-zero) Conv2d init is fine here: the
        zero-init property of APT's merge comes entirely from zero_conv gating
        this layer's output to 0, not from patch_attn itself (see get_apt_zero_conv).
        """
        if getattr(self, "apt_patch_attn", None) is None:
            vt = self.get_vision_tower()
            dim = getattr(vt, "hidden_size", None) or self.config.mm_hidden_size
            conv = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
            self.apt_patch_attn = conv.to(device=vt.device, dtype=vt.dtype)
        return self.apt_patch_attn

    def get_apt_zero_conv(self):
        """APT's trainable ZeroMLP (paper arXiv 2510.18091 Eq. 2, self.zero_conv).

        Same ownership pattern as get_apt_patch_attn() above. Zero-init (weight
        AND bias): a coarse token starts equal to E(Resize_p(P_i)) + pos, i.e.
        enabling APT is a no-op at the seam until fine-tuning learns to fold in
        sub-patch detail via patch_attn.
        """
        if getattr(self, "apt_zero_conv", None) is None:
            vt = self.get_vision_tower()
            dim = getattr(vt, "hidden_size", None) or self.config.mm_hidden_size
            lin = nn.Linear(dim, dim)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.apt_zero_conv = lin.to(device=vt.device, dtype=vt.dtype)
        return self.apt_zero_conv

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        
        # __init__ already builds self.sae, and from_pretrained fills it with the
        # checkpoint's trained weights. Rebuilding it here would silently discard
        # them -- initialize_vision_modules runs *after* from_pretrained. Only
        # construct when genuinely absent (pretrain from a bare LLM).
        if getattr(self, "sae", None) is None:
            self.sae = SiglipAE()
        ##############################################################################
#         self.vision_select=nn.Parameter(
#                 torch.randn((30, self.config.hidden_size), dtype=self.dtype)
#         )
        
#         #self.text_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
#         self.text_select_model = T5EncoderModel.from_pretrained('google-t5/t5-small')
        
#         self.text_mlp=nn.Sequential(
#             nn.Linear(512,self.config.hidden_size),
#             nn.GELU(),
#             # nn.Linear(config.hidden_size,config.hidden_size),
#             # nn.GELU(),
#         )
        ##############################################################################
        
        
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
            
#             self.vision_select.data = mm_projector_weights["model.vision_select"]
            
#             self.text_mlp.load_state_dict(get_w(mm_projector_weights, "text_mlp"))
            
#             self.text_select_model.load_state_dict(get_w(mm_projector_weights, "text_select_model"),strict=False)
            #self.vision_tower.load_state_dict(get_w(mm_projector_weights, "vision_tower"),strict=False)

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def rotary_position_embedding(q):
    seq_len, dim = q.shape

    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)

    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(1000000.0) / dim)).to(q.device)
    
    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    
    q_rotated = q * cos_emb + q_alternate * sin_emb

    return q_rotated

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        #image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    def add_image(self, image_features):
        return torch.repeat_interleave(image_features, repeats=4, dim=0)
    
    def add_video(self, video_features):
        if video_features.size(0)<4:
            last_feature = video_features[-1:]

            repeated_features = last_feature.repeat(4 - video_features.size(0), 1,1,1)
            expanded_x = torch.cat([video_features, repeated_features], dim=0)
            return expanded_x
        
        repeat_counts = torch.ones(video_features.size(0), dtype=torch.long, device=video_features.device)

        sum_counts=torch.sum(repeat_counts)
        if sum_counts % 4!=0:
            padding_size = 4 - (sum_counts % 4)
            random_indices = torch.randperm(repeat_counts.size(0))[:padding_size].to(video_features.device)
            repeat_counts[random_indices] += 1 
            
        expanded_x = torch.repeat_interleave(video_features, repeat_counts, dim=0)

        return expanded_x

    def _get_rlt_module(self):
        """Lazily build + cache the SigLIP-RLT embedding module.

        Stored via __dict__ (not nn.Module's registry) so it does not re-register
        the shared vision tower as a duplicate submodule.
        """
        rlt = self.__dict__.get("_rlt_module", None)
        if rlt is None:
            from .multimodal_encoder.siglip_rlt_embeddings import SiglipRLTEmbeddings
            vt = self.get_model().get_vision_tower()
            rlt = SiglipRLTEmbeddings(
                vt.vision_tower,
                threshold=getattr(self.config, "rlt_threshold", 0.2),
                patch_size=getattr(self.config, "rlt_patch_size", 14),
                max_frames=getattr(self.config, "rlt_max_frames", 512),
                temporal_pos_scale=getattr(self.config, "rlt_temporal_pos_scale", 1.0),
                attn_mode=getattr(self.config, "rlt_attn_mode", "reuse"),
                mask_mode=getattr(self.config, "rlt_mask_mode", "ref"),
                refresh_every=getattr(self.config, "rlt_refresh_every", 0),
                # Where the drop test compares patches. "embed" is NOT plumbed into
                # APT-Temporal, which shares rlt_threshold/rlt_mask_mode but tests
                # sub-tiles of raw pixels -- hence its own knob rather than a third
                # rlt_mask_mode value that would be meaningless over there.
                mask_space=getattr(self.config, "rlt_mask_space", "pixel"),
                embed_threshold=getattr(self.config, "rlt_embed_threshold", 0.34),
                embed_metric=getattr(self.config, "rlt_embed_metric", "l2"),
            )
            # NB: no blanket .to(device/dtype) here. The module aliases the
            # already-placed vision tower; temporal_pos is placed on-device inside
            # forward(). RLT adds no learnable state of its own.
            self.__dict__["_rlt_module"] = rlt
        return rlt

    def encode_multimodals_rlt(self, videos_or_images, video_idx_in_batch, split_sizes):
        """RLT encoder path: per-clip pre-encoder token dropping -> projector.

        Returns a list with one (N, hidden) tensor per clip, bypassing the grid
        pipeline (interpolate / add_video / SAE / 2D pool). Gated by config.use_rlt.
        """
        rlt = self._get_rlt_module()
        if split_sizes is None:
            split_sizes = [videos_or_images.shape[0]]
        per_clip_frames = torch.split(videos_or_images, split_sizes, dim=0)
        features = []
        total_keep = total_dense = 0
        for i, frames in enumerate(per_clip_frames):
            dense, mask_flat = rlt(frames)                         # (T, P, C), (T*P,)
            survivors = dense.reshape(-1, dense.shape[-1])[mask_flat]   # (N, C)
            n_keep, n_dense = int(mask_flat.sum()), int(mask_flat.numel())
            total_keep += n_keep
            total_dense += n_dense
            rank0_print(
                f"[RLT] video={i} frames={frames.shape[0]} "
                f"space={rlt.mask_space} threshold={rlt.active_threshold} "
                f"survivors={n_keep}/{n_dense} ({100.0*n_keep/n_dense:.1f}% kept) "
                f"-> LLM visual tokens={n_keep}"
            )
            survivors = self.get_model().mm_projector(survivors)   # (N, hidden)
            features.append(survivors)
        rank0_print(
            f"[RLT] TOTAL survivors={total_keep}/{total_dense} "
            f"({100.0*total_keep/total_dense:.1f}% kept) -> {total_keep} LLM visual tokens"
        )
        # Accumulate run-wide totals so the eval harness can print a grand summary.
        self._rlt_grand_keep = getattr(self, "_rlt_grand_keep", 0) + total_keep
        self._rlt_grand_dense = getattr(self, "_rlt_grand_dense", 0) + total_dense
        self._rlt_grand_videos = getattr(self, "_rlt_grand_videos", 0) + len(per_clip_frames)
        return features

    def encode_clips_rlt_dense(self, videos_or_images, split_sizes):
        """RLT-encode each clip cheaply, then forward-fill to a dense grid.

        Returns a tuple of (T_i, P, C) tensors -- one per clip -- so the standard
        dense pipeline (interpolate -> add_video -> SAE/DTS -> pool) in
        encode_multimodals runs unchanged. Survivors are NOT projected here;
        projection happens later in that shared loop.
        """
        rlt = self._get_rlt_module()
        if split_sizes is None:
            split_sizes = [videos_or_images.shape[0]]
        per_clip_frames = torch.split(videos_or_images, split_sizes, dim=0)

        dense_clips = []
        total_keep = total_dense = 0
        for i, frames in enumerate(per_clip_frames):
            # The RLT module already returns the forward-filled dense grid: under
            # attn_mode="reuse" the carries ARE the encoder state, so there is nothing
            # left to scatter back here.
            dense, mask_flat = rlt(frames)                                 # (T, P, C)
            n_keep, n_dense = int(mask_flat.sum()), int(mask_flat.numel())
            total_keep += n_keep
            total_dense += n_dense
            dense_clips.append(dense)
        rank0_print(
            f"[RLT+DTS] mode={rlt.attn_mode} space={rlt.mask_space} "
            f"threshold={rlt.active_threshold} "
            f"encoder tokens kept={total_keep}/{total_dense} "
            f"({100.0 * total_keep / max(total_dense, 1):.1f}% kept); DTS compresses the dense grid next."
        )
        # Keep the grand-total counters live so the eval harness summary still works.
        self._rlt_grand_keep = getattr(self, "_rlt_grand_keep", 0) + total_keep
        self._rlt_grand_dense = getattr(self, "_rlt_grand_dense", 0) + total_dense
        self._rlt_grand_videos = getattr(self, "_rlt_grand_videos", 0) + len(per_clip_frames)
        return tuple(dense_clips)

    def _get_apt_module(self):
        """Lazily build + cache the SigLIP-APT embedding module.

        Stored via __dict__ (not nn.Module's registry) so it does not re-register
        the shared vision tower as a duplicate submodule. patch_attn/zero_conv are
        passed in from the base model's own get_apt_patch_attn()/get_apt_zero_conv()
        so they land in named_parameters() -> optimizer + checkpoint instead of being
        orphaned inside this unregistered wrapper.
        """
        apt = self.__dict__.get("_apt_module", None)
        if apt is None:
            from .multimodal_encoder.siglip_apt_embeddings import SiglipAPTEmbeddings
            vt = self.get_model().get_vision_tower()
            apt = SiglipAPTEmbeddings(
                vt.vision_tower,
                thresholds=getattr(self.config, "apt_thresholds", [4.0, 6.0]),
                num_scales=getattr(self.config, "apt_num_scales", 3),
                base_patch_size=getattr(self.config, "apt_base_patch_size", 14),
                image_size=getattr(self.config, "apt_input_res", 392),
                patch_attn=self.get_model().get_apt_patch_attn(),
                zero_conv=self.get_model().get_apt_zero_conv(),
            ).to(device=vt.device, dtype=vt.dtype)
            self.__dict__["_apt_module"] = apt
        return apt

    def encode_clips_apt_dense(self, videos_or_images, split_sizes):
        """APT-encode each clip cheaply, then scatter survivors back to a dense grid.

        APT re-embeds each frame with a hierarchical (quadtree) patchifier so the
        SigLIP encoder runs over a reduced, content-adaptive token set. Its masks
        form a strict spatial partition, so each surviving (possibly coarse) token
        is broadcast over the base cells it covers to rebuild a dense (T, P, C)
        grid. Returns a tuple of (T_i, P, C) tensors -- one per clip -- so the
        standard dense pipeline (interpolate -> add_video -> SAE/DTS -> pool) in
        encode_multimodals runs unchanged. Survivors are NOT projected here.
        """
        from .multimodal_encoder.siglip_apt_embeddings import apt_scatter_back
        apt = self._get_apt_module()
        if split_sizes is None:
            split_sizes = [videos_or_images.shape[0]]
        per_clip_frames = torch.split(videos_or_images, split_sizes, dim=0)

        dense_clips = []
        total_keep = total_dense = 0
        for i, frames in enumerate(per_clip_frames):
            survivors, output_mask, masks, T, P = apt(frames)
            n_keep, n_dense = int(output_mask.numel()), int(T * P)
            total_keep += n_keep
            total_dense += n_dense
            dense_clips.append(
                apt_scatter_back(survivors, output_mask, masks, apt.base_patch_size, apt.image_size)
            )
        # Keep grand-total counters live so the eval harness summary still works.
        self._apt_grand_keep = getattr(self, "_apt_grand_keep", 0) + total_keep
        self._apt_grand_dense = getattr(self, "_apt_grand_dense", 0) + total_dense
        self._apt_grand_videos = getattr(self, "_apt_grand_videos", 0) + len(per_clip_frames)
        return tuple(dense_clips)

    def _get_apt_temporal_module(self):
        """Lazily build + cache the APT-Temporal embedding module.

        Reuses the SAME lazily-built/cached SiglipAPTEmbeddings wrapper
        (_get_apt_module) rather than constructing a second one -- APT-Temporal
        composes plain APT, it does not duplicate it. Stored via __dict__ (not
        nn.Module's registry) so it does not re-register the shared vision
        tower / APT wrapper as a duplicate submodule, same convention as
        _get_rlt_module / _get_apt_module.

        Deliberately reuses config.rlt_threshold (not a separate
        apt_temporal_* field) for the dirty-tile check: APT-Temporal's
        SiglipAPTTemporalEmbeddings operates on the same SigLIP-normalized
        pixel scale RLT itself does, so it's the literal same knob rather than
        a second, differently-scaled threshold for the same underlying
        question ("did this patch change vs. the previous frame?"). Same
        reasoning for config.rlt_mask_mode / config.rlt_refresh_every: the
        drift-bounded "ref" dirty-check fix applies identically to a TAPT
        REDUNDANT carry chain as to an RLT drop run (see dirty_subtile_mask's
        docstring), so it shares RLT's knob and RLT's "ref" default rather
        than a second copy of the same fix.
        """
        tapt = self.__dict__.get("_apt_temporal_module", None)
        if tapt is None:
            from .multimodal_encoder.siglip_apt_temporal_embeddings import SiglipAPTTemporalEmbeddings
            apt = self._get_apt_module()
            tapt = SiglipAPTTemporalEmbeddings(
                apt,
                threshold=getattr(self.config, "rlt_threshold", 0.2),
                # Shares RLT's knob: the starvation bug and its fix are the same in both
                # (events attend over their frame's full partition, not just each other).
                attn_mode=getattr(self.config, "rlt_attn_mode", "reuse"),
                mask_mode=getattr(self.config, "rlt_mask_mode", "ref"),
                refresh_every=getattr(self.config, "rlt_refresh_every", 0),
            )
            self.__dict__["_apt_temporal_module"] = tapt
        return tapt

    def encode_clips_apt_temporal_dense(self, videos_or_images, split_sizes):
        """APT-Temporal-encode each clip, then scatter back to a dense grid.

        Spatial redundancy is resolved first (each frame gets its own
        independent entropy-driven partition, exactly like plain APT).
        Temporal redundancy is resolved second: per base cell, frame t is
        carried forward (REDUNDANT, emitting no token) iff its shape matches
        frame t-1's partition AND nothing inside it changed; otherwise it is
        FRESH and re-embedded at frame t's own scale. See
        apt_temporal_static_tokens.py for the full rationale.

        Returns a tuple of (T_i, P, C) tensors -- one per clip -- so the
        standard dense pipeline (interpolate -> add_video -> SAE/DTS -> pool)
        in encode_multimodals runs unchanged. Survivors are NOT projected here.

        Also logs the classification diagnostics, of which `missed_reuse` is the
        one to watch: base cells that did not change AT ALL but still had to pay
        for a full token because the quadtree boundaries wobbled across an entropy
        threshold between frames. That is pure loss and it is the ceiling on TAPT's
        savings -- a high value points at PARTITION INSTABILITY (fix with threshold
        hysteresis or windowed re-partitioning), not at anything in the classifier.
        """
        from .multimodal_encoder.siglip_apt_temporal_embeddings import apt_temporal_scatter_back
        tapt = self._get_apt_temporal_module()
        if split_sizes is None:
            split_sizes = [videos_or_images.shape[0]]
        per_clip_frames = torch.split(videos_or_images, split_sizes, dim=0)

        dense_clips = []
        total_keep = total_dense = 0
        clip_stats = []
        for i, frames in enumerate(per_clip_frames):
            survivors, origin_index, T, P = tapt(frames)
            n_keep, n_dense = int(survivors.shape[0]), int(T * P)
            total_keep += n_keep
            total_dense += n_dense
            if tapt.last_stats:
                clip_stats.append(tapt.last_stats)
            dense_clips.append(apt_temporal_scatter_back(survivors, origin_index, T, P))
        # Keep grand-total counters live so the eval harness summary still works.
        self._apt_temporal_grand_keep = getattr(self, "_apt_temporal_grand_keep", 0) + total_keep
        self._apt_temporal_grand_dense = getattr(self, "_apt_temporal_grand_dense", 0) + total_dense
        self._apt_temporal_grand_videos = getattr(self, "_apt_temporal_grand_videos", 0) + len(per_clip_frames)
        if clip_stats:
            # Running totals only -- no per-forward logging, matching encode_clips_apt_dense
            # (the RLT paths still log per forward). These fired on EVERY step, three lines
            # per forward, which buries a multi-thousand-step training log. Rates stay
            # recoverable:
            # the eval harness reads _apt_temporal_stat_sum / _apt_temporal_stat_n and the
            # grand-total counters above (lmms_eval/models/simple/videoxlpro.py,
            # scripts/profile_encoder.py) and reports them once at the end, over many clips
            # rather than one clip's noise.
            keys = clip_stats[0].keys()
            prev = getattr(self, "_apt_temporal_stat_sum", {})
            self._apt_temporal_stat_sum = {
                k: prev.get(k, 0.0) + sum(s[k] for s in clip_stats) for k in keys
            }
            self._apt_temporal_stat_n = getattr(self, "_apt_temporal_stat_n", 0) + len(clip_stats)
        return tuple(dense_clips)

    @staticmethod
    def _dist_ready():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @classmethod
    def _sync_max_int(cls, value, device):
        """Global max of a per-rank integer. Identity when not distributed.

        Used to align DATA-dependent call counts of ZeRO-3-partitioned modules across
        ranks -- see the note in encode_multimodals.
        """
        if not cls._dist_ready():
            return int(value)
        t = torch.tensor([int(value)], device=device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return int(t.item())

    @classmethod
    def _check_rank_invariant(cls, value, what, device):
        """Raise loudly if `value` is not identical on every rank.

        A per-rank difference in how many times a partitioned module is called is not a
        recoverable condition -- it desyncs the ZeRO-3 all-gather sequence and the job
        deadlocks until ddp_timeout with an uninformative ALLGATHER_BASE timeout. Better
        to die immediately, naming the rank and the value, than to hang for hours.
        """
        if not cls._dist_ready():
            return int(value)
        rank = torch.distributed.get_rank()
        t = torch.tensor([int(value), -int(value)], device=device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        vmax, vmin = int(t[0].item()), -int(t[1].item())
        if vmax != vmin:
            msg = (
                f"[ZeRO-3 desync guard] rank {rank}: {what} is not rank-invariant "
                f"(this rank={int(value)}, global min={vmin}, max={vmax}). Each rank would "
                f"issue a different number of all-gathers, desyncing the collective sequence "
                f"and deadlocking the job until ddp_timeout. Failing fast instead."
            )
            print(msg, flush=True)
            raise RuntimeError(msg)
        return int(value)

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        #################################################################################
        # if videos_or_images.shape[0] > 360:
        #     random_indices = np.random.choice(videos_or_images.shape[0], size=360, replace=False)
        #     videos_or_images = videos_or_images[random_indices]
        #     split_sizes=videos_or_images.shape[0]
            
        #################################################################################
        # RLT encoder path. RLT makes the SigLIP encoder cheap by running attention
        # only over non-redundant survivor tokens. Two modes:
        #   use_rlt + use_sae  : scatter survivors back to a dense grid so DTS (SAE)
        #                        still does its semantic temporal compression (composed).
        #   use_rlt, no use_sae: legacy ragged RLT-only output (no DTS), kept for A/B.
        use_rlt = getattr(self.config, "use_rlt", False)
        use_apt = getattr(self.config, "use_apt", False)
        use_apt_temporal = getattr(self.config, "use_apt_temporal", False)
        use_sae = getattr(self.config, "use_sae", True)
        # RLT (temporal), APT (spatial), and APT-Temporal (spatial-then-temporal
        # combined) are separate, mutually-exclusive paths so each method's gain
        # can be measured independently.
        assert sum([use_rlt, use_apt, use_apt_temporal]) <= 1, (
            "use_rlt, use_apt, and use_apt_temporal are mutually exclusive; enable one at a time"
        )

        # Clip count decides how many times the encoders below -- and the sae/mm_projector
        # loop further down -- call ZeRO-3-partitioned modules, so it must be checked
        # BEFORE the first of those calls, not after. Checking it later cannot work: the
        # encoders would already have desynced the collective sequence, and this check's
        # own all-reduce would then be just another mismatched op in a jammed group
        # (which is precisely how rank 1 came to hang inside it).
        self._check_rank_invariant(
            len(split_sizes) if split_sizes is not None else 1,
            "clip count (len(split_sizes))",
            videos_or_images.device,
        )

        if use_rlt and not use_sae:
            return self.encode_multimodals_rlt(videos_or_images, video_idx_in_batch, split_sizes)

        if use_rlt:
            # Cheap encoder over survivors -> forward-fill to dense (T, P, C) per clip,
            # then fall through to the shared interpolate/add_video/SAE/pool loop below.
            per_videos_or_images_features = self.encode_clips_rlt_dense(videos_or_images, split_sizes)
        elif use_apt:
            # APT: content-adaptive hierarchical re-embedding makes the SigLIP
            # encoder cheap; scatter survivors back to a dense (T, P, C) grid, then
            # fall through to the shared interpolate/add_video/SAE/pool loop below.
            per_videos_or_images_features = self.encode_clips_apt_dense(videos_or_images, split_sizes)
        elif use_apt_temporal:
            # APT-Temporal: per-frame spatial partition (like APT) THEN temporal
            # redundancy collapsing across frames (like RLT) on top of it; scatter
            # back to a dense (T, P, C) grid, then fall through to the shared
            # interpolate/add_video/SAE/pool loop below.
            per_videos_or_images_features = self.encode_clips_apt_temporal_dense(videos_or_images, split_sizes)
        else:
            # Define the maximum batch size (1024 frames)
            max_batch_size = 60
            num_frames = videos_or_images.shape[0]
            # Initialize a list to store the features from each batch
            videos_or_images_features = []

            # Split videos_or_images into smaller batches if num_frames > max_batch_size
            if num_frames > max_batch_size:
                # Calculate the number of batches needed
                num_batches = (num_frames + max_batch_size - 1) // max_batch_size
                for i in range(num_batches):
                    start_idx = i * max_batch_size
                    end_idx = min((i + 1) * max_batch_size, num_frames)

                    # Process each batch separately
                    batch_videos_or_images = videos_or_images[start_idx:end_idx]
                    batch_features = self.get_model().get_vision_tower()(batch_videos_or_images)
                    videos_or_images_features.append(batch_features)

                # Concatenate the features of all batches
                videos_or_images_features = torch.cat(videos_or_images_features, dim=0)
            else:
                videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)

            per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []

        # ---- ZeRO-3 collective-sequence invariance --------------------------------
        # sae and mm_projector are ZeRO-3-PARTITIONED modules: every call to one issues
        # an all-gather of its shards. DeepSpeed requires all ranks to issue the same
        # all-gathers in the same order; a rank that calls a partitioned module a
        # different number of times desyncs the sequence, and the job then deadlocks
        # until ddp_timeout with a bare ALLGATHER_BASE watchdog timeout naming neither
        # the rank nor the cause (observed on job 351066, hanging at step ~3204 -- and
        # deterministically at the SAME step on rerun, because it is the fixed shuffle
        # order that decides which sample lands on which rank).
        #
        # Two counts here are DATA-dependent, and both must be pinned:
        #   * the loop below runs once per clip and calls sae + mm_projector each pass,
        #   * the sae chunk loop ran once for bc//4 <= 24 but ceil((bc//4)/24) times
        #     above it -- so an image whose anyres tiling yields >24 crops (the
        #     image_grid_pinpoints here reach 49 tiles, and process_images ignores the
        #     "_4" in anyres_max_4, so nothing caps the crop count) issues EXTRA
        #     all-gathers versus a peer rank holding an ordinary image or a 32-frame
        #     video (both of which land at bc//4 <= 24 -> exactly one call).
        #
        # The clip count is checked above, before the encoders run (a mismatch is
        # unrecoverable -- fail fast and name the rank rather than hang for hours); the
        # chunk count is padded up to the global max with dummy calls, the same fix
        # already applied to patch_attn in siglip_apt_embeddings._embed.
        _dev = videos_or_images.device

        for idx, feat in enumerate(per_videos_or_images_features):
            #print(feat.shape,end='1\n')
            feat=self.interpolate(feat)
            #######################################################
            if idx in video_idx_in_batch:
                feat=self.add_video(feat)
            else:
                feat=self.add_image(feat)

            bc,ch,h,w=feat.shape

            feat = feat.view(bc//4,ch,4,h,w)
            if getattr(self.config, 'use_sae', True):
                # torch.split gives a single chunk when bc//4 <= 24, so this one loop
                # reproduces both of the old branches exactly -- minus their divergent
                # call counts.
                chunk_size = 24
                chunks = list(torch.split(feat, chunk_size, dim=0))
                n_chunks = self._sync_max_int(len(chunks), _dev)
                interpolated_chunks = []
                dummy_acc = None
                for i in range(n_chunks):
                    if i < len(chunks):
                        interpolated_chunks.append(self.get_model().sae(chunks[i]).squeeze(2))
                    else:
                        # Padding call: exists solely so this rank's all-gather count
                        # matches the peer that had more chunks.
                        d = self.get_model().sae(chunks[0][:1]).squeeze(2)
                        dummy_acc = d if dummy_acc is None else dummy_acc + d
                feat = torch.cat(interpolated_chunks, dim=0) if len(interpolated_chunks) > 1 else interpolated_chunks[0]
                if dummy_acc is not None:
                    # Fold the padding call into the graph with weight 0: numerically a
                    # no-op, but it keeps sae reachable from the loss on this rank. Without
                    # it, a forward with no corresponding backward would leave sae's grad
                    # hooks unfired whenever sae is TRAINABLE (mm_tunable_parts=
                    # mm_temporal_compressor), desyncing the gradient reduce-scatter and
                    # reintroducing the very deadlock this block exists to prevent.
                    feat = feat + 0.0 * dummy_acc.sum()
                del interpolated_chunks
                del dummy_acc
                del chunks
            else:
                feat = feat.view(bc, ch, h, w)

            feat = feat.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            #print(feat.shape,end='3\n')
            feat = self.get_model().mm_projector(feat)
            #print(feat.shape,end='4\n')
            # Post pooling
            if idx in video_idx_in_batch:
                #print('************************',idx,video_idx_in_batch)
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
            
        del per_videos_or_images_features
        return all_videos_or_images_features
    ########################################################
    def interpolate(self,image_features):
        b, num_tokens, dim = image_features.shape
        
        #print(str(image_features.shape)+' i\n')
        
        target_h = target_w = int(576**0.5)
        h = w = int(num_tokens**0.5)

        image_features = image_features.view(b, h, w, dim)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        chunk_size = 24
        chunks = torch.split(image_features, chunk_size, dim=0)
        interpolated_chunks = []
        for chunk in chunks:
            interpolated_chunk = F.interpolate(
                chunk.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(chunk.dtype)
            interpolated_chunks.append(interpolated_chunk)
        image_features = torch.cat(interpolated_chunks, dim=0)
        del interpolated_chunks
        
        del chunks

        return image_features

    def _videoxl_select_tokens(self, image_features, input_ids, time_embedding):
        """Original Video-XL-Pro time-embedding + text-relevance token selection.
        Skipped when config.use_rlt is set (RLT replaces this compression stage)."""
        video_token_indices=[]
        for image_idx, image_feature in enumerate(image_features):
            if time_embedding[image_idx] is not None:
                mask = (time_embedding[image_idx] == 151654)
                indices = torch.nonzero(mask).squeeze()
                #print(image_features[image_idx].shape,len(time_embedding[image_idx]))
                embed_token=self.get_model().embed_tokens(time_embedding[image_idx])
                embed_token[indices]=image_features[image_idx]
                
                video_token_indices.append(indices)

                image_features[image_idx]=embed_token
            else:
                # language-only sample (dummy "text" image, time_embedding is None):
                # keep video_token_indices index-aligned with the batch so the
                # per-sample lookups below don't run off the end.
                video_token_indices.append(None)

        token_score_features=[]
        for text_ids in range(len(input_ids)):
            #######################################################################
  
            text_per=input_ids[text_ids]

            num_images = (text_per == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = (
                [-1]
                + torch.where(text_per == IMAGE_TOKEN_INDEX)[0].tolist()
                + [text_per.shape[0]]
            )

            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    text_per[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                
            del text_per
            cur_input_ids_noim=torch.cat(cur_input_ids_noim)
            outputs_text_select  = self.get_model().embed_tokens(cur_input_ids_noim)
            
            outputs_text_select=self.get_model().text_mlp(outputs_text_select)

            t_sum,chan_sum=image_features[text_ids].shape
            image_feature_per=image_features[text_ids]
            
            if time_embedding[text_ids] is not None:
                #print(image_feature_per.shape,end='1\n')
                image_feature_per=self.get_model().WindowTimeToTokenAttention(image_feature_per)
                #print(image_feature_per.shape,end='2\n')
                feats=image_feature_per[video_token_indices[text_ids]]
            else:
                # language-only sample: no video tokens to slice, so score the
                # whole (dummy) feature map. Matches the time_embedding-is-None
                # branch in the selection loop below.
                feats=image_feature_per
            #print(image_feature_per.shape)


            select_mat=torch.matmul(feats,outputs_text_select.transpose(0, 1)).mean(dim=-1)
            
            # if time_embedding[text_ids] is not None:
            #     select_mat=select_mat[video_token_indices[text_ids]]
            
            #######################################################################
            min_val = torch.min(select_mat)
            max_val = torch.max(select_mat)
            select_mat = (select_mat - min_val) / (max_val - min_val)
            #######################################################################

            token_score_features.append(select_mat)
            
        for image_ind in range(len(image_features)):
            typ=image_features[image_ind].dtype
            image_features[image_ind]=rotary_position_embedding(image_features[image_ind]).to(typ)

        new_input_embeds = []
        
        drop_rate=random.uniform(0.7, 0.99)
        for image_idx in range(len(image_features)):
            if time_embedding[image_idx] is not None:
                image_per=image_features[image_idx]
                image_per[video_token_indices[image_idx]]+=token_score_features[image_idx].unsqueeze(1)

                t_sum=len(video_token_indices[image_idx])
                #print(t_sum)
                save_time_sum=int(t_sum*drop_rate)

                save_time_sum=save_time_sum-save_time_sum % 16


                _, top_indices = torch.topk(token_score_features[image_idx],save_time_sum)
                top_indices=torch.tensor(sorted(top_indices))
                top_indices=video_token_indices[image_idx][top_indices]
                
                all_indices = torch.arange(len(image_features[image_idx])).to(image_per.device)
                text_indices = all_indices[~torch.isin(all_indices,video_token_indices[image_idx])]
                
                select_token_indices = torch.cat((top_indices, text_indices))  # 合并
                select_token_indices, _ = torch.sort(select_token_indices)  # 排序

                image_per=image_per[select_token_indices]
                new_input_embeds.append(image_per)
            else:
                image_per=image_features[image_idx]+token_score_features[image_idx].unsqueeze(1)

                t_sum,chan_sum=image_per.shape

                save_time_sum=int(t_sum*drop_rate)

                save_time_sum=save_time_sum-save_time_sum % 16


                _, top_indices = torch.topk(token_score_features[image_idx],save_time_sum)
                top_indices=torch.tensor(sorted(top_indices))

                image_per=image_per[top_indices]

                new_input_embeds.append(image_per)
            
        image_features=new_input_embeds
        return image_features

    def _rlt_query_select_tokens(self, image_features, input_ids):
        """Query-aware token selection for RLT's variable-length survivors.

        Mirrors the core of _videoxl_select_tokens but skips the time_embedding
        slot injection and WindowTimeToTokenAttention, which both require a fixed
        144-token grid that RLT's ragged output cannot satisfy.

        What is preserved:
          - question text projected via text_mlp into visual feature space
          - per-token relevance score via matmul + mean over question tokens
          - score normalised to [0,1] and added to the token embedding
          - rotary position embedding applied before selection
          - topk with random drop_rate in [0.70, 0.99], rounded to multiple of 16
        """
        keep_rate = random.uniform(0.7, 0.99)   # fraction of tokens to KEEP
        new_image_features = []

        for image_idx in range(len(image_features)):
            image_feature = image_features[image_idx]   # (N, hidden)

            # extract question tokens — strip IMAGE_TOKEN_INDEX positions
            text_per = input_ids[image_idx]
            image_token_positions = (
                [-1]
                + torch.where(text_per == IMAGE_TOKEN_INDEX)[0].tolist()
                + [text_per.shape[0]]
            )
            cur_input_ids_noim = []
            for i in range(len(image_token_positions) - 1):
                cur_input_ids_noim.append(
                    text_per[image_token_positions[i] + 1 : image_token_positions[i + 1]]
                )
            cur_input_ids_noim = torch.cat(cur_input_ids_noim)

            # project question into visual feature space (same text_mlp as original)
            outputs_text = self.get_model().embed_tokens(cur_input_ids_noim)   # (Q_len, hidden)
            outputs_text = self.get_model().text_mlp(outputs_text)             # (Q_len, hidden)

            # score each visual token: avg similarity to all question tokens
            select_mat = torch.matmul(
                image_feature, outputs_text.transpose(0, 1)
            ).mean(dim=-1)                                                      # (N,)
            min_val, max_val = select_mat.min(), select_mat.max()
            if max_val > min_val:
                select_mat = (select_mat - min_val) / (max_val - min_val)

            # rotary position embedding (same as original pipeline)
            typ = image_feature.dtype
            image_feature = rotary_position_embedding(image_feature).to(typ)

            # add relevance score to embedding so LLM sees the signal
            image_feature = image_feature + select_mat.unsqueeze(1)

            # topk, aligned to multiple of 16
            N = image_feature.shape[0]
            save_sum = int(N * keep_rate)
            save_sum = max(16, save_sum - save_sum % 16)
            save_sum = min(save_sum, N)

            _, top_indices = torch.topk(select_mat, save_sum)
            top_indices = torch.tensor(
                sorted(top_indices.tolist()), device=image_feature.device
            )

            kept = image_feature[top_indices]
            rank0_print(
                f"[RLT+Q] video={image_idx} query-aware: {N} -> {kept.shape[0]} tokens "
                f"({100.0*kept.shape[0]/N:.1f}% kept, keep_rate={keep_rate:.2f})"
            )
            new_image_features.append(kept)

        return new_image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None,time_embedding=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))
            #print(len(images_list),images_list[0].shape)

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)    #16,144,3584

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if getattr(self.config, "use_rlt", False) and not getattr(self.config, "use_sae", True):
                # Legacy ragged RLT-only: features are already final (N, hidden) per
                # clip; "flat" is a no-op on 2D features so they pass straight through.
                # Composed RLT+SAE produces standard dense features and uses the
                # configured merge type, exactly like the non-RLT path.
                mm_patch_merge_type = "flat"

            visual_drop_score=[]
            new_image_features=[]
            
            if mm_patch_merge_type == "flat":
                
                if image_features[0].ndim>2:
                    image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type== "unires":
                #print('unires')
                for image_idx, image_feature in enumerate(image_features):
                    # rank0_print(f"Initial feature size : {image_feature.shape}")
                    if image_idx in video_idx_in_batch:  # video operations
                        #print(image_feature.shape)
                        image_feature = image_feature.flatten(0, 1)
                        
                    elif image_feature.shape[0] > 1:
                        # base image feature is never used in unires
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                        kernel_size = 2
                        image_feature = image_feature.view(image_feature.shape[0], height, width, -1) # [4, 24, 24, 4096]
                        
                        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                        image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                        image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                        image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                    
                        image_feature = image_feature.flatten(0, 1)
                        
                    else:
                        
                        image_feature = image_feature[0]
                        
                    new_image_features.append(image_feature)
                
                image_features = new_image_features
                
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    if image_idx in video_idx_in_batch:  # video operations
                        if "unpad" in mm_patch_merge_type:
                            # image_feature = image_feature.permute(2, 0, 1).contiguous()
                            # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            # image_feature = image_feature.permute(1, 2, 0).contiguous()
                            image_feature = image_feature.flatten(0, 1)
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the image or video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
                model.generate(input_ids, images=[image_tensor],  modalities=["image"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
                model.generate(input_ids, images=image_tensor,  modalities=["image"], **gen_kwargs)
            """
            raise ValueError(error_message)

        #print(time_embedding[0].shape)
        # Composed RLT+SAE behaves like the standard path here (its features carry the
        # same per-group token layout the time-embedding expects). Only legacy ragged
        # RLT-only (use_rlt without use_sae) uses the query-aware token selection.
        if not (getattr(self.config, "use_rlt", False) and not getattr(self.config, "use_sae", True)):
            image_features = self._videoxl_select_tokens(image_features, input_ids, time_embedding)
        else:
            image_features = self._rlt_query_select_tokens(image_features, input_ids)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            #print(image_token_indices) #[-1, 14, 236]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            
            # print(cur_input_ids)
            # print(labels[batch_idx])
            
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            
           #print(torch.cat(cur_input_ids_noim).shape,torch.cat(cur_input_ids_noim))
            
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
    

   

