import logging
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module



warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from videoxlpro.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videoxlpro.mm_utils import tokenizer_image_token
from videoxlpro.demo_utils import load_image_processor, process_video


@register_model("videoxlpro")
class VideoXLPro(lmms):
    def __init__(
        self,
        pretrained: str = "lmms-lab/Video-XL-Pro",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: Optional[str] = "flash_attention_2",
        device_map: Optional[str] = "cuda:0",
        use_cache: Optional[bool] = True,
        max_frames_num: Optional[int] = 128,
        use_sae: Optional[bool] = True,
        use_rlt: Optional[bool] = False,
        rlt_threshold: Optional[float] = 0.2,
        rlt_temporal_pos_scale: Optional[float] = 0.0,
        rlt_attn_mode: Optional[str] = "reuse",
        rlt_mask_mode: Optional[str] = "ref",
        rlt_refresh_every: Optional[int] = 0,
        rlt_mask_space: Optional[str] = "pixel",
        rlt_embed_threshold: Optional[float] = 0.34,
        rlt_embed_metric: Optional[str] = "l2",
        use_apt: Optional[bool] = False,
        apt_threshold: Optional[str] = "4.0:6.0",
        apt_thresholds: Optional[str] = None,
        apt_num_scales: Optional[int] = 3,
        apt_input_res: Optional[int] = 392,
        use_apt_temporal: Optional[bool] = False,
        apt_temporal_max_frames: Optional[int] = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.max_frames_num = max_frames_num
        self.use_cache = use_cache
        self.use_sae = use_sae

        
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], pretrained)

        self._model = model_class.from_pretrained(
            pretrained,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None or self._tokenizer.pad_token_id == self._tokenizer.eos_token_id:
            self._tokenizer.pad_token_id = 0
        self._image_processor = load_image_processor(self._model, self._tokenizer)
        self._config = self._model.config
        self._model.config.use_sae = use_sae
        self._model.config.use_rlt = use_rlt
        self._model.config.rlt_threshold = rlt_threshold
        # How loud the fixed temporal-position sinusoid RLT adds to survivors is.
        # Composed path (use_sae=True): keep 0.0 -- the SAE already supplies temporal
        # info, so the extra sinusoid only perturbs it. Ragged path (use_sae=False):
        # set >0 (e.g. 1.0) since nothing else provides temporal order.
        self._model.config.rlt_temporal_pos_scale = rlt_temporal_pos_scale
        # "reuse" = survivors attend over the full frame (dropped tokens supply carried
        # k/v); "per_frame" = legacy starved attention, kept for the ablation.
        self._model.config.rlt_attn_mode = rlt_attn_mode
        # "ref" = drop test compares against the frame a patch is actually reused from
        # (bounds drift); "consec" = legacy paper behaviour. refresh_every bounds staleness.
        self._model.config.rlt_mask_mode = rlt_mask_mode
        self._model.config.rlt_refresh_every = int(rlt_refresh_every)
        # WHERE the drop test compares patches (orthogonal to rlt_mask_mode's WHAT it
        # compares against). "pixel" = the paper's raw-pixel test; "embed" = distance
        # between patch embeddings, which the patch-embed conv has already denoised.
        # rlt_embed_threshold is its own scale, NOT interchangeable with rlt_threshold --
        # and like rlt_threshold it must match whatever training used, since it sets how
        # much reuse (hence how much distortion) the model is expected to compensate for.
        self._model.config.rlt_mask_space = rlt_mask_space
        self._model.config.rlt_embed_threshold = float(rlt_embed_threshold)
        self._model.config.rlt_embed_metric = rlt_embed_metric
        # APT (spatial adaptive patches) and APT-Temporal (APT + temporal
        # redundancy collapsing on top) -- mutually exclusive with use_rlt
        # and with each other, so each method's gain is measured independently.
        assert sum([use_rlt, use_apt, use_apt_temporal]) <= 1, (
            "use_rlt, use_apt, and use_apt_temporal are mutually exclusive; enable one at a time"
        )
        self._model.config.use_apt = use_apt
        # Thresholds: one per non-base scale (apt_num_scales - 1). Avoid commas --
        # lmms_eval splits --model_args on commas. Either key works, and each may
        # be a single value (broadcast to all non-base scales) or a COLON-separated
        # list, e.g.  apt_threshold=4.0  or  apt_threshold=4.0:5.0.
        # The default must stay in step with train.py's apt_thresholds ("4.0,6.0")
        # and llava_arch._get_apt_module's own fallback ([4.0, 6.0]): entropy grows
        # with patch area, so broadcasting one scalar across scales makes the
        # coarsest scale far stricter than intended and silently suppresses it.
        raw = apt_thresholds if apt_thresholds is not None else apt_threshold
        thresholds = [float(p) for p in str(raw).replace(",", ":").split(":") if p != ""]
        n_needed = int(apt_num_scales) - 1
        if len(thresholds) == 1:
            thresholds = thresholds * n_needed
        assert len(thresholds) == n_needed, (
            f"apt needs apt_num_scales-1 = {n_needed} thresholds, got {thresholds}"
        )
        self._model.config.apt_thresholds = thresholds
        self._model.config.apt_num_scales = apt_num_scales
        self._model.config.apt_input_res = apt_input_res

        # APT-Temporal (TAPT): reuses the same apt_thresholds/apt_num_scales/
        # apt_input_res parsed above for its underlying spatial partition, and
        # reuses rlt_threshold (set above) for its dirty-tile check -- both
        # operate on the same SigLIP-normalized pixel scale, so it's one
        # shared knob rather than a second, differently-scaled threshold.
        self._model.config.use_apt_temporal = use_apt_temporal
        self._model.config.apt_temporal_max_frames = apt_temporal_max_frames
        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "VideoXLPro does not support batched generation."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._world_size = 1
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._config.max_position_embeddings

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except Exception:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        return [j for i in input for j in i]

    def _build_prompt(self, context: str) -> str:
        return (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{DEFAULT_IMAGE_TOKEN}\n{context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.pop("until", None)
            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0.01)
            gen_kwargs.setdefault("top_p", 0.001)
            gen_kwargs.setdefault("do_sample", True)
            gen_kwargs.setdefault("num_beams", 1)
            gen_kwargs.setdefault("use_cache", self.use_cache)
            gen_kwargs.pop("image_aspect_ratio", None)

            visual = batched_visuals[0]
            context = batched_contexts[0]

            if not visual or not isinstance(visual[0], str):
                eval_logger.error("VideoXLPro only supports video inputs (str paths).")
                res.append("")
                pbar.update(1)
                continue

            video_path = visual[0]
            try:
                video_tensor, time_stamps = process_video(
                    video_path, self._tokenizer, self._image_processor, self.device, self.max_frames_num, use_sae=self.use_sae
                )
            except Exception as e:
                eval_logger.error(f"Error loading video {video_path}: {e}")
                res.append("")
                pbar.update(1)
                continue

            prompt = self._build_prompt(context)
            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=[video_tensor],
                        time_embedding=time_stamps,
                        modalities=["video"],
                        **gen_kwargs,
                    )
                # The model's generate() passes inputs_embeds (not input_ids) to
                # super().generate(), so HF returns only the newly generated tokens.
                text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            except Exception as e:
                raise e

            text_outputs = [r.strip() for r in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()

        if getattr(self._model.config, "use_rlt", False):
            gk = getattr(self.model, "_rlt_grand_keep", 0)
            gd = getattr(self.model, "_rlt_grand_dense", 0)
            gv = getattr(self.model, "_rlt_grand_videos", 0)
            if gd > 0:
                print(
                    f"\n[RLT] ===== GRAND TOTAL =====\n"
                    f"[RLT] videos={gv}  tokens kept={gk}/{gd} ({100.0*gk/gd:.1f}%)\n"
                    f"[RLT] avg LLM visual tokens/video={gk//gv if gv else 0}\n"
                    f"[RLT] ==========================\n",
                    flush=True,
                )

        if getattr(self._model.config, "use_apt", False):
            gk = getattr(self.model, "_apt_grand_keep", 0)
            gd = getattr(self.model, "_apt_grand_dense", 0)
            gv = getattr(self.model, "_apt_grand_videos", 0)
            if gd > 0:
                print(
                    f"\n[APT] ===== GRAND TOTAL =====\n"
                    f"[APT] videos={gv}  encoder tokens kept={gk}/{gd} ({100.0*gk/gd:.1f}%)\n"
                    f"[APT] avg encoder survivors/video={gk//gv if gv else 0}\n"
                    f"[APT] ==========================\n",
                    flush=True,
                )

        if getattr(self._model.config, "use_apt_temporal", False):
            gk = getattr(self.model, "_apt_temporal_grand_keep", 0)
            gd = getattr(self.model, "_apt_temporal_grand_dense", 0)
            gv = getattr(self.model, "_apt_temporal_grand_videos", 0)
            if gd > 0:
                print(
                    f"\n[APT-Temporal] ===== GRAND TOTAL =====\n"
                    f"[APT-Temporal] videos={gv}  encoder tokens kept={gk}/{gd} ({100.0*gk/gd:.1f}%)\n"
                    f"[APT-Temporal] avg encoder survivors/video={gk//gv if gv else 0}\n"
                    f"[APT-Temporal] ==========================\n",
                    flush=True,
                )

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for VideoXLPro.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
