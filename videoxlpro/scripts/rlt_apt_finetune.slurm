#!/bin/bash
#SBATCH --job-name=videoxlpro
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/apt_videoxlpro-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=8

# RLT-FIRST composition (apt_temporal_partition_mode=survivor).
#
# The spatial partition is built from RLT SURVIVORSHIP, not from entropy: a coarse
# cell exists only where every base tile changed this frame AND their run lengths
# agree. So spatial merging only ever lands on tokens RLT could not drop, making
# its saving additive on top of temporal reuse instead of competing for the same
# redundancy. The entropy partition merges flat regions, which are
# disproportionately the ones RLT was going to drop anyway -- and a REDUNDANT cell
# emits zero tokens at ANY scale, so merging a still region buys nothing and costs
# fidelity.
#
# Trainable: apt_patch_attn + apt_zero_conv ONLY. mm_tunable_parts="none" freezes
# everything else (train.py:1889 takes the else branch, matches no unfreeze rule),
# and the APT block at train.py:1934 unfreezes the two merge modules regardless of
# mm_tunable_parts because it keys on use_apt_temporal alone. This matches the
# APT-only run, so the two are directly comparable.
#
# GRADIENT WARM-UP -- matters more here than with a wider tunable set. zero_conv is
# zero-initialised, and the merge is zero_conv(patch_attn(children)), so
# d(merged)/d(patch_attn_out) = zero_conv.weight = 0 at step 0. patch_attn therefore
# receives EXACTLY zero gradient until zero_conv leaves zero (verified: |grad
# patch_attn| = 0 at zero_conv=0, 138 at 1e-4, 1.1e4 at 1e-2). With
# mm_tunable_parts="none" that means the only thing learning on step 0 is zero_conv
# itself. This is the zero-init design working as intended, not a defect, but it
# means the first steps are a warm-up rather than real training of the merge.
#
# Watch apt/apt_zero_conv_weight_wnorm in wandb: it starts at exactly 0 and must
# climb off zero, otherwise NOTHING in this run is learning at all.
#
# There is MORE for it to learn here than under the entropy partition. Measured on
# MLVU, the zero-init anchor error ||E(Resize_p(cell)) - mean(E(children))|| is
# ~2.2x larger in survivor mode (0.138/0.162 at 2p/4p vs 0.063/0.073), because this
# partition merges moving, textured content where downsampling aliases badly, while
# entropy merged flat content where it is nearly lossless. Survivor mode also gives
# ~3.4x more merged cells per frame, and it populates the 2p scale that the entropy
# thresholds left essentially dead (1.3 -> 55 cells/frame), so patch_attn finally
# gets a large sample of its single-application 4:1 regime rather than almost always
# doing 16:1 via two recursive applications.


PORT=$((RANDOM % 10000 + 20000))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 10000 + 20000))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
# torch is built with cu121, so load a CUDA 12.x toolkit (default 'cuda' resolves to 13.x and
# breaks DeepSpeedCPUAdam's JIT build). 12.2 is the closest available minor to 12.1.
module load cuda/12.2

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate videoxlpro


echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory
cd /home/av354855/EfficientVideoXLPro/videoxlpro

export NUM_GPUS=2
export NNODES=1
export RANK=0
export ADDR=localhost

# Bypass DeepSpeed's strict CUDA version-string check; safe because the loaded toolkit (12.x)
# matches torch's CUDA major version (cu121). Needed for the cpu_adam JIT build under offload.
export DS_SKIP_CUDA_CHECK=1


# NCCL Flight Recorder: if a collective ever desyncs across ranks again (see
# the step-3204 hang on job 351066 -- ALLGATHER_BASE timeout after the full
# ddp_timeout with zero other diagnostic info), this dumps each rank's recent
# collective history to disk the moment the watchdog times out, instead of
# leaving nothing but a bare "ran for N ms" message. Both env var spellings
# set since the flight-recorder vars were renamed across torch versions.
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_TRACE_BUFFER_SIZE=2000
export NCCL_DUMP_ON_TIMEOUT=1

export WANDB_PROJECT="videoxlpro-apt"
export WANDB_ENTITY="ag8093-university-of-central-florida"
export WANDB_DIR="/home/av354855/EfficientVideoXLPro/videoxlpro"

MODEL_PATH="MINT-SJTU/Video-XL-Pro-3B"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION=qwen_1_5
# Renamed off the old entropy-mode name: output_dir is derived from this, so reusing
# it would write this run's checkpoints on top of the entropy-partition run.
MID_RUN_NAME="videoxlpro-3b-rlt-apt-survivor-embed-llava-ego4D"

echo "Fine-tuning run: ${MID_RUN_NAME}  (partition=survivor, mask_space=embed, persist=False)"

# ACCELERATE_CPU_AFFINITY deliberately NOT set. With it on, accelerate reads the
# GPU's NUMA node, derives its own "nearby cores" list, and hard-sets it with
# os.sched_setaffinity() -- without checking what the SLURM cgroup actually permits.
# On a node where those two disagree (seen on c5-2) the syscall rejects it with
# OSError [Errno 22] Invalid argument and BOTH ranks die. It fails inside
# TrainingArguments.__post_init__, i.e. before any training flag is parsed, so the
# traceback points at hf_argparser and reads like a config error when it is really a
# scheduling one. Node-dependent, so the same script runs fine elsewhere.
#
# Nothing is lost: #SBATCH --gres-flags=enforce-binding + --cpus-per-gpu already bind
# each rank to CPUs on its GPU's NUMA node, cgroup-aware. Accelerate's pass is a
# second, less-informed attempt at a job SLURM has already done correctly.
torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    videoxlpro/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /home/av354855/EfficientVideoXLPro/data_mix.yaml \
    --image_folder /home/c3-0/datasets/llava_665K/playground/data \
    --video_folder /home/c3-0/datasets/Ego4D/videos/h264 \
    --mm_tunable_parts="none" \
    --use_apt_temporal True \
    --apt_temporal_partition_mode survivor \
    --apt_temporal_run_tol 0 \
    --apt_temporal_persist False \
    --apt_num_scales 3 \
    --apt_thresholds "4.0,6.0" \
    --rlt_mask_space embed \
    --rlt_embed_threshold 0.34 \
    --rlt_embed_metric l2 \
    --rlt_threshold 0.2 \
    --rlt_attn_mode reuse \
    --rlt_mask_mode ref \
    --rlt_temporal_pos_scale 0.0 \
    --frames_upbound 48 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1152 \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (336, 2352), (336, 2688), (336, 3024), (336, 3360), (336, 3696), (336, 4032), (336, 4368), (336, 4704), (336, 5040), (336, 5376), (336, 5712), (336, 6048), (336, 6384), (336, 6720), (336, 7056), (336, 7392), (336, 7728), (336, 8064), (336, 8400), (336, 8736), (336, 9072), (336, 9408), (336, 9744), (336, 10080), (336, 10416), (336, 10752), (336, 11088), (336, 11424), (336, 11760), (336, 12096), (336, 12432), (336, 12768), (336, 13104), (336, 13440), (336, 13776), (336, 14112), (336, 14448), (336, 14784), (336, 15120), (336, 15456), (336, 15792), (336, 16128), (336, 16464), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (672, 2352), (672, 2688), (672, 3024), (672, 3360), (672, 3696), (672, 4032), (672, 4368), (672, 4704), (672, 5040), (672, 5376), (672, 5712), (672, 6048), (672, 6384), (672, 6720), (672, 7056), (672, 7392), (672, 7728), (672, 8064), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1008, 2352), (1008, 2688), (1008, 3024), (1008, 3360), (1008, 3696), (1008, 4032), (1008, 4368), (1008, 4704), (1008, 5040), (1008, 5376), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1344, 2352), (1344, 2688), (1344, 3024), (1344, 3360), (1344, 3696), (1344, 4032), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (1680, 2352), (1680, 2688), (1680, 3024), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016), (2016, 2352), (2016, 2688), (2352, 336), (2352, 672), (2352, 1008), (2352, 1344), (2352, 1680), (2352, 2016), (2352, 2352), (2688, 336), (2688, 672), (2688, 1008), (2688, 1344), (2688, 1680), (2688, 2016), (2688, 2352), (3024, 336), (3024, 672), (3024, 1008), (3024, 1344), (3024, 1680), (3360, 336), (3360, 672), (3360, 1008), (3360, 1344), (3696, 336), (3696, 672), (3696, 1008), (3696, 1344), (4032, 336), (4032, 672), (4032, 1008), (4032, 1344), (4368, 336), (4368, 672), (4368, 1008), (4704, 336), (4704, 672), (4704, 1008), (5040, 336), (5040, 672), (5040, 1008), (5376, 336), (5376, 672), (5376, 1008), (5712, 336), (5712, 672), (6048, 336), (6048, 672), (6384, 336), (6384, 672), (6720, 336), (6720, 672), (7056, 336), (7056, 672), (7392, 336), (7392, 672), (7728, 336), (7728, 672), (8064, 336), (8064, 672), (8400, 336), (8736, 336), (9072, 336), (9408, 336), (9744, 336), (10080, 336), (10416, 336), (10752, 336), (11088, 336), (11424, 336), (11760, 336), (12096, 336), (12432, 336), (12768, 336), (13104, 336), (13440, 336), (13776, 336), (14112, 336), (14448, 336), (14784, 336), (15120, 336), (15456, 336), (15792, 336), (16128, 336), (16464, 336)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/home/av354855/EfficientVideoXLPro/videoxlpro/outputs/checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --ddp_timeout 14400 \
    --mlvu_eval_on_save True \
    --mlvu_eval_at_start False \
    --mlvu_eval_frames 128 \
    --mlvu_eval_limit 100 \
    --mlvu_eval_timeout 10800 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2
  conda deactivate