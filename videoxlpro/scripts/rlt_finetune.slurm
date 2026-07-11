#!/bin/bash
#SBATCH --job-name=videoxlpro
#SBATCH --output=/home/av354855/EfficientVideoXLPro/slurmlogs/videoxlpro-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=8

# Starting from the pre-trained Video-XL-Pro-3B checkpoint (skips pretrain stage)


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
# Reduce CUDA allocator fragmentation (the original OOM had >1GB reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# /mnt/SSD2/huggingface/hub/models--MINT-SJTU--Video-XL-Pro-3B/snapshots/d9914fb2249a9de39daac33a3e02a34a991524db
MODEL_PATH="MINT-SJTU/Video-XL-Pro-3B"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION=qwen_1_5
MID_RUN_NAME="videoxlpro-3b-finetune-rlt-ego4d"

echo "Fine-tuning run: ${MID_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    videoxlpro/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --use_rlt True \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /home/av354855/EfficientVideoXLPro/data_mix.yaml \
    --image_folder /home/c3-0/datasets/llava_665K/playground/data \
    --video_folder /home/c3-0/datasets/Ego4D/videos/h264 \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --frames_upbound 32 \
    --video_fps 1 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1152 \
    --group_by_modality_length False \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "[(336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (336, 2352), (336, 2688), (336, 3024), (336, 3360), (336, 3696), (336, 4032), (336, 4368), (336, 4704), (336, 5040), (336, 5376), (336, 5712), (336, 6048), (336, 6384), (336, 6720), (336, 7056), (336, 7392), (336, 7728), (336, 8064), (336, 8400), (336, 8736), (336, 9072), (336, 9408), (336, 9744), (336, 10080), (336, 10416), (336, 10752), (336, 11088), (336, 11424), (336, 11760), (336, 12096), (336, 12432), (336, 12768), (336, 13104), (336, 13440), (336, 13776), (336, 14112), (336, 14448), (336, 14784), (336, 15120), (336, 15456), (336, 15792), (336, 16128), (336, 16464), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (672, 2352), (672, 2688), (672, 3024), (672, 3360), (672, 3696), (672, 4032), (672, 4368), (672, 4704), (672, 5040), (672, 5376), (672, 5712), (672, 6048), (672, 6384), (672, 6720), (672, 7056), (672, 7392), (672, 7728), (672, 8064), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1008, 2352), (1008, 2688), (1008, 3024), (1008, 3360), (1008, 3696), (1008, 4032), (1008, 4368), (1008, 4704), (1008, 5040), (1008, 5376), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1344, 2352), (1344, 2688), (1344, 3024), (1344, 3360), (1344, 3696), (1344, 4032), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (1680, 2352), (1680, 2688), (1680, 3024), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016), (2016, 2352), (2016, 2688), (2352, 336), (2352, 672), (2352, 1008), (2352, 1344), (2352, 1680), (2352, 2016), (2352, 2352), (2688, 336), (2688, 672), (2688, 1008), (2688, 1344), (2688, 1680), (2688, 2016), (3024, 336), (3024, 672), (3024, 1008), (3024, 1344), (3024, 1680), (3360, 336), (3360, 672), (3360, 1008), (3360, 1344), (3696, 336), (3696, 672), (3696, 1008), (3696, 1344), (4032, 336), (4032, 672), (4032, 1008), (4032, 1344), (4368, 336), (4368, 672), (4368, 1008), (4704, 336), (4704, 672), (4704, 1008), (5040, 336), (5040, 672), (5040, 1008), (5376, 336), (5376, 672), (5376, 1008), (5712, 336), (5712, 672), (6048, 336), (6048, 672), (6384, 336), (6384, 672), (6720, 336), (6720, 672), (7056, 336), (7056, 672), (7392, 336), (7392, 672), (7728, 336), (7728, 672), (8064, 336), (8064, 672), (8400, 336), (8736, 336), (9072, 336), (9408, 336), (9744, 336), (10080, 336), (10416, 336), (10752, 336), (11088, 336), (11424, 336), (11760, 336), (12096, 336), (12432, 336), (12768, 336), (13104, 336), (13440, 336), (13776, 336), (14112, 336), (14448, 336), (14784, 336), (15120, 336), (15456, 336), (15792, 336), (16128, 336), (16464, 336)]" \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/home/av354855/EfficientVideoXLPro/videoxlpro/outputs/checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_steps 3000 \
    --save_total_limit 2 \
    --mlvu_eval_on_save True \
    --mlvu_eval_at_start False \
    --mlvu_eval_task mlvu_test \
    --mlvu_eval_limit 100 \
    --mlvu_eval_frames 128 \
    --mlvu_eval_timeout 7200 \
    --ddp_timeout 14400 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2
conda deactivate