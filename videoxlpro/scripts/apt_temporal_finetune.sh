export NUM_GPUS=1
export NNODES=1
export RANK=0
export ADDR=localhost
export PORT=12345

export WANDB_PROJECT="videoxlpro-apt"
export WANDB_ENTITY="ag8093-university-of-central-florida"

# Starting from the pre-trained Video-XL-Pro-3B checkpoint (skips pretrain stage)
MODEL_PATH="MINT-SJTU/Video-XL-Pro-3B"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# The point of this run: zero_conv (merge correction) is zero-initialised, so untrained
# APT-Temporal has its compensation switched off entirely -- the coarse token is just a
# raw downsampled patch embed. train.py unfreezes patch_attn / zero_conv automatically
# whenever use_apt_temporal is set, so this run is what actually turns that on.
#
# TAPT carries NO learnable temporal state, matching plain RLT: the trained modules here
# are exactly the two APT trains, so an APT-only run and this one are directly
# comparable and a result that holds for APT should transfer.
#
# THRESHOLD must match the eval script (lmms-eval/scripts/eval_videoxl_pro.sh). It sets how
# much temporal reuse happens, hence how much there is to compensate FOR. Training at one
# reuse rate and evaluating at another teaches the model to invert the wrong distortion.
THRESHOLD=0.2

# reuse = each frame's events attend over its FULL partition (fresh + carried tokens).
# Train with `per_frame` ONLY for the ablation: it starves events of their own frame's
# context, so zero_conv would learn to compensate for that BUG as well as for the
# legitimate reuse -- and the checkpoint is then invalid once the bug is fixed.
ATTN_MODE=reuse

PROMPT_VERSION=qwen_1_5
MID_RUN_NAME="videoxlpro-3b-apt-temporal-${ATTN_MODE}-thr${THRESHOLD}"

echo "Fine-tuning run: ${MID_RUN_NAME}  (attn_mode=${ATTN_MODE}, threshold=${THRESHOLD})"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    videoxlpro/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path /home/av354855/EfficientVideoXLPro/data_mix.yaml \
    --image_folder /home/c3-0/datasets/llava_665K/playground/data \
    --video_folder /home/c3-0/datasets/Ego4D/videos/h264 \
    --mm_tunable_parts="mm_mlp_adapter,mm_temporal_compressor" \
    --use_apt_temporal True \
    --apt_thresholds "4.0,6.0" \
    --apt_num_scales 3 \
    --rlt_threshold ${THRESHOLD} \
    --rlt_attn_mode ${ATTN_MODE} \
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
    --save_steps 200 \
    --save_total_limit 2 \
    --ddp_timeout 14400 \
    --mlvu_eval_on_save False \
    --mlvu_eval_at_start False \
    --mlvu_eval_frames 128 \
    --mlvu_eval_limit 100 \
    --mlvu_eval_timeout 10800 \
    --learning_rate 1e-5 \
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