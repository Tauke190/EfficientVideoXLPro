
# For dev server
export PYTHONPATH=/home/av354855/projects/Video-XL-Pro:$PYTHONPATH
export HF_HOME=/mnt/SSD2/huggingface

# For HPC 
# export PYTHONPATH=/home/av354855/EfficientVideoXLPro:$PYTHONPATH
# export HF_HOME=/home/c3-0/datasets/huggingface

export CUDA_VISIBLE_DEVICES=0,2
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=True,use_rlt=True,rlt_threshold=0.2 \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_apt=True,apt_threshold=4.0:6.0,apt_num_scales=3 \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=True,use_rlt=True,rlt_threshold=0.2 \
# --model_args pretrained=/home/av354855/projects/Video-XL-Pro/videoxlpro/outputs/checkpoints/videoxlpro-3b-finetune-llava665k-without-DTS,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=False \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=False \

# use_apt_temporal
LOG_DIR="./logs/videoxlpro_mlvu"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_apttemporal_$(date +%Y%m%d_%H%M%S).log"

#[ use_apt , use_rlt , use_apt_temporal]

accelerate launch --num_processes=2 --main_process_port 12345 \
    -m lmms_eval \
    --model videoxlpro \
    --tasks mlvu_test \
    --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_apt_temporal=True,rlt_threshold=0.2,rlt_temporal_pos_scale=0.0 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoxlpro_mlvu \
    --output_path "$LOG_DIR" \
    --verbosity=DEBUG 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Full log saved to: $LOG_FILE"
echo "========================================"

# Num of sample at 32 frames without SAE - Accuracy MLVU Test
# 100 - 8.74 
# 200 - 
# All -

# Num of sample at 32 frames with SAE - Accuracy MLVU Test
# 100 - 13.98
# 200 - 
# All -

# Num of sample at 128 frames with new RLT (position encoding dropped because SAE adds it anyways) - Accuracy MLVU Test
# 100 - 15
# 200 - 
# All -

# Num of sample at 128 frames - Accuracy MLVU Test
# 100 - 16
# 200 - 22
# All - 47.4

# Num of sample at 128 frames With RLT and SAE enabled 9 ( 55 % of token dropped in encoding)
# 100 - 9.47
# 200 - 11.93
# All - 

# Num of sample at 128 frames With APT and SAE enabled 9 ( 44 % of token dropped in encoding) Threshold : 4 - 6
# 100 - 14.7
# 200 - 
# All - 42.7

# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: sportsQA: 38.9%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: order: 44.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: topic_reasoning: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: needleQA: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: tutorialQA: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: plotQA: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: count: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: anomaly_reco: 61.5%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:173 - Evaluation on Task Categories: ego: 0.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:181 - Average Performance Across All Task Categories: 16.0%
# 2026-06-25 07:08:07 | INFO     | utils:mlvu_aggregate_results_test:182 - Videos skipped (not found): 0