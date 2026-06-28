
# For dev server
export PYTHONPATH=/home/av354855/projects/Video-XL-Pro:$PYTHONPATH
export HF_HOME=/mnt/SSD2/huggingface

# For HPC 
# export PYTHONPATH=/home/av354855/lmms-eval/Video-XL/Video-XL-Pro:$PYTHONPATH
# export HF_HOME=/home/c3-0/datasets/huggingface
# use_sae=True

export CUDA_VISIBLE_DEVICES=0,2

accelerate launch --num_processes=2 --main_process_port 12345 \
    -m lmms_eval \
    --model videoxlpro \
    --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=True,use_rlt=True,rlt_threshold=0.2 \
    --tasks mlvu_test \
    --batch_size 1 \
    --limit 200 \
    --log_samples \
    --log_samples_suffix videoxlpro_mlvu \
    --output_path ./logs/videoxlpro_mlvu \
    --verbosity=DEBUG

# Num of sample at 32 frames without SAE - Accuracy MLVU Test
# 100 - 8.74 
# 200 - 
# All -

# Num of sample at 32 frames with SAE - Accuracy MLVU Test
# 100 - 13.98
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