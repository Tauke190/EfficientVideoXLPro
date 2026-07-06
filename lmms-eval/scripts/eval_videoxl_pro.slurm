#!/bin/bash
#SBATCH --job-name=videoxlpro_eval
#SBATCH --output=/home/av354855/EfficientVideoXLPro/lmms-eval/logs/slurm/videoxlpro_eval-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# NOTE: the --output dir above must exist before submitting, otherwise slurm
# silently drops the log:  mkdir -p /home/av354855/EfficientVideoXLPro/lmms-eval/logs/slurm

PORT=$((RANDOM % 10000 + 20000))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 10000 + 20000))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate lmms_eval

# Set UTF-8 encoding to handle Unicode characters
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory (HPC checkout root; relative ./logs output resolves under here)
cd /home/av354855/EfficientVideoXLPro/lmms-eval

export PYTHONPATH=/home/av354855/EfficientVideoXLPro:$PYTHONPATH

# ---- Ablation variants (uncomment one --model_args below; keep the rest commented) ----
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=True,use_rlt=True,rlt_threshold=0.2 \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_apt=True,apt_threshold=4.0:6.0,apt_num_scales=3 \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=False,use_rlt=True,rlt_threshold=0.3 \
# --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=True \

LOG_DIR="./logs/videoxlpro_mlvu"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_$(date +%Y%m%d_%H%M%S).log"

accelerate launch --num_processes=2 --main_process_port "$PORT" \
    -m lmms_eval \
    --model videoxlpro \
    --tasks mlvu_test \
    --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=128,attn_implementation=flash_attention_2,use_sae=False \
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

conda deactivate
