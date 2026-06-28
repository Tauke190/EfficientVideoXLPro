#!/bin/bash
#SBATCH --job-name=videoxlpro
#SBATCH --output=/home/av354855/projects/lmms-eval/logs/slurm/videoxlpro-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

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

# Change directory
cd /home/av354855/lmms-eval


export PYTHONPATH=/home/av354855/lmms-eval/Video-XL/Video-XL-Pro:$PYTHONPATH
# export HF_HOME=/home/av354855/.cache/huggingface


accelerate launch --num_processes=1 --main_process_port $PORT \
    -m lmms_eval \
    --model videoxlpro \
    --model_args pretrained=MINT-SJTU/Video-XL-Pro-3B,max_frames_num=256,attn_implementation=flash_attention_2 \
    --tasks mlvu_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix videoxlpro_mlvu \
    --output_path ./logs/videoxlpro_mlvu \
    --verbosity=DEBUG

conda deactivate