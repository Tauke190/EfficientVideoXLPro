#!/bin/bash
#SBATCH --job-name=videoxlpro_sweep
#SBATCH --output=/home/av354855/EfficientVideoXLPro/lmms-eval/logs/slurm/videoxlpro_sweep-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8

# NOTE: the --output dir above must exist before submitting, otherwise slurm
# silently drops the log:  mkdir -p /home/av354855/EfficientVideoXLPro/lmms-eval/logs/slurm
#
# HPC counterpart of eval_videoxl_pro_F.sh: sweeps {RLT, APT, APT+RLT} x
# {256,512,1024} frames on the base MINT-SJTU/Video-XL-Pro-3B checkpoint.
# No --limit: runs the FULL mlvu_test split (unlike the mlvu_eval_limit=100
# convention used during training's periodic eval) -- expect this to take a
# lot longer per combo than a --limit 100 run. Runs strictly one combo at a
# time; each combo's accuracy + token-drop is parsed and appended to
# results.csv the moment that combo finishes, so results are readable as the
# sweep progresses instead of only after the whole job ends.
#
# NUM_GPUS below must match #SBATCH --gres=gpu:N above -- accelerate launches
# exactly that many ranks and Slurm's enforce-binding only exposes that many
# devices, so the two silently drifting apart is a hard crash, not a slowdown.
NUM_GPUS=1

PORT=$((RANDOM % 10000 + 20000))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 10000 + 20000))
done
echo "Free port found: $PORT"

# Load necessary modules
module load anaconda3
# torch is built with cu121, so load a CUDA 12.x toolkit (default 'cuda' resolves to 13.x)
module load cuda/12.2

eval "$(conda shell.bash hook)"
conda activate videoxlpro

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

# Change directory (HPC checkout root; relative ./logs output resolves under here)
cd /home/av354855/EfficientVideoXLPro/lmms-eval

export PYTHONPATH=/home/av354855/EfficientVideoXLPro:$PYTHONPATH
export HF_HOME=/home/av354855/.cache/huggingface

PRETRAINED="MINT-SJTU/Video-XL-Pro-3B"
FRAME_COUNTS=(256 512 1024)
METHODS=(rlt apt apt_rlt)

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="./logs/videoxlpro_mlvu/sweep_${SWEEP_TS}"
mkdir -p "$SWEEP_DIR"
echo "Sweep logs -> ${SWEEP_DIR}"

RESULTS_CSV="${SWEEP_DIR}/results.csv"
echo "method,frames,accuracy_pct,drop_pct,tokens_kept,tokens_dense,status" > "$RESULTS_CSV"

for METHOD in "${METHODS[@]}"; do
  for FRAMES in "${FRAME_COUNTS[@]}"; do

    case "$METHOD" in
      rlt)
        # RLT in embedding space (siglip_rlt_embeddings.py dirty test on patch embeds)
        args=(
          "pretrained=${PRETRAINED}"
          "max_frames_num=${FRAMES}"
          "use_rlt=True"
          "rlt_mask_space=embed"
          "rlt_embed_threshold=0.34"
          "rlt_embed_metric=l2"
          "rlt_attn_mode=reuse"
          "attn_implementation=flash_attention_2"
        )
        ;;
      apt)
        # Spatial-only adaptive patches, no temporal reuse
        args=(
          "pretrained=${PRETRAINED}"
          "max_frames_num=${FRAMES}"
          "use_apt=True"
          "apt_num_scales=3"
          "apt_threshold=4.0:6.0"
          "attn_implementation=flash_attention_2"
        )
        ;;
      apt_rlt)
        # APT-Temporal: RLT's embed-space dirty test gates which cells are eligible
        # to merge (entropy partition, the eval_videoxl_pro.slurm default -- NOT
        # apt_temporal_partition_mode=survivor, which is tied to a specific finetune)
        args=(
          "pretrained=${PRETRAINED}"
          "max_frames_num=${FRAMES}"
          "use_apt_temporal=True"
          "rlt_mask_space=embed"
          "rlt_mask_mode=ref"
          "rlt_attn_mode=reuse"
          "rlt_embed_threshold=0.34"
          "apt_threshold=4.0:6.0"
          "attn_implementation=flash_attention_2"
        )
        ;;
    esac

    MODEL_ARGS=$(IFS=,; echo "${args[*]}")
    LOG_FILE="${SWEEP_DIR}/${METHOD}_f${FRAMES}.log"

    echo ""
    echo "=== ${METHOD} @ ${FRAMES} frames -> ${LOG_FILE} ==="
    accelerate launch --num_processes="${NUM_GPUS}" --main_process_port "$PORT" \
        -m lmms_eval \
        --model videoxlpro \
        --tasks mlvu_test \
        --model_args "$MODEL_ARGS" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videoxlpro_mlvu_sweep \
        --output_path "$SWEEP_DIR" \
        --verbosity=DEBUG 2>&1 | tee "$LOG_FILE"

    RUN_STATUS="ok"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "!!! ${METHOD} @ ${FRAMES} frames FAILED (see ${LOG_FILE}) -- continuing sweep"
      RUN_STATUS="failed"
    fi

    # Parse this combo's own log right now instead of waiting to parse the whole
    # sweep afterward. kept/dense are summed across --num_processes ranks (NUM_GPUS=1
    # here, so a single rank sees the full mlvu_test split).
    ACC=$(grep -oP 'Average Performance Across All Task Categories:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)
    read -r KEPT DENSE <<< "$(grep -oP 'tokens kept=\K[0-9]+/[0-9]+' "$LOG_FILE" \
      | awk -F/ '{k+=$1; d+=$2} END {print k+0, d+0}')"
    DROP="N/A"
    if [ -n "$DENSE" ] && [ "$DENSE" -gt 0 ]; then
      DROP=$(awk -v k="$KEPT" -v d="$DENSE" 'BEGIN {printf "%.1f", 100*(1-k/d)}')
    fi

    if [ -n "$ACC" ]; then
      echo ">>> ${METHOD} @ ${FRAMES}: accuracy=${ACC}% token_drop=${DROP}% (kept ${KEPT}/${DENSE}) [${RUN_STATUS}]"
    else
      ACC=""
      echo ">>> ${METHOD} @ ${FRAMES}: no accuracy line found in log (see ${LOG_FILE}) [${RUN_STATUS}]"
    fi
    echo "${METHOD},${FRAMES},${ACC},${DROP},${KEPT},${DENSE},${RUN_STATUS}" >> "$RESULTS_CSV"

  done
done

echo ""
echo "Sweep complete. Results (accumulated as each combo finished): ${RESULTS_CSV}"

conda deactivate
