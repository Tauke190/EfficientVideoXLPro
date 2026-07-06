export PYTHONPATH=/home/av354855/projects/Video-XL-Pro:$PYTHONPATH
export HF_HOME=/mnt/SSD2/huggingface

python scripts/profile_encoder.py \
    --model MINT-SJTU/Video-XL-Pro-3B \
    --video /mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4 \
    --frames 16 32 64 \
    --methods original rlt apt \
    --sae both \
    --rlt_threshold 0.2 \
    --apt_threshold 4.0:6.0 \
    --apt_num_scales 3 \
    --warmup 2 \
    --runs 3