export PYTHONPATH=/home/av354855/projects/lmms-eval/Video-XL/Video-XL-Pro:$PYTHONPATH
export HF_HOME=/mnt/SSD2/huggingface

python scripts/profile_encoder.py \
    --model MINT-SJTU/Video-XL-Pro-3B \
    --video /mnt/SSD2/huggingface/mlvu_test/test_surveil_27.mp4 \
    --frames 64 128 256 \
    --warmup 1 \
    --runs 3