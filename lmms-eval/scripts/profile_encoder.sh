export PYTHONPATH=/home/av354855/projects/Video-XL-Pro:$PYTHONPATH
export HF_HOME=/mnt/SSD2/huggingface

# Sampled over N distinct mlvu_test videos rather than one clip: what each remover
# saves is content-dependent (a static surveillance shot lets APT-Temporal reuse nearly
# everything, a moving camera lets it reuse nothing), so one video measures one video.
# The seed is fixed, so every method is compared on the exact same footage.
# --runs is repeats PER VIDEO (timing jitter only); the sample axis is --num_videos.
python scripts/profile_encoder.py \
    --model MINT-SJTU/Video-XL-Pro-3B \
    --video_dir /mnt/SSD2/huggingface/mlvu_test \
    --num_videos 5 \
    --video_seed 0 \
    --frames 128 \
    --methods original rlt apt apt_temporal \
    --sae on \
    --rlt_threshold 0.2 \
    --apt_threshold 4.0:6.0 \
    --apt_num_scales 3 \
    --warmup 2 \
    --runs 1