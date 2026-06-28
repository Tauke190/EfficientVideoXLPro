python3 -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=./Video-XL-Pro-3B,conv_template=qwen_1_5,max_frames_num=128 \
  --tasks mlvu_dev  \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix videoxlpro_lvb \
  --output_path ./logs/lvb/