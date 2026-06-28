export FORCE_QWENVL_VIDEO_READER=decord

python -m lmms_eval \
  --model qwen2_vl \
  --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
  --tasks temporalbench_long_qa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix Qwen2-7B \
  --output_path ./logs/Qwen2-7B \
  --nframes 1


#limit 8
  