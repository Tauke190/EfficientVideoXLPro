
export PYTHONPATH=/home/av354855/projects/lmms-eval/Video-XL/Video-XL/videoxl:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,2

accelerate launch --num_processes=1 --main_process_port 12345 \
      -m lmms_eval \
      --model videoxl \
      --model_args pretrained=/mnt/SSD2/huggingface/Video_XL/VideoXL_weight_8,model_name=llava_qwen,conv_template=qwen_1_5,max_frames_num=64,video_decode_backend=decord,attn_implementation=flash_attention_2 \
      --tasks mlvu_test \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix videoxl_mlvu \
      --output_path ./logs/videoxl_mlvu