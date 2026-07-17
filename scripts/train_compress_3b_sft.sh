nproc_per_node=4

# Run in background with: nohup bash scripts/train_compress_3b_sft.sh > logs/train.log 2>&1 &
# NOTE: The --model and --dataset values below are placeholders. Replace them
# with your own pretrained checkpoint and processed SFT data files (ms-swift
# jsonl/json format) before running.

COPY_INIT_BIDIR=1 \
FORCE_QWENVL_VIDEO_READER=torchvision \
USE_AUDIO_IN_VIDEO=True \
ENABLE_AUDIO_OUTPUT=0 \
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=105369 \
FPS_MAX_FRAMES=120 \
SOFTMAX=vanilla \
MROPE_SECTION=18,18,18,10 \
swift sft \
    --custom_register_path tools/register.py \
    --model /path/to/pretrained_checkpoint \
    --train_type full \
    --model_kwargs '{"fps_max_frames": 120}' \
    --freeze_parameters ['thinker.audio_tower','thinker.visual','token2wav'] \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 200 \
    --save_total_limit 100 \
    --save_only_model \
    --logging_steps 5 \
    --max_length 16384 \
    --model_type 'qwen2_5_omni_compress' \
    --warmup_ratio 0.05 \
    --lr_scheduler_type 'cosine' \
    --output_dir output/compress_3b_sft \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataset data/your_dataset.jsonl \
    --split_dataset_ratio 0.01 \
    --eval_strategy steps \
    --eval_steps 300
