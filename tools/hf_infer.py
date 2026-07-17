import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
import sys
sys.path.append(os.getcwd())

# NOTE: The video path below is a placeholder. Replace `assets/example.mp4`
# with your own local or publicly available video file before running.

import time

import numpy as np
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from qwen2_5_omni_compress.modeling_qwen2_5_omni_compress import Qwen2_5OmniCompressForConditionalGeneration
from qwen2_5_omni_compress.configuration_qwen2_5_omni_compress import Qwen2_5OmniCompressConfig

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Omni-3B", help="Path to the model checkpoint.")
parser.add_argument("--use_compression", action="store_true", help="Whether to use the compression model.")
parser.add_argument("--video_path", type=str)

args = parser.parse_args()

if args.use_compression:
    model = Qwen2_5OmniCompressForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
else:
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

print(f"model.device: {model.device}")
min_pixels = 50176
max_pixels = 50176
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)

conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": args.video_path, "max_frames": 80, "max_pixels": 50176},
            {"type": "text", "text": "Describe the video"},
        ],
    },
]

# Use the audio track embedded in the video.
USE_AUDIO_IN_VIDEO = True
timings = []

# Run inference for a few iterations to measure timing.
for i in range(2):
    print(f"--- Starting Iteration {i + 1} ---", flush=True)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate the output text (no audio output).
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_audio=False,
        thinker_max_new_tokens=10,
        use_cache=True,
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration = end_time - start_time
    timings.append(duration)
    print(f"Iteration {i + 1} generation took: {duration:.4f} seconds", flush=True)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated Text:", text, flush=True)
    print("-" * 20)

print("\n--- Inference Timing Summary ---")
for i, t in enumerate(timings):
    print(f"Run {i + 1}: {t:.4f} seconds")

if timings:
    average_time = np.mean(timings)
    std_dev = np.std(timings)
    print(f"\nAverage generation time over {len(timings)} runs: {average_time:.4f} seconds")
    print(f"Standard Deviation: {std_dev:.4f} seconds")

    print(f"First run (cold start) took: {timings[0]:.4f} seconds")
    if len(timings) > 1:
        average_time_warm = np.mean(timings[1:])
        print(f"Average time (excluding first run): {average_time_warm:.4f} seconds")
