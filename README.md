# EchoingPixels: Aliasing-Resistant Joint Token Reduction for Audio-Visual LLMs

Official code for the ICML 2026 paper *"EchoingPixels: Aliasing-Resistant Joint Token Reduction for Audio-Visual LLMs"*.

[![Paper](https://img.shields.io/badge/arXiv-2512.10324-b31b1b)](https://arxiv.org/abs/2512.10324)
[![Conference](https://img.shields.io/badge/ICML-2026-0044cc)](https://icml.cc/virtual/2026/poster/62697)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Introduction

Audio-Visual LLMs (AV-LLMs) face prohibitive computational cost from massive,
redundant audio-visual token streams. Compressing each modality independently
ignores the synergistic information between sight and sound, and any aggressive
sparse reduction suffers from a previously overlooked bottleneck:
**Positional Aliasing** — sparse sampling makes adjacent token gaps large enough
that RoPE's high-frequency temporal components violate the Nyquist limit,
collapsing distant timestamps into one another and corrupting the model's sense
of time.

**EchoingPixels** is an aliasing-resistant framework for joint audio-visual
token reduction. It selects tokens on the *synergistic* audio-visual stream
rather than per modality, and adapts the positional-encoding bandwidth to the
sparse sampling rate. It matches the full model using only **5–20%** of the
original tokens.

Two components, built on top of [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni):

- **CS2 (Cross-Modal Semantic Sieve)** — a bidirectional cross-modal encoder
  inserted after the audio/video encoders and before the LLM; it fuses the two
  streams then performs extractive top-k selection by joint-modality saliency.
- **Sync-RoPE `[T, H, W, T]`** — a spectral low-pass filter for RoPE that assigns
  the low-frequency channels to the temporal axis, preserving monotonic temporal
  order in the reduced stream.

## Installation

We recommend a conda environment with Python 3.10:

```bash
conda create -n echoingpixels python=3.10 -y
conda activate echoingpixels
pip install -r requirements.txt
```

This installs the pinned versions the code is developed and tested with:
`torch==2.7.1`, `torchvision==0.22.1`, `transformers==4.54.1`, `ms-swift==3.7.0`,
`accelerate==1.5.1`, `deepspeed==0.17.3`, `numpy==1.26.4`, `soundfile==0.13.1`,
`qwen-omni-utils[decord]`. Flash-Attention 2 is required for the `flash_attention_2`
path in the example script; install it matching your CUDA build if you use it.

## Repository structure

```
.
├── qwen2_5_omni_compress/                  # model implementation
│   ├── configuration_qwen2_5_omni_compress.py   # config (av_refiner_layers, compression_ratio)
│   └── modeling_qwen2_5_omni_compress.py        # CS2 + top-k compressor + Sync-RoPE [T,H,W,T]
├── tools/
│   ├── register.py                         # ms-swift model/template registration
│   └── hf_infer.py                         # HuggingFace inference example
├── scripts/
│   └── train_compress_3b_sft.sh            # ms-swift SFT training script
└── requirements.txt
```

## Model configuration

The compressed model is built on top of a standard Qwen2.5-Omni checkpoint.
To enable compression, you need to add two fields to the model's `config.json`
under `thinker_config` → `text_config`:

```jsonc
{
  "thinker_config": {
    "text_config": {
      // ... existing fields ...
      "av_refiner_layers": 4,       // number of bidirectional CS² refiner layers
      "compression_ratio": 0.2      // fraction of audio-visual tokens to keep (e.g. 0.2 = keep 20%)
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `av_refiner_layers` | `int` | Number of bidirectional cross-modal refiner (CS2) layers inserted before the causal decoder. Set to `0` to disable the refiner. |
| `compression_ratio` | `float` | Target keep-ratio for audio-visual tokens. `0.2` keeps 20% of the original AV tokens; `1.0` disables compression. |

## Quick inference

`tools/hf_infer.py` supports both the vanilla Qwen2.5-Omni and the compressed
variant:

```bash
# vanilla Qwen2.5-Omni (works out of the box with public weights)
CUDA_VISIBLE_DEVICES=0 python tools/hf_infer.py --model_name_or_path Qwen/Qwen2.5-Omni-3B --video_path your_video_path.mp4

# compressed model (use a checkpoint trained with this code)
CUDA_VISIBLE_DEVICES=0 python tools/hf_infer.py --model_name_or_path /path/to/Qwen2.5-Omni-Compress-3B --use_compression --video_path your_video_path.mp4
```

## Training

Fine-tune the 3B compressed model with ms-swift:

```bash
bash scripts/train_compress_3b_sft.sh
```

Edit the script first to set `--model` to your pretrained checkpoint and
`--dataset` to your processed SFT data files (ms-swift `jsonl`/`json` format).

### Runtime knobs

A few behavior switches are read from environment variables (and echoed at
import time) so a checkpoint's training configuration can be matched exactly:

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `SOFTMAX` | `vanilla` \| `gumbel` | `vanilla` | softmax variant used for token scoring |
| `RETAIN_TOKEN` | `v` \| `av` | `av` | compression candidates: video-only (`v`) or audio+video (`av`) |
| `MROPE_SECTION` | four comma-separated ints | `18,18,18,10` | per-axis channel split for `[T,H,W,T]`; must sum to `head_dim // 2` |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gongechoingpixels,
  title={EchoingPixels: Aliasing-Resistant Joint Token Reduction for Audio-Visual LLMs},
  author={Gong, Chao and Wang, Depeng and Wei, Zhipeng and Guo, Ya and Zhu, Huijia and Chen, Jingjing},
  booktitle={Forty-third International Conference on Machine Learning}
}
```

## Acknowledgements

Built on top of [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) and the
[ms-swift](https://github.com/modelscope/ms-swift) training framework.

## License

Released under the [MIT License](LICENSE).
