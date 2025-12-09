# Neural Chroma Reconstruction

*A research project by **Bruno Rios**, University of Texas Rio Grande Valley*

---

## Overview
Neural Chroma Reconstruction explores how a lightweight convolutional network can
rebuild full 4:4:4 color fidelity from 4:2:0–subsampled images.
The long-term goal is to extend this model toward video, incorporating
bitrate-aware refinement and temporal consistency.

---

## Current Objectives
-Merge a model using version 5 and version 6.
Version 5 Model: Trained in Y and low_cb, low_cr of 4:2:0. This model is intended to hallucinate images.
PSNR Avg: 47.82 dB
SSIM avg: .998
================================================================================
COMPARISON REPORT (3703 images) | Mode: VAL
================================================================================
METRIC             | METHOD     | AVG        | MIN        | MAX
--------------------------------------------------------------------------------
RGB PSNR           | Bilinear   | 40.42 dB    | 24.19 dB   | 80.00 dB
RGB PSNR           | AI Model   | 37.11 dB    | 23.86 dB   | 47.27 dB
--------------------------------------------------------------------------------
RGB SSIM           | Bilinear   | 0.9816     | 0.6372     | 1.0000
RGB SSIM           | AI Model   | 0.9761     | 0.6341     | 0.9996
================================================================================
CHROMA ONLY PSNR   | Bilinear   | 42.88 dB    | 26.83 dB   | 80.00 dB
CHROMA ONLY PSNR   | AI Model   | 40.98 dB    | 26.54 dB   | 53.39 dB
--------------------------------------------------------------------------------
CHROMA ONLY SSIM   | Bilinear   | 0.9530     | 0.5556     | 1.0000
CHROMA ONLY SSIM   | AI Model   | 0.9536     | 0.5863     | 0.9995
================================================================================
VERDICT:
RGB Gain:    +-3.31 dB
Chroma Gain: +-1.90 dB (This is the pure color improvement!)

Version 6 Model: Trained in Y and low_cb, low_cr difference in an area. This model is intended to keep consistency and not force hallucinations.
=====================================================================================
RESIDUAL MODEL (v6) EVALUATION REPORT (3703 images)
=====================================================================================
METRIC             | METHOD     | AVG        | MIN        | MAX
-------------------------------------------------------------------------------------
RGB PSNR           | Bilinear   | 40.42 dB    | 24.19 dB   | 80.00 dB
RGB PSNR           | AI Model   | 44.04 dB    | 25.50 dB   | 80.00 dB
-------------------------------------------------------------------------------------
CHROMA ONLY PSNR   | Bilinear   | 42.88 dB    | 26.83 dB   | 80.00 dB
CHROMA ONLY PSNR   | AI Model   | 46.76 dB    | 28.29 dB   | 80.00 dB
=====================================================================================
VERDICT: Chroma PSNR Gain: +3.88 dB (AI vs Bilinear)

Video Metrics:
================================================================================
METRIC     | METHOD     | AVG        | MIN        | MAX
--------------------------------------------------------------------------------
PSNR       | Bilinear   | 57.54 dB    | 51.66 dB   | 64.05 dB
PSNR       | AI Model   | 58.55 dB    | 51.83 dB   | 67.81 dB
--------------------------------------------------------------------------------
SSIM       | Bilinear   | 0.9985     | 0.9960     | 0.9996
SSIM       | AI Model   | 0.9988     | 0.9966     | 0.9998
================================================================================
Chroma PSNR Gain: +1.01 dB

---

## Environment
```bash
conda create -n chroma python=3.10
conda activate chroma
pip install -r requirements.txt
```

---

## Run Dataset Prep
```bash
python scripts/prepare_dataset.py --input data/raw --output data/processed
```

## Train
```bash
python scripts/train.py --epochs 50 --batch-size 8
```

## Testing
```bash
Model 5
python evaluate_blur_rgb.py --src "D:\colorrecon02\unlabeled2017\unlabeled2017" --weights checkpoints_v5_yuv_gan/epoch_010.pth --batch 256 --split val

Model 6
python video_metric_test.py --video "video.mp4" --weights "model_upscailing_v6/res_epoch_030.pth" --output "ai_output.mp4"
```

## Inference
```bash
python scripts/inference.py --image sample.jpg
```

---

## License
MIT License © 2025 Bruno Rios
