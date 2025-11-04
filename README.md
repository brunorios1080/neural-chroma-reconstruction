# Neural Chroma Reconstruction

*A research project by **Bruno Rios**, University of Texas Rio Grande Valley*

---

## Overview
Neural Chroma Reconstruction explores how a lightweight convolutional network can
rebuild full 4:4:4 color fidelity from 4:2:0–subsampled images.
The long-term goal is to extend this model toward video, incorporating
bitrate-aware refinement and temporal consistency—similar in spirit to DLSS-style
spatiotemporal reconstruction.

---

## Current Objectives
- Convert RGB datasets (COCO, DIV2K) to YCbCr 4:2:0 pairs  
- Train a baseline CNN to upsample Cb/Cr to 4:4:4  
- Compare against bilinear and bicubic interpolation  
- Log PSNR / SSIM gains and visualize color-edge recovery  

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

## Inference
```bash
python scripts/inference.py --image sample.jpg
```

---

## License
MIT License © 2025 Bruno Rios
