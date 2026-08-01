# TabPFN V3 COCO chroma pilot

Run date: 2026-08-01

This pilot tested whether image-adaptive TabPFN V3 imputation generalizes beyond
the initially favorable center crop. Twenty 64x64 crops were sampled from sorted
COCO 2017 validation images with seed 2026. All methods received the same
synthetic 4:2:0 chroma observations. TabPFN fitted Cr and Cb separately using
1,024 observed rows and predicted 4,096 pixel rows with 16 luma/spatial features.
The hosted model identifier was `v3_default`; Thinking mode was disabled.

## Mean results

| Method | Chroma PSNR | Chroma SSIM | Chroma MAE | Edge MAE | Gradient MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bilinear | 47.0957 | 0.951135 | 0.005202 | 0.008581 | 0.002587 |
| Bicubic | 48.1573 | 0.959445 | 0.004745 | 0.007726 | 0.002292 |
| V5 | 42.9576 | 0.959965 | 0.006497 | 0.008563 | 0.002380 |
| V6 | **50.5231** | **0.978180** | **0.003444** | **0.005190** | **0.001569** |
| TabPFN V3 | 46.8985 | 0.961636 | 0.004773 | 0.007287 | 0.002342 |

## Paired findings

Positive paired improvements always favor TabPFN, including for error metrics
where the sign is reversed so that an error reduction is positive.

- Against bilinear, TabPFN changed chroma PSNR by -0.1972 dB (95% bootstrap CI
  -0.9569 to +0.6950) and won 35% of crops. Its edge-MAE reduction was +0.001293
  (CI +0.000025 to +0.002893), although it won only 40% of individual crops.
- Against bicubic, TabPFN changed chroma PSNR by -1.2588 dB (CI -2.1759 to
  -0.2645) and won 25% of crops.
- Against V6, TabPFN changed chroma PSNR by -3.6246 dB (CI -4.2389 to -3.0489)
  and lost all 20 crops. It also lost all crops on chroma SSIM, MAE, edge MAE,
  and gradient MAE.
- V6 beat bilinear on all 20 crops, with a mean chroma-PSNR gain of +3.4274 dB
  (CI +2.7926 to +4.1407).

## Interpretation

The first hand-selected crop overstated TabPFN's general performance. The
tabular formulation appears capable of aligning chroma with some luma edges, but
its result is image-dependent and does not outperform bicubic or V6 overall in
this pilot. A defensible follow-up would test a learned hybrid that uses V6 or
bicubic chroma as a coarse prior and asks TabPFN to predict bounded residuals,
rather than asking an image-local tabular model to reconstruct absolute chroma.

This is still a small crop-level pilot, not a full COCO benchmark. The sampling,
per-image caching, paired statistics, and exact command are implemented in
`scripts/benchmark_coco_tabpfn.py` for larger runs.
