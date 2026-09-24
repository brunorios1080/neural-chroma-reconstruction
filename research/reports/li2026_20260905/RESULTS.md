# Comparison with Li et al. (2026)

**Status: complete.** 5040 completed image/method pairs from 264 images.

All requested job groups are finalized.

This is an independent implementation of the described experiment. The exhaustive color-transform test exactly matches Table 1. Image-table reproduction remains unverified because sampling, borders, lifting/interpolation interfaces, and metric settings are incompletely specified. Published values and our measurements must be interpreted separately.

Source: [Li, Zhang, and Huang, IET Image Processing (2026)](https://doi.org/10.1049/ipr2.70338). UHD images: [the authors' Zenodo record](https://doi.org/10.5281/zenodo.17649711).

## uhd240 / cosited_point

All 240 native images. RGB CPSNR averages per-image dB; higher is better.

| Method | RGB CPSNR (dB) | Gain vs bilinear | Gain vs best independent scaled method | Wins vs scaled |
|---|---:|---:|---:|---:|
| Conventional bilinear | 51.24215 | — | -5.57514 | — |
| scaled_decode_first/bicubic | 56.81730 | +5.57514 | — | — |
| V5 | 41.64610 | -9.59605 | -15.17120 | 0/240 |
| V5.1 | 45.88247 | -5.35968 | -10.93483 | 0/240 |
| V6 | 50.27362 | -0.96854 | -6.54368 | 0/240 |
| V7 | 50.70133 | -0.54082 | -6.11597 | 0/240 |
| Prism Residual | 51.04452 | -0.19763 | -5.77278 | 0/240 |
| Prism Polar | 50.93381 | -0.30835 | -5.88349 | 0/240 |
| Prism Polar Prob | 50.73028 | -0.51187 | -6.08702 | 0/240 |
| Prism Cartesian Prob | 51.12490 | -0.11726 | -5.69240 | 0/240 |

The strongest independent comparator is selected by dataset mean, not separately for each image. This is not an author-code comparison.

### Learned models: secondary metrics

| Model | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM (higher) | Luma FSIM (higher) |
|---|---:|---:|---:|---:|
| V5 | 1.53810 | 1.69109 | 0.9977310 | 0.9995364 |
| V5.1 | 0.79844 | 0.98224 | 0.9978968 | 0.9995832 |
| V6 | 0.54691 | 0.49432 | 0.9985697 | 0.9998065 |
| V7 | 0.52211 | 0.46955 | 0.9985723 | 0.9998105 |
| Prism Residual | 0.50123 | 0.44827 | 0.9985781 | 0.9998050 |
| Prism Polar | 0.50673 | 0.45447 | 0.9985734 | 0.9998051 |
| Prism Polar Prob | 0.51593 | 0.46638 | 0.9985700 | 0.9998061 |
| Prism Cartesian Prob | 0.49727 | 0.44389 | 0.9985796 | 0.9998049 |

### All independent interpolation variants

| Method | RGB CPSNR | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM | Luma FSIM |
|---|---:|---:|---:|---:|---:|
| conventional/bicubic | 51.27593 | 0.48791 | 0.43509 | 0.9985800 | 0.9998026 |
| conventional/bilinear | 51.24215 | 0.48398 | 0.43294 | 0.9985816 | 0.9998011 |
| scaled_decode_first/bicubic | 56.81730 | 0.13246 | 0.11879 | 0.9998106 | 0.9999990 |
| scaled_decode_first/bilinear | 56.42125 | 0.13562 | 0.12357 | 0.9997938 | 0.9999989 |
| scaled_hybrid/bicubic | 51.90211 | 0.38111 | 0.34323 | 0.9994808 | 0.9999660 |
| scaled_hybrid/bilinear | 52.27503 | 0.34999 | 0.31537 | 0.9995128 | 0.9999646 |
| scaled_lifting/bicubic | 51.84197 | 0.38116 | 0.34495 | 0.9994814 | 0.9999642 |
| scaled_lifting/bilinear | 52.21541 | 0.35019 | 0.31719 | 0.9995149 | 0.9999628 |
| scaled_matrix/bicubic | 54.61254 | 0.23223 | 0.20048 | 0.9996564 | 0.9999764 |
| scaled_matrix/bilinear | 54.45641 | 0.22965 | 0.20031 | 0.9996723 | 0.9999761 |

Published CPSNR: conventional bilinear 50.49; conventional bicubic 50.69; scaled/lifting bilinear 54.40; scaled/lifting bicubic 54.82 dB.
Our conventional bilinear differs from the published value by +0.75215 dB.

## kodak / cosited_point

All 24 native images. RGB CPSNR averages per-image dB; higher is better.

| Method | RGB CPSNR (dB) | Gain vs bilinear | Gain vs best independent scaled method | Wins vs scaled |
|---|---:|---:|---:|---:|
| Conventional bilinear | 47.24729 | — | -1.17397 | — |
| scaled_decode_first/bilinear | 48.42126 | +1.17397 | — | — |
| V5 | 38.29608 | -8.95121 | -10.12518 | 0/24 |
| V5.1 | 40.52858 | -6.71871 | -7.89268 | 0/24 |
| V6 | 43.18509 | -4.06220 | -5.23617 | 0/24 |
| V7 | 43.87302 | -3.37427 | -4.54824 | 0/24 |
| Prism Residual | 43.08891 | -4.15838 | -5.33235 | 0/24 |
| Prism Polar | 43.48914 | -3.75815 | -4.93211 | 0/24 |
| Prism Polar Prob | 43.98658 | -3.26071 | -4.43468 | 0/24 |
| Prism Cartesian Prob | 43.53417 | -3.71312 | -4.88709 | 0/24 |

The strongest independent comparator is selected by dataset mean, not separately for each image. This is not an author-code comparison.

### Learned models: secondary metrics

| Model | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM (higher) | Luma FSIM (higher) |
|---|---:|---:|---:|---:|
| V5 | 1.97560 | 2.04621 | 0.9983119 | 0.9995131 |
| V5.1 | 1.38110 | 1.50543 | 0.9984059 | 0.9995386 |
| V6 | 1.08025 | 1.04230 | 0.9989924 | 0.9998678 |
| V7 | 0.94372 | 0.90681 | 0.9989908 | 0.9998638 |
| Prism Residual | 1.02528 | 1.00231 | 0.9990181 | 0.9998798 |
| Prism Polar | 0.96746 | 0.94865 | 0.9990180 | 0.9998810 |
| Prism Polar Prob | 0.91551 | 0.89023 | 0.9990136 | 0.9998783 |
| Prism Cartesian Prob | 0.96805 | 0.93918 | 0.9990139 | 0.9998779 |

### All independent interpolation variants

| Method | RGB CPSNR | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM | Luma FSIM |
|---|---:|---:|---:|---:|---:|
| conventional/bicubic | 46.67323 | 0.73365 | 0.65436 | 0.9989842 | 0.9998538 |
| conventional/bilinear | 47.24729 | 0.69438 | 0.60461 | 0.9989883 | 0.9998553 |
| scaled_decode_first/bicubic | 47.72881 | 0.52371 | 0.46452 | 0.9996276 | 0.9999398 |
| scaled_decode_first/bilinear | 48.42126 | 0.48547 | 0.41602 | 0.9996040 | 0.9999397 |
| scaled_hybrid/bicubic | 46.88130 | 0.63485 | 0.56879 | 0.9995943 | 0.9999368 |
| scaled_hybrid/bilinear | 47.61003 | 0.58060 | 0.50575 | 0.9996023 | 0.9999392 |
| scaled_lifting/bicubic | 46.87359 | 0.63492 | 0.56939 | 0.9995913 | 0.9999361 |
| scaled_lifting/bilinear | 47.60074 | 0.58073 | 0.50645 | 0.9995994 | 0.9999386 |
| scaled_matrix/bicubic | 47.47628 | 0.57695 | 0.50878 | 0.9995864 | 0.9999344 |
| scaled_matrix/bilinear | 48.14969 | 0.53565 | 0.45729 | 0.9995990 | 0.9999373 |

Published CPSNR: conventional bilinear 47.33; conventional bicubic 46.28; scaled/lifting bilinear 48.31; scaled/lifting bicubic 47.42 dB.
Our conventional bilinear differs from the published value by -0.08271 dB.

## kodak / center_box

All 24 native images. RGB CPSNR averages per-image dB; higher is better.

| Method | RGB CPSNR (dB) | Gain vs bilinear | Gain vs best independent scaled method | Wins vs scaled |
|---|---:|---:|---:|---:|
| Conventional bilinear | 45.54629 | — | -1.59549 | — |
| scaled_matrix/bicubic | 47.14178 | +1.59549 | — | — |
| V5 | 40.71356 | -4.83274 | -6.42823 | 0/24 |
| V5.1 | 44.27623 | -1.27006 | -2.86555 | 0/24 |
| V6 | 47.94419 | +2.39790 | +0.80241 | 18/24 |
| V7 | 46.22099 | +0.67470 | -0.92079 | 3/24 |
| Prism Residual | 46.43228 | +0.88599 | -0.70950 | 3/24 |
| Prism Polar | 46.50336 | +0.95706 | -0.63843 | 5/24 |
| Prism Polar Prob | 46.42881 | +0.88252 | -0.71297 | 3/24 |
| Prism Cartesian Prob | 46.54530 | +0.99901 | -0.59648 | 3/24 |

The strongest independent comparator is selected by dataset mean, not separately for each image. This is not an author-code comparison.

### Learned models: secondary metrics

| Model | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM (higher) | Luma FSIM (higher) |
|---|---:|---:|---:|---:|
| V5 | 1.64075 | 1.72018 | 0.9984608 | 0.9995568 |
| V5.1 | 0.89511 | 1.04392 | 0.9985247 | 0.9995766 |
| V6 | 0.68848 | 0.62826 | 0.9990369 | 0.9998834 |
| V7 | 0.77044 | 0.71834 | 0.9990186 | 0.9998784 |
| Prism Residual | 0.75020 | 0.70540 | 0.9990342 | 0.9998832 |
| Prism Polar | 0.74611 | 0.70179 | 0.9990318 | 0.9998834 |
| Prism Polar Prob | 0.74938 | 0.70486 | 0.9990289 | 0.9998832 |
| Prism Cartesian Prob | 0.74123 | 0.69657 | 0.9990320 | 0.9998824 |

### All independent interpolation variants

| Method | RGB CPSNR | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM | Luma FSIM |
|---|---:|---:|---:|---:|---:|
| conventional/bicubic | 46.59564 | 0.71681 | 0.66908 | 0.9990152 | 0.9998707 |
| conventional/bilinear | 45.54629 | 0.78484 | 0.74884 | 0.9990117 | 0.9998659 |
| scaled_matrix/bicubic | 47.14178 | 0.61105 | 0.56852 | 0.9995761 | 0.9999477 |
| scaled_matrix/bilinear | 45.93562 | 0.69356 | 0.66076 | 0.9995628 | 0.9999420 |

## Interpretation limits

- Models receive the same conventional bilinear observation. Scaled methods alter the encoder representation as described in the paper; equal bitrate and equal intermediate bit depth are not established.
- Kodak centered-box sampling is a sensitivity experiment, not the paper's confirmed protocol.
- No model was trained or selected using these test scores. Checkpoints were frozen before evaluation.
- The legacy training-image manifests remain unknown. Prior overlap-audit fields in checkpoint metadata refer to the earlier COCO campaign; they do not certify absence of UHD/Kodak training overlap.
- These RGB scores differ in metric and degradation from the earlier COCO chroma-PSNR experiment.

## Model size

The paper's proposed method uses analytically computed transform coefficients and interpolation, with zero trainable parameters and no neural-network checkpoint.

| Model | Trainable parameters | FP32 weights only (MiB) |
|---|---:|---:|
| V5 | 1,925,667 | 7.346 |
| V5.1 | 1,925,667 | 7.346 |
| V6 | 593,794 | 2.265 |
| V7 | 595,525 | 2.272 |
| Prism Residual | 593,794 | 2.265 |
| Prism Polar | 594,371 | 2.267 |
| Prism Polar Prob | 595,525 | 2.272 |
| Prism Cartesian Prob | 594,948 | 2.270 |

Weight storage excludes activations, optimizer state, and other checkpoint contents. These are not runtime-memory or measured-speed comparisons.

## Downloads

- [Aggregate metrics](comparison.csv)
- [Per-image metrics](per_image.csv)
- [Results by resolution](by_resolution.csv)
- [Chart (PDF)](comparison_plot.pdf)
- [Coverage and provenance audit](audit.json)
