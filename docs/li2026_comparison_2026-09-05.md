# Li et al. (2026) comparison

The requested comparison targets Li, Zhang, and Huang, *High-Fidelity Ultra-High-Definition Chroma Reconstruction in the Scaled YCbCr Colour Space via LUL0 Factorisation and Generalised Lifting*, IET Image Processing 20 (2026), e70338, <https://doi.org/10.1049/ipr2.70338>.

## Evidence and scope

- Paper data: UHD240, Zenodo record 17649711, <https://doi.org/10.5281/zenodo.17649711>.
- Additional complete benchmark: the 24 Kodak PNGs from <https://r0k.us/graphics/kodak/>.
- Local data: `/ocean/projects/cis260224p/shared/brios/data/li2026`.
- Results and frozen source: `/ocean/projects/cis260224p/shared/brios/evaluations/li2026_20260905`.
- All 240 UHD files are PNGs. Each archive contains exactly 80 images. Downloads passed the authors' MD5 checks and ZIP CRC validation. The data directory contains the complete Zenodo metadata, source URLs, byte counts, SHA-256 hashes, and member lists.
- This campaign covers Tables 2 and 3. The BSD split and SCID PPT membership are unspecified; Table 6 also describes a different block-prediction comparison. Those tables are not represented as reproduced.

No executable implementation accompanied the cited Zenodo record. All named `scaled_*` methods here are **independent implementations**, with their differences retained in results. Scores must not be represented as confirmed reproduction of the authors' image tables.

## Confirmed transform reproduction

The exhaustive 256^3-color check exactly reproduces all three Table 1 component counts when using the matrix printed to four decimals:

| Transform | Paper | Reproduced |
|---|---:|---:|
| Conventional matrix | 30,378,628 | 30,378,628 |
| Scaled matrix | 45,095,625 | 45,095,625 |
| Reversible lifting | 50,331,648 | 50,331,648 |

The scale vector is approximately `(1.8387701823, 1.8954645560, 1.8324918886)`. The conventional matrix, scaling optimization, rounding convention, and integer reversibility are therefore supported by an exact independent check.

The online rendering of Equation 27 has `+M31`. With the paper's Equation 21 definition, that gives a factorization error of approximately 18.15656 in max absolute matrix entry. Using `-M31` reconstructs the specified matrix. The implementation explicitly records this correction; it does not silently claim to execute the printed expression unchanged. The floating-point inverse uses the actual inverse matrix with offsets canceled, rather than the misplaced offset subtraction displayed in Equation 2.

Scaled integer values are retained in their actual numerical range, including negatives and values above 255. Clipping these intermediate channels to 8 bits would invalidate the transform reproduction. Fixed-bit-depth or compression-rate equivalence is not established by these quality-only tests.

## Image protocol

Images retain full native dimensions and decoded RGB samples. No crops, resizing, gamma reapplication, training, or test-dependent checkpoint selection are performed. Standard RGB CPSNR uses all channels and all pixels, `10 log10(255^2 / MSE)`, then averages per-image dB. MAE is in 0–255 RGB units.

The primary interpretation uses point samples on the even-row/even-column lattice, spacing two, and bilinear or Keys bicubic (`a=-0.5`) interpolation with edge replication. Kodak additionally uses centered 2x2 box downsampling as a sensitivity check. Sampling, borders, and the interface between lifting and interpolation are not completely specified in the paper.

Retained interpretation variants:

- `conventional`: rounded conventional matrix, chroma sampling/interpolation, rounded inverse.
- `scaled_matrix`: same operations with the scaled matrix.
- `scaled_hybrid`: lifting codes at retained sites, matrix luminance elsewhere; interpolation of coded chroma; inverse lifting at retained sites and matrix inverse elsewhere.
- `scaled_lifting`: lifting at every forward pixel, followed by the same mixed inverse.
- `scaled_decode_first`: recover retained RGB from the transmitted lifting triplets first; convert those decoded samples to unrounded scaled chroma for interpolation. This uses only decoded retained samples and available luminance. An invariance test changes unavailable chroma while preserving the observation and verifies the output cannot depend on it.

For every learned method, input comes from the conventional bilinear observation. The adapter converts the observed studio-range YCbCr to the trained normalized YCrCb representation; legacy models use chroma neutral `128/255`, Prism/V7 use `0.5`. No original off-grid chroma enters a model. Outputs use each checkpoint's mean prediction, then common RGB rounding and clipping. Thus these compare complete reconstruction pipelines; they are not claims about identical encoded bitstreams or bitrate.

The eight evaluated checkpoints are the frozen V5, V5.1, V6, V7, and four main Prism checkpoints from the completed COCO campaign. Their weights, epochs, and hashes are recorded. Smoke checkpoints are excluded. Historical training-manifest limitations persist.

## Verification and reporting

Six numerical invariants passed. Full-versus-tiled inference passed for all eight real checkpoints: max discrepancy was zero for V6/V7/Prism and at most 1.8e-7 for V5-family models. Tiles have a 512-pixel core and 64-pixel halo; inference uses float32 with TF32 disabled. Transforms and CPSNR use float64.

Secondary metrics explicitly document their implementation: CIEDE2000 over all sRGB/D65 Lab pixels, luminance SSIM with valid 11x11 Gaussian windows, and PIQ 0.8.0 nonchromatic FSIM with its default pooling. The paper does not supply enough implementation detail to call those exact metric reproductions.

The initial native Kodak baseline is 47.24729 dB for bilinear and 46.67323 dB for bicubic, versus the paper's 47.33 and 46.28 dB. Matrix-only scaling gives 48.14969 and 47.47628 dB. The different lifting/interpolation interpretations yield different scores. Consequently, exact image-table reproduction remains **unverified** even though Table 1 matches exactly.

The collector pairs each model with every independently implemented comparator on identical image IDs. Its output includes `exact_published_protocol_verified: false`. A result against an independent implementation must be described that way, with the published figures listed separately. This prevents a protocol mismatch from being presented as proof of superiority over the paper.

## Completed Kodak results

All 24 native images and all eight frozen checkpoints completed. These are RGB CPSNR values, not the chroma-only PSNR used in the previous COCO evaluation.

| Method | Point sampling | Centered 2x2 box sampling |
|---|---:|---:|
| Conventional bilinear | 47.24729 | 45.54629 |
| Conventional bicubic | 46.67323 | 46.59564 |
| Scaled matrix bilinear | 48.14969 | 45.93562 |
| Scaled matrix bicubic | 47.47628 | 47.14178 |
| Scaled decode-first bilinear | 48.42126 | Not defined |
| V5 | 38.29608 | 40.71356 |
| V5.1 | 40.52858 | 44.27623 |
| V6 | 43.18509 | 47.94419 |
| V7 | 43.87302 | 46.22099 |
| Prism Residual | 43.08891 | 46.43228 |
| Prism Polar | 43.48914 | 46.50336 |
| Prism Polar Prob | 43.98658 | 46.42881 |
| Prism Cartesian Prob | 43.53417 | 46.54530 |

Sampling changes both the degradation and model ranking. V6 gains 2.39790 dB over bilinear on centered-box Kodak but loses 4.06220 dB on point-sampled Kodak. Neither result can be transferred to an unspecified sampling protocol. The centered-box experiment does not implement the retained-site lifting construction because box samples do not correspond to retained source pixels.

The training configurations provide relevant context: V6's historical pipeline uses OpenCV area downsampling and bilinear upsampling. The main Prism and V7 configurations include box, triangle, Gaussian, and Lanczos filtering at three sitings; direct point decimation is absent. Point sampling is therefore outside these documented degradation configurations. That is a plausible contributor to the observed change, not an isolated causal measurement. Also, V6 preserves observed luminance, whereas the paper changes the encoder transform for luminance as well as chroma. Full RGB reconstruction scores include that difference.

## Execution and result audit

Kodak job `45301760_0` completed all 720 image/method pairs. A native 8K probe (`45301761_3`) completed all 18 methods. The perceptual-metric accelerator passed CPU/reference comparisons, including a full 7680x4320 image: maximum discrepancy across metric outputs was 4.44e-16. CPSNR, MAE, and FSIM agreed exactly in that check. This permits equivalent CPU and GPU metric rows to be combined with their source provenance retained.

The original UHD jobs `45302813`, `45302814`, and `45302817` were superseded because measured runtimes exceeded the initial estimates. Array `45303007` partitions all 240 UHD images into 16 disjoint sets of 15 images, with five images per resolution in each set. Its initial concurrency limit was eight, subsequently raised to 16 after checking available V100 capacity. Each task has a one-hour cap (16 GPU-hours combined); raising concurrency does not increase the task count or that combined cap. A total of 229 completed rows were imported without rerunning; `sharding.json` records their original configurations and hashes. Original results remain available. Source snapshots are separate from the actively edited repository.

Tasks 8 and 9 failed after 27 seconds on `v005`, before inference, with CUDA error 803 (unsupported display/CUDA driver combination). Node `v005` was excluded from remaining tasks. Replacement array `45303339_[8-9]` uses the same frozen code and shard output directories. The two failed attempts consumed 0.015 GPU-hours combined and produced no evaluation rows.

`scripts/report_li2026.py` verifies expected image/method coverage, unique pairs, identical ground-truth RGB hashes, the same eight checkpoint hashes, and finite metrics. It produces `audit.json`, `RESULTS.md`, and, after completion, `by_resolution.csv`. The final campaign must contain 5040 image/method pairs: 4320 from UHD and 720 from Kodak. Partial UHD coverage is explicitly excluded from final rankings.

The legacy training-image manifests remain unknown. The inherited `overlap_audit` fields in checkpoint metadata refer to the previous COCO evaluation, and do not certify an overlap audit on the present UHD/Kodak images.

## Final UHD240 results

The campaign completed on September 6, 2026. All 240 UHD images and all 24 Kodak images are present, with 5040 image/method pairs, eight matching checkpoint hashes, consistent settings, identical decoded ground truth across methods, and no missing or duplicate pairs. Both failed node attempts were recovered. All 16 required UHD partitions completed successfully.

| Method | UHD240 RGB CPSNR (dB) |
|---|---:|
| Conventional bilinear | 51.24215 |
| Conventional bicubic | 51.27593 |
| Scaled matrix bilinear | 54.45641 |
| Scaled matrix bicubic | 54.61254 |
| Scaled decode-first bilinear | 56.42125 |
| Scaled decode-first bicubic | 56.81730 |
| V5 | 41.64610 |
| V5.1 | 45.88247 |
| V6 | 50.27362 |
| V7 | 50.70133 |
| Prism Residual | 51.04452 |
| Prism Polar | 50.93381 |
| Prism Polar Prob | 50.73028 |
| Prism Cartesian Prob | 51.12490 |

Under the explicitly chosen point-sampling protocol, every learned checkpoint has a lower mean CPSNR than conventional bilinear on both UHD240 and Kodak. Prism Cartesian Prob is the strongest learned model on UHD240, losing 0.11726 dB on average and beating bilinear on 47/240 images. V6 loses 0.96854 dB and beats bilinear on 0/240 images. Scaled-matrix bilinear exceeds every learned model on all 240 UHD images. Every retained scaled-transform interpretation also exceeds every learned model in dataset mean CPSNR.

The published UHD conventional bilinear score is 50.49 dB, whereas this implementation gives 51.24215 dB, a difference of +0.75215 dB. Scaled-matrix bilinear happens to be close to the paper's reported 54.40 dB, but this does not resolve the baseline or lifting-interface discrepancies. Exact reproduction of the image tables remains unverified. The independently measured results are not proof that the authors' exact executable method has been reproduced.

Final artifacts in the case directory are `RESULTS.md`, `comparison.json`, `comparison.csv`, `per_image.csv` (5040 rows), `by_resolution.csv`, `comparison_plot.png`, `comparison_plot.pdf`, and `audit.json`. The reporting scripts and their hashes are preserved under `report_code/`. `PROTOCOL.md` is a copy of this document. The `4k_interim.json` and `6k_interim.json` files retain the earlier complete-resolution snapshots; the final aggregate supersedes those interim reports.
