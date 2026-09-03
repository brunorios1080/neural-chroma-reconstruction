# V7 design and repository audit

V7 tests whether a polar/complex chroma representation with separate magnitude
and circular-phase uncertainty is useful for 4:2:0 reconstruction. It does not
claim an improvement over V6, and the procedural fixture is software validation,
not scientific evidence.

## Existing repository conventions

- `chroma/models.py::ChromaRefiner` is V6: `[Y, Cr, Cb]` input, a 3x3
  3-to-64 stem, eight `conv -> ReLU -> conv` residual blocks, and a 3x3
  64-to-2 Cartesian residual tail. Y is passed through. It has 593,794 trainable
  parameters. `scripts/train.py --model v6` and the historical
  `scripts/Model V6/train.py` wrapper use `chroma/training.py`.
- Legacy checkpoints are handled by `chroma/checkpoints.py`. Current checkpoints
  contain strict model state, optimizer state, epoch, configuration, and model
  version; legacy V5 `G` and V6 `model` layouts remain supported.
- The legacy image path (`chroma/data.py`) uses OpenCV
  `COLOR_RGB2YCrCb`, uint8 conversion, then division by 255. Its tensor order is
  exactly `[Y, Cr, Cb]`; neutral chroma is therefore the code value 128, i.e.
  **128/255 = 0.5019607843**, not 0.5. Its synthetic operator uses OpenCV area
  resize to `ceil(W/2) x ceil(H/2)` followed by configurable resize (bilinear by
  default).
- The publication path (`chroma/research_data.py`) deliberately uses a floating
  analytical full-range BT.601-style conversion. Its order is also `[Y, Cr,
  Cb]`, but its equations add exactly **0.5**, so its neutral chroma is **0.5**.
  It requires even crops and implements explicit center, left, and cosited sample
  locations; point/box/triangle/Gaussian/Lanczos3 downsampling; and
  nearest/bilinear/bicubic/Lanczos3 upsampling. `simulate_420` returns both the
  full-resolution model input and the actual low-resolution observed chroma.
- `chroma/research_codecs.py` performs real Pillow JPEG 4:2:0 and optional local
  FFmpeg H.264, HEVC, and AV1 round trips.
- `chroma/research_models.py` and `chroma/research_training.py` contain the V6
  ablation family, checkpoint format, training matrix, parameter/FLOP counting,
  and SRCNN/NAF-style controls. The full and smoke matrices are under
  `research/configs/`.
- `chroma/research_benchmark.py` is the manifest-driven publication benchmark.
  It retains classical, V5/V6, retrained V6-family, SRCNN, and NAF-style methods,
  selects difficult boundaries using baseline edge error, and writes
  `per_image.jsonl`, aggregates, provenance, and qualitative sheets.
  `chroma/research_metrics.py` implements chroma/RGB PSNR and SSIM, chroma MAE,
  edge/gradient error, and mean/p95 CIEDE2000. The older COCO pilot is
  `scripts/benchmark_coco_tabpfn.py` with notes in
  `notes/tabpfn_v3_coco_pilot.md`.

V7 uses the manifest-driven publication path and therefore defaults to
`neutral_chroma = 0.5`. This value is stored in every V7 checkpoint and checked
strictly on load. Callers using legacy OpenCV tensors must explicitly construct a
V7 configuration with `neutral_chroma = 128/255`; conversion is never inferred.

## V7 controlled architecture

The V6 trunk is preserved exactly: 3-to-64 stem and eight 64-channel residual
blocks. Only the tail changes from two Cartesian maps to five maps:

1. amplitude residual mean `delta_A`;
2. raw cosine coordinate for phase residual;
3. raw sine coordinate for phase residual;
4. raw Laplace amplitude scale;
5. raw von Mises phase concentration.

For centered chroma `u = Cr - c0`, `v = Cb - c0`,

```
A = sqrt(u^2 + v^2)
phi = atan2(v, u)
Cr = c0 + A cos(phi)
Cb = c0 + A sin(phi)
```

The mean is residual: `A_hat = clamp(A0 + delta_A, 0, A_max)` and
`phi_hat = wrap(phi0 + atan2(q_sin, q_cos))`. The tail is initialized so
`delta_A=0`, `(q_cos,q_sin)=(1,0)`, making the initial mean exactly the bilinear
input up to floating-point roundoff. The physical radial bound is the
direction-dependent distance from `(c0,c0)` to the edge of `[0,1]^2`, so the
inverse Cartesian chroma remains in range without post-hoc clipping.

The V7 trunk plus five-map head has 595,525 trainable parameters at width 64 and
depth 8. This is 1,731 more than V6, solely from the three additional output
maps.

The controlled `v7_ablations.json` matrix matches the existing retrained V6
control's manifest, crops, batch size, degradation draw, AdamW settings, seed,
and validation selection. AMP and gradient clipping are disabled in that matrix
because the existing V6 control does not use them. The standalone V7 config may
enable both for later non-attribution runs.

## Objective and numerical policy

The configured supervised loss is

```
L = lambda_cart * mean(|C_hat-C|)
  + lambda_amp * mean(|A-A_hat|/b_A + log(2 b_A))
  + lambda_phase * sum(w_phi * L_vm) / sum(w_phi)
  + lambda_forward * mean(|D(C_hat)-C_low|)
```

where `b_A = softplus(raw_scale) + eps`,
`kappa = clamp(softplus(raw_kappa) + eps, kappa_min, kappa_max)`, and

```
L_vm = -kappa cos(phi-phi_hat) + log(2 pi) + log(I0(kappa)).
```

`log(I0(kappa))` uses `log(i0e(kappa)) + |kappa|`. Float32 computation is
forced for the special function under mixed precision. The initial documented
cap is `kappa_max=100`; it is a numerical guard, not an empirical optimum.
An accidentally zero-length raw phase vector is mapped to the identity direction
`(1, 0)` before `atan2`, avoiding its undefined gradient at `(0, 0)`.
Hue is undefined close to neutral, so `w_phi = clamp(A/phase_reference, 0, 1)`.
The reference amplitude is configurable. Forward consistency uses the exact
publication degradation matrices for the observation's siting and prefilter.

Amplitude central intervals use the exact Laplace half-width
`-b_A log(1-p)`. Phase intervals are circular and evaluation-only: a numerical
von Mises CDF lookup maps kappa to a symmetric angular half-width. Maps also
export kappa and circular variance `1-I1(kappa)/I0(kappa)`; approximate intervals
are labeled as such.

Safe inference is not used by the training loss. It attenuates magnitude
residuals with `exp(-b_A / scale_reference)` and phase residuals with
`kappa/(kappa+kappa_reference)`, optionally modulated by predicted amplitude near
neutral. All references are configuration values. Phase interpolation uses the
short wrapped circular difference, never a linear interpolation across ±pi.

Optional self-training uses an EMA teacher, confidence-weighted accepted pseudo
labels, magnitude-scale, phase-concentration, forward-consistency, and (only for
center-sited observations) horizontal-flip consistency tests. It excludes test
records/hashes, enforces supervised updates, caps pseudo-label loss contribution,
logs acceptance/disagreement, and preserves the starting teacher checkpoint.
It is disabled by default and is a separate second-stage command.
