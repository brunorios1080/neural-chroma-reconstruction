# Neural Chroma Reconstruction paper

This directory contains the research-style paper for the repository's V6
residual chroma refiner. The paper keeps three evidence tiers separate:

- the reproducible, paired 20-crop COCO/TabPFN pilot documented at repository
  revision `f4c8563`; and
- the archived 3,703-image evaluation recorded in the project README; and
- the repository-local procedural fixture audit, which validates the expanded
  benchmark but is not treated as scientific performance evidence.

Build the PDF from this directory with:

```bash
latexmk -pdf main.tex
```

Clean generated LaTeX intermediates with:

```bash
latexmk -c
```

The draft now documents the lossless manifest contract, 12-condition
siting/filter grid, real codec round trips, classical and learned baselines,
ablation matrix, expanded quality/resource metrics, and qualitative
difficult-boundary export. The included qualitative figure comes from the
procedural fixture and carries an explicit evidence disclaimer.

Before submission, supply a curated local lossless corpus, commit its manifest,
run the full ablation and benchmark configurations, commit the resulting
per-image data and natural-image crops, then replace the intentionally absent
publication-scale results. The compiled `main.pdf` should also be adapted to the
target venue's document class.
