# Dataset directory

Suggested layout:

```text
data/
├── raw/
│   └── coco/
│       ├── unlabeled2017.zip  # original archive; keep compressed on shared storage
│       ├── test2014.zip       # held-out model testing against bilinear interpolation
│       └── annotations/       # COCO image metadata
└── processed/                     # optional validated copies
```

Run `python scripts/prepare_dataset.py --input data/raw --output data/processed`
from the project root, or pass any image directory directly to the trainer.

On Bridges-2, do not store the dataset in this `$HOME` repository. Keep the packed
archive at `/ocean/projects/cis260224p/shared/$USER/data/coco/unlabeled2017.zip` and use
the Slurm launcher under `scripts/bridges2/`. Each job extracts it only to
node-local storage rather than creating 123,403 files on either shared filesystem.

The COCO 2014 test archive and its image metadata are stored beside that training
archive. `scripts/prepare_coco_test2014.py` reads the ZIP without extracting it,
checks image CRCs, hashes and metadata dimensions, and writes an all-test manifest
under the shared `data/coco/manifests/` directory. Evaluation uses distinct images
large enough for 256-pixel crops, excluding any training/validation image ID or
content hash found in the supplied model manifests. Original downloaded images
remain intact in the archive.

Use `research/configs/prism_coco_test2014.json` for this test set. Its evaluation
baseline is bilinear interpolation. The batch script
`scripts/bridges2/evaluate_prism_test2014.sbatch` stages images on node-local storage
and takes explicit checkpoint paths; it does not train or select checkpoints.
