# Dataset directory

Suggested layout:

```text
data/
├── raw/        # original images
└── processed/  # validated images; ignored by Git
```

Run `python scripts/prepare_dataset.py --input data/raw --output data/processed`
from the project root, or pass any image directory directly to the trainer.
