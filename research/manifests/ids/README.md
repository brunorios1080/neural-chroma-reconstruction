# Image IDs used by experiments

Each text file contains one image ID per line. The prefix names the source corpus and the suffix names the split. These files contain no image data. The source files are JPEGs, despite some earlier references to PNGs.

The IDs are taken from the exact manifests below; SHA-256 values identify those original manifests. V5/V6 historical training IDs are unavailable, as noted in the main README.

| ID list | Count | Source manifest SHA-256 |
| --- | ---: | --- |
| [coco_unlabeled2017_test.txt](coco_unlabeled2017_test.txt) | 2,450 | `da0e993e233183bc27e2324b482c516c785f187a92ded05acacb4141e30ac780` |
| [coco_unlabeled2017_train.txt](coco_unlabeled2017_train.txt) | 115,196 | `da0e993e233183bc27e2324b482c516c785f187a92ded05acacb4141e30ac780` |
| [coco_unlabeled2017_validation.txt](coco_unlabeled2017_validation.txt) | 4,903 | `da0e993e233183bc27e2324b482c516c785f187a92ded05acacb4141e30ac780` |
| [coco_smoke100_test.txt](coco_smoke100_test.txt) | 2 | `a328d265b4911f05522518a645a29fd7a6d7b35ec6623f6d828c7eaff9ea69fd` |
| [coco_smoke100_train.txt](coco_smoke100_train.txt) | 94 | `a328d265b4911f05522518a645a29fd7a6d7b35ec6623f6d828c7eaff9ea69fd` |
| [coco_smoke100_validation.txt](coco_smoke100_validation.txt) | 4 | `a328d265b4911f05522518a645a29fd7a6d7b35ec6623f6d828c7eaff9ea69fd` |
| [coco_li2026_pilot_train.txt](coco_li2026_pilot_train.txt) | 256 | `fd8680c672d841ee3e8edf9074d934007ca2733a47a750f39fdb12b087835217` |
| [coco_li2026_pilot_validation.txt](coco_li2026_pilot_validation.txt) | 64 | `fd8680c672d841ee3e8edf9074d934007ca2733a47a750f39fdb12b087835217` |
| [coco_test2014_test.txt](coco_test2014_test.txt) | 40,499 | `e3b7b309898e2fb5f92414977edb3bb5592c7052abe9bcc09a47aa5dcea70340` |
| [coco_v7_full_train.txt](coco_v7_full_train.txt) | 117,664 | Reconstructed from archive and metadata (see below) |
| [coco_v7_full_validation.txt](coco_v7_full_validation.txt) | 4,903 | Reconstructed from archive and metadata (see below) |
| [coco_v7_smoke100_train.txt](coco_v7_smoke100_train.txt) | 96 | Reconstructed from archive and metadata (see below) |
| [coco_v7_smoke100_validation.txt](coco_v7_smoke100_validation.txt) | 4 | Reconstructed from archive and metadata (see below) |
| [coco_v7_smoke2000_train.txt](coco_v7_smoke2000_train.txt) | 1,920 | Reconstructed from archive and metadata (see below) |
| [coco_v7_smoke2000_validation.txt](coco_v7_smoke2000_validation.txt) | 80 | Reconstructed from archive and metadata (see below) |
| [coco_v5_1_full_train.txt](coco_v5_1_full_train.txt) | 118,897 | Reconstructed from archive and metadata (see below) |
| [coco_v5_1_full_validation.txt](coco_v5_1_full_validation.txt) | 3,670 | Reconstructed from archive and metadata (see below) |
| [coco_v5_1_smoke100_train.txt](coco_v5_1_smoke100_train.txt) | 96 | Reconstructed from archive and metadata (see below) |
| [coco_v5_1_smoke100_validation.txt](coco_v5_1_smoke100_validation.txt) | 3 | Reconstructed from archive and metadata (see below) |
| [li2026_uhd240_test.txt](li2026_uhd240_test.txt) | 240 | Evaluation rows (see below) |
| [li2026_kodak_test.txt](li2026_kodak_test.txt) | 24 | Evaluation rows (see below) |

The COCO unlabeled2017 list covers the full Prism COCO runs, including the 100-epoch U-Net runs. The smoke and Li2026 lists record the smaller experiment subsets. The test2014 list records the held-out evaluation set.

The V7 lists were reconstructed because the launch script built its manifest on node-local storage and did not retain a copy. They use `scripts/build_coco_manifest.py`'s exact ordering, minimum image size of 256 pixels, seed 2026, and 4% validation / 0% test split. The source archive contains 123,403 JPEGs; 122,567 pass that size filter. The metadata file's SHA-256 is `659476821f469b90e8e47f44e95aee7afe7d59961bdb8d61418556a31ac5a11d`. The smoke lists apply the first 100 or 2,000 eligible sorted filenames before shuffling. These correspond to the launcher defaults; runs with environment overrides could have used a different split.

The V5.1 lists were reconstructed from `scripts/bridges2/train_v5_1.sbatch` and `chroma/data.py`: all 123,403 sorted JPEGs, seed 1337, and 3% validation. Small images are removed from these ID lists after splitting because the 256-pixel crop loader skips them. The full split contains 119,701 training and 3,702 validation files before that crop check. The smoke list applies the first 100 sorted filenames before splitting. As with V7, environment overrides could have changed an individual run.

The Li2026 UHD and Kodak test lists contain actual PNG filenames used in the completed 264-image comparison. They were taken from the unique `(dataset, id)` pairs in [`research/reports/li2026_20260905/per_image.csv`](../../reports/li2026_20260905/per_image.csv), whose SHA-256 is `3ca8d8f804f31c5a2950df5cdd47ba234bcbb42dfeccff365c7075d1904e2981` after normalizing CSV line endings.
