# Experiment checkpoints

This directory contains the `best.pth` checkpoint from each saved experiment run under the Bridges-2 and home checkpoint directories. Paths below this directory preserve the original run names. The accompanying JSON and JSONL files preserve configuration, provenance, summaries, and training history where available.

Historical V5 and V6 weights are already tracked in `models/version5/` and `models/version6/`.

The frozen weights actually used for the COCO test2014 comparison are under [`models/evaluation/test2014_all_20260905`](../evaluation/test2014_all_20260905/README.md). Four of those were selected before the later Prism runs advanced, so they are kept separately.

Only image IDs and split membership are committed under [`research/manifests/ids`](../../research/manifests/ids/README.md). No COCO images or image archives are included.
