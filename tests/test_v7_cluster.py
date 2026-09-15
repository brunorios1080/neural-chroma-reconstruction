from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_coco_manifest import assign_splits, build_coco_manifest
from scripts.train_v7 import apply_overrides


class V7ClusterTests(unittest.TestCase):
    def test_coco_split_is_exact_and_deterministic(self) -> None:
        paths = [Path(f"{index:012d}.jpg") for index in range(100)]
        first = assign_splits(paths, 0.03, 0.02, 2026)
        second = assign_splits(list(reversed(paths)), 0.03, 0.02, 2026)
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "train" for value in first.values()), 95)
        self.assertEqual(sum(value == "validation" for value in first.values()), 3)
        self.assertEqual(sum(value == "test" for value in first.values()), 2)

    def test_coco_manifest_supports_flat_image_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "unlabeled2017"
            images.mkdir()
            for index in range(10):
                Image.new("RGB", (16, 12), (index, 2 * index, 3 * index)).save(
                    images / f"{index:012d}.jpg"
                )
            output = root / "manifest.jsonl"
            records = build_coco_manifest(images, root, output, None, 0.2, 0.1, 7)
            self.assertEqual(len(records), 10)
            self.assertEqual(sum(record.split == "train" for record in records), 7)
            self.assertEqual(sum(record.split == "validation" for record in records), 2)
            self.assertEqual(sum(record.split == "test" for record in records), 1)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".jsonl.meta.json").is_file())

    def test_coco_manifest_can_limit_a_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "unlabeled2017"
            images.mkdir()
            for index in range(20):
                Image.new("RGB", (16, 12)).save(images / f"{index:012d}.jpg")
            records = build_coco_manifest(
                images, root, root / "manifest.jsonl", None, 0.2, 0.0, 7, 10
            )
            self.assertEqual(len(records), 10)
            self.assertEqual(sum(record.split == "train" for record in records), 8)
            self.assertEqual(sum(record.split == "validation" for record in records), 2)

    def test_coco_manifest_excludes_images_smaller_than_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "unlabeled2017"
            images.mkdir()
            for index, size in enumerate(((8, 16), (16, 8), (16, 16), (20, 18))):
                Image.new("RGB", size).save(images / f"{index:012d}.jpg")
            records = build_coco_manifest(
                images, root, root / "manifest.jsonl", None, 0.5, 0.0, 7, None, 16
            )
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.width >= 16 for record in records))
            self.assertTrue(all(record.height >= 16 for record in records))

    def test_training_cli_overrides_are_non_destructive(self) -> None:
        source = {
            "manifest": "old.jsonl",
            "dataset_root": "old-data",
            "output_dir": "old-output",
            "training": {"epochs": 30, "amp": True},
        }
        args = argparse.Namespace(
            manifest=Path("new.jsonl"),
            dataset_root=Path("new-data"),
            output_dir=Path("new-output"),
            resume=Path("last.pth"),
            epochs=2,
            batch_size=8,
            workers=4,
            device="cuda",
            no_amp=True,
            skip_manifest_verification=True,
        )
        updated = apply_overrides(source, args)
        self.assertEqual(source["manifest"], "old.jsonl")
        self.assertEqual(updated["manifest"], "new.jsonl")
        self.assertEqual(updated["training"]["epochs"], 2)
        self.assertFalse(updated["training"]["amp"])
        self.assertFalse(updated["training"]["verify_manifest"])


if __name__ == "__main__":
    unittest.main()
