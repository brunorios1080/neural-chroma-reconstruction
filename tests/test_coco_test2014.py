"""Test-only archive preparation keeps training content out of evaluation."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts.prepare_coco_test2014 import prepare


class CocoTest2014Tests(unittest.TestCase):
    def test_test_only_manifest_excludes_overlap_small_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, payloads = [], []
            for identifier, (size, color) in enumerate(
                [(32, 10), (32, 20), (16, 30), (32, 40), (32, 40), (32, 60)], 1
            ):
                encoded = io.BytesIO()
                Image.new("RGB", (size, size), (color, 0, 0)).save(encoded, format="JPEG")
                payloads.append(encoded.getvalue())
                images.append({"file_name": f"COCO_test2014_{identifier:012d}.jpg",
                               "id": identifier, "width": size, "height": size})
            archive, metadata = root / "test.zip", root / "metadata.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for info, raw in zip(images, payloads):
                    output.writestr("test2014/" + info["file_name"], raw)
            with zipfile.ZipFile(metadata, "w") as output:
                output.writestr("annotations/image_info_test2014.json", json.dumps({"images": images}))
            used = root / "training.jsonl"
            used.write_text(json.dumps({"split": "train", "relative_path": "unlabeled2017/000000000001.jpg",
                                        "sha256": "different"}) + "\n" +
                            json.dumps({"split": "validation", "relative_path": "unlabeled2017/000000000099.jpg",
                                        "sha256": hashlib.sha256(payloads[1]).hexdigest()}) + "\n")
            manifest = root / "manifest.jsonl"
            report = prepare(archive, metadata, manifest, [used], min_size=32)
            self.assertEqual(report["archive_images"], 6)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["excluded"], {"training_validation_overlap": 2,
                                                  "too_small": 1, "duplicate_content": 1})
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual({row["split"] for row in rows}, {"test"})
            self.assertEqual(report["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertTrue(manifest.with_suffix(".jsonl.meta.json").is_file())


if __name__ == "__main__":
    unittest.main()
