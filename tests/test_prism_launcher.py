"""Exercise launcher limits without allocating a GPU or importing torch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PrismLauncherTests(unittest.TestCase):
    def test_submission_overrides_site_export_none(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/bridges2/submit_prism.sh"),
             "--dry-run", "prism_residual"],
            env={**os.environ, "SBATCH_EXPORT": "NONE", "NCR_PYTHON": sys.executable},
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--export=ALL", result.stdout)

    def launch(self, count, config="prism_gpu_smoke.json", limit=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text(''.join(json.dumps({"split": "train"}) + '\n' for _ in range(count)))
            commands = {
                "python3": '#!/bin/bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nexec "$TEST_PYTHON" "$@"\n',
                "srun": '#!/bin/bash\nprintf "TRAINING %s\\n" "$*"\n',
                "nvidia-smi": '#!/bin/bash\nexit 0\n',
            }
            for name, source in commands.items():
                path = root / name
                path.write_text(source)
                path.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("NCR_", "SLURM_"))}
            env.pop("BASH_ENV", None)
            env.update(
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                TEST_PYTHON=sys.executable,
                SLURM_JOB_ID="test",
                SLURM_SUBMIT_DIR=str(ROOT),
                NCR_MANIFEST=str(manifest),
                NCR_OUTPUT_ROOT=str(root / "outputs"),
            )
            if limit is not None:
                env["NCR_MAX_IMAGES"] = limit
            return subprocess.run(
                ["bash", "-c", 'module() { :; }; export -f module; exec bash "$@"',
                 "test", str(ROOT / "scripts/bridges2/train_prism.sbatch"),
                 str(ROOT / "research/configs" / config), "all"],
                env=env, text=True, capture_output=True,
            )

    def test_config_limits_smoke_without_environment_variable(self):
        result = self.launch(100)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"smoke_max_images": 100', result.stdout)
        self.assertIn('outputs_smoke100', result.stdout)
        self.assertIn('TRAINING ', result.stdout)

    def test_oversized_manifest_never_starts_training(self):
        result = self.launch(101)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing smoke training', result.stderr)
        self.assertNotIn('TRAINING ', result.stdout)

    def test_production_config_has_no_smoke_limit(self):
        result = self.launch(101, config="prism_coco.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('smoke_max_images', result.stdout)
        self.assertIn('TRAINING ', result.stdout)

    def test_explicit_limit_and_invalid_limit(self):
        result = self.launch(80, limit="80")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"smoke_max_images": 80', result.stdout)
        for limit in ("0", "49", "invalid"):
            result = self.launch(100, limit=limit)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('TRAINING ', result.stdout)


if __name__ == "__main__":
    unittest.main()
