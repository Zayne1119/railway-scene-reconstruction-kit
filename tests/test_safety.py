import tempfile
import unittest
from pathlib import Path

from railway_recon.safety import safety_check


class SafetyTests(unittest.TestCase):
    def test_safety_accepts_plain_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("synthetic example", encoding="utf-8")
            self.assertTrue(safety_check(root)["passed"])

    def test_safety_rejects_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.laz").write_bytes(b"not a real point cloud")
            report = safety_check(root)
            self.assertFalse(report["passed"])
            self.assertEqual(report["findings"][0]["kind"], "banned_extension")

    def test_safety_skips_private_local_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "benchmarks" / "local" / "site-a"
            private.mkdir(parents=True)
            (private / "manifest.json").write_text(
                '{"source":"D:\\\\Railway\\\\private.laz"}', encoding="utf-8"
            )
            self.assertTrue(safety_check(root)["passed"])

    def test_safety_skips_local_dependency_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / ".uv-cache" / "archive"
            cache.mkdir(parents=True)
            (cache / "large-secret-looking.bin").write_bytes(b"x" * 1024)
            (cache / "example.py").write_text(
                "api" + '_key = "dependency-fixture-value"\n', encoding="utf-8"
            )
            self.assertTrue(safety_check(root, maximum_file_bytes=128)["passed"])
