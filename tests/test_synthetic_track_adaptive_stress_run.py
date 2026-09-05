from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from railway_recon.synthetic_track_adaptive_stress_run import run_adaptive_stress


class AdaptiveStressRunTests(unittest.TestCase):
    def test_complete_negative_results_and_artifact_integrity_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "stress"
            report = run_adaptive_stress(output)
            self.assertEqual(report["case_count"], 18)
            self.assertFalse(report["registered_test_generated"])
            self.assertEqual(report["base_layout_count"], 6)
            self.assertEqual(len(report["summary_by_mode"]), 4)
            for summary in report["summary_by_mode"].values():
                self.assertEqual(summary["truth_defect_count"], 18)
            clutter = report["by_condition"]["wrong_connection_bridge_clutter"]["adaptive_evidence"]
            self.assertGreater(clutter["diagnosis"]["fn"], 0)
            manifest = json.loads((output / "manifest.json").read_text())
            for item in manifest["files"]:
                raw = (output / item["path"]).read_bytes()
                self.assertEqual(len(raw), item["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
            with self.assertRaises(FileExistsError):
                run_adaptive_stress(output)
