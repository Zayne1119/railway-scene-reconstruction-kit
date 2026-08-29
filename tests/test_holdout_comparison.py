from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.holdout_comparison import compare_rail_holdout_reports
from railway_recon.io import load_json, write_json


class HoldoutComparisonTests(unittest.TestCase):
    def _report(self, path: Path, lateral: float, vertical: float) -> None:
        write_json(
            path,
            {
                "schema_version": "railway.rail-holdout-evaluation.v1",
                "holdout_manifest_sha256": "a" * 64,
                "parameters": {
                    "core_length_m": 50.0,
                    "sample_step_m": 0.25,
                    "holdout_seed": 1,
                    "voxel_size_m": 0.5,
                },
                "summary": {
                    "train_line_match_recall": 0.5,
                    "holdout_line_match_precision": 0.5,
                    "line_lateral_repeatability": {"p90_m": lateral},
                    "line_vertical_repeatability": {"p90_m": vertical},
                    "raw_gauge_repeatability": {"p90_m": lateral},
                    "model_to_holdout_support_distance": {"p90_m": lateral},
                    "model_to_holdout_coverage_at_0_05m": 0.5,
                    "model_to_holdout_coverage_at_0_10m": 0.5,
                },
            },
        )

    def test_reports_wins_and_weaknesses_on_same_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.json"
            method = root / "method.json"
            output = root / "comparison.json"
            self._report(baseline, lateral=0.1, vertical=0.01)
            self._report(method, lateral=0.02, vertical=0.03)
            compare_rail_holdout_reports(baseline, method, output)
            result = load_json(output)
            self.assertEqual(result["metrics"]["lateral_repeatability_p90_m"]["winner"], "method")
            self.assertEqual(result["metrics"]["vertical_repeatability_p90_m"]["winner"], "baseline")


if __name__ == "__main__":
    unittest.main()
