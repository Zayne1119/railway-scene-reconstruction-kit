from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.holdout_bootstrap import bootstrap_rail_holdout_comparison
from railway_recon.io import load_json, write_json


def _segment(segment_id: str, value: float) -> dict:
    stats = {"p90_m": value}
    return {
        "segment_id": segment_id,
        "line_repeatability": {
            "train_match_recall": value,
            "holdout_match_precision": value,
            "lateral_difference": stats,
            "vertical_difference": stats,
        },
        "track_pair_repeatability": {"raw_gauge_difference": stats},
        "model_to_holdout_support": {
            "distance": stats,
            "coverage_at_0_05m": value,
            "coverage_at_0_10m": value,
        },
    }


class HoldoutBootstrapTests(unittest.TestCase):
    def test_bootstrap_is_paired_deterministic_and_exploratory_for_four_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.json"
            method = root / "method.json"
            output = root / "bootstrap.json"
            common = {
                "schema_version": "railway.rail-holdout-evaluation.v1",
                "holdout_manifest_sha256": "a" * 64,
            }
            write_json(
                baseline,
                {**common, "segments": [_segment(f"s{i}", 0.5) for i in range(4)]},
            )
            write_json(
                method,
                {**common, "segments": [_segment(f"s{i}", 0.7) for i in range(4)]},
            )
            bootstrap_rail_holdout_comparison(
                baseline, method, output, iterations=1000, seed=7
            )
            result = load_json(output)
            self.assertEqual(
                result["overall_status"],
                "exploratory_insufficient_independent_blocks",
            )
            recall = result["metrics"]["train_line_match_recall"]
            self.assertEqual(recall["bootstrap_probability_method_better"], 1.0)
            lateral = result["metrics"]["lateral_repeatability_p90_m"]
            self.assertEqual(lateral["bootstrap_probability_method_better"], 0.0)


if __name__ == "__main__":
    unittest.main()
