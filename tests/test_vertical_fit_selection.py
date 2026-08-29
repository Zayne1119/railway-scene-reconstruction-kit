from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.io import load_json, write_json
from railway_recon.vertical_fit_selection import select_rail_vertical_fit


def _evaluation(vertical: float, precision: float = 1.0) -> dict:
    return {
        "schema_version": "railway.rail-holdout-evaluation.v1",
        "holdout_manifest_sha256": "a" * 64,
        "parameters": {
            "maximum_match_distance_m": 0.3,
            "core_length_m": 50.0,
            "sample_step_m": 0.25,
            "voxel_size_m": 0.5,
            "holdout_fraction": 0.25,
            "holdout_seed": 7,
        },
        "summary": {
            "train_line_match_recall": 0.8,
            "holdout_line_match_precision": precision,
            "line_lateral_repeatability": {"p90_m": 0.02},
            "line_vertical_repeatability": {"p90_m": vertical},
            "raw_gauge_repeatability": {"p90_m": 0.03},
            "model_to_holdout_support_distance": {"p90_m": 0.2},
            "model_to_holdout_coverage_at_0_10m": 0.8,
        },
    }


class VerticalFitSelectionTests(unittest.TestCase):
    def test_selects_lowest_vertical_p90_that_passes_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variants = []
            for name, vertical, precision in (
                ("legacy", 0.05, 1.0),
                ("winner", 0.03, 1.0),
                ("rejected", 0.01, 0.7),
            ):
                evaluation = root / f"{name}-evaluation.json"
                settings = root / f"{name}-settings.json"
                write_json(evaluation, _evaluation(vertical, precision))
                write_json(settings, {"rail_top_fit_method": name})
                variants.append((name, evaluation, settings))
            output = root / "selection.json"

            select_rail_vertical_fit(variants, "legacy", output)

            report = load_json(output)
            self.assertEqual(report["selected_variant"], "winner")
            self.assertFalse(report["variants"]["rejected"]["passes_all_guardrails"])
            self.assertEqual(
                report["selection_scope"], "nested_validation_within_outer_train_only"
            )


if __name__ == "__main__":
    unittest.main()
