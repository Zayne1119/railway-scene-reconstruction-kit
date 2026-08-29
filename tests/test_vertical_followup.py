from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.io import load_json, write_json
from railway_recon.vertical_followup import audit_rail_vertical_followup


def _evaluation(manifest_hash: str, vertical: float, precision: float, support: float) -> dict:
    return {
        "schema_version": "railway.rail-holdout-evaluation.v1",
        "holdout_manifest_sha256": manifest_hash,
        "summary": {
            "train_line_match_recall": 0.8,
            "holdout_line_match_precision": precision,
            "line_lateral_repeatability": {"p90_m": 0.02},
            "line_vertical_repeatability": {"p90_m": vertical},
            "raw_gauge_repeatability": {"p90_m": 0.03},
            "model_to_holdout_support_distance": {"p90_m": support},
            "model_to_holdout_coverage_at_0_10m": 0.9 if vertical > 0.04 else 0.95,
        },
    }


class VerticalFollowupTests(unittest.TestCase):
    def test_audit_records_post_holdout_limitation_and_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / f"{name}.json" for name in (
                "baseline", "selection", "trigger", "nested", "final", "settings", "output"
            )}
            write_json(paths["baseline"], _evaluation("a" * 64, 0.06, 1.0, 0.08))
            write_json(paths["trigger"], _evaluation("a" * 64, 0.04, 0.9, 0.05))
            write_json(paths["final"], _evaluation("a" * 64, 0.04, 1.0, 0.05))
            write_json(paths["nested"], _evaluation("b" * 64, 0.04, 1.0, 0.2))
            write_json(
                paths["selection"],
                {
                    "schema_version": "railway.rail-vertical-fit-selection.v1",
                    "selected_variant": "anchored_q50",
                },
            )
            write_json(
                paths["settings"],
                {
                    "rail_top_fit_method": "anchored_quantile",
                    "rail_pair_top_method": "height_grid_max",
                },
            )

            audit_rail_vertical_followup(
                paths["baseline"],
                paths["selection"],
                paths["trigger"],
                paths["nested"],
                paths["final"],
                paths["settings"],
                paths["output"],
            )

            report = load_json(paths["output"])
            self.assertEqual(report["status"], "passes_development_guardrails")
            self.assertEqual(
                report["evaluation_scope"],
                "single_site_post_holdout_development_exploratory",
            )


if __name__ == "__main__":
    unittest.main()
