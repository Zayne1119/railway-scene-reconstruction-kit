import json
import tempfile
import unittest
from pathlib import Path

from railway_recon.config import ProjectConfig
from railway_recon.platform_candidate_gate import evaluate_platform_candidate_gate


class PlatformCandidateGateTest(unittest.TestCase):
    def test_gate_passes_only_for_resolved_low_residual_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project.json"
            project = ProjectConfig(
                path=project_path,
                value={
                    "project": {"id": "test"},
                    "quality": {"platform_fit_p90_max_m": 0.08},
                },
            )
            report_path = root / "platform.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "railway.platform-surface.v1",
                        "segment_id": "s0000_0050m",
                        "platform_components": [
                            {
                                "id": "PLATFORM-1",
                                "status": "automatic_platform_surface_hypothesis_review_required",
                                "longitudinal_range_m": [0.0, 50.0],
                                "fit_segments": [
                                    {"absolute_residual_p90_m": 0.03}
                                ],
                                "unresolved_interior_gap_count": 0,
                                "interior_gap_candidates": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_platform_candidate_gate(
                project,
                "s0000_0050m",
                report_path,
                root / "gate.json",
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["mesh_gate"], "passed_automatic_candidate")

    def test_ownership_plan_excludes_buffer_only_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = ProjectConfig(
                path=root / "project.json",
                value={
                    "project": {"id": "test"},
                    "quality": {"platform_fit_p90_max_m": 0.08},
                },
            )
            report_path = root / "platform.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "railway.platform-surface.v1",
                        "segment_id": "s0000_0050m",
                        "frame": {
                            "origin_xy": [0.0, 0.0],
                            "along_xy": [1.0, 0.0],
                            "cross_xy": [0.0, 1.0],
                        },
                        "platform_components": [
                            {
                                "id": "PLATFORM-1",
                                "status": "automatic_platform_surface_hypothesis_review_required",
                                "longitudinal_range_m": [-50.0, 50.0],
                                "cross_range_m": [2.0, 8.0],
                                "fit_segments": [
                                    {
                                        "longitudinal_range_m": [-50.0, -45.0],
                                        "observed_cross_range_m": [2.0, 8.0],
                                        "absolute_residual_p90_m": 0.20,
                                    },
                                    {
                                        "longitudinal_range_m": [0.0, 5.0],
                                        "observed_cross_range_m": [2.0, 8.0],
                                        "absolute_residual_p90_m": 0.02,
                                    },
                                ],
                                "interior_gap_candidates": [
                                    {
                                        "longitudinal_range_m": [-48.0, -47.0],
                                        "cross_range_m": [3.0, 4.0],
                                        "gap_classification": "unique_enclosed_gap_review_required",
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ownership_path = root / "ownership.json"
            ownership_path.write_text(
                json.dumps(
                    {
                        "frame": {"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0]},
                        "segments": [
                            {
                                "segment_id": "s0000_0050m",
                                "owned_interval_m": [-1.0, 10.0],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = evaluate_platform_candidate_gate(
                project,
                "s0000_0050m",
                report_path,
                root / "gate.json",
                ownership_plan_value=ownership_path,
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["evaluated_fit_segment_count"], 1)
            self.assertEqual(result["evaluated_gap_count"], 0)


if __name__ == "__main__":
    unittest.main()
