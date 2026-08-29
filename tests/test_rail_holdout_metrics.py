import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.io import load_json, write_json
from railway_recon.rail_holdout_metrics import evaluate_rail_holdout


def _line(identifier: str, cross: float, z: float) -> dict[str, float | str | int]:
    return {
        "id": identifier,
        "cross_position_m": cross,
        "median_z_m": z,
        "point_count": 100,
        "cross_fit_slope_m_per_m": 0.0,
        "cross_fit_intercept_m": cross,
        "cross_fit_sample_count": 40,
        "cross_fit_residual_p90_m": 0.01,
        "z_fit_slope_m_per_m": 0.0,
        "z_fit_intercept_m": z,
        "z_fit_sample_count": 40,
        "z_fit_residual_p90_m": 0.01,
    }


class RailHoldoutMetricTests(unittest.TestCase):
    def test_repeatability_and_raw_point_support_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            }
            train_path = root / "train.json"
            holdout_path = root / "holdout.json"
            base = {
                "schema_version": "railway.rail-candidates.v1",
                "segment_id": "s0000_0010m",
                "frame": frame,
                "longitudinal_range_m": [0.0, 10.0],
                "search_z_range_m": [0.5, 1.5],
                "rail_pairs": [
                    {
                        "observed_cross_positions_m": [-0.75, 0.75],
                        "observed_gauge_m": 1.427,
                    }
                ],
            }
            write_json(
                train_path,
                {**base, "rail_lines": [_line("T1", -0.75, 1.0), _line("T2", 0.75, 1.0)]},
            )
            write_json(
                holdout_path,
                {
                    **base,
                    "rail_lines": [_line("H1", -0.74, 1.01), _line("H2", 0.76, 1.01)],
                    "rail_pairs": [
                        {
                            "observed_cross_positions_m": [-0.74, 0.76],
                            "observed_gauge_m": 1.427,
                        }
                    ],
                },
            )
            coordinate = np.arange(0.0, 10.001, 0.25)
            cloud = laspy.create(point_format=3, file_version="1.2")
            cloud.x = np.concatenate((coordinate, coordinate))
            cloud.y = np.concatenate(
                (np.full_like(coordinate, -0.75), np.full_like(coordinate, 0.75))
            )
            cloud.z = np.ones(len(coordinate) * 2)
            cloud_path = root / "holdout.las"
            cloud.write(cloud_path)
            holdout_manifest_path = root / "holdout-manifest.json"
            write_json(
                holdout_manifest_path,
                {
                    "schema_version": "railway.point-holdout.v1",
                    "voxel_size_m": 0.5,
                    "requested_holdout_fraction": 0.5,
                    "seed": 42,
                    "segments": [
                        {
                            "segment_id": "s0000_0010m",
                            "holdout": {"path": str(cloud_path)},
                        }
                    ],
                },
            )

            output = evaluate_rail_holdout(
                [("s0000_0010m", train_path)],
                [("s0000_0010m", holdout_path)],
                [("s0000_0010m", cloud_path)],
                holdout_manifest_path,
                root / "report.json",
                core_length_m=10.0,
                sample_step_m=0.25,
            )
            report = load_json(output)
            self.assertEqual(report["summary"]["matched_line_count"], 2)
            self.assertEqual(report["summary"]["train_line_match_recall"], 1.0)
            self.assertAlmostEqual(
                report["summary"]["line_lateral_repeatability"]["p90_m"], 0.01
            )
            self.assertLess(
                report["summary"]["model_to_holdout_support_distance"]["p90_m"],
                1e-6,
            )
            self.assertEqual(
                report["summary"]["model_to_holdout_coverage_at_0_02m"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
