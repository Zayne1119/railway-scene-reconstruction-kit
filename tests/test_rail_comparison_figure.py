from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from railway_recon.io import load_json, write_json
from railway_recon.rail_comparison_figure import render_rail_method_comparison


class RailComparisonFigureTests(unittest.TestCase):
    def test_output_is_explicitly_a_prediction_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "neutral.png"
            Image.new("RGB", (1102, 532), (4, 18, 27)).save(image_path)
            camera_path = root / "camera.csv"
            with camera_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
                writer.writerow([0, 0, "0.jpg", 0, 0, 2])
                writer.writerow([1, 1, "1.jpg", 50, 0, 2])
            evidence_path = root / "evidence.json"
            write_json(
                evidence_path,
                {
                    "schema_version": "railway.rail-neutral-evidence.v1",
                    "model_overlay": False,
                    "candidate_overlay": False,
                    "camera_pose_path": str(camera_path),
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "stations": [
                        {
                            "segment_id": "s0000_0050m",
                            "chainage_m": 25.0,
                            "cross_section_image": str(image_path),
                            "cross_min_m": -2.0,
                            "cross_max_m": 2.0,
                            "z_min_m": 0.0,
                            "z_max_m": 1.0,
                        }
                    ],
                },
            )
            report = {
                "segment_id": "s0000_0050m",
                "frame": {
                    "origin_xy": [0.0, 0.0],
                    "along_xy": [1.0, 0.0],
                    "cross_xy": [0.0, 1.0],
                },
                "rail_lines": [
                    {
                        "cross_fit_slope_m_per_m": 0.0,
                        "cross_fit_intercept_m": 0.0,
                        "z_fit_slope_m_per_m": 0.0,
                        "z_fit_intercept_m": 0.5,
                    }
                ],
            }
            baseline_path = root / "baseline.json"
            method_path = root / "method.json"
            write_json(baseline_path, report)
            write_json(method_path, report)
            output = root / "figures"
            manifest_path = render_rail_method_comparison(
                evidence_path,
                [("s0000_0050m", baseline_path)],
                [("s0000_0050m", method_path)],
                output,
            )
            manifest = load_json(manifest_path)
            self.assertEqual(
                manifest["visualization_role"], "prediction_overlay_not_ground_truth"
            )
            self.assertTrue(Path(manifest["contact_sheet"]).is_file())


if __name__ == "__main__":
    unittest.main()
