from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.defect_injection import evaluate_defect_injection
from railway_recon.io import load_json, write_json
from railway_recon.segments import plan_segments
from railway_recon.track_graph import build_track_graph


class DefectInjectionTests(unittest.TestCase):
    def test_known_defects_are_detected_by_expected_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = initialize_project(root / "sample", "sample", "Sample")
            project = load_project(project_path)
            camera_path = project.input_path("camera_csv")
            assert camera_path is not None
            with camera_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
                writer.writerow([0, 0, "0.jpg", 0, 0, 2])
                writer.writerow([1, 1, "1.jpg", 50, 0, 2])
                writer.writerow([2, 2, "2.jpg", 100, 0, 2])
            plan_segments(project)
            reports = project.workspace_path("reports")
            sources = []
            for segment_id, origin_x in (("s0000_0050m", 25.0), ("s0050_0100m", 75.0)):
                path = reports / f"{segment_id}.json"
                centers = [0.0] if segment_id == "s0000_0050m" else [0.0, 4.0]
                rail_pairs = []
                rail_lines = []
                peaks = []
                for index, center in enumerate(centers, start=1):
                    positions = [center - 0.754, center + 0.754]
                    rail_pairs.append(
                        {
                            "track_id": f"TRACK-{index:04d}",
                            "cross_positions_m": positions,
                            "gauge_m": 1.435,
                            "separation_m": 1.508,
                            "rail_center_spacing_m": 1.508,
                            "score": 1.0,
                        }
                    )
                    rail_lines.extend(
                        {"cross_position_m": position, "median_z_m": 0.0}
                        for position in positions
                    )
                    peaks.extend(
                        {"cross_position_m": position, "coverage": 1.0}
                        for position in positions
                    )
                write_json(
                    path,
                    {
                        "schema_version": "railway.rail-candidates.v1",
                        "project_id": "sample",
                        "segment_id": segment_id,
                        "frame": {
                            "origin_xy": [origin_x, 0.0],
                            "along_xy": [1.0, 0.0],
                            "cross_xy": [0.0, 1.0],
                        },
                        "longitudinal_range_m": [-25.0, 25.0],
                        "rail_pairs": rail_pairs,
                        "rail_lines": rail_lines,
                        "peaks": peaks,
                        "review_status": "accepted",
                        "status": "geometry_candidates_only_manual_review_required",
                    },
                )
                sources.append((segment_id, path))
            built = build_track_graph(project, sources)
            result_path = root / "injection.json"
            evaluate_defect_injection(
                built["output_graph_path"],
                project.resolve(project.value["algorithms"]["track_graph"]),
                result_path,
            )
            result = load_json(result_path)
            self.assertEqual(result["summary"]["defect_recall"], 1.0)
            self.assertEqual(result["summary"]["unexpected_failed_check_count"], 0)


if __name__ == "__main__":
    unittest.main()
