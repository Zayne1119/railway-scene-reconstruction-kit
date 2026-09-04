from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.config import initialize_project, load_project, validate_project_value
from railway_recon.geometry import fit_segment_corridor_frame
from railway_recon.multi_source import prepare_input_sources
from railway_recon.segments import (
    crop_segment_context_sources,
    crop_segments,
    plan_segments,
)


def _write_cloud(path: Path, xs: list[float]) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.asarray(xs, dtype=np.float64)
    cloud.y = np.zeros(len(xs), dtype=np.float64)
    cloud.z = np.zeros(len(xs), dtype=np.float64)
    cloud.write(path)


class MultiSourceInputTests(unittest.TestCase):
    def test_prepare_plan_and_crop_assigns_one_owner_per_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "multi", "multi-project", "Multi")
            project = load_project(config_path)
            camera_path = project.input_path("camera_csv")
            assert camera_path is not None
            with camera_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
                for index, x in enumerate(range(-20, 121, 10)):
                    writer.writerow([index, index, f"{index}.jpg", x, 0, 2])

            first = root / "source-a.las"
            second = root / "source-b.las"
            _write_cloud(first, [0, 10, 25, 50, 60])
            _write_cloud(second, [40, 50, 75, 90, 100])
            project.value["inputs"].pop("point_cloud")
            project.value["inputs"]["point_clouds"] = [
                {"id": "source-a", "path": str(first), "priority": 0},
                {"id": "source-b", "path": str(second), "priority": 1},
            ]
            project.value["segmentation"]["length_m"] = 25.0

            self.assertEqual(validate_project_value(project.value), [])
            prepared = prepare_input_sources(project)
            self.assertEqual(prepared["status"], "ready_for_segment_planning")
            self.assertEqual(prepared["camera"]["filtered_row_count"], 11)
            self.assertEqual(prepared["camera"]["first_index"], 2)
            self.assertEqual(prepared["camera"]["last_index"], 12)
            self.assertEqual(prepared["seams"][0]["status"], "overlap_context_available")

            manifest = plan_segments(project)
            self.assertEqual(len(manifest["segments"]), 4)
            self.assertEqual(
                [item["primary_source_id"] for item in manifest["segments"]],
                ["source-a", "source-a", "source-b", "source-b"],
            )
            self.assertEqual(
                manifest["segments"][1]["context_source_ids"],
                ["source-a", "source-b"],
            )
            _, frame_cameras = fit_segment_corridor_frame(project, "s0000_0025m")
            self.assertGreaterEqual(min(item["index"] for item in frame_cameras), 2)
            self.assertLessEqual(max(item["index"] for item in frame_cameras), 12)

            report = crop_segments(project)
            self.assertEqual(report["source_point_counts"], {"source-a": 5, "source-b": 5})
            self.assertTrue(all(item["point_count"] > 0 for item in report["segments"]))

            context = crop_segment_context_sources(project, "s0025_0050m")
            self.assertEqual(context["status"], "evidence_only_no_duplicate_mesh_authority")
            self.assertEqual(
                [item["role"] for item in context["sources"]],
                ["owner", "context_only"],
            )
            self.assertTrue(all(item["point_count"] > 0 for item in context["sources"]))

            with camera_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Camera CSV changed"):
                plan_segments(project)

    def test_tiny_route_tail_is_merged_into_previous_segment(self) -> None:
        intervals = __import__(
            "railway_recon.segments", fromlist=["_segment_intervals"]
        )._segment_intervals(4400.191, 50.0, [])
        self.assertEqual(len(intervals), 88)
        self.assertEqual(intervals[-1], (4350.0, 4400.191))


if __name__ == "__main__":
    unittest.main()
