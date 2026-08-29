import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.annotation_tasks import (
    create_vertical_annotation_package,
    validate_vertical_annotation_package,
)
from railway_recon.io import load_json, write_json


class AnnotationTaskTests(unittest.TestCase):
    def test_vertical_package_is_double_blind_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            (benchmark / "annotations").mkdir(parents=True)
            evidence = root / "evidence"
            evidence.mkdir()
            photo_pages = [evidence / "photo-01.jpg", evidence / "photo-02.jpg"]
            geometry_pages = [evidence / "geometry-01.png"]
            top_map = evidence / "top.png"
            for path in [*photo_pages, *geometry_pages, top_map]:
                path.write_bytes(b"synthetic evidence")
            candidates = []
            for candidate_id in range(1, 8):
                candidates.append(
                    {
                        "id": candidate_id,
                        "center_x": float(candidate_id),
                        "center_y": float(candidate_id + 10),
                        "minimum_z": 0.0,
                        "maximum_z": 5.0,
                        "height_m": 5.0,
                        "footprint_x_m": 0.5,
                        "footprint_y_m": 0.6,
                        "point_count": 100,
                        "cross_position_m": 1.0,
                        "nearest_track_distance_m": 2.0,
                        "trajectory_distance_m": 3.0,
                        "base_near_ground": True,
                        "vertical_occupied_ratio": 0.9,
                        "vertical_longest_run_ratio": 0.8,
                        "evidence_views": [{"camera_index": 100 + candidate_id}],
                        "existing_prediction": "must_not_leak",
                    }
                )
            report_path = root / "features.json"
            write_json(
                report_path,
                {
                    "candidate_count": len(candidates),
                    "annotated_top_map": str(top_map),
                    "contact_sheets": [str(path) for path in photo_pages],
                    "geometry_profile_sheets": [str(path) for path in geometry_pages],
                    "candidates": candidates,
                },
            )
            package = create_vertical_annotation_package(
                benchmark, [("s000_50m", report_path)], seed=42
            )
            manifest = load_json(package / "manifest.json")
            self.assertEqual(manifest["task_count"], 7)
            self.assertTrue(manifest["blind_to_existing_classification"])

            rows_by_reviewer = []
            for name in ("reviewer_a.csv", "reviewer_b.csv"):
                with (package / name).open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 7)
                self.assertTrue(all(not row["asset_type"] for row in rows))
                self.assertTrue(all("existing_prediction" not in row for row in rows))
                rows_by_reviewer.append(rows)
            self.assertNotEqual(
                [row["task_id"] for row in rows_by_reviewer[0]],
                [row["task_id"] for row in rows_by_reviewer[1]],
            )
            validation = validate_vertical_annotation_package(package)
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["evidence_file_count"], 4)


if __name__ == "__main__":
    unittest.main()
