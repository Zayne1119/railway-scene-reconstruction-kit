from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy

from railway_recon.coordinate_preflight import (
    CoordinatePreflightError,
    CoordinatePreflightThresholds,
    evaluate_point_model_consistency,
    inspect_obj_geometry,
    preflight_point_cloud_obj_consistency,
    require_point_cloud_obj_consistency,
)


class CoordinatePreflightTests(unittest.TestCase):
    def test_overlapping_bounds_pass(self) -> None:
        report = evaluate_point_model_consistency(
            [0.0, 0.0, 0.0],
            [100.0, 20.0, 10.0],
            [10.0, 2.0, 1.0],
            [90.0, 18.0, 9.0],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["minimum_axis_overlap_ratio"], 1.0)

    def test_obvious_origin_error_fails_with_actual_metrics(self) -> None:
        report = evaluate_point_model_consistency(
            [0.0, 0.0, 0.0],
            [100.0, 20.0, 10.0],
            [1000.0, 1000.0, 1000.0],
            [1100.0, 1020.0, 1010.0],
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["metrics"]["minimum_axis_overlap_ratio"], 0.0)
        self.assertGreater(report["metrics"]["normalized_center_distance"], 1.0)
        self.assertIn(
            "point_model_axis_overlap_too_small",
            {row["code"] for row in report["failures"]},
        )

    def test_configured_overlap_threshold_can_reject_axis_swap(self) -> None:
        report = evaluate_point_model_consistency(
            [0.0, 0.0, 0.0],
            [100.0, 5.0, 3.0],
            [0.0, 0.0, 0.0],
            [5.0, 100.0, 3.0],
            thresholds=CoordinatePreflightThresholds(minimum_smaller_volume_overlap_ratio=0.20),
        )
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["metrics"]["smaller_volume_overlap_ratio"], 0.05)

    def test_obj_face_arity_is_part_of_consumer_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            triangle = root / "triangle.obj"
            triangle.write_text(
                "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
                encoding="utf-8",
            )
            polygon = root / "polygon.obj"
            polygon.write_text(
                "v 0 0 0\nv 2 0 0\nv 2 2 2\nv 0 2 2\nf 1 2 3 4\n",
                encoding="utf-8",
            )
            self.assertEqual(inspect_obj_geometry(triangle)["triangle_face_ratio"], 1.0)
            self.assertEqual(inspect_obj_geometry(polygon)["triangle_face_ratio"], 0.0)

    def test_las_translation_and_triangle_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud_path = root / "cloud.las"
            cloud = laspy.create(point_format=3, file_version="1.2")
            cloud.x = [100.0, 102.0, 100.0, 100.0]
            cloud.y = [200.0, 200.0, 202.0, 200.0]
            cloud.z = [10.0, 10.0, 10.0, 12.0]
            cloud.write(cloud_path)
            model_path = root / "model.obj"
            model_path.write_text(
                "v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
                encoding="utf-8",
            )
            thresholds = CoordinatePreflightThresholds(minimum_triangle_face_ratio=0.95)
            report = preflight_point_cloud_obj_consistency(
                [cloud_path],
                model_path,
                point_translation=(-100.0, -200.0, -10.0),
                thresholds=thresholds,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["point_cloud"]["source_point_count"], 4)
            self.assertEqual(report["point_cloud"]["evaluation_bounds"]["minimum"], [0.0, 0.0, 0.0])

    def test_require_raises_with_structured_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud_path = root / "cloud.las"
            cloud = laspy.create(point_format=3, file_version="1.2")
            cloud.x = [0.0, 2.0, 0.0, 0.0]
            cloud.y = [0.0, 0.0, 2.0, 0.0]
            cloud.z = [0.0, 0.0, 0.0, 2.0]
            cloud.write(cloud_path)
            model_path = root / "wrong.obj"
            model_path.write_text(
                "v 100 100 100\nv 102 100 100\nv 100 102 100\nv 100 100 102\n"
                "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
                encoding="utf-8",
            )
            with self.assertRaises(CoordinatePreflightError) as context:
                require_point_cloud_obj_consistency([cloud_path], model_path)
            self.assertFalse(context.exception.report["passed"])


if __name__ == "__main__":
    unittest.main()
