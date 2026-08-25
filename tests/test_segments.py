import csv
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.config import initialize_project, load_project
from railway_recon.segments import crop_segments, plan_segments


class SegmentTests(unittest.TestCase):
    def test_plan_and_crop_synthetic_las(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "sample", "sample-project", "Sample")
            project = load_project(config_path)
            camera_path = project.input_path("camera_csv")
            self.assertIsNotNone(camera_path)
            assert camera_path is not None
            with camera_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
                writer.writerow([0, 0, "a.jpg", 0, 0, 2])
                writer.writerow([1, 1, "b.jpg", 50, 0, 2])
                writer.writerow([2, 2, "c.jpg", 100, 0, 2])

            cloud_path = project.input_path("point_cloud")
            self.assertIsNotNone(cloud_path)
            assert cloud_path is not None
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            cloud.x = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 200.0])
            cloud.y = np.zeros(6)
            cloud.z = np.zeros(6)
            cloud.write(cloud_path)

            manifest = plan_segments(project)
            self.assertEqual(len(manifest["segments"]), 2)
            report = crop_segments(project)
            self.assertGreaterEqual(
                sum(item["point_count"] for item in report["segments"]), 5
            )

