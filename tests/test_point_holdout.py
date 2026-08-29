import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.benchmark import initialize_benchmark
from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.point_holdout import (
    spatial_voxel_holdout_mask,
    split_point_cloud_holdout,
    validate_point_cloud_holdout,
)


class PointHoldoutTests(unittest.TestCase):
    def test_same_voxel_never_crosses_train_and_holdout(self) -> None:
        x = np.asarray([0.1, 0.2, 1.1, 1.2, 2.1, 2.2])
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        mask = spatial_voxel_holdout_mask(x, y, z, 1.0, 0.5, 42)
        self.assertEqual(bool(mask[0]), bool(mask[1]))
        self.assertEqual(bool(mask[2]), bool(mask[3]))
        self.assertEqual(bool(mask[4]), bool(mask[5]))

    def test_split_is_hash_bound_and_complementary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = initialize_benchmark(root / "benchmark", "dataset-a", "scene-a")
            project_path = initialize_project(root / "project", "project-a", "Project A")
            point_path = root / "cloud.las"
            cloud = laspy.create(point_format=3, file_version="1.2")
            cloud.x = np.repeat(np.arange(100, dtype=np.float64), 2) + np.tile(
                [0.1, 0.2], 100
            )
            cloud.y = np.zeros(200)
            cloud.z = np.zeros(200)
            cloud.write(point_path)
            project = load_json(project_path)
            project["inputs"]["point_cloud"] = str(point_path)
            project["segmentation"]["chunk_size_points"] = 1000
            write_json(project_path, project)
            segment_manifest = (
                root / "project" / "workspace" / "manifests" / "segments.generated.json"
            )
            write_json(
                segment_manifest,
                {
                    "schema_version": "railway.segments.v1",
                    "project_id": "project-a",
                    "segments": [
                        {
                            "id": "s0000_0100m",
                            "chainage_start_m": 0.0,
                            "chainage_end_m": 100.0,
                            "bounds": {
                                "min_x": 0.0,
                                "max_x": 100.0,
                                "min_y": -1.0,
                                "max_y": 1.0,
                                "min_z": -1.0,
                                "max_z": 1.0,
                            },
                        }
                    ],
                },
            )
            manifest_path = split_point_cloud_holdout(
                load_project(project_path),
                benchmark,
                {"s0000_0100m"},
                output_name="holdout_test_v1",
                holdout_fraction=0.2,
                voxel_size_m=1.0,
                seed=42,
            )
            manifest = load_json(manifest_path)
            segment = manifest["segments"][0]
            self.assertEqual(
                segment["train"]["point_count"] + segment["holdout"]["point_count"],
                segment["selected_point_count"],
            )
            self.assertGreater(segment["train"]["point_count"], 0)
            self.assertGreater(segment["holdout"]["point_count"], 0)
            self.assertTrue(validate_point_cloud_holdout(manifest_path)["passed"])


if __name__ == "__main__":
    unittest.main()
