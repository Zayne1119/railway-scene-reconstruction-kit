from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.io import load_json, write_json
from railway_recon.open3d_baseline import detect_open3d_rail_baseline


@unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D baseline is optional")
class Open3dBaselineTests(unittest.TestCase):
    def test_plane_removal_and_dbscan_find_synthetic_rail_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample.las"
            reference = root / "reference.json"
            output = root / "result.json"
            ground_long, ground_cross = np.meshgrid(
                np.arange(0.0, 20.01, 0.25), np.arange(-3.0, 3.01, 0.25)
            )
            rail_long = np.arange(0.0, 20.01, 0.05)
            x = np.concatenate((ground_long.ravel(), rail_long, rail_long))
            y = np.concatenate(
                (
                    ground_cross.ravel(),
                    np.full_like(rail_long, -0.754),
                    np.full_like(rail_long, 0.754),
                )
            )
            z = np.concatenate(
                (
                    np.zeros(ground_long.size),
                    np.full_like(rail_long, 0.2),
                    np.full_like(rail_long, 0.2),
                )
            )
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            cloud = laspy.LasData(header)
            cloud.x = x
            cloud.y = y
            cloud.z = z
            cloud.write(source)
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0020m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 20.0],
                    "search_z_range_m": [-0.1, 0.3],
                },
            )
            detect_open3d_rail_baseline(
                source,
                reference,
                "s0000_0020m",
                output,
                voxel_size_m=0.04,
                plane_distance_m=0.02,
                dbscan_eps_m=0.15,
                dbscan_min_points=3,
                longitudinal_scale=0.02,
                minimum_cluster_length_m=10.0,
                maximum_cluster_width_m=0.1,
                maximum_input_points=100_000,
            )
            result = load_json(output)
            self.assertEqual(result["rail_pair_count"], 1)
            self.assertEqual(len(result["rail_lines"]), 2)
            self.assertAlmostEqual(result["rail_pairs"][0]["observed_gauge_m"], 1.435, places=2)


if __name__ == "__main__":
    unittest.main()
