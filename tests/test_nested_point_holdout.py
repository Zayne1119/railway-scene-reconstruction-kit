from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.io import load_json
from railway_recon.point_holdout import (
    split_existing_point_cloud_holdout,
    validate_point_cloud_holdout,
)


class NestedPointHoldoutTests(unittest.TestCase):
    def test_existing_cloud_split_is_complementary_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.asarray([0.001, 0.001, 0.001])
            cloud = laspy.LasData(header)
            cloud.x = np.arange(100, dtype=np.float64) * 0.2
            cloud.y = np.zeros(100)
            cloud.z = np.zeros(100)
            cloud.write(source)
            manifest_path = split_existing_point_cloud_holdout(
                [("segment-a", source)],
                root / "nested",
                holdout_fraction=0.25,
                voxel_size_m=0.5,
                seed=11,
            )
            manifest = load_json(manifest_path)
            record = manifest["segments"][0]
            self.assertEqual(
                record["train"]["point_count"] + record["holdout"]["point_count"],
                100,
            )
            self.assertTrue(validate_point_cloud_holdout(manifest_path)["passed"])


if __name__ == "__main__":
    unittest.main()
