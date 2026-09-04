import unittest

import numpy as np

from railway_recon.canopy_photo_evidence import (
    _local_to_world,
    _wrapped_delta,
    directional_edge_evidence,
)


class CanopyPhotoEvidenceTests(unittest.TestCase):
    def test_directional_edge_prefers_edge_aligned_with_probe(self) -> None:
        image = np.zeros((200, 240, 3), dtype=np.uint8)
        image[100:] = 255
        aligned = directional_edge_evidence(
            image, (120.0, 100.0), (60.0, 100.0), (180.0, 100.0), 50
        )
        crossed = directional_edge_evidence(
            image, (120.0, 100.0), (120.0, 40.0), (120.0, 160.0), 50
        )
        self.assertTrue(aligned["valid"])
        self.assertGreater(
            aligned["directional_orientation_ratio"],
            crossed["directional_orientation_ratio"],
        )

    def test_local_to_world_uses_corridor_frame(self) -> None:
        frame = {
            "origin_xy": [100.0, 200.0],
            "along_xy": [0.0, 1.0],
            "cross_xy": [-1.0, 0.0],
        }
        result = _local_to_world(np.asarray([[10.0, 3.0, 5.0]]), frame)
        np.testing.assert_allclose(result, [[97.0, 210.0, 5.0]])

    def test_wrapped_delta_uses_shortest_panorama_direction(self) -> None:
        self.assertAlmostEqual(_wrapped_delta(5.0, 5755.0, 5760), 10.0)


if __name__ == "__main__":
    unittest.main()
