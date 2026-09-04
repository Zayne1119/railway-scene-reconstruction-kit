import unittest

import numpy as np

from railway_recon.platform_photo_evidence import (
    _gradient_fields,
    _platform_gap_boundary,
    polyline_edge_evidence,
    polyline_warm_color_fraction,
)


class PlatformPhotoEvidenceTests(unittest.TestCase):
    def test_gap_boundary_uses_nearest_segment_plane(self) -> None:
        component = {
            "fit_segments": [
                {
                    "longitudinal_range_m": [0.0, 5.0],
                    "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.2],
                }
            ]
        }
        gap = {"longitudinal_range_m": [1.0, 2.0], "cross_range_m": [3.0, 4.0]}
        boundary = _platform_gap_boundary(component, gap, {"gap_edge_projection_samples": 4})
        self.assertEqual(boundary.shape, (16, 3))
        self.assertTrue(np.allclose(boundary[:, 2], 1.2))

    def test_polyline_edge_response_finds_parallel_boundary(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[100:] = 255
        gx, gy = _gradient_fields(image)
        u = np.linspace(20.0, 280.0, 80)
        v = np.full(u.size, 100.0)
        result = polyline_edge_evidence(gx, gy, u, v, 2)
        self.assertTrue(result["valid"])
        self.assertGreater(result["directional_response"], 0.1)
        self.assertGreater(result["directional_orientation_ratio"], 0.9)

    def test_warm_color_probe_distinguishes_orange_line(self) -> None:
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        image[58:63, :, :] = np.asarray([230, 150, 20], dtype=np.uint8)
        u = np.linspace(10.0, 230.0, 80)
        on_line = polyline_warm_color_fraction(image, u, np.full(u.size, 60.0), 2)
        off_line = polyline_warm_color_fraction(image, u, np.full(u.size, 30.0), 2)
        self.assertGreater(on_line, 0.9)
        self.assertEqual(off_line, 0.0)


if __name__ == "__main__":
    unittest.main()
