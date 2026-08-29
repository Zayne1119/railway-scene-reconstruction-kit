from __future__ import annotations

import unittest

import numpy as np

from railway_recon.geometry import fit_corridor_frame_dominant_axis_regression


class CorridorFrameRegressionTests(unittest.TestCase):
    def test_vertical_route_uses_y_as_predictor_and_preserves_direction(self) -> None:
        trajectory = [
            {"x": 10.00, "y": 100.0},
            {"x": 10.12, "y": 75.0},
            {"x": 10.18, "y": 50.0},
            {"x": 10.32, "y": 25.0},
        ]
        frame = fit_corridor_frame_dominant_axis_regression(trajectory)
        self.assertLess(frame.along_xy[1], 0.0)
        self.assertAlmostEqual(float(np.linalg.det(
            np.column_stack((frame.along_xy, frame.cross_xy))
        )), 1.0)


if __name__ == "__main__":
    unittest.main()
