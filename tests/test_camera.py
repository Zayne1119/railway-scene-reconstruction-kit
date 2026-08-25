import unittest

from railway_recon.camera import audit_camera_rows, camera_trajectory


class CameraTests(unittest.TestCase):
    def test_camera_trajectory_accumulates_3d_distance(self) -> None:
        rows = [
            {"index": "0", "timestamp": "0", "file": "a.jpg", "x": "0", "y": "0", "z": "0"},
            {"index": "1", "timestamp": "1", "file": "b.jpg", "x": "3", "y": "4", "z": "0"},
        ]
        trajectory = camera_trajectory(rows)
        self.assertEqual(trajectory[-1]["distance_m"], 5.0)
        audit = audit_camera_rows(rows)
        self.assertTrue(audit["indexes_strictly_increasing"])
        self.assertEqual(audit["trajectory_length_m"], 5.0)
