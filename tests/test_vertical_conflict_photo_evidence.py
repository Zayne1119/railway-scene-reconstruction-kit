import unittest

import numpy as np

from railway_recon.vertical_conflict_photo_evidence import candidate_markers


class VerticalConflictPhotoEvidenceTests(unittest.TestCase):
    def test_candidate_markers_cover_full_height_and_square_footprint(self) -> None:
        markers, names = candidate_markers(
            {
                "longitudinal_position_m": 10.0,
                "cross_position_m": 3.0,
                "minimum_z": 1.2,
                "maximum_z": 6.2,
                "footprint_m": 0.8,
            }
        )
        self.assertEqual(markers.shape, (7, 3))
        self.assertEqual(names[:3], ("base", "middle", "top"))
        self.assertTrue(np.allclose(markers[0], [10.0, 3.0, 1.2]))
        self.assertTrue(np.allclose(markers[2], [10.0, 3.0, 6.2]))
        footprint = markers[3:, :2]
        self.assertAlmostEqual(float(np.ptp(footprint[:, 0])), 0.8)
        self.assertAlmostEqual(float(np.ptp(footprint[:, 1])), 0.8)


if __name__ == "__main__":
    unittest.main()
