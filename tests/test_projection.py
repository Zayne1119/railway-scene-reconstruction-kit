import unittest

import numpy as np

from railway_recon.projection import project_equirectangular, signed_permutations


class ProjectionTests(unittest.TestCase):
    def test_canonical_directions(self) -> None:
        vectors = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        u, v = project_equirectangular(vectors, 1000, 500)
        self.assertAlmostEqual(u[0], 500.0)
        self.assertAlmostEqual(v[0], 250.0)
        self.assertAlmostEqual(u[1], 750.0)
        self.assertAlmostEqual(v[1], 250.0)

    def test_signed_permutation_count(self) -> None:
        self.assertEqual(len(signed_permutations()), 48)

