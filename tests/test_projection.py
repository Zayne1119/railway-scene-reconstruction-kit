import unittest

import numpy as np

from railway_recon.projection import (
    aggregate_consensus_scores,
    project_equirectangular,
    signed_permutations,
)


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

    def test_consensus_uses_shared_rank_not_first_camera_winner(self) -> None:
        def record(name: str, score: float) -> dict[str, object]:
            return {
                "euler_order": name,
                "pose_direction": "camera_to_world",
                "canonical_axes_from_local": "x+,y+,z+",
                "score_median_rgb_mae": score,
            }

        cameras = [
            {"sample_id": "a", "results": [record("xyz", 1), record("zyx", 9)]},
            {"sample_id": "b", "results": [record("xyz", 8), record("zyx", 2)]},
            {"sample_id": "c", "results": [record("xyz", 7), record("zyx", 3)]},
        ]
        ranked = aggregate_consensus_scores(cameras)
        self.assertEqual(ranked[0]["euler_order"], "zyx")
        self.assertEqual(len(ranked[0]["per_camera"]), 3)

    def test_consensus_requires_three_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            aggregate_consensus_scores([])

    def test_consensus_gives_equal_scores_equal_rank(self) -> None:
        records = [
            {
                "euler_order": order,
                "pose_direction": "camera_to_world",
                "canonical_axes_from_local": "x+,y+,z+",
                "score_median_rgb_mae": 5.0,
            }
            for order in ("xyz", "XYZ")
        ]
        ranked = aggregate_consensus_scores(
            [{"sample_id": name, "results": records} for name in ("a", "b", "c")]
        )
        self.assertEqual(
            ranked[0]["consensus_median_rank_fraction"],
            ranked[1]["consensus_median_rank_fraction"],
        )
