from __future__ import annotations

import unittest

import numpy as np

from railway_recon.algorithms.rail_candidates import (
    _anchored_rail_top_samples,
    _core_longitudinal_mask,
    _height_mask,
    _local_rail_top_samples,
    _ordered_pair_candidates,
    _pair_continuity_settings,
    _paired_longitudinal_support,
    _pairing_rail_top_z,
)


class RailTopFitTests(unittest.TestCase):
    def test_anchored_quantile_rejects_high_outlier_and_returns_one_sample_per_bin(self) -> None:
        longitudinal = np.asarray([0.1, 0.1, 0.1, 1.1, 1.1, 1.1, 1.1])
        cross = np.zeros(len(longitudinal))
        z = np.asarray([1.00, 1.01, 1.50, 1.02, 1.03, 1.04, 0.2])
        sample_long, sample_z = _anchored_rail_top_samples(
            longitudinal,
            cross,
            z,
            np.ones(len(z), dtype=bool),
            position_m=0.0,
            anchor_z_m=1.02,
            long_min_m=0.0,
            longitudinal_bin_m=1.0,
            half_width_m=0.1,
            below_m=0.1,
            above_m=0.1,
            quantile=0.5,
            minimum_bin_points=2,
        )
        np.testing.assert_allclose(sample_long, [0.5, 1.5])
        np.testing.assert_allclose(sample_z, [1.005, 1.03])

    def test_invalid_quantile_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "rail_top_quantile"):
            _anchored_rail_top_samples(
                np.asarray([0.0]),
                np.asarray([0.0]),
                np.asarray([1.0]),
                np.asarray([True]),
                0.0,
                1.0,
                0.0,
                1.0,
                0.1,
                0.1,
                0.1,
                1.5,
                1,
            )

    def test_pairing_height_can_be_decoupled_from_selected_vertical_fit(self) -> None:
        self.assertEqual(_pairing_rail_top_z(1.0, 1.4, "height_grid_max"), 1.0)
        self.assertEqual(_pairing_rail_top_z(1.0, 1.4, "selected_fit"), 1.4)
        with self.assertRaisesRegex(ValueError, "rail_pair_top_method"):
            _pairing_rail_top_z(1.0, 1.4, "unknown")

    def test_core_longitudinal_mask_excludes_context_points(self) -> None:
        result = _core_longitudinal_mask(
            np.asarray([-1.0, 0.0, 4.0, 10.0, 11.0]),
            (0.0, 10.0),
        )
        np.testing.assert_array_equal(result, [False, True, True, True, False])

    def test_paired_support_records_joint_gap_without_generating_geometry(self) -> None:
        prominent = np.zeros((8, 8), dtype=bool)
        prominent[:, 2] = True
        prominent[:, 5] = True
        prominent[3:5, 5] = False
        result = _paired_longitudinal_support(prominent, 2, 5, 0, 0.5)
        self.assertEqual(result["longitudinal_bin_count"], 8)
        self.assertAlmostEqual(result["left_support_ratio"], 1.0)
        self.assertAlmostEqual(result["right_support_ratio"], 0.75)
        self.assertAlmostEqual(result["joint_support_ratio"], 0.75)
        self.assertAlmostEqual(result["asymmetric_support_ratio"], 0.25)
        self.assertAlmostEqual(result["maximum_internal_joint_gap_m"], 1.0)

    def test_paired_support_fails_closed_when_no_joint_bin_exists(self) -> None:
        prominent = np.zeros((4, 6), dtype=bool)
        prominent[:2, 1] = True
        prominent[2:, 4] = True
        result = _paired_longitudinal_support(prominent, 1, 4, 0, 1.0)
        self.assertEqual(result["joint_supported_bin_count"], 0)
        self.assertEqual(result["maximum_internal_joint_gap_m"], 4.0)

    def test_paired_support_uses_a_density_tolerant_longitudinal_window(self) -> None:
        prominent = np.zeros((8, 6), dtype=bool)
        prominent[[0, 4], 1] = True
        prominent[[1, 5], 4] = True
        exact = _paired_longitudinal_support(prominent, 1, 4, 0, 0.25)
        grouped = _paired_longitudinal_support(
            prominent, 1, 4, 0, 0.25, support_bin_m=1.0
        )
        self.assertEqual(exact["joint_supported_bin_count"], 0)
        self.assertEqual(grouped["joint_supported_bin_count"], 2)
        self.assertAlmostEqual(grouped["joint_support_ratio"], 1.0)
        self.assertAlmostEqual(grouped["support_longitudinal_bin_m"], 1.0)

    def test_pair_continuity_settings_reject_invalid_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_joint_coverage"):
            _pair_continuity_settings({"minimum_pair_joint_coverage": 1.1})
        with self.assertRaisesRegex(ValueError, "maximum_internal_gap_m"):
            _pair_continuity_settings({"maximum_pair_internal_gap_m": -0.1})
        with self.assertRaisesRegex(ValueError, "joint_support_score_weight"):
            _pair_continuity_settings({"pair_joint_support_score_weight": -1.0})

    def test_pair_continuity_selection_policy_can_prefer_or_require_pass(self) -> None:
        candidates = [
            {"id": "review-high", "score": 10.0, "pair_continuity_status": "review_required"},
            {"id": "pass-low", "score": 5.0, "pair_continuity_status": "pass"},
        ]
        diagnostic = _ordered_pair_candidates(candidates, "diagnostic_only")
        preferred = _ordered_pair_candidates(candidates, "prefer_pass")
        required = _ordered_pair_candidates(candidates, "require_pass")
        self.assertEqual([item["id"] for item in diagnostic], ["review-high", "pass-low"])
        self.assertEqual([item["id"] for item in preferred], ["pass-low", "review-high"])
        self.assertEqual([item["id"] for item in required], ["pass-low"])
        with self.assertRaisesRegex(ValueError, "pair_continuity_selection_policy"):
            _ordered_pair_candidates(candidates, "unknown")

    def test_local_quantile_uses_full_search_window_without_anchor(self) -> None:
        longitudinal = np.asarray([0.1, 0.1, 0.1, 1.1, 1.1, 1.1])
        cross = np.zeros(len(longitudinal))
        z = np.asarray([0.2, 1.0, 1.2, 0.3, 1.1, 1.3])
        sample_long, sample_z = _local_rail_top_samples(
            longitudinal,
            cross,
            z,
            np.ones(len(z), dtype=bool),
            position_m=0.0,
            long_min_m=0.0,
            longitudinal_bin_m=1.0,
            half_width_m=0.1,
            quantile=0.75,
            minimum_bin_points=3,
        )
        np.testing.assert_allclose(sample_long, [0.5, 1.5])
        np.testing.assert_allclose(sample_z, [1.1, 1.2])

    def test_percentile_height_range_is_estimated_from_core_only(self) -> None:
        z = np.asarray([-100.0, 1.0, 2.0, 3.0, 100.0])
        mask, z_range = _height_mask(
            z,
            {
                "z_mode": "percentile",
                "z_percentile_min": 0.0,
                "z_percentile_max": 100.0,
            },
            estimation_mask=np.asarray([False, True, True, True, False]),
        )
        self.assertEqual(z_range, (1.0, 3.0))
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_trajectory_relative_height_mask_tracks_grade(self) -> None:
        z = np.asarray([7.0, 7.5, 8.0, 8.5])
        reference = np.asarray([10.0, 10.5, 11.0, 11.5])
        mask, z_range = _height_mask(
            z,
            {
                "z_mode": "trajectory_relative",
                "trajectory_z_min_offset_m": -3.2,
                "trajectory_z_max_offset_m": -2.8,
            },
            reference_z=reference,
        )
        self.assertEqual(z_range, (6.8, 8.7))
        np.testing.assert_array_equal(mask, [True, True, True, True])


if __name__ == "__main__":
    unittest.main()
