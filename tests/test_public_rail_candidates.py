from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from railway_recon.public_rail_candidates import (
    DEFAULT_PUBLIC_RAIL_POLICY,
    PUBLIC_RAIL_CANDIDATE_MODES,
    extract_public_rail_candidates,
)


def mathematical_corridor(
    *,
    spacing: float = 1.50,
    unmatched_line: bool = False,
    gap: bool = False,
    disjoint_support: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Independent mathematical surfaces; no public/customer sample is used."""
    rng = np.random.default_rng(731)
    ground_x, ground_y = np.meshgrid(
        np.linspace(-15.0, 15.0, 241), np.linspace(-4.0, 4.0, 81), indexing="ij"
    )
    ground = np.column_stack(
        (ground_x.ravel(), ground_y.ravel(), rng.normal(0.0, 0.002, ground_x.size))
    )
    rails = []
    for side in (-1.0, 1.0):
        x = np.linspace(-14.98, 14.98, 600)
        if gap:
            x = x[(x < -1.5) | (x > 1.5)]
        if disjoint_support:
            x = x[x < -0.5] if side < 0.0 else x[x > 0.5]
        rails.append(
            np.column_stack(
                (
                    np.repeat(x, 3),
                    side * spacing / 2.0 + np.tile([-0.012, 0.0, 0.012], len(x)),
                    np.full(len(x) * 3, 0.18),
                )
            )
        )
    rail = np.concatenate(rails)
    high_line = np.column_stack((np.linspace(-15.0, 15.0, 400), np.zeros(400), np.full(400, 5.0)))
    clutter = rng.uniform([-15.0, -4.0, -0.01], [15.0, 4.0, 3.0], (600, 3))
    pieces = {"ground": ground, "rail": rail, "high_line": high_line, "clutter": clutter}
    if unmatched_line:
        x = np.linspace(-14.98, 14.98, 600)
        pieces["unmatched"] = np.column_stack((x, np.full(len(x), 3.0), np.full(len(x), 0.18)))
    point_masks = {}
    count = sum(len(piece) for piece in pieces.values())
    offset = 0
    for name, piece in pieces.items():
        mask = np.zeros(count, dtype=bool)
        mask[offset : offset + len(piece)] = True
        point_masks[name] = mask
        offset += len(piece)
    return np.concatenate(list(pieces.values())), point_masks


class PublicRailCandidateTests(unittest.TestCase):
    def test_mathematical_double_rail_is_detected_without_high_wire_or_ground(self) -> None:
        points, truth = mathematical_corridor()
        result = extract_public_rail_candidates(points)
        report = result["report"]
        self.assertEqual(result["point_mask"].dtype, np.bool_)
        self.assertEqual(result["point_mask"].shape, (len(points),))
        self.assertEqual(len(report["rail_pairs"]), 1)
        self.assertGreater(np.mean(result["point_mask"][truth["rail"]]), 0.90)
        self.assertFalse(np.any(result["point_mask"][truth["ground"]]))
        self.assertFalse(np.any(result["point_mask"][truth["high_line"]]))
        self.assertFalse(report["disclosure"]["uses_labels"])
        self.assertNotIn("probability", report["candidate_pool"][0])
        json.dumps(report, allow_nan=False)

    def test_modes_share_candidate_pool_and_pairing_only_selects_subset(self) -> None:
        points, truth = mathematical_corridor(unmatched_line=True)
        local = extract_public_rail_candidates(points, mode="height_prominence")
        paired = extract_public_rail_candidates(points, mode="paired_geometry")
        self.assertEqual(local["report"]["candidate_pool"], paired["report"]["candidate_pool"])
        self.assertFalse(np.any(paired["point_mask"] & ~local["point_mask"]))
        self.assertGreater(np.count_nonzero(local["point_mask"][truth["unmatched"]]), 400)
        self.assertFalse(np.any(paired["point_mask"][truth["unmatched"]]))
        self.assertEqual(len(paired["report"]["rail_pairs"]), 1)

    def test_observed_spacing_is_not_forced_to_nominal_gauge(self) -> None:
        points, _ = mathematical_corridor(spacing=1.56)
        pair = extract_public_rail_candidates(points)["report"]["rail_pairs"][0]
        self.assertAlmostEqual(pair["observed_center_spacing_m"], 1.56, delta=0.06)
        self.assertGreater(abs(pair["observed_center_spacing_m"] - 1.508), 0.025)
        self.assertFalse(pair["nominal_gauge_correction"])

    def test_disjoint_support_does_not_create_a_paired_track(self) -> None:
        points, _ = mathematical_corridor(disjoint_support=True)
        local = extract_public_rail_candidates(points, mode="height_prominence")
        paired = extract_public_rail_candidates(points)
        self.assertGreaterEqual(len(local["report"]["candidate_pool"]), 2)
        self.assertFalse(np.any(paired["point_mask"]))
        self.assertEqual(paired["report"]["rail_pairs"], [])

    def test_output_polylines_do_not_bridge_an_unsupported_gap(self) -> None:
        points, _ = mathematical_corridor(gap=True)
        report = extract_public_rail_candidates(points)["report"]
        self.assertEqual(len(report["rail_pairs"]), 1)
        for candidate in report["candidate_pool"]:
            self.assertEqual(candidate["unsupported_bins_bridged"], 0)
            self.assertGreaterEqual(len(candidate["polylines_xyz"]), 2)
            for polyline in candidate["polylines_xyz"]:
                x = np.asarray(polyline)[:, 0]
                self.assertFalse(np.min(x) < -1.0 and np.max(x) > 1.0)

    def test_translation_and_quarter_turn_preserve_predictions(self) -> None:
        points, _ = mathematical_corridor()
        original = extract_public_rail_candidates(points)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed = points @ rotation.T + np.array([480_000.0, 5_700_000.0, 125.0])
        revised = extract_public_rail_candidates(transformed)
        np.testing.assert_array_equal(original["point_mask"], revised["point_mask"])
        original_spacing = original["report"]["rail_pairs"][0]["observed_center_spacing_m"]
        revised_spacing = revised["report"]["rail_pairs"][0]["observed_center_spacing_m"]
        self.assertAlmostEqual(original_spacing, revised_spacing, places=6)

    def test_repeated_and_reordered_input_is_unchanged(self) -> None:
        points, _ = mathematical_corridor()
        saved = points.copy()
        first = extract_public_rail_candidates(points)
        second = extract_public_rail_candidates(points)
        np.testing.assert_array_equal(points, saved)
        np.testing.assert_array_equal(first["point_mask"], second["point_mask"])
        self.assertEqual(first["report"], second["report"])
        order = np.random.default_rng(919).permutation(len(points))
        permuted = extract_public_rail_candidates(points[order])
        np.testing.assert_array_equal(first["point_mask"][order], permuted["point_mask"])

    def test_empty_and_degenerate_scenes_return_empty_json_safe_reports(self) -> None:
        isotropic = np.array([[x, y, 0.0] for x in range(10) for y in range(10)], dtype=float)
        for points in (np.empty((0, 3)), np.zeros((100, 3)), isotropic):
            with self.subTest(shape=points.shape):
                result = extract_public_rail_candidates(points)
                self.assertFalse(np.any(result["point_mask"]))
                self.assertEqual(result["report"]["status"], "no_candidates")
                json.dumps(result["report"], allow_nan=False)

    def test_flat_ground_never_requires_a_fabricated_model(self) -> None:
        points, truth = mathematical_corridor()
        result = extract_public_rail_candidates(points[truth["ground"]])
        self.assertFalse(np.any(result["point_mask"]))
        self.assertEqual(result["report"]["candidate_pool"], [])

    def test_invalid_coordinate_arrays_are_rejected(self) -> None:
        for invalid in (
            [[0.0, 0.0, 0.0]],
            np.zeros((3, 4)),
            np.zeros(3),
            np.full((2, 3), np.nan),
            np.full((2, 3), np.inf),
            np.zeros((2, 3), dtype=complex),
            np.zeros((2, 3), dtype=bool),
            np.array([["1", "2", "3"]]),
        ):
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                extract_public_rail_candidates(invalid)

    def test_policy_validation_and_no_policy_mutation(self) -> None:
        points, _ = mathematical_corridor()
        default_copy = copy.deepcopy(DEFAULT_PUBLIC_RAIL_POLICY)
        policy = {"minimum_pair_joint_coverage": 0.2}
        policy_copy = policy.copy()
        extract_public_rail_candidates(points, policy)
        self.assertEqual(policy, policy_copy)
        self.assertEqual(DEFAULT_PUBLIC_RAIL_POLICY, default_copy)
        invalids = [
            {"classification": 10},
            {"cross_bin_m": 0.0},
            {"z_percentile_min": 36.0},
            {"gaussian_sigma_bins": float("nan")},
            {"maximum_grid_cells": 4.5},
            {"maximum_grid_cells": True},
            {"height_baseline_filter_bins": 4},
            {"minimum_direction_eigenvalue_ratio": 1.0},
            {"minimum_pair_joint_coverage": 1.1},
            {"minimum_line_sample_count": 1},
            {"pair_support_longitudinal_bin_m": 0.01},
            {"rail_pair_maximum_m": 1.0},
        ]
        for policy in invalids:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                extract_public_rail_candidates(points, policy)
        with self.assertRaises(ValueError):
            extract_public_rail_candidates(points, mode="labels")
        with self.assertRaises(ValueError):
            extract_public_rail_candidates(points, policy=[])
        self.assertEqual(len(PUBLIC_RAIL_CANDIDATE_MODES), 2)

    def test_point_and_grid_resource_limits_are_explicit(self) -> None:
        points, _ = mathematical_corridor()
        with self.assertRaisesRegex(ValueError, "maximum_point_count"):
            extract_public_rail_candidates(points, {"maximum_point_count": 10})
        result = extract_public_rail_candidates(points, {"maximum_grid_cells": 10})
        self.assertFalse(np.any(result["point_mask"]))
        self.assertEqual(
            result["report"]["diagnostics"]["empty_reason"], "grid_cell_resource_limit"
        )
        huge = points * 1e100
        result = extract_public_rail_candidates(huge)
        self.assertFalse(np.any(result["point_mask"]))
        self.assertEqual(
            result["report"]["diagnostics"]["empty_reason"], "coordinate_extent_resource_limit"
        )
        json.dumps(result["report"], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
