import unittest

import numpy as np

from railway_recon.targeted_canopy_recovery import (
    _profile_runs,
    apply_grid_review,
    audit_confirmed_column_roof_interfaces,
    audit_local_node_chains,
    audit_roof_profile_mesh_continuity,
    infer_column_grid,
    recover_local_column_roof_nodes,
    reviewed_canopy_seeds,
    split_roof_surfaces,
)


class TargetedCanopyRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = [
            {
                "id": "C1",
                "longitudinal_position_m": -30.7,
                "cross_position_m": 7.3,
                "minimum_z": 21.1,
                "maximum_z": 26.0,
                "footprint_m": 0.8,
            },
            {
                "id": "C2",
                "longitudinal_position_m": 23.3,
                "cross_position_m": 7.3,
                "minimum_z": 21.1,
                "maximum_z": 26.0,
                "footprint_m": 0.8,
            },
            {
                "id": "C3",
                "longitudinal_position_m": -57.8,
                "cross_position_m": 5.9,
                "minimum_z": 21.1,
                "maximum_z": 25.4,
                "footprint_m": 0.55,
            },
        ]

    def test_review_overlay_does_not_modify_source_class(self) -> None:
        vertical = {"candidates": [{**self.seeds[0], "predicted_class": "false_positive"}]}
        review = {
            "decisions": [
                {
                    "candidate_id": "C1",
                    "reviewed_class": "canopy_column",
                    "confidence": 0.95,
                }
            ]
        }
        result = reviewed_canopy_seeds(vertical, review, 0.85)
        self.assertEqual(result[0]["review_overlay"]["reviewed_class"], "canopy_column")
        self.assertEqual(vertical["candidates"][0]["predicted_class"], "false_positive")

    def test_long_baseline_is_subdivided_into_realistic_spacing(self) -> None:
        grid, fit = infer_column_grid(self.seeds, [-59.0, 55.0], 9.0, 7.0, 11.0, 0.6)
        self.assertAlmostEqual(fit["refined_spacing_m"], 9.02, delta=0.15)
        self.assertGreaterEqual(len(grid), 12)
        self.assertEqual(sum(item["reviewed_seed_id"] is not None for item in grid), 3)
        differences = np.diff([item["predicted_longitudinal_position_m"] for item in grid])
        self.assertTrue(np.allclose(differences, fit["refined_spacing_m"]))

    def test_single_seed_uses_periodic_platform_gaps_for_spacing_only(self) -> None:
        grid, fit = infer_column_grid(
            self.seeds[:1],
            [-59.0, 55.0],
            9.0,
            7.0,
            11.0,
            0.6,
            phase_support_positions=[-53.5, -44.5, -35.5, -26.5, -17.5],
        )
        self.assertEqual(
            fit["fit_basis"],
            "single_reviewed_seed_plus_periodic_platform_gap_support",
        )
        self.assertAlmostEqual(fit["refined_spacing_m"], 9.0)
        self.assertEqual(sum(item["reviewed_seed_id"] is not None for item in grid), 1)
        self.assertTrue(
            all(
                item["reviewed_seed_id"] is not None
                or item["evidence_level"] == "grid_inferred_pending_support"
                for item in grid
            )
        )

    def test_single_seed_without_periodic_support_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "periodic platform-gap"):
            infer_column_grid(
                self.seeds[:1], [-59.0, 55.0], 9.0, 7.0, 11.0, 0.6
            )

    def test_roof_surface_split_preserves_unobserved_cross_gap(self) -> None:
        s = np.tile(np.linspace(0.0, 10.0, 101), 30)
        first_c = np.repeat(np.linspace(0.0, 2.0, 15), 101)
        second_c = np.repeat(np.linspace(4.0, 6.0, 15), 101)
        c = np.concatenate((first_c, second_c))
        s = np.concatenate((s[: first_c.size], s[: second_c.size]))
        z = np.where(c < 3.0, 5.0 + 0.1 * c, 6.0 - 0.05 * c)
        component = {"id": "ROOF-1"}
        settings = {
            "roof_surface_cross_bin_m": 0.5,
            "roof_surface_cross_bin_minimum_points": 100,
            "roof_surface_median_jump_m": 0.35,
            "roof_surface_minimum_width_m": 0.75,
            "roof_grid_m": 0.5,
            "roof_surface_cell_z_percentile": 90.0,
            "roof_surface_profile_minimum_cells": 3,
            "outline_profile_bin_m": 2.0,
            "minimum_profile_points": 30,
            "planar_residual_p90_maximum_m": 0.15,
            "shallow_curve_residual_p90_maximum_m": 0.35,
        }
        surfaces = split_roof_surfaces(
            component, np.arange(s.size), s, c, z, settings
        )
        self.assertEqual(len(surfaces), 2)
        self.assertLessEqual(surfaces[0]["cross_range_m"][1], 2.5)
        self.assertGreaterEqual(surfaces[1]["cross_range_m"][0], 4.0)

    def test_profile_mesh_splits_abrupt_endpoint_jump(self) -> None:
        def profile(start: float, low_cross: float, height: float) -> dict:
            return {
                "longitudinal_range_m": [start, start + 2.0],
                "cross_range_m": [low_cross, 4.0],
                "z_equals_a_cross_plus_d": [0.0, height],
            }

        surface = {
            "id": "R1",
            "footprint_profiles": [
                profile(0.0, 0.0, 5.0),
                profile(2.0, 0.1, 5.02),
                profile(4.0, 1.2, 5.5),
                profile(6.0, 1.1, 5.48),
            ],
        }
        runs = _profile_runs(surface)
        self.assertEqual([len(run) for run in runs], [2, 2])
        audit = audit_roof_profile_mesh_continuity([surface])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["unsafe_transition_split_count"], 1)
        self.assertEqual(audit["discarded_isolated_profile_count"], 0)

    def test_profile_mesh_does_not_bridge_isolated_outlier(self) -> None:
        surface = {
            "id": "R1",
            "footprint_profiles": [
                {
                    "longitudinal_range_m": [0.0, 2.0],
                    "cross_range_m": [0.0, 4.0],
                    "z_equals_a_cross_plus_d": [0.0, 5.0],
                },
                {
                    "longitudinal_range_m": [2.0, 4.0],
                    "cross_range_m": [2.0, 4.0],
                    "z_equals_a_cross_plus_d": [0.0, 6.0],
                },
                {
                    "longitudinal_range_m": [4.0, 6.0],
                    "cross_range_m": [0.0, 4.0],
                    "z_equals_a_cross_plus_d": [0.0, 5.0],
                },
            ],
        }
        self.assertEqual(_profile_runs(surface), [])
        audit = audit_roof_profile_mesh_continuity([surface])
        self.assertEqual(audit["discarded_isolated_profile_count"], 3)

    def test_rejected_grid_never_emits_inferred_geometry(self) -> None:
        report = {
            "segment_id": "s0000_0050m",
            "column_grid": [
                {"id": "C1", "reviewed_seed_id": "V1", "status": "confirmed_photo_seed"},
                {
                    "id": "C2",
                    "reviewed_seed_id": None,
                    "status": "candidate_geometry_allowed_review_required",
                },
            ],
            "evidence_policy": {},
        }
        apply_grid_review(
            report,
            {
                "schema_version": "railway.targeted-canopy-grid-review.v1",
                "segment_id": "s0000_0050m",
                "decision": "reject_constant_spacing_grid",
            },
        )
        self.assertEqual(report["geometry_column_count"], 1)
        self.assertEqual(
            report["column_grid"][1]["status"],
            "grid_position_rejected_by_photo_review_no_geometry",
        )

    def test_interface_gate_blocks_column_outside_observed_roof(self) -> None:
        report = {
            "settings": {
                "roof_candidate_thickness_m": 0.12,
                "column_roof_interface_tolerance_m": 0.15,
            },
            "column_grid": [
                {
                    "id": "C1",
                    "reviewed_seed_id": "V1",
                    "status": "confirmed_photo_seed",
                    "longitudinal_position_m": 5.0,
                    "cross_position_m": 3.0,
                    "maximum_z": 5.0,
                }
            ],
            "roof_surfaces": [
                {
                    "id": "R1",
                    "footprint_profiles": [
                        {
                            "longitudinal_range_m": [0.0, 10.0],
                            "cross_range_m": [0.0, 2.0],
                            "z_equals_a_cross_plus_d": [0.0, 5.12],
                        }
                    ],
                }
            ],
        }
        result = audit_confirmed_column_roof_interfaces(report)
        self.assertEqual(result["merge_gate"], "blocked")
        self.assertEqual(
            result["interfaces"][0]["status"],
            "column_axis_outside_observed_roof_surface",
        )

    def test_local_node_is_uniform_thickness_and_remains_candidate_only(self) -> None:
        report = {
            "settings": {
                "local_node_maximum_cross_distance_m": 3.5,
                "capital_height_m": 0.32,
                "capital_footprint_scale": 1.35,
                "capital_minimum_footprint_m": 0.75,
                "capital_maximum_footprint_m": 1.25,
                "local_underroof_minimum_thickness_m": 0.12,
                "local_underroof_along_half_extent_m": 1.0,
                "local_node_candidate_confidence": 0.72,
            },
            "column_grid": [
                {
                    "id": "C1",
                    "reviewed_seed_id": "V1",
                    "status": "confirmed_photo_seed",
                    "longitudinal_position_m": 5.0,
                    "cross_position_m": 3.0,
                    "maximum_z": 5.0,
                    "footprint_m": 0.8,
                }
            ],
            "roof_surfaces": [
                {
                    "id": "R1",
                    "footprint_profiles": [
                        {
                            "longitudinal_range_m": [0.0, 10.0],
                            "cross_range_m": [4.0, 6.0],
                            "z_equals_a_cross_plus_d": [0.1, 5.2],
                        }
                    ],
                }
            ],
        }
        report["local_column_roof_nodes"] = recover_local_column_roof_nodes(report)
        self.assertEqual(len(report["local_column_roof_nodes"]), 1)
        sections = report["local_column_roof_nodes"][0]["cross_sections"]
        for section in sections:
            self.assertAlmostEqual(
                section["top_z_m"] - section["bottom_z_m"], 0.12
            )
        audit = audit_local_node_chains(report)
        self.assertEqual(audit["candidate_chain_gate"], "passed")
        self.assertEqual(
            audit["formal_merge_gate"],
            "blocked_pending_fixed_view_visual_review",
        )
        self.assertFalse(
            audit["contact_surface_policy"][
                "coplanar_internal_contact_faces_emitted"
            ]
        )


if __name__ == "__main__":
    unittest.main()
