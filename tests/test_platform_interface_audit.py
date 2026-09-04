import unittest

from railway_recon.platform_interface_audit import audit_platform_interfaces_data


class PlatformInterfaceAuditTests(unittest.TestCase):
    def test_contacting_false_positive_is_reopened_as_semantic_conflict(self) -> None:
        platform = {
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "rail_lines": [
                {"id": "RAIL-1", "cross_position_m": -0.5, "median_z_m": 0.0},
                {"id": "RAIL-2", "cross_position_m": 0.5, "median_z_m": 0.0},
            ],
            "platform_components": [
                {
                    "id": "PLATFORM-SURFACE-CANDIDATE-001",
                    "side": "right",
                    "longitudinal_range_m": [0.0, 10.0],
                    "cross_range_m": [1.0, 5.0],
                    "fit_segments": [
                        {
                            "longitudinal_range_m": [0.0, 10.0],
                            "observed_cross_range_m": [1.0, 5.0],
                            "rail_side_edge_cross_m": 1.0,
                            "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.2],
                        }
                    ],
                }
            ],
        }
        vertical = {
            "candidates": [
                {
                    "id": "VERTICAL-HYPOTHESIS-0001",
                    "predicted_class": "false_positive",
                    "confidence": "medium",
                    "longitudinal_position_m": 5.0,
                    "cross_position_m": 3.0,
                    "minimum_z": 1.22,
                    "footprint_m": 0.5,
                }
            ]
        }
        canopy = {"stable_column_ids": [], "roof_components": []}
        gate = {
            "platform_component_id": "PLATFORM-SURFACE-CANDIDATE-001",
            "dispositions": [],
        }
        result = audit_platform_interfaces_data(
            platform,
            vertical,
            canopy,
            gate,
            {
                "vertical_base_contact_tolerance_m": 0.08,
                "maximum_adjacent_platform_edge_jump_m": 0.4,
            },
        )
        self.assertEqual(result["semantic_conflict_count"], 1)
        self.assertEqual(result["vertical_interface_count"], 1)
        self.assertEqual(
            result["vertical_interfaces"][0]["status"],
            "semantic_reclassification_required",
        )
        self.assertEqual(
            result["rail_interface"]["status"],
            "pass_internal_continuity_absolute_clearance_not_certified",
        )

    def test_vertical_outside_platform_is_not_reported_as_interface(self) -> None:
        platform = {
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "rail_lines": [{"id": "RAIL-1", "cross_position_m": 0.0, "median_z_m": 0.0}],
            "platform_components": [
                {
                    "id": "PLATFORM-SURFACE-CANDIDATE-001",
                    "side": "right",
                    "longitudinal_range_m": [0.0, 10.0],
                    "cross_range_m": [1.0, 5.0],
                    "fit_segments": [
                        {
                            "longitudinal_range_m": [0.0, 10.0],
                            "observed_cross_range_m": [1.0, 5.0],
                            "rail_side_edge_cross_m": 1.0,
                            "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.2],
                        }
                    ],
                }
            ],
        }
        vertical = {
            "candidates": [
                {
                    "id": "VERTICAL-HYPOTHESIS-0002",
                    "predicted_class": "catenary_support",
                    "confidence": "high",
                    "longitudinal_position_m": 5.0,
                    "cross_position_m": -2.0,
                    "minimum_z": 0.0,
                    "footprint_m": 0.4,
                }
            ]
        }
        result = audit_platform_interfaces_data(
            platform,
            vertical,
            {"stable_column_ids": [], "roof_components": []},
            {
                "platform_component_id": "PLATFORM-SURFACE-CANDIDATE-001",
                "dispositions": [],
            },
            {
                "vertical_base_contact_tolerance_m": 0.08,
                "maximum_adjacent_platform_edge_jump_m": 0.4,
            },
        )
        self.assertEqual(result["vertical_interface_count"], 0)
        self.assertEqual(result["catenary_support_intersection_count"], 0)


if __name__ == "__main__":
    unittest.main()
