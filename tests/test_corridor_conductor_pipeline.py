from __future__ import annotations

import unittest

import numpy as np

from railway_recon.corridor_conductor_pipeline import (
    bind_conductor_spans_to_track_centerlines,
    select_conductor_sequence_options,
    select_conductor_spans,
    snap_passing_conductor_seams,
)


class CorridorConductorPipelineTests(unittest.TestCase):
    def test_selects_contact_and_messenger_by_track_relative_height(self) -> None:
        manifest = {
            "segments": [
                {"id": "s0000_0050m", "chainage_start_m": 0.0, "chainage_end_m": 50.0}
            ]
        }
        reports = {
            "s0000_0050m": {
                "frame": {
                    "origin_xy": [100.0, 200.0],
                    "along_xy": [1.0, 0.0],
                    "cross_xy": [0.0, 1.0],
                },
                "cable_candidates": [
                    {
                        "id": "CONTACT",
                        "cross_position_m": -2.05,
                        "elevation_z": 15.3,
                        "longitudinal_coverage": 0.8,
                    },
                    {
                        "id": "MESSENGER",
                        "cross_position_m": -1.95,
                        "elevation_z": 16.5,
                        "longitudinal_coverage": 0.7,
                    },
                    {
                        "id": "GROUND-EDGE",
                        "cross_position_m": -2.0,
                        "elevation_z": 10.3,
                        "longitudinal_coverage": 0.9,
                    },
                ],
            }
        }
        graph = {
            "observations": [
                {
                    "segment_id": "s0000_0050m",
                    "global_track_id": "TRACK-0001",
                    "lateral_offset_m": -2.0,
                    "rail_top_z_m": 10.0,
                }
            ]
        }
        settings = {
            "maximum_track_cross_difference_m": 0.45,
            "minimum_longitudinal_coverage": 0.15,
            "cross_error_weight": 0.7,
            "coverage_reward_weight": 0.2,
            "sequence_continuity_cross_weight": 0.75,
            "sequence_continuity_height_weight": 1.5,
            "maximum_adjacent_chainage_gap_m": 0.05,
            "wire_types": {
                "contact_wire": {
                    "minimum_height_above_rail_m": 4.6,
                    "maximum_height_above_rail_m": 5.9,
                    "target_height_above_rail_m": 5.3,
                },
                "messenger_wire": {
                    "minimum_height_above_rail_m": 6.0,
                    "maximum_height_above_rail_m": 7.3,
                    "target_height_above_rail_m": 6.5,
                },
            },
        }

        result = select_conductor_spans(
            manifest,
            reports,
            {"s0000_0050m": "linear.json"},
            graph,
            settings,
        )

        self.assertEqual(result["span_count"], 2)
        by_type = {item["wire_type"]: item for item in result["spans"]}
        self.assertEqual(by_type["contact_wire"]["source_candidate_id"], "CONTACT")
        self.assertEqual(by_type["messenger_wire"]["source_candidate_id"], "MESSENGER")
        self.assertEqual(by_type["contact_wire"]["start_xyz"], [100.0, 197.95, 15.3])
        self.assertEqual(by_type["contact_wire"]["end_xyz"], [150.0, 197.95, 15.3])
        self.assertAlmostEqual(by_type["contact_wire"]["wire_track_offset_m"], -0.05)

    def test_joint_sequence_selection_can_reject_local_greedy_outlier(self) -> None:
        def option(
            segment: str,
            start: float,
            candidate: str,
            score: float,
            height: float,
        ) -> dict:
            return {
                "id": f"CONTACT_WIRE-TRACK-0001-{segment.upper()}",
                "segment_id": segment,
                "track_id": "TRACK-0001",
                "wire_type": "contact_wire",
                "source_candidate_id": candidate,
                "chainage_start_m": start,
                "chainage_end_m": start + 50.0,
                "wire_track_offset_m": 0.0,
                "height_above_rail_m": height,
                "selection_score": score,
            }

        options = [
            option("s0000_0050m", 0.0, "A", 0.05, 5.3),
            option("s0050_0100m", 50.0, "B-LOCAL", 0.01, 5.8),
            option("s0050_0100m", 50.0, "B-SMOOTH", 0.20, 5.3),
            option("s0100_0150m", 100.0, "C", 0.05, 5.3),
        ]
        settings = {
            "maximum_adjacent_chainage_gap_m": 0.05,
            "sequence_continuity_cross_weight": 0.75,
            "sequence_continuity_height_weight": 1.5,
        }

        selected, audit = select_conductor_sequence_options(options, settings)

        self.assertEqual(
            [item["source_candidate_id"] for item in selected],
            ["A", "B-SMOOTH", "C"],
        )
        self.assertEqual(audit["changed_from_local_greedy_count"], 1)

    def test_binds_local_evidence_span_to_global_track_centerline(self) -> None:
        spans = [
            {
                "id": "CONTACT-TRACK-0001-S0025_0075M",
                "track_id": "TRACK-0001",
                "chainage_start_m": 25.0,
                "chainage_end_m": 75.0,
                "wire_track_offset_m": 0.1,
                "height_above_rail_m": 5.2,
                "start_xyz": [999.0, 999.0, 15.2],
                "end_xyz": [1049.0, 999.0, 15.4],
            }
        ]
        track_centerlines = {
            "TRACK-0001": (
                np.asarray([0.0, 50.0, 100.0]),
                np.asarray(
                    [[0.0, 0.0, 10.0], [50.0, 0.0, 10.1], [100.0, 0.0, 10.2]]
                ),
            )
        }

        bound, audit = bind_conductor_spans_to_track_centerlines(
            spans, track_centerlines
        )

        self.assertAlmostEqual(bound[0]["start_xyz"][0], 25.0)
        self.assertAlmostEqual(bound[0]["start_xyz"][1], 0.1)
        self.assertAlmostEqual(bound[0]["start_xyz"][2], 15.25)
        self.assertAlmostEqual(bound[0]["end_xyz"][0], 75.0)
        self.assertAlmostEqual(bound[0]["end_xyz"][1], 0.1)
        self.assertAlmostEqual(bound[0]["end_xyz"][2], 15.35)
        self.assertEqual(bound[0]["evidence_start_xyz"], [999.0, 999.0, 15.2])
        self.assertEqual(bound[0]["geometry_binding"], "global_track_graph_centerline")
        self.assertEqual(audit["span_count"], 1)
        self.assertTrue(audit["source_evidence_endpoints_retained"])

    def test_snaps_only_seams_that_passed_the_audit_gate(self) -> None:
        spans = [
            {"id": "A", "start_xyz": [0.0, 0.0, 10.0], "end_xyz": [1.0, 0.0, 10.0]},
            {"id": "B", "start_xyz": [1.2, 0.0, 10.1], "end_xyz": [2.0, 0.0, 10.1]},
            {"id": "C", "start_xyz": [2.8, 0.0, 10.8], "end_xyz": [3.0, 0.0, 10.8]},
        ]
        seams = [
            {
                "left_span_id": "A",
                "right_span_id": "B",
                "endpoint_xy_error_m": 0.2,
                "endpoint_z_error_m": 0.1,
                "passed": True,
            },
            {
                "left_span_id": "B",
                "right_span_id": "C",
                "endpoint_xy_error_m": 0.8,
                "endpoint_z_error_m": 0.7,
                "passed": False,
            },
        ]

        repaired, audit = snap_passing_conductor_seams(spans, seams)
        by_id = {span["id"]: span for span in repaired}

        self.assertEqual(by_id["A"]["end_xyz"], [1.1, 0.0, 10.05])
        self.assertEqual(by_id["B"]["start_xyz"], [1.1, 0.0, 10.05])
        self.assertEqual(by_id["B"]["end_xyz"], [2.0, 0.0, 10.1])
        self.assertEqual(by_id["C"]["start_xyz"], [2.8, 0.0, 10.8])
        self.assertEqual(audit["repaired_seam_count"], 1)


if __name__ == "__main__":
    unittest.main()
