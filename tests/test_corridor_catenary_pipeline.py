from __future__ import annotations

import unittest

from railway_recon.corridor_catenary_pipeline import (
    select_corridor_catenary_supports,
)


def _candidate(
    identifier: str,
    local_s: float,
    x: float,
    *,
    cable: bool = True,
    rail_distance_m: float = 2.0,
):
    return {
        "id": identifier,
        "center_x": x,
        "center_y": -8.0,
        "minimum_z": 20.0,
        "maximum_z": 27.5,
        "height_m": 7.5,
        "footprint_m": 0.6,
        "longitudinal_position_m": local_s,
        "cross_position_m": -8.0,
        "predicted_class": "building_edge",
        "features": {
            "vertical_axis_z": 0.999,
            "vertical_occupied_ratio": 0.30,
            "linearity": 0.99,
        },
        "cable_context": {
            "available": cable,
            "cross_difference_m": 0.05,
        },
        "rail_context": {
            "available": True,
            "nearest_rail_distance_m": rail_distance_m,
        },
    }


class CorridorCatenaryPipelineTests(unittest.TestCase):
    def test_deduplicates_boundaries_and_requires_periodic_run(self) -> None:
        manifest = {
            "segments": [
                {"id": "s0000_0050m", "chainage_start_m": 0.0, "chainage_end_m": 50.0},
                {"id": "s0050_0100m", "chainage_start_m": 50.0, "chainage_end_m": 100.0},
                {"id": "s0100_0150m", "chainage_start_m": 100.0, "chainage_end_m": 150.0},
                {"id": "s0150_0200m", "chainage_start_m": 150.0, "chainage_end_m": 200.0},
                {"id": "s0250_0300m", "chainage_start_m": 250.0, "chainage_end_m": 300.0},
            ]
        }
        frame = {"along_xy": [1.0, 0.0], "cross_xy": [0.0, 1.0]}
        reports = {
            "s0000_0050m": {"frame": frame, "candidates": [_candidate("A", 10.0, 10.0)]},
            "s0050_0100m": {
                "frame": frame,
                "candidates": [
                    _candidate("B", 10.0, 60.0),
                    _candidate("B-DUP", 11.0, 60.5),
                ],
            },
            "s0100_0150m": {"frame": frame, "candidates": [_candidate("C", 10.0, 110.0)]},
            "s0150_0200m": {"frame": frame, "candidates": [_candidate("D", 10.0, 160.0)]},
            "s0250_0300m": {"frame": frame, "candidates": [_candidate("ISO", 10.0, 260.0)]},
        }
        settings = {
            "minimum_height_m": 5.0,
            "maximum_height_m": 12.0,
            "minimum_footprint_m": 0.15,
            "maximum_footprint_m": 1.2,
            "minimum_vertical_axis_z": 0.97,
            "minimum_vertical_occupied_ratio": 0.2,
            "minimum_linearity": 0.8,
            "maximum_cable_cross_difference_m": 0.35,
            "duplicate_station_tolerance_m": 3.0,
            "duplicate_xy_tolerance_m": 2.5,
            "minimum_periodic_spacing_m": 35.0,
            "maximum_periodic_spacing_m": 65.0,
            "maximum_periodic_cross_change_m": 1.2,
            "minimum_periodic_run_count": 3,
            "minimum_mast_distance_from_nearest_rail_m": 1.0,
            "maximum_mast_distance_from_nearest_rail_m": 5.0,
            "minimum_renderable_mast_height_m": 6.5,
        }
        paths = {key: f"{key}.json" for key in reports}

        result = select_corridor_catenary_supports(
            manifest, reports, paths, settings
        )

        self.assertEqual(result["geometric_seed_count"], 6)
        self.assertEqual(result["deduplicated_seed_count"], 5)
        self.assertEqual(result["periodic_run_count"], 1)
        self.assertEqual(result["periodic_support_count"], 4)
        self.assertEqual(result["accepted_support_count"], 4)
        accepted = [
            item
            for item in result["candidates"]
            if item["accepted_for_candidate_mesh"]
        ]
        self.assertEqual([item["chainage_m"] for item in accepted], [10.0, 60.0, 110.0, 160.0])
        self.assertEqual(accepted[1]["duplicate_candidate_count"], 2)

    def test_withholds_periodic_seed_that_conflicts_with_track_clearance(self) -> None:
        manifest = {
            "segments": [
                {
                    "id": f"s{start:04d}_{start + 50:04d}m",
                    "chainage_start_m": float(start),
                    "chainage_end_m": float(start + 50),
                }
                for start in (0, 50, 100)
            ]
        }
        frame = {"along_xy": [1.0, 0.0], "cross_xy": [0.0, 1.0]}
        reports = {
            "s0000_0050m": {
                "frame": frame,
                "candidates": [_candidate("A", 10.0, 10.0)],
            },
            "s0050_0100m": {
                "frame": frame,
                "candidates": [
                    _candidate("B", 10.0, 60.0, rail_distance_m=0.2)
                ],
            },
            "s0100_0150m": {
                "frame": frame,
                "candidates": [_candidate("C", 10.0, 110.0)],
            },
        }
        settings = {
            "minimum_height_m": 5.0,
            "maximum_height_m": 12.0,
            "minimum_footprint_m": 0.15,
            "maximum_footprint_m": 1.2,
            "minimum_vertical_axis_z": 0.97,
            "minimum_vertical_occupied_ratio": 0.2,
            "minimum_linearity": 0.8,
            "maximum_cable_cross_difference_m": 0.35,
            "duplicate_station_tolerance_m": 3.0,
            "duplicate_xy_tolerance_m": 2.5,
            "minimum_periodic_spacing_m": 35.0,
            "maximum_periodic_spacing_m": 65.0,
            "maximum_periodic_cross_change_m": 1.2,
            "minimum_periodic_run_count": 3,
            "minimum_mast_distance_from_nearest_rail_m": 1.0,
            "maximum_mast_distance_from_nearest_rail_m": 5.0,
            "minimum_renderable_mast_height_m": 6.5,
        }

        result = select_corridor_catenary_supports(
            manifest,
            reports,
            {key: f"{key}.json" for key in reports},
            settings,
        )

        self.assertEqual(result["periodic_support_count"], 3)
        self.assertEqual(result["accepted_support_count"], 2)
        self.assertEqual(result["withheld_by_clearance_count"], 1)
        withheld = next(
            item for item in result["candidates"] if item["candidate_id"] == "B"
        )
        self.assertEqual(
            withheld["candidate_mesh_decision"],
            "withheld_track_clearance_conflict",
        )


if __name__ == "__main__":
    unittest.main()
