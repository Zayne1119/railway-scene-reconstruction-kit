import unittest

import numpy as np

from railway_recon.platform_surface import (
    _classify_repeated_gap_patterns,
    _elevation_connected_record_groups,
    _fit_segments,
    _interior_gap_candidates,
    _resource_settings,
    analyze_platform_surface_data,
    build_platform_candidate_graph,
)


class PlatformSurfaceTests(unittest.TestCase):
    def test_touching_surfaces_at_different_elevations_are_separated(self) -> None:
        records = []
        for row in range(4):
            for column in range(8):
                records.append(
                    {
                        "row": row,
                        "column": column,
                        "z_p50_m": 1.0 if column < 4 else 1.4,
                    }
                )

        groups = _elevation_connected_record_groups(records, _resource_settings())

        self.assertEqual(sorted(len(group) for group in groups), [16, 16])

    def test_robust_fit_rejects_sparse_height_outliers(self) -> None:
        records = [
            {"s_m": float(index % 10), "c_m": float(index // 10), "z_p50_m": 1.0}
            for index in range(100)
        ]
        for index in range(10):
            records[index]["z_p50_m"] = 1.5
        settings = _resource_settings()
        settings["longitudinal_fit_segment_m"] = 20.0
        settings["minimum_fit_segment_cells"] = 20

        fits = _fit_segments(records, "right", settings)

        self.assertEqual(len(fits), 1)
        self.assertLess(fits[0]["absolute_residual_p90_m"], 0.01)
        self.assertLess(fits[0]["robust_inlier_fraction"], 1.0)

    def test_enclosed_surface_gap_is_preserved_as_candidate(self) -> None:
        records = []
        for row in range(12):
            for column in range(16):
                if 4 <= row <= 7 and 6 <= column <= 10:
                    continue
                records.append({"row": row, "column": column})
        settings = _resource_settings()
        settings["surface_grid_m"] = 0.25
        settings["minimum_opening_area_m2"] = 0.5
        candidates = _interior_gap_candidates("PLATFORM-001", records, (12, 16), 0.0, 0.0, settings)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["area_m2"], 1.25)
        self.assertGreaterEqual(candidates[0]["boundary_coverage_ratio"], 0.7)
        self.assertEqual(candidates[0]["gap_classification"], "unique_enclosed_gap_review_required")

    def test_regular_repeated_gaps_are_suppressed_as_occlusion_pattern(self) -> None:
        records = []
        missing = {
            (row, column)
            for column_start in (8, 24, 40, 56)
            for row in range(6, 10)
            for column in range(column_start, column_start + 4)
        }
        for row in range(16):
            for column in range(72):
                if (row, column) not in missing:
                    records.append({"row": row, "column": column})
        settings = _resource_settings()
        settings["surface_grid_m"] = 0.25
        settings["minimum_opening_area_m2"] = 0.5
        candidates = _interior_gap_candidates("PLATFORM-001", records, (16, 72), 0.0, 0.0, settings)
        self.assertEqual(len(candidates), 4)
        self.assertTrue(
            all(
                item["gap_classification"] == "periodic_occlusion_pattern_not_opening_candidate"
                for item in candidates
            )
        )
        self.assertTrue(
            all(
                item["mesh_action"] == "preserve_platform_surface_do_not_create_opening"
                for item in candidates
            )
        )

    def test_one_transverse_outlier_does_not_break_periodic_occlusion_pattern(self) -> None:
        settings = _resource_settings()
        candidates = []
        for index, center in enumerate((0.0, 9.0, 18.0, 22.5, 27.0, 36.0), start=1):
            candidates.append(
                {
                    "id": f"GAP-{index:03d}",
                    "longitudinal_range_m": [center - 0.5, center + 0.5],
                    "cross_range_m": [4.5, 6.25],
                    "interpretation": "candidate",
                    "status": "review",
                }
            )

        _classify_repeated_gap_patterns(candidates, settings)

        periodic = [
            item
            for item in candidates
            if item["gap_classification"]
            == "periodic_occlusion_pattern_not_opening_candidate"
        ]
        self.assertEqual(len(periodic), 5)
        self.assertTrue(all(item["periodic_pattern_outlier_count"] == 1 for item in periodic))

    def test_disjoint_observed_surfaces_are_not_forced_through_gap(self) -> None:
        points = []
        for s_start, s_end in ((0.0, 8.0), (14.0, 22.0)):
            for longitudinal in np.arange(s_start, s_end, 0.2):
                for cross in np.arange(-5.0, -3.0, 0.2):
                    for offset in (0.0, 0.01):
                        points.append((longitudinal, cross, 1.2 + offset))
        xyz = np.asarray(points)
        vertical = {
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "rail_lines": [
                {"id": "R1", "cross_position_m": -1.0, "median_z_m": 0.0},
                {"id": "R2", "cross_position_m": 1.0, "median_z_m": 0.0},
            ],
        }
        settings = _resource_settings()
        settings.update(
            {
                "surface_grid_m": 0.25,
                "minimum_cell_points": 1,
                "minimum_component_cells": 20,
                "minimum_component_length_m": 3.0,
                "minimum_component_width_m": 0.5,
                "minimum_fit_segment_cells": 4,
                "longitudinal_search_padding_m": 0.0,
            }
        )
        report = analyze_platform_surface_data(vertical, xyz[:, 0], xyz[:, 1], xyz[:, 2], settings)
        self.assertEqual(report["platform_component_count"], 2)
        intervals = [
            component["longitudinal_observed_intervals_m"]
            for component in report["platform_components"]
        ]
        self.assertTrue(all(len(value) == 1 for value in intervals))
        graph = build_platform_candidate_graph(report)
        self.assertEqual(graph["status"], "candidate_graph_only_no_asset_registry_or_model_write")

    def test_missing_rail_frame_fails_closed(self) -> None:
        vertical = {
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "rail_lines": [],
        }
        result = analyze_platform_surface_data(
            vertical,
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
            _resource_settings(),
        )
        self.assertEqual(result["status"], "insufficient_rail_frame")


if __name__ == "__main__":
    unittest.main()
