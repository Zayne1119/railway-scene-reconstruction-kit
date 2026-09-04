import unittest

import numpy as np

from railway_recon.canopy_structure import (
    _resource_settings,
    analyze_canopy_structure_data,
    build_canopy_candidate_graph,
)


class CanopyStructureTests(unittest.TestCase):
    def test_curved_roof_is_not_forced_to_plane_and_columns_contact(self) -> None:
        points = []
        for longitudinal in np.arange(-1.0, 20.01, 0.25):
            for cross in np.arange(-5.0, 5.01, 0.25):
                roof_z = 5.0 + 0.03 * (cross + 5.0) ** 2
                points.append((longitudinal, cross, roof_z))
                points.append((longitudinal + 0.01, cross, roof_z + 0.01))
        xyz = np.asarray(points, dtype=np.float64)
        lane = {
            "id": "VERTICAL-LANE-001",
            "strong_vertical_median_spacing_m": 9.0,
            "periodic_column_grid_candidate": True,
            "periodic_pattern_confidence": "high",
        }
        columns = []
        for index, longitudinal in enumerate((0.0, 9.0, 18.0), start=1):
            columns.append(
                {
                    "id": f"COLUMN-{index}",
                    "predicted_class": "canopy_column",
                    "longitudinal_position_m": longitudinal,
                    "cross_position_m": -5.0,
                    "maximum_z": 5.0,
                    "lane": lane,
                    "rail_context": {
                        "available": True,
                        "track_envelope_cross_range_m": [-1.0, 2.0],
                    },
                }
            )
        vertical_report = {
            "schema_version": "railway.vertical-hypotheses.v1",
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "rail_lines": [],
            "candidates": columns,
        }
        settings = _resource_settings()
        settings["roof_minimum_cell_points"] = 1
        settings["roof_minimum_component_points"] = 100
        settings["minimum_profile_points"] = 5
        report = analyze_canopy_structure_data(
            vertical_report, xyz[:, 0], xyz[:, 1], xyz[:, 2], settings
        )
        self.assertEqual(report["roof_component_count"], 1)
        self.assertEqual(
            report["roof_components"][0]["surface_type"], "curved_or_multi_surface"
        )
        self.assertEqual(report["column_roof_contact_counts"]["pass"], 3)
        self.assertEqual(report["distinct_transverse_member_candidate_count"], 0)
        graph = build_canopy_candidate_graph(report)
        self.assertEqual(graph["status"], "candidate_graph_only_no_asset_registry_write")
        self.assertEqual(
            sum(edge["type"] == "column_top_near_roof_observation" for edge in graph["edges"]),
            3,
        )

    def test_missing_column_grid_fails_closed(self) -> None:
        report = {
            "schema_version": "railway.vertical-hypotheses.v1",
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "rail_lines": [],
            "candidates": [],
        }
        xyz = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = analyze_canopy_structure_data(
            report, xyz[:, 0], xyz[:, 1], xyz[:, 2], _resource_settings()
        )
        self.assertEqual(result["status"], "no_stable_canopy_column_grid")
        self.assertEqual(result["roof_component_count"], 0)


if __name__ == "__main__":
    unittest.main()
