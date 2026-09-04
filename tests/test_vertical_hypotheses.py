import unittest

import numpy as np

from railway_recon.vertical_hypotheses import (
    _resource_settings,
    analyze_vertical_candidate_data,
    build_vertical_candidate_graph,
)


def _points(x: float, y: float, levels: list[float]) -> list[tuple[float, float, float]]:
    return [(x + (index % 3) * 0.003, y, z) for index, z in enumerate(levels)]


class VerticalHypothesisTests(unittest.TestCase):
    def test_periodic_columns_catenary_edge_and_noise_are_separated(self) -> None:
        candidates = []
        cloud: list[tuple[float, float, float]] = []

        def add(candidate_id: str, x: float, y: float, levels: list[float]) -> None:
            points = _points(x, y, levels)
            cloud.extend(points)
            candidates.append(
                {
                    "id": candidate_id,
                    "center_x": x,
                    "center_y": y,
                    "minimum_z": min(levels),
                    "maximum_z": max(levels),
                    "height_m": max(levels) - min(levels),
                    "footprint_m": 0.25,
                    "point_count": len(points),
                    "status": "unclassified_requires_photo_evidence",
                }
            )

        full_column = [value * 0.25 for value in range(21)]
        for index, x in enumerate((0.0, 9.0, 18.0), start=1):
            add(f"CANOPY-{index}", x, -5.0, full_column)
        add("CATENARY", 10.0, 0.75, [value * 0.25 for value in range(33)])
        for index, x in enumerate((0.0, 1.0, 2.0, 3.0, 4.0), start=1):
            add(f"EDGE-{index}", x, 4.0, [0.0, 5.0])
        add("NOISE", 16.0, 10.0, [0.0, 4.0])

        xyz = np.asarray(cloud, dtype=np.float64)
        report = {
            "schema_version": "railway.linear-candidates.v1",
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "vertical_candidates": candidates,
            "cable_candidates": [
                {
                    "id": "CABLE-1",
                    "cross_position_m": 0.75,
                    "elevation_z": 7.0,
                    "longitudinal_coverage": 0.8,
                }
            ],
        }
        rail = {
            "frame": report["frame"],
            "rail_lines": [
                {"id": "RAIL-1", "cross_position_m": 0.0},
                {"id": "RAIL-2", "cross_position_m": 1.5},
            ],
        }
        result = analyze_vertical_candidate_data(
            report, xyz[:, 0], xyz[:, 1], xyz[:, 2], _resource_settings(), rail
        )
        labels = {
            source_id: candidate["predicted_class"]
            for candidate in result["candidates"]
            for source_id in candidate["source_candidate_ids"]
        }
        self.assertEqual(labels["CATENARY"], "catenary_support")
        self.assertEqual(labels["CANOPY-1"], "canopy_column")
        self.assertEqual(labels["CANOPY-2"], "canopy_column")
        self.assertEqual(labels["CANOPY-3"], "canopy_column")
        self.assertEqual(labels["EDGE-3"], "building_edge")
        self.assertEqual(labels["NOISE"], "false_positive")
        self.assertEqual(
            result["status"], "automatic_hypotheses_only_no_asset_registry_write"
        )
        graph = build_vertical_candidate_graph(result)
        self.assertEqual(graph["status"], "candidate_graph_only_no_asset_registry_write")
        self.assertTrue(
            any(edge["type"] == "potential_supports_linear_candidate" for edge in graph["edges"])
        )

    def test_anchor_merges_nearby_fragment_without_transitive_lane_collapse(self) -> None:
        settings = _resource_settings()
        candidates = []
        cloud: list[tuple[float, float, float]] = []
        for identifier, x, levels, points in (
            ("ANCHOR", 0.0, [value * 0.25 for value in range(21)], 21),
            ("FRAGMENT", 0.7, [value * 0.25 for value in range(17)], 17),
            ("DISTINCT", 1.8, [value * 0.25 for value in range(21)], 21),
        ):
            local = _points(x, -5.0, levels)
            cloud.extend(local)
            candidates.append(
                {
                    "id": identifier,
                    "center_x": x,
                    "center_y": -5.0,
                    "minimum_z": min(levels),
                    "maximum_z": max(levels),
                    "height_m": max(levels) - min(levels),
                    "footprint_m": 0.25,
                    "point_count": points,
                    "status": "unclassified_requires_photo_evidence",
                }
            )
        xyz = np.asarray(cloud, dtype=np.float64)
        report = {
            "schema_version": "railway.linear-candidates.v1",
            "project_id": "synthetic",
            "segment_id": "s0000_0050m",
            "source": "synthetic.laz",
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            "vertical_candidates": candidates,
            "cable_candidates": [],
        }
        result = analyze_vertical_candidate_data(
            report, xyz[:, 0], xyz[:, 1], xyz[:, 2], settings
        )
        self.assertEqual(result["raw_candidate_count"], 3)
        self.assertEqual(result["merged_candidate_count"], 2)
        self.assertIn(
            {"ANCHOR", "FRAGMENT"},
            [set(candidate["source_candidate_ids"]) for candidate in result["candidates"]],
        )


if __name__ == "__main__":
    unittest.main()
