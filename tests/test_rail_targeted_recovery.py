from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.io import load_json, sha256_file, write_json
from railway_recon.rail_targeted_recovery import (
    apply_targeted_rail_recovery,
    evaluate_targeted_rail_recovery,
)


class RailTargetedRecoveryTests(unittest.TestCase):
    def test_fixed_center_recovery_uses_raw_support_without_moving_rails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud_path = root / "segment.las"
            x: list[float] = []
            y: list[float] = []
            z: list[float] = []
            for row in range(20):
                for rail_y in (-0.75, 0.75):
                    for offset in (0.10, 0.20, 0.30):
                        x.append(row * 0.5 + offset)
                        y.append(rail_y)
                        z.append(1.0)
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.asarray([0.001, 0.001, 0.001])
            cloud = laspy.LasData(header)
            cloud.x = np.asarray(x)
            cloud.y = np.asarray(y)
            cloud.z = np.asarray(z)
            cloud.write(cloud_path)

            report_path = root / "rail-report.json"
            report = {
                "segment_id": "s0000_0010m",
                "source": str(cloud_path),
                "frame": {
                    "origin_xy": [0.0, 0.0],
                    "along_xy": [1.0, 0.0],
                    "cross_xy": [0.0, 1.0],
                },
                "core_longitudinal_range_m": [0.0, 10.0],
                "rail_lines": [
                    {
                        "id": "left",
                        "cross_position_m": -0.75,
                        "median_z_m": 1.0,
                        "cross_fit_slope_m_per_m": 0.0,
                        "cross_fit_intercept_m": -0.75,
                        "z_fit_slope_m_per_m": 0.0,
                        "z_fit_intercept_m": 1.0,
                    },
                    {
                        "id": "right",
                        "cross_position_m": 0.75,
                        "median_z_m": 1.0,
                        "cross_fit_slope_m_per_m": 0.0,
                        "cross_fit_intercept_m": 0.75,
                        "z_fit_slope_m_per_m": 0.0,
                        "z_fit_intercept_m": 1.0,
                    },
                ],
            }
            write_json(report_path, report)
            graph_path = root / "track-graph.json"
            graph = {
                "schema_version": "railway.track-graph.v1",
                "sources": [
                    {
                        "segment_id": "s0000_0010m",
                        "path": str(report_path),
                        "sha256": sha256_file(report_path),
                    }
                ],
                "observations": [
                    {
                        "id": "s0000_0010m:TRACK-0001",
                        "segment_id": "s0000_0010m",
                        "global_track_id": "TRACK-0001",
                        "rail_cross_positions_local_m": [-0.75, 0.75],
                    }
                ],
                "pair_continuity_recovery": [
                    {
                        "observation_id": "s0000_0010m:TRACK-0001",
                        "status": "unresolved",
                    }
                ],
            }
            write_json(graph_path, graph)
            output = root / "recovery.json"
            evaluate_targeted_rail_recovery(graph_path, output)
            result = load_json(output)
            self.assertEqual(result["summary"]["recovered_count"], 1)
            recovery = result["recoveries"][0]
            self.assertEqual(recovery["status"], "fixed_center_evidence_recovered")
            self.assertEqual(recovery["rail_cross_positions_local_m"], [-0.75, 0.75])
            self.assertAlmostEqual(recovery["support"]["joint_support_ratio"], 1.0)
            updated_graph_path = root / "track-graph.recovered.json"
            apply_targeted_rail_recovery(graph_path, output, updated_graph_path)
            updated_graph = load_json(updated_graph_path)
            updated_observation = updated_graph["observations"][0]
            self.assertEqual(
                updated_observation["pair_recovery_status"],
                "fixed_center_evidence_recovered",
            )
            self.assertEqual(
                updated_observation["rail_cross_positions_local_m"], [-0.75, 0.75]
            )
            self.assertFalse(updated_graph["targeted_pair_recovery"]["geometry_changed"])

    def test_fixed_center_recovery_promotes_rule_inferred_edge_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_path = root / "track-graph.json"
            recovery_path = root / "recovery.json"
            output_path = root / "track-graph.recovered.json"
            observation_id = "s0000_0010m:TRACK-0001"
            write_json(
                graph_path,
                {
                    "schema_version": "railway.track-graph.v1",
                    "observations": [
                        {
                            "id": observation_id,
                            "segment_id": "s0000_0010m",
                            "global_track_id": "TRACK-0001",
                            "rail_cross_positions_local_m": [-0.75, 0.75],
                            "evidence_level": "rule_inferred",
                        }
                    ],
                    "edges": [
                        {
                            "id": "edge-1",
                            "source_observation_id": observation_id,
                            "evidence_level": "rule_inferred",
                        }
                    ],
                    "pair_continuity_recovery": [],
                },
            )
            write_json(
                recovery_path,
                {
                    "schema_version": "railway.targeted-rail-recovery.v1",
                    "graph_sha256": sha256_file(graph_path),
                    "recoveries": [
                        {
                            "observation_id": observation_id,
                            "status": "fixed_center_evidence_recovered",
                            "rail_cross_positions_local_m": [-0.75, 0.75],
                            "support": {"joint_support_ratio": 1.0},
                        }
                    ],
                },
            )
            apply_targeted_rail_recovery(graph_path, recovery_path, output_path)
            updated = load_json(output_path)
            observation = updated["observations"][0]
            edge = updated["edges"][0]
            self.assertEqual(observation["evidence_level"], "observed")
            self.assertEqual(observation["source_evidence_level"], "rule_inferred")
            self.assertEqual(edge["evidence_level"], "observed")
            self.assertEqual(edge["source_evidence_level"], "rule_inferred")
            self.assertEqual(
                updated["targeted_pair_recovery"]
                ["rule_inferred_observations_promoted_after_fixed_center_support"],
                1,
            )

    def test_explicit_observation_can_recheck_a_resolved_fixed_center(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud_path = root / "segment.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.asarray([0.001, 0.001, 0.001])
            cloud = laspy.LasData(header)
            longitudinal = np.repeat(np.arange(0.1, 10.0, 0.5), 6)
            cloud.x = longitudinal
            cloud.y = np.tile(np.repeat([-0.75, 0.75], 3), 20)
            cloud.z = np.ones(len(longitudinal))
            cloud.write(cloud_path)

            report_path = root / "rail-report.json"
            report = {
                "segment_id": "s0000_0010m",
                "source": str(cloud_path),
                "frame": {
                    "origin_xy": [0.0, 0.0],
                    "along_xy": [1.0, 0.0],
                    "cross_xy": [0.0, 1.0],
                },
                "core_longitudinal_range_m": [0.0, 10.0],
                "rail_lines": [
                    {
                        "cross_position_m": position,
                        "median_z_m": 1.0,
                        "cross_fit_slope_m_per_m": 0.0,
                        "cross_fit_intercept_m": position,
                        "z_fit_slope_m_per_m": 0.0,
                        "z_fit_intercept_m": 1.0,
                    }
                    for position in (-0.75, 0.75)
                ],
            }
            write_json(report_path, report)
            graph_path = root / "track-graph.json"
            observation_id = "s0000_0010m:TRACK-0001"
            write_json(
                graph_path,
                {
                    "schema_version": "railway.track-graph.v1",
                    "sources": [
                        {
                            "segment_id": "s0000_0010m",
                            "path": str(report_path),
                            "sha256": sha256_file(report_path),
                        }
                    ],
                    "observations": [
                        {
                            "id": observation_id,
                            "segment_id": "s0000_0010m",
                            "global_track_id": "TRACK-0001",
                            "rail_cross_positions_local_m": [-0.75, 0.75],
                        }
                    ],
                    "pair_continuity_recovery": [],
                },
            )
            output = root / "recovery.json"
            evaluate_targeted_rail_recovery(
                graph_path,
                output,
                observation_ids=[observation_id],
            )
            result = load_json(output)
            self.assertEqual(result["summary"]["candidate_count"], 1)
            self.assertEqual(result["summary"]["recovered_count"], 1)
            self.assertEqual(
                result["parameters"]["requested_observation_ids"], [observation_id]
            )


if __name__ == "__main__":
    unittest.main()
