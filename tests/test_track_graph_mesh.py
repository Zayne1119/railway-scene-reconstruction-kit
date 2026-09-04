from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from railway_recon.algorithms.track_graph_mesh import (
    _candidate_inferred_gap_intervals,
    _evidence_interval_gaps,
    _hermite_turnout_centerline,
    _interval_component_id,
    _polyline_interval,
    _track_evidence_intervals,
    build_track_graph_mesh,
)
from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.mesh_audit import audit_obj
from railway_recon.qa import quality_report
from railway_recon.registry import validate_registry_file
from railway_recon.segments import plan_segments
from railway_recon.track_graph import build_track_graph


class TrackGraphMeshTests(unittest.TestCase):
    def test_evidence_interval_components_and_gaps_remain_explicit(self) -> None:
        intervals = [
            {
                "chainage_start_m": 0.0,
                "chainage_end_m": 50.0,
                "evidence_level": "observed",
            },
            {
                "chainage_start_m": 75.0,
                "chainage_end_m": 100.0,
                "evidence_level": "observed",
            },
        ]

        self.assertEqual(
            _interval_component_id("TRACK-1", "BED", intervals[0], 1, 2),
            "TRACK-1-BED-OBSERVED-001",
        )
        self.assertEqual(
            _evidence_interval_gaps(intervals),
            [
                {
                    "chainage_start_m": 50.0,
                    "chainage_end_m": 75.0,
                    "length_m": 25.0,
                }
            ],
        )

    def test_candidate_gap_hypotheses_are_bounded_and_explicit(self) -> None:
        intervals = [
            {
                "chainage_start_m": 0.0,
                "chainage_end_m": 50.0,
                "evidence_level": "observed",
                "source_observation_ids": ["A"],
            },
            {
                "chainage_start_m": 75.0,
                "chainage_end_m": 100.0,
                "evidence_level": "observed",
                "source_observation_ids": ["B"],
            },
            {
                "chainage_start_m": 400.0,
                "chainage_end_m": 450.0,
                "evidence_level": "observed",
                "source_observation_ids": ["C"],
            },
        ]

        inferred = _candidate_inferred_gap_intervals(intervals, 50.0)

        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["chainage_start_m"], 50.0)
        self.assertEqual(inferred[0]["chainage_end_m"], 75.0)
        self.assertEqual(inferred[0]["evidence_level"], "rule_inferred")
        self.assertEqual(inferred[0]["source_observation_ids"], ["A", "B"])

    def test_evidence_intervals_are_merged_without_hiding_inference(self) -> None:
        observations = [
            {
                "id": "A",
                "chainage_start_m": 0.0,
                "chainage_end_m": 5.0,
                "evidence_level": "observed",
            },
            {
                "id": "B",
                "chainage_start_m": 5.0,
                "chainage_end_m": 10.0,
                "evidence_level": "rule_inferred",
            },
            {
                "id": "C",
                "chainage_start_m": 10.0,
                "chainage_end_m": 15.0,
                "evidence_level": "rule_inferred",
            },
            {
                "id": "D",
                "chainage_start_m": 15.0,
                "chainage_end_m": 20.0,
                "evidence_level": "observed",
            },
        ]

        intervals = _track_evidence_intervals(observations)

        self.assertEqual(
            [(item["chainage_start_m"], item["chainage_end_m"]) for item in intervals],
            [(0.0, 5.0), (5.0, 15.0), (15.0, 20.0)],
        )
        self.assertEqual(
            [item["evidence_level"] for item in intervals],
            ["observed", "rule_inferred", "observed"],
        )
        self.assertEqual(intervals[1]["source_observation_ids"], ["B", "C"])

    def test_polyline_interval_inserts_exact_boundary_samples(self) -> None:
        chainages = np.asarray([0.0, 5.0, 10.0])
        polyline = np.asarray([[0.0, 0.0, 0.0], [5.0, 1.0, 2.0], [10.0, 2.0, 4.0]])

        clipped = _polyline_interval(chainages, polyline, 2.5, 7.5)

        np.testing.assert_allclose(
            clipped,
            np.asarray([[2.5, 0.5, 1.0], [5.0, 1.0, 2.0], [7.5, 1.5, 3.0]]),
        )

    def test_turnout_connector_matches_branch_and_target_endpoints(self) -> None:
        branch_chainages = np.asarray([0.0, 5.0, 10.0])
        branch = np.asarray([[0.0, 0.0, 0.0], [5.0, -1.0, 0.1], [10.0, -2.0, 0.2]])
        target_chainages = np.asarray([0.0, 10.0, 20.0, 30.0])
        target = np.asarray(
            [[0.0, 4.0, 0.0], [10.0, 4.0, 0.1], [20.0, 4.0, 0.2], [30.0, 4.0, 0.3]]
        )

        chainages, connector = _hermite_turnout_centerline(
            branch_chainages, branch, target_chainages, target, 30.0, 0.5
        )

        self.assertEqual(chainages[0], 10.0)
        self.assertEqual(chainages[-1], 30.0)
        np.testing.assert_allclose(connector[0], branch[-1])
        np.testing.assert_allclose(connector[-1], target[-1])
        self.assertTrue(np.all(np.linalg.norm(np.diff(connector[:, :2], axis=0), axis=1) > 0))

    def _project_with_graph(self, root: Path, *, reviewed: bool = True):
        project = load_project(
            initialize_project(root / "sample", "sample-project", "Sample")
        )
        camera_path = project.input_path("camera_csv")
        assert camera_path is not None
        with camera_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
            writer.writerow([0, 0, "000.jpg", 0, 0, 2])
            writer.writerow([1, 1, "001.jpg", 50, 0, 2])
            writer.writerow([2, 2, "002.jpg", 100, 0, 2])
        plan_segments(project)
        report_paths: list[Path] = []
        for segment_id, origin_x in (("s0000_0050m", 25.0), ("s0050_0100m", 75.0)):
            report = {
                "schema_version": "railway.rail-candidates.v1",
                "project_id": project.project_id,
                "segment_id": segment_id,
                "frame": {
                    "origin_xy": [origin_x, 0.0],
                    "along_xy": [1.0, 0.0],
                    "cross_xy": [0.0, 1.0],
                },
                "longitudinal_range_m": [-25.0, 25.0],
                "rail_pairs": [
                    {
                        "track_id": "TRACK-0001",
                        "peak_indexes": [0, 1],
                        "cross_positions_m": [-0.754, 0.754],
                        "separation_m": 1.508,
                        "rail_center_spacing_m": 1.508,
                        "gauge_m": 1.435,
                        "score": 1.9,
                    }
                ],
                "rail_lines": [
                    {
                        "id": "RAIL-0001",
                        "cross_position_m": -0.754,
                        "median_z_m": 0.0,
                        "point_count": 1000,
                    },
                    {
                        "id": "RAIL-0002",
                        "cross_position_m": 0.754,
                        "median_z_m": 0.0,
                        "point_count": 1000,
                    },
                ],
                "peaks": [
                    {"grid_index": 0, "cross_position_m": -0.754, "coverage": 0.95},
                    {"grid_index": 1, "cross_position_m": 0.754, "coverage": 0.95},
                ],
                "status": "geometry_candidates_only_manual_review_required",
            }
            if reviewed:
                report["review_status"] = "accepted"
            path = project.workspace_path("reports") / f"{segment_id}.json"
            write_json(path, report)
            report_paths.append(path)
        graph_result = build_track_graph(
            project,
            [
                ("s0000_0050m", report_paths[0]),
                ("s0050_0100m", report_paths[1]),
            ],
        )
        return project, graph_result

    def test_builds_one_continuous_mesh_set_per_global_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _ = self._project_with_graph(Path(temporary))

            report = build_track_graph_mesh(project)

            self.assertEqual(report["status"], "review_required")
            self.assertTrue(report["automatic_checks_passed"])
            self.assertEqual(report["visible_node_count"], 4)
            self.assertEqual(len(report["tracks"]), 1)
            stats = report["tracks"][0]
            self.assertAlmostEqual(stats["sleeper_minimum_spacing_m"], 0.6)
            self.assertAlmostEqual(stats["sleeper_maximum_spacing_m"], 0.6)
            mesh_report = audit_obj(Path(report["output_obj"]))
            self.assertTrue(mesh_report["passed"])
            self.assertEqual(mesh_report["object_count"], 4)

            object_names = []
            with Path(report["output_obj"]).open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("o "):
                        object_names.append(line[2:].strip())
            self.assertEqual(
                set(object_names),
                {
                    "TRACK-0001-RAIL-LEFT",
                    "TRACK-0001-RAIL-RIGHT",
                    "TRACK-0001-SLEEPERS",
                    "TRACK-0001-BED",
                },
            )
            self.assertFalse(any("SLEEPER-0001" in name for name in object_names))

            registry, errors = validate_registry_file(project.workspace_path("asset_registry"))
            self.assertEqual(errors, [])
            self.assertEqual(len(registry["assets"]), 5)
            self.assertEqual(len(registry["relations"]), 4)
            sidecar, sidecar_errors = validate_registry_file(
                Path(report["asset_registry_sidecar"])
            )
            self.assertEqual(sidecar_errors, [])
            self.assertEqual(len(sidecar["assets"]), 5)
            self.assertEqual(len(sidecar["relations"]), 4)
            visible = set(load_json(Path(report["mesh_audit"]))["visible_node_ids"])
            self.assertTrue({item["id"] for item in registry["assets"]}.issuperset(visible))
            project_qa = quality_report(project)
            track_mesh_check = next(
                item for item in project_qa["checks"] if item["id"] == "track_graph_mesh.pass"
            )
            self.assertTrue(track_mesh_check["passed"])

    def test_nonpassing_track_graph_cannot_generate_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, graph_result = self._project_with_graph(Path(temporary), reviewed=False)
            self.assertEqual(graph_result["status"], "review_required")
            with self.assertRaisesRegex(ValueError, "audit is not pass"):
                build_track_graph_mesh(project)

    def test_nonpassing_measurement_audit_can_only_generate_unregistered_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, graph_result = self._project_with_graph(Path(temporary))
            audit_path = Path(graph_result["output_audit_path"])
            audit = load_json(audit_path)
            gauge_check = next(
                item for item in audit["checks"] if item["id"] == "rail_gauge"
            )
            gauge_check["status"] = "fail"
            gauge_check["failure_count"] = 1
            gauge_check["failures"] = [{"reason": "test_measurement_failure"}]
            audit["status"] = "fail"
            audit["passed"] = False
            write_json(audit_path, audit)

            report = build_track_graph_mesh(
                project,
                update_registry=False,
                candidate_only_nonpassing_audit=True,
            )

            self.assertEqual(
                report["status"],
                "candidate_review_required_nonpassing_source_audit",
            )
            self.assertFalse(report["automatic_checks_passed"])
            self.assertTrue(report["automatic_mesh_checks_passed"])
            self.assertFalse(report["formal_release"])
            self.assertFalse(report["registry_updated"])
            self.assertEqual(
                [item["id"] for item in report["source_track_graph_nonpassing_checks"]],
                ["rail_gauge"],
            )
            self.assertFalse(project.workspace_path("asset_registry").exists())

    def test_nonpassing_core_audit_blocks_candidate_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, graph_result = self._project_with_graph(Path(temporary))
            audit_path = Path(graph_result["output_audit_path"])
            audit = load_json(audit_path)
            structure_check = next(
                item for item in audit["checks"] if item["id"] == "track_graph_structure"
            )
            structure_check["status"] = "fail"
            structure_check["failure_count"] = 1
            structure_check["failures"] = [{"reason": "test_core_failure"}]
            audit["status"] = "fail"
            audit["passed"] = False
            write_json(audit_path, audit)

            with self.assertRaisesRegex(ValueError, "core audit checks"):
                build_track_graph_mesh(
                    project,
                    update_registry=False,
                    candidate_only_nonpassing_audit=True,
                )

    def test_graph_edit_after_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, graph_result = self._project_with_graph(Path(temporary))
            graph_path = Path(graph_result["output_graph_path"])
            graph = load_json(graph_path)
            graph["observations"][1]["lateral_offset_m"] = 3.0
            write_json(graph_path, graph)
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                build_track_graph_mesh(project)

    def test_camera_trajectory_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _ = self._project_with_graph(Path(temporary))
            camera_path = project.input_path("camera_csv")
            assert camera_path is not None
            with camera_path.open("a", encoding="utf-8", newline="") as stream:
                stream.write("3,3,003.jpg,101,0,2\n")
            with self.assertRaisesRegex(ValueError, "project bindings are stale"):
                build_track_graph_mesh(project)

    def test_rail_sleeper_contact_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _ = self._project_with_graph(Path(temporary))
            config_path = project.resolve(project.value["algorithms"]["track_build"])
            config = load_json(config_path)
            config["sleeper"]["top_below_rail_m"] = 0.145
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "do not contact"):
                build_track_graph_mesh(project)

    def test_track_graph_threshold_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _ = self._project_with_graph(Path(temporary))
            settings_path = project.resolve(project.value["algorithms"]["track_graph"])
            settings = load_json(settings_path)
            settings["quality"]["maximum_seam_3d_error_m"] = 0.04
            write_json(settings_path, settings)
            with self.assertRaisesRegex(ValueError, "project bindings are stale"):
                build_track_graph_mesh(project)


if __name__ == "__main__":
    unittest.main()
