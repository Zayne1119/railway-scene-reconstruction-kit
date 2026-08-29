from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.segments import plan_segments
from railway_recon.track_graph import (
    _classify_pair_continuity_recovery,
    audit_track_graph,
    build_track_graph,
)


class TrackGraphTests(unittest.TestCase):
    def test_pair_recovery_uses_only_adjacent_same_track_pass_evidence(self) -> None:
        settings = {
            "quality": {
                "maximum_pair_recovery_lateral_difference_m": 0.30,
                "maximum_pair_recovery_gauge_difference_m": 0.03,
                "maximum_pair_recovery_chainage_gap_m": 5.0,
                "maximum_pair_recovery_rail_top_difference_m": 0.15,
            }
        }

        def observation(
            observation_id: str,
            track_id: str,
            start: float,
            end: float,
            lateral: float,
            status: str,
        ) -> dict[str, Any]:
            return {
                "id": observation_id,
                "global_track_id": track_id,
                "chainage_start_m": start,
                "chainage_end_m": end,
                "lateral_offset_m": lateral,
                "gauge_m": 1.435,
                "rail_top_z_m": 23.0,
                "rail_top_start_z_m": 23.0,
                "rail_top_end_z_m": 23.0,
                "paired_support_evaluated": True,
                "pair_continuity_status": status,
            }

        values = [
            observation("pass", "TRACK-0001", 0.0, 50.0, -1.10, "pass"),
            observation(
                "recover", "TRACK-0001", 50.0, 100.0, -1.15, "review_required"
            ),
            observation(
                "wrong-track", "TRACK-0002", 50.0, 100.0, -1.14, "review_required"
            ),
            observation(
                "too-far", "TRACK-0001", 100.0, 150.0, -2.00, "review_required"
            ),
        ]
        recovery = _classify_pair_continuity_recovery(values, settings)
        by_id = {item["observation_id"]: item for item in recovery}
        self.assertEqual(by_id["recover"]["status"], "adjacent_track_evidence")
        self.assertEqual(by_id["recover"]["support"]["observation_id"], "pass")
        self.assertEqual(by_id["wrong-track"]["status"], "unresolved")
        self.assertEqual(by_id["too-far"]["status"], "unresolved")

    def _project(self, root: Path):
        config_path = initialize_project(root / "sample", "sample-project", "Sample")
        project = load_project(config_path)
        camera_path = project.input_path("camera_csv")
        assert camera_path is not None
        with camera_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
            writer.writerow([0, 0, "000.jpg", 0, 0, 2])
            writer.writerow([1, 1, "001.jpg", 50, 0, 2])
            writer.writerow([2, 2, "002.jpg", 100, 0, 2])
        manifest = plan_segments(project)
        self.assertEqual(
            [item["id"] for item in manifest["segments"]], ["s0000_0050m", "s0050_0100m"]
        )
        return project, manifest

    def _report(
        self,
        path: Path,
        segment_id: str,
        origin_x: float,
        centers: list[float] | None = None,
        *,
        reversed_frame: bool = False,
        reviewed: bool = True,
        z_m: float = 0.0,
        rail_z_difference_m: float = 0.0,
    ) -> Path:
        centers = centers or [0.0]
        rail_pairs: list[dict[str, Any]] = []
        rail_lines: list[dict[str, Any]] = []
        peaks: list[dict[str, Any]] = []
        for index, center in enumerate(centers, start=1):
            rail_center_spacing = 1.435 + 0.073
            positions = [
                center - rail_center_spacing / 2.0,
                center + rail_center_spacing / 2.0,
            ]
            rail_pairs.append(
                {
                    "track_id": f"TRACK-{index:04d}",
                    "peak_indexes": [2 * index - 2, 2 * index - 1],
                    "cross_positions_m": positions,
                    "separation_m": rail_center_spacing,
                    "rail_center_spacing_m": rail_center_spacing,
                    "gauge_m": 1.435,
                    "score": 1.9,
                }
            )
            for rail_index, position in enumerate(positions, start=1):
                rail_lines.append(
                    {
                        "id": f"RAIL-{index:02d}-{rail_index:02d}",
                        "cross_position_m": position,
                        "median_z_m": z_m + (rail_z_difference_m if rail_index == 2 else 0.0),
                        "point_count": 500,
                    }
                )
                peaks.append(
                    {
                        "grid_index": len(peaks),
                        "cross_position_m": position,
                        "coverage": 0.95,
                    }
                )
        report = {
            "schema_version": "railway.rail-candidates.v1",
            "project_id": "sample-project",
            "segment_id": segment_id,
            "frame": {
                "origin_xy": [origin_x, 0.0],
                "along_xy": [-1.0, 0.0] if reversed_frame else [1.0, 0.0],
                "cross_xy": [0.0, -1.0] if reversed_frame else [0.0, 1.0],
            },
            "longitudinal_range_m": [-25.0, 25.0],
            "rail_pairs": rail_pairs,
            "rail_lines": rail_lines,
            "peaks": peaks,
            "status": "geometry_candidates_only_manual_review_required",
        }
        if reviewed:
            report["review_status"] = "accepted"
        write_json(path, report)
        return path

    def _build(
        self,
        root: Path,
        *,
        second_center: float = 0.0,
        reversed_second: bool = False,
        duplicate_centers: list[float] | None = None,
        reviewed: bool = True,
    ):
        project, _ = self._project(root)
        reports = project.workspace_path("reports")
        centers_a = duplicate_centers or [0.0]
        centers_b = duplicate_centers or [second_center]
        first = self._report(reports / "a.json", "s0000_0050m", 25.0, centers_a, reviewed=reviewed)
        second = self._report(
            reports / "b.json",
            "s0050_0100m",
            75.0,
            centers_b,
            reversed_frame=reversed_second,
            reviewed=reviewed,
        )
        result = build_track_graph(
            project,
            [("s0000_0050m", first), ("s0050_0100m", second)],
        )
        graph = load_json(Path(result["output_graph_path"]))
        audit = load_json(Path(result["output_audit_path"]))
        return project, result, graph, audit

    def test_stable_global_identity_passes_reviewed_straight_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, graph, audit = self._build(Path(temporary))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(graph["tracks"]), 1)
            self.assertEqual(
                {item["global_track_id"] for item in graph["observations"]}, {"TRACK-0001"}
            )
            self.assertEqual(len(graph["seams"]), 1)
            self.assertTrue(audit["passed"])

    def test_reversed_segment_fails_without_force_connecting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, graph, audit = self._build(Path(temporary), reversed_second=True)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(len(graph["tracks"]), 2)
            check = next(item for item in audit["checks"] if item["id"] == "canonical_direction")
            self.assertEqual(check["status"], "fail")

    def test_wrong_track_family_becomes_explicit_break_not_connector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, graph, audit = self._build(Path(temporary), second_center=4.0)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(len(graph["tracks"]), 2)
            self.assertEqual(graph["identity_events"][0]["match_count"], 0)
            self.assertEqual(graph["seams"], [])
            check = next(
                item for item in audit["checks"] if item["id"] == "interior_track_termination"
            )
            self.assertEqual(check["status"], "fail")

    def test_half_metre_seam_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, graph, audit = self._build(Path(temporary), second_center=0.5)
            self.assertEqual(len(graph["tracks"]), 1)
            self.assertEqual(result["status"], "fail")
            seam = graph["seams"][0]
            self.assertAlmostEqual(seam["lateral_difference_m"], 0.5)
            check = next(item for item in audit["checks"] if item["id"] == "segment_seams")
            self.assertEqual(check["status"], "fail")

    def test_small_seam_is_reconciled_with_explicit_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, graph, audit = self._build(Path(temporary), second_center=0.1)
            self.assertEqual(result["status"], "pass")
            seam = graph["seams"][0]
            self.assertAlmostEqual(seam["lateral_difference_m"], 0.0)
            self.assertAlmostEqual(seam["reconciliation"]["raw_lateral_difference_m"], 0.1)
            self.assertAlmostEqual(seam["required_correction_m"], 0.05)
            self.assertTrue(seam["reconciliation"]["applied"])
            check = next(item for item in audit["checks"] if item["id"] == "segment_seams")
            self.assertEqual(check["status"], "pass")

    def test_overlapping_duplicate_tracks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, _, audit = self._build(Path(temporary), duplicate_centers=[0.0, 0.2])
            self.assertEqual(result["status"], "fail")
            check = next(item for item in audit["checks"] if item["id"] == "duplicate_tracks")
            self.assertEqual(check["status"], "fail")
            self.assertGreater(check["failure_count"], 0)

    def test_long_inferred_edge_cannot_become_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, graph, _ = self._build(Path(temporary))
            graph["edges"][0]["evidence_level"] = "rule_inferred"
            settings = load_json(project.resolve(project.value["algorithms"]["track_graph"]))
            audit = audit_track_graph(graph, settings)
            check = next(item for item in audit["checks"] if item["id"] == "inferred_spans")
            self.assertEqual(check["status"], "fail")

    def test_asymmetric_paired_support_fails_continuity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _ = self._project(root)
            reports = project.workspace_path("reports")
            first = self._report(reports / "a.json", "s0000_0050m", 25.0)
            second = self._report(reports / "b.json", "s0050_0100m", 75.0)
            for path in (first, second):
                value = load_json(path)
                value["rail_pairs"][0].update(
                    {
                        "joint_support_ratio": 0.08,
                        "asymmetric_support_ratio": 0.40,
                        "maximum_internal_joint_gap_m": 7.0,
                        "pair_continuity_status": "review_required",
                    }
                )
                write_json(path, value)

            settings_path = project.resolve(project.value["algorithms"]["track_graph"])
            settings = load_json(settings_path)
            settings["quality"]["paired_rail_continuity_gate_enabled"] = True
            write_json(settings_path, settings)

            result = build_track_graph(
                project,
                [("s0000_0050m", first), ("s0050_0100m", second)],
            )
            self.assertEqual(result["status"], "fail")
            audit = load_json(Path(result["output_audit_path"]))
            check = next(
                item for item in audit["checks"] if item["id"] == "paired_rail_continuity"
            )
            self.assertEqual(check["status"], "fail")
            self.assertEqual(check["failure_count"], 2)

    def test_paired_support_is_diagnostic_only_until_project_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _ = self._project(root)
            reports = project.workspace_path("reports")
            first = self._report(reports / "a.json", "s0000_0050m", 25.0)
            second = self._report(reports / "b.json", "s0050_0100m", 75.0)
            for path in (first, second):
                value = load_json(path)
                value["rail_pairs"][0].update(
                    {
                        "joint_support_ratio": 0.02,
                        "asymmetric_support_ratio": 0.60,
                        "maximum_internal_joint_gap_m": 12.0,
                    }
                )
                write_json(path, value)

            result = build_track_graph(
                project,
                [("s0000_0050m", first), ("s0050_0100m", second)],
            )
            audit = load_json(Path(result["output_audit_path"]))
            check = next(
                item for item in audit["checks"] if item["id"] == "paired_rail_continuity"
            )
            self.assertEqual(check["status"], "pass")
            self.assertEqual(check["enforcement"], "diagnostic_only")
            self.assertEqual(check["diagnostic_count"], 2)

    def test_fixed_center_recovery_satisfies_enabled_continuity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, _, graph, _ = self._build(Path(temporary))
            observation = graph["observations"][0]
            observation.update(
                {
                    "paired_support_evaluated": True,
                    "joint_support_ratio": 0.02,
                    "asymmetric_support_ratio": 0.60,
                    "maximum_internal_joint_gap_m": 12.0,
                    "pair_recovery_status": "fixed_center_evidence_recovered",
                }
            )
            settings = load_json(
                project.resolve(project.value["algorithms"]["track_graph"])
            )
            settings["quality"]["paired_rail_continuity_gate_enabled"] = True
            audit = audit_track_graph(graph, settings)
            check = next(
                item for item in audit["checks"] if item["id"] == "paired_rail_continuity"
            )
            self.assertEqual(check["status"], "pass")

    def test_unreviewed_geometry_never_reports_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, result, _, audit = self._build(Path(temporary), reviewed=False)
            self.assertEqual(result["status"], "review_required")
            self.assertFalse(audit["passed"])

    def test_empty_source_segment_writes_fail_closed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _ = self._project(root)
            reports = project.workspace_path("reports")
            first = self._report(reports / "a.json", "s0000_0050m", 25.0)
            second = self._report(reports / "b.json", "s0050_0100m", 75.0, centers=[])
            second_value = load_json(second)
            second_value["rail_pairs"] = []
            second_value["rail_lines"] = []
            second_value["peaks"] = []
            write_json(second, second_value)

            result = build_track_graph(
                project,
                [("s0000_0050m", first), ("s0050_0100m", second)],
            )
            self.assertEqual(result["status"], "fail")
            audit = load_json(Path(result["output_audit_path"]))
            check = next(
                item for item in audit["checks"] if item["id"] == "source_rail_pair_presence"
            )
            self.assertEqual(check["status"], "fail")
            self.assertEqual(check["failures"][0]["segment_id"], "s0050_0100m")

    def test_implausible_rail_top_crosslevel_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, _ = self._project(root)
            reports = project.workspace_path("reports")
            first = self._report(
                reports / "a.json",
                "s0000_0050m",
                25.0,
                rail_z_difference_m=0.35,
            )
            second = self._report(
                reports / "b.json",
                "s0050_0100m",
                75.0,
                rail_z_difference_m=0.35,
            )
            result = build_track_graph(
                project,
                [("s0000_0050m", first), ("s0050_0100m", second)],
            )
            self.assertEqual(result["status"], "fail")
            audit = load_json(Path(result["output_audit_path"]))
            check = next(item for item in audit["checks"] if item["id"] == "rail_top_crosslevel")
            self.assertEqual(check["status"], "fail")


if __name__ == "__main__":
    unittest.main()
