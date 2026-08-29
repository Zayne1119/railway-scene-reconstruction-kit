from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.gislab_baseline import (
    normalize_gislab_railtrack_points,
    prepare_gislab_railtrack_input,
    record_gislab_no_detection,
    record_gislab_timeout,
)
from railway_recon.io import load_json, write_json


class GislabBaselineTests(unittest.TestCase):
    def test_proven_timeout_becomes_explicit_non_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run.json"
            log = root / "combined.log"
            reference = root / "reference.json"
            output = root / "result.json"
            write_json(
                run,
                {
                    "schema_version": "railway.external-baseline-run.v1",
                    "method": "GISLab-ELTE/railroad:RailTrack",
                    "upstream_commit": "3557ecff1bd284108c7a833cce2b50ec9fcd5189",
                    "lastools_commit": "9bdc92c73047b46be25e5c2ed4abda2521e30fba",
                    "input": str(root / "input.laz"),
                    "input_sha256": "0" * 64,
                    "elapsed_seconds": 600.1,
                    "timeout_seconds": 600,
                    "exit_code": 124,
                },
            )
            log.write_text("still running", encoding="utf-8")
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0010m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 10.0],
                    "search_z_range_m": [0.0, 1.0],
                },
            )
            record_gislab_timeout(run, reference, "s0000_0010m", output)
            result = load_json(output)
            self.assertEqual(result["status"], "upstream_timeout_no_output")
            self.assertEqual(result["rail_lines"], [])

    def test_proven_empty_upstream_crash_becomes_zero_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run.json"
            log = root / "combined.log"
            reference = root / "reference.json"
            output = root / "result.json"
            write_json(
                run,
                {
                    "schema_version": "railway.external-baseline-run.v1",
                    "method": "GISLab-ELTE/railroad:RailTrack",
                    "upstream_commit": "3557ecff1bd284108c7a833cce2b50ec9fcd5189",
                    "lastools_commit": "9bdc92c73047b46be25e5c2ed4abda2521e30fba",
                    "input": str(root / "input.laz"),
                    "input_sha256": "0" * 64,
                    "exit_code": 139,
                },
            )
            log.write_text("Pairs: 0\nFiltered points: 100 (100 -> 0)\n", encoding="utf-8")
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0010m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 10.0],
                    "search_z_range_m": [0.0, 1.0],
                },
            )
            record_gislab_no_detection(run, reference, "s0000_0010m", output)
            result = load_json(output)
            self.assertEqual(result["rail_pair_count"], 0)
            self.assertEqual(result["rail_lines"], [])
            self.assertEqual(result["upstream_exit_code"], 139)
            self.assertEqual(result["zero_detection_proof"], "pair_filter")

    def test_zero_candidate_crash_becomes_zero_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run.json"
            log = root / "combined.log"
            reference = root / "reference.json"
            output = root / "result.json"
            write_json(
                run,
                {
                    "schema_version": "railway.external-baseline-run.v1",
                    "method": "GISLab-ELTE/railroad:RailTrack",
                    "upstream_commit": "3557ecff1bd284108c7a833cce2b50ec9fcd5189",
                    "lastools_commit": "9bdc92c73047b46be25e5c2ed4abda2521e30fba",
                    "input": str(root / "input.laz"),
                    "input_sha256": "0" * 64,
                    "exit_code": 134,
                },
            )
            log.write_text("railH95 size: 0\ncvHough size: 0\n", encoding="utf-8")
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0010m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 10.0],
                    "search_z_range_m": [0.0, 1.0],
                },
            )
            record_gislab_no_detection(run, reference, "s0000_0010m", output)
            result = load_json(output)
            self.assertEqual(result["rail_pair_count"], 0)
            self.assertEqual(result["zero_detection_proof"], "rail_candidate_extraction")

    def test_shared_preprocess_applies_only_frozen_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.las"
            reference = root / "reference.json"
            output = root / "prepared.laz"
            manifest = root / "prepared.json"
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            cloud.x = np.array([-1.0, 1.0, 5.0, 11.0, 5.0])
            cloud.y = np.zeros(5)
            cloud.z = np.array([0.2, 0.2, 0.2, 0.2, 2.0])
            cloud.write(source)
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0010m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 10.0],
                    "search_z_range_m": [0.0, 1.0],
                },
            )
            prepare_gislab_railtrack_input(
                source, reference, "s0000_0010m", output, manifest
            )
            prepared = laspy.read(output)
            report = load_json(manifest)
            self.assertEqual(len(prepared.points), 2)
            self.assertEqual(report["selected_point_count"], 2)
            self.assertFalse(report["manual_seed_used"])

    def test_corridor_local_preprocess_rotates_without_rail_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.las"
            reference = root / "reference.json"
            output = root / "prepared.laz"
            manifest = root / "prepared.json"
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            cloud.x = np.array([10.0, 10.0])
            cloud.y = np.array([21.0, 25.0])
            cloud.z = np.array([0.2, 0.2])
            cloud.write(source)
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0010m",
                    "frame": {
                        "origin_xy": [10.0, 20.0],
                        "along_xy": [0.0, 1.0],
                        "cross_xy": [-1.0, 0.0],
                    },
                    "longitudinal_range_m": [0.0, 10.0],
                    "search_z_range_m": [0.0, 1.0],
                },
            )
            prepare_gislab_railtrack_input(
                source,
                reference,
                "s0000_0010m",
                output,
                manifest,
                coordinate_space="corridor_local_xy",
            )
            prepared = laspy.read(output)
            report = load_json(manifest)
            np.testing.assert_allclose(np.asarray(prepared.x), [1.0, 5.0], atol=0.001)
            np.testing.assert_allclose(np.asarray(prepared.y), [0.0, 0.0], atol=0.001)
            self.assertEqual(report["coordinate_space"], "corridor_local_xy")

    def test_point_output_is_normalized_into_one_rail_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "RailTrack.las"
            reference = root / "reference.json"
            output = root / "result.json"
            longitudinal = np.arange(0.0, 20.01, 0.04)
            x = np.concatenate((longitudinal, longitudinal))
            y = np.concatenate(
                (
                    -0.754 + 0.001 * longitudinal,
                    0.754 + 0.001 * longitudinal,
                )
            )
            z = np.concatenate(
                (
                    0.20 + 0.0005 * longitudinal,
                    0.20 + 0.0005 * longitudinal,
                )
            )
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.001, 0.001, 0.001])
            cloud = laspy.LasData(header)
            cloud.x = x
            cloud.y = y
            cloud.z = z
            cloud.write(source)
            write_json(
                reference,
                {
                    "schema_version": "railway.rail-candidates.v1",
                    "project_id": "sample",
                    "segment_id": "s0000_0020m",
                    "frame": {
                        "origin_xy": [0.0, 0.0],
                        "along_xy": [1.0, 0.0],
                        "cross_xy": [0.0, 1.0],
                    },
                    "longitudinal_range_m": [0.0, 20.0],
                    "search_z_range_m": [0.0, 0.4],
                },
            )
            normalize_gislab_railtrack_points(
                source,
                reference,
                "s0000_0020m",
                output,
                minimum_cluster_length_m=10.0,
            )
            result = load_json(output)
            self.assertEqual(result["rail_pair_count"], 1)
            self.assertEqual(len(result["rail_lines"]), 2)
            self.assertAlmostEqual(result["rail_pairs"][0]["observed_gauge_m"], 1.435, places=2)


if __name__ == "__main__":
    unittest.main()
