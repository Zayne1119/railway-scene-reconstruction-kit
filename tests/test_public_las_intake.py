from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np
from jsonschema import validate

from railway_recon.public_las_intake import PUBLIC_LAS_INTAKE_SCHEMA, inspect_public_las


class PublicLasIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="railway-public-intake-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def make_cloud(self, filename: str = "fixture.las", *, point_format: int = 3) -> Path:
        path = self.directory / filename
        cloud = laspy.create(
            point_format=point_format, file_version="1.4" if point_format == 6 else "1.2"
        )
        cloud.header.scales = np.array([0.01, 0.02, 0.05])
        cloud.header.offsets = np.array([100.0, 200.0, 10.0])
        cloud.x = [100, 101, 102, 103, 104]
        cloud.y = [200, 202, 204, 206, 208]
        cloud.z = [10, 11, 12, 13, 14]
        cloud.classification = [0, 1, 1, 7, 9]
        cloud.point_source_id = [0, 2, 2, 65535, 0]
        cloud.write(path)
        return path

    def digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def inspect(self, path: Path, output_name: str = "report.json", **kwargs) -> dict:
        return inspect_public_las(path, self.digest(path), self.directory / output_name, **kwargs)

    def test_chunked_actual_fields_counts_bounds_and_schema(self) -> None:
        path = self.make_cloud()
        digest = self.digest(path)
        report = self.inspect(path, chunk_size=2)
        validate(report, PUBLIC_LAS_INTAKE_SCHEMA)
        self.assertEqual(report["header"]["las_version"], "1.2")
        self.assertEqual(report["header"]["point_format_id"], 3)
        self.assertEqual(report["header"]["scales"], [0.01, 0.02, 0.05])
        self.assertEqual(report["reader"]["chunks_decoded"], 3)
        self.assertEqual(report["decoded"]["point_count"], 5)
        self.assertEqual(report["decoded"]["minimum"], [100.0, 200.0, 10.0])
        self.assertEqual(report["decoded"]["maximum"], [104.0, 208.0, 14.0])
        self.assertTrue(report["decoded"]["header_bounds_match"])
        self.assertEqual(
            report["decoded"]["classification"]["counts"], {"0": 1, "1": 2, "7": 1, "9": 1}
        )
        self.assertEqual(
            report["decoded"]["point_source_id"]["counts"], {"0": 2, "2": 2, "65535": 1}
        )
        self.assertEqual(report["decoded"]["point_source_id"]["unique_values_including_zero"], 3)
        self.assertIn("red", {item["name"] for item in report["header"]["dimensions"]})
        self.assertEqual(self.digest(path), digest)
        saved = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report, saved)
        self.assertFalse(report["model_evaluation_performed"])
        self.assertEqual(
            report["field_interface"]["relation_ground_truth"], "not_provided_or_inferred"
        )
        self.assertEqual(
            report["field_interface"]["coordinate_units"], "not_asserted_by_this_intake"
        )

    def test_bad_hash_rejects_before_decode_and_creates_no_report(self) -> None:
        path = self.make_cloud()
        with patch("railway_recon.public_las_intake.laspy.open") as open_las:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                inspect_public_las(path, "0" * 64, self.directory / "bad.json")
            open_las.assert_not_called()
        self.assertFalse((self.directory / "bad.json").exists())

    def test_point_limit_rejects_full_file_instead_of_silently_sampling(self) -> None:
        path = self.make_cloud()
        with self.assertRaisesRegex(ValueError, "exceeds max_points"):
            self.inspect(path, max_points=4)
        self.assertFalse((self.directory / "report.json").exists())

    def test_existing_output_is_preserved(self) -> None:
        path = self.make_cloud()
        output = self.directory / "existing.json"
        output.write_text("existing contents", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.inspect(path, output.name)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing contents")

    def test_nonfinite_gps_dimension_is_rejected(self) -> None:
        path = self.make_cloud()
        cloud = laspy.read(path)
        cloud.gps_time[2] = np.nan
        cloud.write(path)
        with self.assertRaisesRegex(ValueError, "Non-finite floating dimension gps_time"):
            self.inspect(path)
        self.assertFalse((self.directory / "report.json").exists())

    def test_nonfinite_header_scale_is_rejected(self) -> None:
        path = self.make_cloud()
        with path.open("r+b") as stream:
            # LAS 1.2: X scale is a float64 at byte offset 131.
            stream.seek(131)
            stream.write(np.asarray([np.inf], dtype="<f8").tobytes())
        with self.assertRaisesRegex(ValueError, "scales.*finite"):
            self.inspect(path)
        self.assertFalse((self.directory / "report.json").exists())

    def test_actual_new_point_format_and_extra_dimensions_are_preserved(self) -> None:
        path = self.make_cloud(point_format=6)
        cloud = laspy.read(path)
        cloud.add_extra_dim(laspy.ExtraBytesParams(name="test_confidence", type=np.float32))
        cloud.test_confidence = [0.1, 0.2, 0.3, 0.4, 0.5]
        cloud.write(path)
        report = self.inspect(path)
        self.assertEqual(report["header"]["las_version"], "1.4")
        self.assertEqual(report["header"]["point_format_id"], 6)
        dimensions = {item["name"]: item for item in report["header"]["dimensions"]}
        self.assertNotIn("red", dimensions)
        self.assertTrue(dimensions["test_confidence"]["extra_dimension"])
        self.assertEqual(dimensions["test_confidence"]["dtype"], "float32")

    @unittest.skipUnless(laspy.LazBackend.detect_available(), "LAZ backend not installed")
    def test_laz_decodes_same_statistics_as_las(self) -> None:
        las_path = self.make_cloud()
        laz_path = self.directory / "fixture.laz"
        laspy.read(las_path).write(laz_path)
        first = self.inspect(las_path, "las.json", chunk_size=2)
        second = self.inspect(laz_path, "laz.json", chunk_size=3)
        self.assertEqual(first["decoded"], second["decoded"])
        self.assertFalse(first["header"]["compressed"])
        self.assertTrue(second["header"]["compressed"])

    def test_empty_cloud_reports_null_bounds_without_infinite_json(self) -> None:
        path = self.directory / "empty.las"
        laspy.create(point_format=3, file_version="1.2").write(path)
        report = self.inspect(path)
        self.assertEqual(report["decoded"]["point_count"], 0)
        self.assertIsNone(report["decoded"]["minimum"])
        self.assertIsNone(report["decoded"]["maximum"])
        self.assertIsNone(report["decoded"]["header_bounds_match"])
        self.assertEqual(report["decoded"]["classification"]["counts"], {})

    def test_invalid_limits_and_hash_shapes_are_rejected(self) -> None:
        path = self.make_cloud()
        for kwargs in (
            {"max_points": 0},
            {"chunk_size": 0},
            {"chunk_size": 1_000_001},
            {"max_points": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.inspect(path, **kwargs)
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            inspect_public_las(path, "missing", self.directory / "bad.json")


if __name__ == "__main__":
    unittest.main()
