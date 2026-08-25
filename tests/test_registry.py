import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.registry import (
    import_registry_records,
    initialize_registry,
    new_registry,
    validate_registry_file,
    validate_registry_value,
)


class RegistryTests(unittest.TestCase):
    def test_empty_registry_is_valid(self) -> None:
        self.assertEqual(validate_registry_value(new_registry("sample-project")), [])

    def test_duplicate_asset_id_is_rejected(self) -> None:
        registry = new_registry("sample-project")
        asset = {
            "id": "TRACK-0001",
            "type": "track",
            "status": "accepted",
            "evidence_level": "observed",
            "confidence": 0.9,
            "sources": [{"kind": "point_cloud", "reference": "synthetic"}],
        }
        registry["assets"] = [asset, dict(asset)]
        errors = validate_registry_value(registry)
        self.assertTrue(any("duplicate asset ids" in error for error in errors))

    def test_import_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "sample", "sample-project", "Sample")
            project = load_project(config_path)
            initialize_registry(project)
            source = root / "review.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "id", "type", "status", "evidence_level", "confidence",
                        "source_kind", "source_reference",
                    ]
                )
                writer.writerow(
                    ["MAST-0001", "catenary_mast", "reviewed", "observed", "0.9", "point_cloud", "synthetic"]
                )
            report = import_registry_records(project, source)
            self.assertEqual(report["imported_asset_count"], 1)
            _, errors = validate_registry_file(project.workspace_path("asset_registry"))
            self.assertEqual(errors, [])
