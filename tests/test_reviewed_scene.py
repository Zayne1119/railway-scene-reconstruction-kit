import tempfile
import unittest
from pathlib import Path

from railway_recon.algorithms.reviewed_scene import build_reviewed_scene
from railway_recon.config import initialize_project, load_project
from railway_recon.mesh_audit import audit_obj
from railway_recon.registry import initialize_registry, validate_registry_file


class ReviewedSceneTests(unittest.TestCase):
    def test_build_synthetic_reviewed_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = load_project(
                initialize_project(root / "sample", "sample-project", "Sample")
            )
            initialize_registry(project)
            repository = Path(__file__).resolve().parents[1]
            layout = repository / "examples/minimal_project/reviewed_scene.example.json"
            report = build_reviewed_scene(project, layout)
            self.assertEqual(report["asset_count"], 5)
            self.assertTrue(audit_obj(Path(report["output_obj"]))["passed"])
            _, errors = validate_registry_file(project.workspace_path("asset_registry"))
            self.assertEqual(errors, [])

