import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project


class ConfigTests(unittest.TestCase):
    def test_initialize_and_load_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "sample", "sample-project", "Sample")
            project = load_project(config_path)
            self.assertEqual(project.project_id, "sample-project")
            self.assertEqual(
                project.input_path("point_cloud"),
                (root / "sample/input/pointcloud/site.laz").resolve(),
            )
            self.assertTrue((root / "sample/track_graph.json").is_file())
            self.assertTrue((root / "sample/vertical_hypotheses.json").is_file())
            self.assertTrue((root / "sample/canopy_structure.json").is_file())
            self.assertTrue((root / "sample/platform_mesh.json").is_file())
            self.assertTrue((root / "sample/platform_interface_audit.json").is_file())
            self.assertTrue((root / "sample/vertical_conflict_photo_evidence.json").is_file())

    def test_workspace_cannot_escape_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "sample", "sample-project", "Sample")
            project = load_project(config_path)
            project.value["workspace"]["reports"] = "../outside"
            with self.assertRaises(ValueError):
                project.workspace_path("reports")
