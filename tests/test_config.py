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

    def test_workspace_cannot_escape_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = initialize_project(root / "sample", "sample-project", "Sample")
            project = load_project(config_path)
            project.value["workspace"]["reports"] = "../outside"
            with self.assertRaises(ValueError):
                project.workspace_path("reports")
