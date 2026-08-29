import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.registry import freeze_release_registry, new_registry
from railway_recon.release import build_web_acceptance_config


class ReleaseBindingTests(unittest.TestCase):
    def _project_with_registry(self, root: Path, status: str = "accepted"):
        config_path = initialize_project(root / "sample", "sample-project", "Sample")
        project = load_project(config_path)
        registry = new_registry(project.project_id)
        registry["assets"] = [
            {
                "id": "TRACK-001",
                "type": "track",
                "status": status,
                "evidence_level": "observed",
                "confidence": 0.95,
                "sources": [{"kind": "point_cloud", "reference": "pilot.laz"}],
            }
        ]
        write_json(project.workspace_path("asset_registry"), registry)
        return project

    def test_freeze_registry_and_build_hash_bound_web_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project_with_registry(Path(temporary))
            registry_path = project.root / "workspace" / "exports" / "release-registry.json"
            model_path = project.root / "workspace" / "exports" / "scene.glb"
            config_path = project.root / "workspace" / "exports" / "project.json"
            model_path.write_bytes(b"glTF-test")

            freeze = freeze_release_registry(
                project, "release-001", registry_path
            )
            result = build_web_acceptance_config(
                project,
                "release-001",
                "Acceptance",
                model_path,
                registry_path,
                config_path,
                model_url="/models/scene.glb",
                registry_url="/data/registry.json",
            )

            frozen = load_json(registry_path)
            config = load_json(config_path)
            self.assertEqual(frozen["release_id"], "release-001")
            self.assertEqual(freeze["asset_set_sha256"], config["asset_set_sha256"])
            self.assertTrue(result["acceptance_mode"])
            self.assertEqual(config["release_id"], "release-001")

    def test_candidate_registry_cannot_be_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project_with_registry(Path(temporary), status="candidate")
            output = project.root / "workspace" / "exports" / "release-registry.json"

            with self.assertRaisesRegex(ValueError, "non-accepted"):
                freeze_release_registry(project, "release-002", output)


if __name__ == "__main__":
    unittest.main()
