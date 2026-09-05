"""Offline synthetic onboarding tests; no public/customer data or research cases."""
from __future__ import annotations

import copy
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from railway_recon import onboarding_demo as demo
from railway_recon.registry import validate_registry_value
from railway_recon.web_glb_export import inspect_web_glb


class OnboardingDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="railway-onboarding-fixture-")
        cls.parent = Path(cls.temporary.name)
        cls.project = cls.parent / "project"
        cls.result = demo.build_onboarding_demo(cls.project)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway-onboarding-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def copy_project(self) -> Path:
        destination = self.root / "copied-demo"
        shutil.copytree(self.project, destination)
        return destination

    def test_real_extraction_mesh_conversion_and_registry_are_nonempty(self):
        self.assertEqual(self.result["status"], "built_synthetic_demo")
        self.assertGreater(self.result["point_count"], 2000)
        self.assertGreater(self.result["triangle_count"], 100)
        inspection = inspect_web_glb(self.project / "viewer/scene.glb")
        self.assertEqual(inspection["triangle_count"], self.result["triangle_count"])
        registry = json.loads((self.project / "viewer/asset_registry.json").read_bytes())
        self.assertEqual(validate_registry_value(registry), [])
        self.assertEqual({asset["geometry"]["node"] for asset in registry["assets"]}, set(inspection["node_names"]))
        self.assertEqual(len(registry["assets"]), inspection["node_count"])
        self.assertTrue({"rail", "sleeper", "track_bed", "platform", "canopy_column", "canopy_beam", "canopy_roof"}
                        <= {asset["type"] for asset in registry["assets"]})
        self.assertTrue(all(asset["status"] == "candidate" for asset in registry["assets"]))
        state = json.loads((self.project / "run_state.json").read_bytes())
        self.assertEqual([item["stage"] for item in state["events"]], [
            "initialize", "generate_independent_synthetic_inputs", "segment_and_extract_synthetic_track",
            "build_existing_parametric_track", "build_authored_synthetic_station_layout", "assemble_audit_and_export_glb"])
        self.assertEqual(state["events"][2]["rail_pair_count"], 1)

    def test_glb_positions_are_finite_and_have_actual_nondegenerate_triangles(self):
        payload = (self.project / "viewer/scene.glb").read_bytes()
        json_length = struct.unpack_from("<I", payload, 12)[0]
        document = json.loads(payload[20 : 20 + json_length])
        binary_start = 20 + json_length + 8
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                accessor = document["accessors"][primitive["attributes"]["POSITION"]]
                view = document["bufferViews"][accessor["bufferView"]]
                values = np.frombuffer(payload, dtype="<f4", count=accessor["count"] * 3,
                                       offset=binary_start + view["byteOffset"]).reshape(-1, 3, 3)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.all(np.linalg.norm(np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]), axis=1) > 0))

    def test_manifest_binds_inputs_settings_sources_and_all_outputs(self):
        manifest = demo.verify_onboarding_demo(self.project)
        self.assertFalse(manifest["customer_data_used"])
        self.assertFalse(manifest["network_used"])
        self.assertFalse(manifest["real_station_extraction_demonstrated"])
        self.assertFalse(manifest["survey_accuracy_claimed"])
        self.assertEqual(manifest["settings"], demo.DEMO_SETTINGS)
        for name in ("src/railway_recon/algorithms/rail_candidates.py", "src/railway_recon/algorithms/track.py",
                     "src/railway_recon/algorithms/reviewed_scene.py", "src/railway_recon/web_glb_export.py",
                     "src/railway_recon/resources/project.template.json", "scripts/build_onboarding_demo.py"):
            self.assertIn(name, manifest["source_sha256"])
        cloud = laspy.read(self.project / "input/pointcloud/site.laz")
        self.assertTrue(np.all(np.asarray(cloud.classification) == 0))
        self.assertEqual(len(cloud.points), manifest["input"]["point_count"])
        self.assertEqual(manifest["input"]["sha256"], demo._sha(self.project / "input/pointcloud/site.laz"))

    def test_new_runs_have_identical_synthetic_cloud_geometry_and_viewer_registry(self):
        second = self.root / "second"
        result = demo.build_onboarding_demo(second)
        self.assertEqual(result["model_sha256"], self.result["model_sha256"])
        for relative in ("input/pointcloud/site.laz", "viewer/scene.glb", "viewer/asset_registry.json",
                         "viewer/project.json", "viewer/mesh_audit.json", "authored_station_layout.json"):
            self.assertEqual((second / relative).read_bytes(), (self.project / relative).read_bytes(), relative)

    def test_verified_reuse_does_not_reexecute_pipeline(self):
        with patch.object(demo, "detect_rail_candidates") as extractor, patch.object(demo, "initialize_project") as initialize:
            result = demo.build_onboarding_demo(self.project)
        self.assertEqual(result["status"], "verified_existing")
        extractor.assert_not_called()
        initialize.assert_not_called()

    def test_tamper_or_unowned_file_refuses_reuse_without_overwrite(self):
        project = self.copy_project()
        path = project / "viewer/scene.glb"
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            demo.build_onboarding_demo(project)
        self.assertEqual(path.read_bytes(), original + b"tamper")
        path.write_bytes(original)
        (project / "owner-note.txt").write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unowned"):
            demo.build_onboarding_demo(project)
        self.assertEqual((project / "owner-note.txt").read_text(), "keep me")

    def test_changed_settings_or_source_refuses_reuse(self):
        settings = {**demo.DEMO_SETTINGS, "seed": 100}
        with patch.object(demo, "DEMO_SETTINGS", settings), self.assertRaisesRegex(ValueError, "settings changed"):
            demo.verify_onboarding_demo(self.project)
        with patch.object(demo, "_source_bindings", return_value={}), self.assertRaisesRegex(ValueError, "source or dependency"):
            demo.verify_onboarding_demo(self.project)

    def test_path_traversal_overlap_and_existing_unowned_outputs_are_rejected(self):
        for path in (self.root / "new/../escape", demo.REPOSITORY_ROOT, self.root / ".git/demo"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                demo.build_onboarding_demo(path)
        with self.assertRaisesRegex(ValueError, "overlap"):
            demo.build_onboarding_demo(self.root / "project", self.root / "project/web")
        occupied = self.root / "owner"
        occupied.mkdir()
        (occupied / "keep.txt").write_text("owner", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            demo.build_onboarding_demo(occupied)
        with self.assertRaises(FileExistsError):
            demo.build_onboarding_demo(self.root / "new", occupied)
        self.assertFalse((self.root / "new").exists())
        self.assertEqual((occupied / "keep.txt").read_text(), "owner")

    def test_manifest_traversal_is_rejected_before_accessing_external_files(self):
        project = self.copy_project()
        manifest_path = project / "demo_manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"][0]["path"] = "../unowned-secret.txt"
        manifest_path.write_bytes(demo._json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "traversal"):
            demo.verify_onboarding_demo(project)

    def test_publish_only_same_origin_synthetic_payloads_and_verified_reuse(self):
        target = self.root / "demo"
        result = demo.publish_onboarding_demo(self.project, target)
        self.assertEqual(result["status"], "written_local_synthetic_viewer_bundle")
        self.assertEqual({path.name for path in target.iterdir()}, set(demo.WEB_FILES) | {"demo_manifest.json"})
        config = json.loads((target / "project.json").read_bytes())
        self.assertTrue(config["synthetic_demo"])
        self.assertIn("SYNTHETIC", config["title"])
        urls = [value for key, value in config.items() if key.endswith("_url")]
        urls += [source["url"] for source in config["registry_sources"]]
        self.assertTrue(all(url.startswith("/demo/") and ".." not in url and ":" not in url for url in urls))
        for path in target.glob("*.json"):
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.project), payload)
            self.assertNotIn("/@fs/", payload)
        again = demo.publish_onboarding_demo(self.project, target)
        self.assertEqual(again["status"], "verified_existing")
        registry = target / "asset_registry.json"
        registry.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            demo.publish_onboarding_demo(self.project, target)

    def test_failure_records_stage_and_never_writes_completed_manifest(self):
        output = self.root / "failed"
        with patch.object(demo, "export_obj_to_web_glb", side_effect=ValueError("fixture failure")), \
             self.assertRaisesRegex(ValueError, "fixture failure"):
            demo.build_onboarding_demo(output)
        failure = json.loads((output / "run_failure.json").read_bytes())
        self.assertEqual(failure["stage"], "assemble_audit_and_export_glb")
        self.assertFalse((output / "demo_manifest.json").exists())
        self.assertTrue((output / "input/pointcloud/site.laz").is_file())

    def test_authored_station_is_never_mislabeled_as_observed_point_truth(self):
        layout = demo._authored_station_layout()
        self.assertEqual(layout["layout_role"], "authored_not_automatically_extracted")
        self.assertTrue(all(asset["evidence_level"] == "rule_inferred" for asset in layout["assets"]))
        self.assertTrue(all(asset["parameters"]["extracted_from_points"] is False for asset in layout["assets"]))
        self.assertEqual(layout, copy.deepcopy(layout))


if __name__ == "__main__":
    unittest.main()
