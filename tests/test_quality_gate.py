import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, sha256_file, write_json
from railway_recon.quality_gate import evaluate_quality_gate, validate_gate_result
from railway_recon.registry import freeze_release_registry, new_registry
from railway_recon.release import build_web_acceptance_config


class QualityGateTests(unittest.TestCase):
    def _project(self, root: Path):
        config_path = initialize_project(root / "sample", "sample-project", "Sample")
        return load_project(config_path)

    def _accepted_registry(self, project, path: Path, release_id: str) -> None:
        registry = new_registry(project.project_id)
        registry["release_id"] = release_id
        registry["assets"] = [
            {
                "id": "TRACK-001",
                "type": "track",
                "status": "accepted",
                "evidence_level": "observed",
                "confidence": 0.95,
                "sources": [{"kind": "point_cloud", "reference": "pilot.laz"}],
            }
        ]
        write_json(path, registry)

    def test_local_pass_cannot_hide_global_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            reports = project.root / "workspace" / "reports"
            local = reports / "local.json"
            global_audit = reports / "global.json"
            artifact = project.root / "workspace" / "exports" / "scene.glb"
            registry = project.root / "workspace" / "registry" / "release.json"
            artifact.write_bytes(b"glTF-test")
            write_json(local, {"status": "pass"})
            write_json(global_audit, {"status": "fail"})
            self._accepted_registry(project, registry, "release-001")

            result_path, result = evaluate_quality_gate(
                project,
                "release-001",
                "QG7",
                [("track.local", local), ("track.full_length", global_audit)],
                [("model", artifact)],
                registry=registry,
            )

            self.assertEqual(result["status"], "FAIL")
            self.assertIn("track.full_length", result["failed_check_ids"])
            self.assertTrue(validate_gate_result(result_path)["valid"])

    def test_historical_failure_statuses_never_collapse_to_pass(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures" / "quality_gate"
        blocking = [
            "track_full_length_fail.json",
            "canopy_interfaces_fail.json",
            "mesh_requires_optimization.json",
            "segment_conditional.json",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            reports = project.root / "workspace" / "reports"
            local = reports / "release_local_pass.json"
            local.write_bytes((fixture_root / "release_local_pass.json").read_bytes())
            for index, name in enumerate(blocking, start=1):
                defect = reports / name
                defect.write_bytes((fixture_root / name).read_bytes())
                _, result = evaluate_quality_gate(
                    project,
                    f"regression-{index:03d}",
                    "QG0",
                    [("release.local", local), ("independent.audit", defect)],
                    [],
                )
                self.assertEqual(result["status"], "FAIL", name)
                self.assertIn("independent.audit", result["failed_check_ids"])

    def test_candidate_and_conditional_states_are_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            report = project.root / "workspace" / "reports" / "candidate.json"
            artifact = project.root / "workspace" / "exports" / "scene.glb"
            registry_path = project.root / "workspace" / "registry" / "release.json"
            artifact.write_bytes(b"glTF-test")
            write_json(report, {"status": "conditional_pass"})
            registry = new_registry(project.project_id)
            registry["release_id"] = "release-002"
            registry["assets"] = [
                {
                    "id": "TRACK-001",
                    "type": "track",
                    "status": "candidate",
                    "evidence_level": "observed",
                    "confidence": 0.8,
                    "sources": [{"kind": "point_cloud", "reference": "pilot.laz"}],
                }
            ]
            write_json(registry_path, registry)

            _, result = evaluate_quality_gate(
                project,
                "release-002",
                "QG7",
                [("track.segment", report)],
                [("model", artifact)],
                registry=registry_path,
            )

            self.assertEqual(result["status"], "FAIL")
            self.assertIn("track.segment", result["failed_check_ids"])
            self.assertIn("registry.clean_states", result["failed_check_ids"])

    def test_waiver_is_explicit_and_bound_to_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            report = project.root / "workspace" / "reports" / "runtime.json"
            artifact = project.root / "workspace" / "exports" / "scene.glb"
            waiver = project.root / "workspace" / "reports" / "waiver.json"
            artifact.write_bytes(b"glTF-test")
            write_json(report, {"status": "review_required"})
            write_json(
                waiver,
                {
                    "schema_version": "railway.quality-waiver.v2",
                    "waiver_id": "WAIVER-001",
                    "check_id": "visual.runtime",
                    "artifact_sha256": [sha256_file(artifact)],
                    "reason": "Target UE workstation is temporarily unavailable",
                    "risk": "Temporal-AA flicker remains unverified",
                    "mitigation": "Keep release internal and run UE review before publication",
                    "approved_by": ["project-owner", "independent-qa"],
                    "expires_at": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
                },
            )

            _, result = evaluate_quality_gate(
                project,
                "release-003",
                "QG0",
                [("visual.runtime", report)],
                [("model", artifact)],
                waiver_sources=[waiver],
                allow_waiver={"visual.runtime"},
            )

            self.assertEqual(result["status"], "PASS_WITH_WAIVER")
            self.assertEqual(result["waived_check_ids"], ["visual.runtime"])

    def test_hash_validation_detects_changed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            report = project.root / "workspace" / "reports" / "audit.json"
            artifact = project.root / "workspace" / "exports" / "scene.glb"
            artifact.write_bytes(b"glTF-test")
            write_json(report, {"status": "pass"})
            result_path, result = evaluate_quality_gate(
                project,
                "release-004",
                "QG0",
                [("mesh.audit", report)],
                [("model", artifact)],
            )
            self.assertEqual(result["status"], "PASS")
            artifact.write_bytes(b"tampered")

            validation = validate_gate_result(result_path)
            self.assertFalse(validation["valid"])
            self.assertTrue(any("artifact" in error for error in validation["errors"]))

    def test_previous_gate_hash_chain_and_immutable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            report = project.root / "workspace" / "reports" / "audit.json"
            write_json(report, {"passed": True})
            previous_path, previous = evaluate_quality_gate(
                project,
                "release-005",
                "QG0",
                [("pilot.audit", report)],
                [],
            )
            self.assertEqual(previous["status"], "PASS")
            _, current = evaluate_quality_gate(
                project,
                "release-005",
                "QG1",
                [("production.audit", report)],
                [],
                previous_gate=previous_path,
            )
            self.assertEqual(current["status"], "PASS")

            with self.assertRaises(FileExistsError):
                evaluate_quality_gate(
                    project,
                    "release-005",
                    "QG0",
                    [("pilot.audit", report)],
                    [],
                )

            saved = load_json(previous_path)
            saved["status"] = "FAIL"
            write_json(previous_path, saved)
            current_path = (
                project.root
                / "workspace"
                / "gates"
                / "release-005"
                / "QG1"
                / "gate-result.json"
            )
            validation = validate_gate_result(current_path)
            self.assertFalse(validation["valid"])
            self.assertIn("previous gate result hash changed", validation["errors"])

    def test_release_and_web_gates_bind_exact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = self._project(Path(temporary))
            report = project.root / "workspace" / "reports" / "pass.json"
            write_json(report, {"status": "pass"})
            self._accepted_registry(
                project,
                project.workspace_path("asset_registry"),
                "working-registry",
            )
            release_root = project.root / "workspace" / "exports" / "release-006"
            model = release_root / "scene.glb"
            registry = release_root / "asset_registry.json"
            web_config = release_root / "project.json"
            release_root.mkdir(parents=True)
            model.write_bytes(b"glTF-test")
            freeze_release_registry(project, "release-006", registry)
            build_web_acceptance_config(
                project,
                "release-006",
                "Acceptance",
                model,
                registry,
                web_config,
                model_url="/models/scene.glb",
                registry_url="/data/asset_registry.json",
            )

            previous = None
            for index in range(7):
                previous, result = evaluate_quality_gate(
                    project,
                    "release-006",
                    f"QG{index}",
                    [("stage.audit", report)],
                    [],
                    previous_gate=previous,
                )
                self.assertEqual(result["status"], "PASS")
            previous, qg7 = evaluate_quality_gate(
                project,
                "release-006",
                "QG7",
                [("release.audit", report)],
                [("model", model), ("registry", registry)],
                previous_gate=previous,
                registry=registry,
            )
            self.assertEqual(qg7["status"], "PASS")
            qg8_path, qg8 = evaluate_quality_gate(
                project,
                "release-006",
                "QG8",
                [("web.audit", report)],
                [("model", model), ("registry", registry), ("web_config", web_config)],
                previous_gate=previous,
                registry=registry,
            )
            self.assertEqual(qg8["status"], "PASS")
            self.assertTrue(validate_gate_result(qg8_path)["valid"])


if __name__ == "__main__":
    unittest.main()
