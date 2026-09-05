import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railway_recon.io import load_json, sha256_file, sha256_json, write_json
from railway_recon.research_protocol import verify_research_baseline
from railway_recon.synthetic_track_adaptive_protocol import (
    ADAPTATION_DISCLOSURE,
    ADAPTIVE_EXECUTION_FILES,
    ADAPTIVE_MODES,
    REQUIRED_EXECUTION_FILES,
    create_adaptive_protocol,
    freeze_adaptive_protocol,
    validate_adaptive_protocol,
    verify_adaptive_freeze,
)
from railway_recon.synthetic_track_study_protocol import (
    create_study_protocol,
    freeze_study_protocol,
    verify_study_freeze,
)


class AdaptiveProtocolTests(unittest.TestCase):
    def test_default_embeds_exact_v1_groups_without_generating_cases(self) -> None:
        with patch(
            "railway_recon.synthetic_track_study_scene.generate_study_layout",
            side_effect=AssertionError("Protocol operations must never generate study cases"),
        ):
            base = create_study_protocol()
            value = create_adaptive_protocol()
            validate_adaptive_protocol(value)
        self.assertEqual(value["schema_version"], "railway.synthetic-track-adaptive-protocol.v1")
        self.assertEqual(value["base_protocol"], base)
        self.assertEqual(value["base_protocol_sha256"], sha256_json(base))
        self.assertNotEqual(value["protocol_id"], base["protocol_id"])
        self.assertEqual(value["modes"], list(ADAPTIVE_MODES))
        self.assertEqual(value["policy"], base["policy"])
        self.assertEqual(value["evaluation"], base["evaluation"])
        self.assertEqual(
            value["evaluation"]["primary_endpoint"], "diagnosis_recall_including_abstentions"
        )
        self.assertEqual(value["base_protocol"]["counts"], {
            "development": 40, "validation": 20, "test": 60,
        })
        self.assertEqual(len(value["base_protocol"]["conditions"]), 12)
        self.assertEqual(value, create_adaptive_protocol())

    def test_supplied_base_is_preserved_without_shared_mutable_references(self) -> None:
        base = create_study_protocol(seed=12345, counts={"development": 1, "test": 1})
        base["optional_registration_note"] = "Synthetic unit-test metadata"
        original = copy.deepcopy(base)
        value = create_adaptive_protocol(base)
        self.assertEqual(base, original)
        self.assertEqual(value["base_protocol"], original)
        value["base_protocol"]["groups"][0]["seed"] = 7
        value["policy"]["maximum_connection_gap_m"] = 9.0
        self.assertEqual(base, original)
        self.assertEqual(value["base_protocol"]["policy"], original["policy"])

    def test_base_and_policy_tampering_is_rejected(self) -> None:
        original = create_adaptive_protocol(create_study_protocol(counts={"development": 1}))
        for key, replacement in (
            ("schema_version", "v1"), ("base_protocol", {}),
            ("base_protocol_sha256", "0" * 64), ("protocol_id", "old-id"),
            ("modes", list(ADAPTIVE_MODES[:-1])), ("policy", {}),
            ("evaluation", {}), ("adaptation_disclosure", {}),
        ):
            value = copy.deepcopy(original)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_adaptive_protocol(value)
        value = copy.deepcopy(original)
        value["policy"]["maximum_connection_gap_m"] += 0.001
        with self.assertRaisesRegex(ValueError, "thresholds"):
            validate_adaptive_protocol(value)
        value = copy.deepcopy(original)
        value["base_protocol"]["groups"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_adaptive_protocol(value)

    def test_observed_validation_and_nonprobabilistic_fallback_are_explicit(self) -> None:
        value = create_adaptive_protocol(create_study_protocol(counts={"development": 1}))
        self.assertEqual(value["adaptation_disclosure"], ADAPTATION_DISCLOSURE)
        self.assertTrue(value["adaptation_disclosure"]["v1_development_results_observed"])
        self.assertTrue(value["adaptation_disclosure"]["v1_validation_results_observed"])
        self.assertEqual(value["adaptation_disclosure"]["confidence_semantics"],
                         "qualitative_not_calibrated_probability")
        self.assertEqual(value["adaptation_disclosure"]["fallback_semantics"],
                         "topology_inference_not_observation_verified")
        for key in ADAPTATION_DISCLOSURE:
            modified = copy.deepcopy(value)
            modified["adaptation_disclosure"][key] = "different claim"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "disclosures"):
                validate_adaptive_protocol(modified)


class AdaptiveFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repository"
        for relative in REQUIRED_EXECUTION_FILES:
            self._fixture(relative)
        write_json(self.root / "src/railway_recon/resources/unit.schema.json", {"type": "object"})
        self.base = create_study_protocol(counts={"development": 1, "test": 1})
        self.protocol = self.root / "benchmarks/protocols/adaptive.json"
        write_json(self.protocol, create_adaptive_protocol(self.base))
        self.output = self.root / "research_freezes/adaptive.json"

    def _fixture(self, relative: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic unit-test fixture only\n", encoding="utf-8")

    def _freeze(self) -> dict:
        freeze_adaptive_protocol(self.protocol, self.output)
        return load_json(self.output)

    def _rehash(self, value: dict) -> None:
        value.pop("payload_sha256", None)
        value["payload_sha256"] = sha256_json(value)
        write_json(self.output, value)

    def test_freeze_includes_all_execution_files_and_excludes_internal_drafts(self) -> None:
        self._fixture("paper/customer-draft.md")
        before = sha256_file(self.root / "src/railway_recon/synthetic_track_study.py")
        manifest = self._freeze()
        paths = {record["path"] for record in manifest["files"]}
        self.assertTrue(set(REQUIRED_EXECUTION_FILES) <= paths)
        self.assertIn("benchmarks/protocols/adaptive.json", paths)
        self.assertIn("src/railway_recon/resources/unit.schema.json", paths)
        self.assertNotIn("paper/customer-draft.md", paths)
        self.assertEqual(before, sha256_file(self.root / "src/railway_recon/synthetic_track_study.py"))
        self.assertEqual(verify_adaptive_freeze(self.protocol, self.output)["status"], "pass")
        with self.assertRaises(FileExistsError):
            freeze_adaptive_protocol(self.protocol, self.output)

    def test_missing_adaptive_executable_refuses_freeze(self) -> None:
        for relative in ADAPTIVE_EXECUTION_FILES:
            (self.root / relative).unlink()
            with self.subTest(relative=relative), self.assertRaisesRegex(
                FileNotFoundError, "required adaptive execution"
            ):
                self._freeze()
            self.assertFalse(self.output.exists())
            self._fixture(relative)

    def test_missing_hash_binding_cannot_authorize_new_method(self) -> None:
        original = self._freeze()
        for relative in (
            *ADAPTIVE_EXECUTION_FILES,
            "benchmarks/protocols/adaptive.json",
            "src/railway_recon/synthetic_track_pilot.py",
            "src/railway_recon/resources/unit.schema.json",
        ):
            manifest = copy.deepcopy(original)
            manifest["files"] = [item for item in manifest["files"] if item["path"] != relative]
            manifest["file_count"] = len(manifest["files"])
            self._rehash(manifest)
            result = verify_adaptive_freeze(self.protocol, self.output)
            with self.subTest(relative=relative):
                self.assertEqual(result["status"], "fail")
                self.assertTrue(any("Missing required hash binding" in i for i in result["issues"]))

    def test_changed_file_or_new_dependency_invalidates_binding(self) -> None:
        self._freeze()
        for relative in (
            "src/railway_recon/synthetic_track_adaptive_audit.py",
            "src/railway_recon/resources/unit.schema.json",
            "benchmarks/protocols/adaptive.json",
        ):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(relative=relative):
                self.assertEqual(verify_adaptive_freeze(self.protocol, self.output)["status"], "fail")
            path.write_bytes(original)
        self._fixture("src/railway_recon/new_dependency.py")
        self.assertEqual(verify_adaptive_freeze(self.protocol, self.output)["status"], "fail")

    def test_v1_content_integrity_is_not_v2_authorization(self) -> None:
        for relative in ADAPTIVE_EXECUTION_FILES:
            (self.root / relative).unlink()
        base_path = self.root / "benchmarks/protocols/base.json"
        write_json(base_path, self.base)
        old_freeze = self.root / "research_freezes/base.json"
        freeze_study_protocol(base_path, old_freeze)
        old_hash = sha256_file(old_freeze)
        for relative in ADAPTIVE_EXECUTION_FILES:
            self._fixture(relative)
        self.assertEqual(verify_research_baseline(self.root, old_freeze)["status"], "pass")
        self.assertEqual(verify_study_freeze(base_path, old_freeze)["status"], "fail")
        self.assertEqual(verify_adaptive_freeze(self.protocol, old_freeze)["status"], "fail")
        self._freeze()
        self.assertEqual(verify_adaptive_freeze(self.protocol, self.output)["status"], "pass")
        self.assertEqual(sha256_file(old_freeze), old_hash)

    def test_malformed_or_outside_freeze_fails_closed(self) -> None:
        original = self._freeze()
        for key, replacement in (
            ("files", []), ("files", [None]), ("file_count", 0),
            ("status", "draft"), ("freeze_id", self.base["protocol_id"]),
            ("schema_version", "other"),
        ):
            manifest = copy.deepcopy(original)
            manifest[key] = replacement
            self._rehash(manifest)
            with self.subTest(key=key):
                self.assertEqual(verify_adaptive_freeze(self.protocol, self.output)["status"], "fail")
        outside = self.root.parent / "outside.json"
        write_json(outside, {})
        self.assertEqual(verify_adaptive_freeze(self.protocol, outside)["status"], "fail")
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            freeze_adaptive_protocol(self.protocol, outside)


if __name__ == "__main__":
    unittest.main()
