import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railway_recon.io import load_json, sha256_file, sha256_json, write_json
from railway_recon.research_protocol import verify_research_baseline
from railway_recon.synthetic_track_adaptive_protocol import (
    ADAPTIVE_EXECUTION_FILES,
    ADAPTIVE_MODES,
    create_adaptive_protocol,
    freeze_adaptive_protocol,
)
from railway_recon.synthetic_track_guarded_protocol import (
    ADAPTATION_DISCLOSURE,
    GUARDED_DEPENDENCY_FILES,
    GUARDED_EXECUTION_FILES,
    GUARDED_MODES,
    METHOD_CONTRACT,
    REQUIRED_EXECUTION_FILES,
    create_guarded_protocol,
    freeze_guarded_protocol,
    validate_guarded_protocol,
    verify_guarded_freeze,
)
from railway_recon.synthetic_track_study_protocol import (
    create_study_protocol,
    freeze_study_protocol,
)


class GuardedProtocolTests(unittest.TestCase):
    def test_default_preserves_v1_without_generating_any_split(self) -> None:
        with patch(
            "railway_recon.synthetic_track_study_scene.generate_study_layout",
            side_effect=AssertionError("Protocol operations cannot generate study cases"),
        ):
            base = create_study_protocol()
            value = create_guarded_protocol()
            validate_guarded_protocol(value)
        self.assertEqual(value["schema_version"], "railway.synthetic-track-guarded-protocol.v1")
        self.assertEqual(value["method_revision"], "v3")
        self.assertEqual(value["base_protocol"], base)
        self.assertEqual(value["base_protocol_sha256"], sha256_json(base))
        self.assertEqual(value["modes"], [*ADAPTIVE_MODES, "direction_geometry", "guarded_evidence"])
        self.assertEqual(value["modes"], list(GUARDED_MODES))
        self.assertEqual(value["policy"], base["policy"])
        self.assertEqual(value["evaluation"], base["evaluation"])
        self.assertEqual(value["evaluation"]["primary_endpoint"],
                         "diagnosis_recall_including_abstentions")
        self.assertEqual(value["base_protocol"]["counts"],
                         {"development": 40, "validation": 20, "test": 60})
        self.assertEqual(len(value["base_protocol"]["conditions"]), 12)
        self.assertEqual(value, create_guarded_protocol())
        self.assertNotEqual(value["protocol_id"], create_adaptive_protocol(base)["protocol_id"])

    def test_supplied_base_has_no_shared_mutable_references(self) -> None:
        base = create_study_protocol(seed=42, counts={"development": 1, "test": 1})
        base["optional_registration_note"] = "Unit-test fixture metadata"
        original = copy.deepcopy(base)
        value = create_guarded_protocol(base)
        self.assertEqual(base, original)
        self.assertEqual(value["base_protocol"], original)
        value["base_protocol"]["groups"][0]["seed"] = 7
        value["policy"]["minimum_direction_cosine"] = 0.1
        value["method_contract"]["guard_scope"] = "not a supported claim"
        self.assertEqual(base, original)
        self.assertEqual(value["base_protocol"]["policy"], original["policy"])
        self.assertNotEqual(value["method_contract"], METHOD_CONTRACT)

    def test_payload_tampering_and_unknown_fields_are_rejected(self) -> None:
        original = create_guarded_protocol(create_study_protocol(counts={"development": 1}))
        for key, replacement in (
            ("schema_version", "v2"), ("method_revision", "v4"),
            ("base_protocol", {}), ("base_protocol_sha256", "0" * 64),
            ("protocol_id", "old-id"), ("modes", list(GUARDED_MODES[:-1])),
            ("policy", {}), ("evaluation", {}), ("adaptation_disclosure", {}),
            ("method_contract", {}),
        ):
            value = copy.deepcopy(original)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_guarded_protocol(value)
        for candidate in (None, [], "protocol"):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_guarded_protocol(candidate)
        value = copy.deepcopy(original)
        value["unregistered_claim"] = "blind field accuracy"
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_guarded_protocol(value)
        value = copy.deepcopy(original)
        value.pop("method_revision")
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_guarded_protocol(value)
        value = copy.deepcopy(original)
        value["base_protocol"]["groups"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_guarded_protocol(value)

    def test_no_new_thresholds_even_with_self_consistent_embedded_hash(self) -> None:
        original = create_guarded_protocol(create_study_protocol(counts={"development": 1}))
        value = copy.deepcopy(original)
        value["policy"]["minimum_direction_cosine"] -= 0.001
        with self.assertRaisesRegex(ValueError, "thresholds"):
            validate_guarded_protocol(value)
        base = copy.deepcopy(original["base_protocol"])
        base["policy"]["minimum_direction_cosine"] -= 0.001
        with self.assertRaisesRegex(ValueError, "original StudyAuditPolicy thresholds"):
            create_guarded_protocol(base)
        value = copy.deepcopy(original)
        value["base_protocol"] = base
        value["base_protocol_sha256"] = sha256_json(base)
        value["policy"] = copy.deepcopy(base["policy"])
        value["protocol_id"] = "p2-guarded-v3-" + sha256_json({
            key: item for key, item in value.items() if key != "protocol_id"
        })[:24]
        with self.assertRaisesRegex(ValueError, "original StudyAuditPolicy thresholds"):
            validate_guarded_protocol(value)

    def test_observed_data_and_limited_guard_disclosures_cannot_be_weakened(self) -> None:
        value = create_guarded_protocol(create_study_protocol(counts={"development": 1}))
        disclosure = value["adaptation_disclosure"]
        for field in (
            "v1_development_results_observed", "v1_validation_results_observed",
            "v2_development_results_observed", "v2_validation_results_observed",
            "legacy_stress_results_observed",
        ):
            self.assertIs(disclosure[field], True)
        self.assertEqual(disclosure["legacy_stress_case_count"], 18)
        self.assertIs(disclosure["development_replay_is_blind_test"], False)
        self.assertIs(disclosure["production_defaults_modified"], False)
        self.assertEqual(disclosure["test_result_use_for_this_revision"], "none")
        self.assertEqual(disclosure["confidence_semantics"],
                         "qualitative_not_calibrated_probability")
        self.assertEqual(value["method_contract"]["direction_threshold"],
                         "minimum_direction_cosine_from_embedded_policy")
        for container, declaration in (
            ("adaptation_disclosure", ADAPTATION_DISCLOSURE),
            ("method_contract", METHOD_CONTRACT),
        ):
            for key in declaration:
                changed = copy.deepcopy(value)
                changed[container][key] = "a different claim"
                with self.subTest(container=container, key=key), self.assertRaises(ValueError):
                    validate_guarded_protocol(changed)
        # Canonical hashing must distinguish numeric 1 from a required boolean.
        changed = copy.deepcopy(value)
        changed["adaptation_disclosure"]["v1_development_results_observed"] = 1
        with self.assertRaisesRegex(ValueError, "disclosures"):
            validate_guarded_protocol(changed)


class GuardedFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repository"
        for relative in REQUIRED_EXECUTION_FILES:
            self._fixture(relative)
        write_json(self.root / "src/railway_recon/resources/unit.schema.json", {"type": "object"})
        self.base = create_study_protocol(counts={"development": 1, "test": 1})
        self.protocol = self.root / "benchmarks/protocols/guarded.json"
        write_json(self.protocol, create_guarded_protocol(self.base))
        self.output = self.root / "research_freezes/guarded.json"

    def _fixture(self, relative: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic unit-test fixture only\n", encoding="utf-8")

    def _freeze(self) -> dict:
        freeze_guarded_protocol(self.protocol, self.output)
        return load_json(self.output)

    def _rehash(self, value: dict) -> None:
        value.pop("payload_sha256", None)
        value["payload_sha256"] = sha256_json(value)
        write_json(self.output, value)

    def test_complete_nonmutating_binding_and_no_generation(self) -> None:
        self._fixture("paper/customer-draft.md")
        before = {relative: sha256_file(self.root / relative)
                  for relative in REQUIRED_EXECUTION_FILES}
        with patch(
            "railway_recon.synthetic_track_study_scene.generate_study_layout",
            side_effect=AssertionError("Freezing must never generate any study case"),
        ):
            manifest = self._freeze()
            result = verify_guarded_freeze(self.protocol, self.output)
        paths = {record["path"] for record in manifest["files"]}
        self.assertTrue(set(REQUIRED_EXECUTION_FILES) <= paths)
        self.assertIn("benchmarks/protocols/guarded.json", paths)
        self.assertIn("src/railway_recon/resources/unit.schema.json", paths)
        self.assertNotIn("paper/customer-draft.md", paths)
        self.assertEqual(before, {relative: sha256_file(self.root / relative)
                                  for relative in REQUIRED_EXECUTION_FILES})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["repository_root"], str(self.root.resolve()))
        self.assertEqual(result["checked_files"], manifest["file_count"])
        with self.assertRaises(FileExistsError):
            freeze_guarded_protocol(self.protocol, self.output)

    def test_missing_new_executable_or_legacy_dependency_refuses_freeze(self) -> None:
        for relative in (*GUARDED_EXECUTION_FILES, *GUARDED_DEPENDENCY_FILES):
            (self.root / relative).unlink()
            with self.subTest(relative=relative), self.assertRaisesRegex(
                FileNotFoundError, "required guarded execution"
            ):
                self._freeze()
            self.assertFalse(self.output.exists())
            self._fixture(relative)

    def test_missing_required_bindings_fail_even_after_rehash(self) -> None:
        original = self._freeze()
        for relative in (
            *GUARDED_EXECUTION_FILES, *GUARDED_DEPENDENCY_FILES, *ADAPTIVE_EXECUTION_FILES,
            "benchmarks/protocols/guarded.json", "uv.lock",
            "src/railway_recon/synthetic_track_pilot.py",
            "src/railway_recon/resources/unit.schema.json",
        ):
            manifest = copy.deepcopy(original)
            manifest["files"] = [item for item in manifest["files"] if item["path"] != relative]
            manifest["file_count"] = len(manifest["files"])
            self._rehash(manifest)
            result = verify_guarded_freeze(self.protocol, self.output)
            with self.subTest(relative=relative):
                self.assertEqual(result["status"], "fail")
                self.assertTrue(any("Missing required hash binding" in i for i in result["issues"]))

    def test_changed_file_or_new_dependency_invalidates_binding(self) -> None:
        self._freeze()
        for relative in (
            "src/railway_recon/synthetic_track_guarded_audit.py", "uv.lock",
            "src/railway_recon/resources/unit.schema.json", "benchmarks/protocols/guarded.json",
        ):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(relative=relative):
                self.assertEqual(verify_guarded_freeze(self.protocol, self.output)["status"], "fail")
            path.write_bytes(original)
        self._fixture("src/railway_recon/new_transitive_dependency.py")
        self.assertEqual(verify_guarded_freeze(self.protocol, self.output)["status"], "fail")

    def test_v1_and_v2_content_integrity_does_not_authorize_v3(self) -> None:
        guarded_files = (*GUARDED_EXECUTION_FILES, *GUARDED_DEPENDENCY_FILES)
        for relative in (*guarded_files, *ADAPTIVE_EXECUTION_FILES):
            (self.root / relative).unlink()
        base_path = self.root / "benchmarks/protocols/base.json"
        write_json(base_path, self.base)
        v1_freeze = self.root / "research_freezes/base.json"
        freeze_study_protocol(base_path, v1_freeze)
        for relative in ADAPTIVE_EXECUTION_FILES:
            self._fixture(relative)
        adaptive_path = self.root / "benchmarks/protocols/adaptive.json"
        write_json(adaptive_path, create_adaptive_protocol(self.base))
        v2_freeze = self.root / "research_freezes/adaptive.json"
        freeze_adaptive_protocol(adaptive_path, v2_freeze)
        previous_hashes = {source: sha256_file(source) for source in (v1_freeze, v2_freeze)}
        for relative in guarded_files:
            self._fixture(relative)
        for source in (v1_freeze, v2_freeze):
            with self.subTest(source=source.name):
                self.assertEqual(verify_research_baseline(self.root, source)["status"], "pass")
                self.assertEqual(verify_guarded_freeze(self.protocol, source)["status"], "fail")
        self._freeze()
        self.assertEqual(verify_guarded_freeze(self.protocol, self.output)["status"], "pass")
        self.assertEqual(previous_hashes, {source: sha256_file(source)
                                           for source in previous_hashes})

    def test_malformed_manifests_and_bindings_fail_closed(self) -> None:
        original = self._freeze()
        for key, replacement in (
            ("files", []), ("files", None), ("files", [None]),
            ("file_count", 0), ("file_count", True), ("status", "draft"),
            ("freeze_id", self.base["protocol_id"]), ("schema_version", "other"),
        ):
            manifest = copy.deepcopy(original)
            manifest[key] = replacement
            self._rehash(manifest)
            with self.subTest(key=key, replacement=replacement):
                self.assertEqual(verify_guarded_freeze(self.protocol, self.output)["status"], "fail")
        for key, replacement in (
            ("path", "../outside.py"), ("path", "src/../pyproject.toml"),
            ("path", ""), ("bytes", -1), ("bytes", True), ("sha256", "x" * 64),
        ):
            manifest = copy.deepcopy(original)
            manifest["files"][0][key] = replacement
            self._rehash(manifest)
            with self.subTest(key=key, replacement=replacement):
                self.assertEqual(verify_guarded_freeze(self.protocol, self.output)["status"], "fail")
        manifest = copy.deepcopy(original)
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        manifest["file_count"] = len(manifest["files"])
        self._rehash(manifest)
        result = verify_guarded_freeze(self.protocol, self.output)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("Duplicate freeze binding" in item for item in result["issues"]))

    def test_out_of_scope_binding_fails_even_with_valid_content_hashes(self) -> None:
        self._fixture("paper/customer-draft.md")
        manifest = self._freeze()
        path = self.root / "paper/customer-draft.md"
        manifest["files"].append({
            "path": "paper/customer-draft.md", "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        manifest["file_count"] = len(manifest["files"])
        self._rehash(manifest)
        self.assertEqual(verify_research_baseline(self.root, self.output)["status"], "pass")
        result = verify_guarded_freeze(self.protocol, self.output)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("Out-of-scope guarded freeze binding" in item
                            for item in result["issues"]))

    def test_unsafe_output_and_outside_manifest_are_rejected(self) -> None:
        outside = self.root.parent / "outside.json"
        write_json(outside, {})
        self.assertEqual(verify_guarded_freeze(self.protocol, outside)["status"], "fail")
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            freeze_guarded_protocol(self.protocol, outside)
        for output in (self.protocol, self.root / "src/forbidden.json"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "inside source"):
                freeze_guarded_protocol(self.protocol, output)
        self.assertFalse((self.root / "src/forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
