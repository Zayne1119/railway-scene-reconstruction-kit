import copy
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from railway_recon.io import load_json, sha256_file, sha256_json, write_json
from railway_recon.synthetic_track_study_audit import STUDY_MODES, StudyAuditPolicy
from railway_recon.synthetic_track_study_protocol import (
    DEFAULT_COUNTS,
    REQUIRED_EXECUTION_FILES,
    create_study_protocol,
    freeze_study_protocol,
    validate_study_protocol,
    verify_study_freeze,
)
from railway_recon.synthetic_track_study_scene import STUDY_VARIANTS


class StudyProtocolTests(unittest.TestCase):
    def test_default_groups_are_deterministic_disjoint_and_do_not_generate_cases(self) -> None:
        with patch(
            "railway_recon.synthetic_track_study_scene.generate_study_layout",
            side_effect=AssertionError("Protocol creation must never generate cases"),
        ):
            first = create_study_protocol()
            self.assertEqual(first, create_study_protocol())
            validate_study_protocol(first)
        self.assertEqual(first["schema_version"], "railway.synthetic-track-study-protocol.v1")
        self.assertEqual(first["counts"], DEFAULT_COUNTS)
        self.assertEqual(first["modes"], list(STUDY_MODES))
        self.assertEqual(first["policy"], asdict(StudyAuditPolicy()))
        self.assertEqual(first["conditions"], list(STUDY_VARIANTS))
        self.assertEqual(Counter(group["split"] for group in first["groups"]), DEFAULT_COUNTS)
        self.assertEqual(len({group["seed"] for group in first["groups"]}), 120)
        self.assertEqual(len({group["group_id"] for group in first["groups"]}), 120)
        self.assertTrue(all(0 <= group["family_index"] <= 5 for group in first["groups"]))
        self.assertTrue(all(len(group["group_id"]) == 32 for group in first["groups"]))
        self.assertNotEqual(first["groups"], create_study_protocol(seed=20260907)["groups"])

    def test_split_seeds_are_stable_when_development_count_changes(self) -> None:
        first = create_study_protocol(counts={"development": 2, "validation": 1, "test": 3})
        second = create_study_protocol(counts={"development": 4, "validation": 1, "test": 3})
        self.assertEqual(
            [group for group in first["groups"] if group["split"] == "test"],
            [group for group in second["groups"] if group["split"] == "test"],
        )

    def test_nonempty_positive_split_counts_are_required(self) -> None:
        for counts in ({}, {"train": 1}, {"test": 0}, {"test": -1}, {"test": True}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                create_study_protocol(counts=counts)
        with self.assertRaises(ValueError):
            create_study_protocol(seed=True)
        small = create_study_protocol(counts={"development": 1})
        self.assertEqual(len(small["groups"]), 1)
        validate_study_protocol(small)

    def test_all_group_records_are_checked(self) -> None:
        original = create_study_protocol(counts={"development": 1, "validation": 1, "test": 1})
        mutations = (
            ("seed", original["groups"][0]["seed"]),
            ("group_id", original["groups"][0]["group_id"]),
            ("seed", True), ("seed", "123"), ("seed", -1),
            ("family_index", 6), ("family_index", True), ("family_index", -1),
            ("split", "training"), ("group_id", ""),
        )
        for key, item in mutations:
            value = copy.deepcopy(original)
            value["groups"][-1][key] = item
            with self.subTest(key=key, item=item), self.assertRaises(ValueError):
                validate_study_protocol(value)
        value = copy.deepcopy(original)
        value["groups"][-1].pop("seed")
        with self.assertRaises(ValueError):
            validate_study_protocol(value)
        value = copy.deepcopy(original)
        value["groups"][-1]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_study_protocol(value)

    def test_protocol_contract_cannot_silently_change(self) -> None:
        original = create_study_protocol(counts={"development": 1})
        for key, replacement in (
            ("schema_version", "old"), ("modes", list(reversed(STUDY_MODES))),
            ("conditions", list(STUDY_VARIANTS[:-1])), ("groups", []),
            ("policy", {}), ("counts", {"development": 2}), ("evaluation", {}),
        ):
            value = copy.deepcopy(original)
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_study_protocol(value)
        for item in (0, -1, float("nan"), float("inf"), True, "0.1"):
            value = copy.deepcopy(original)
            value["policy"]["maximum_connection_gap_m"] = item
            with self.subTest(policy=item), self.assertRaises(ValueError):
                validate_study_protocol(value)


class StudyFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repository"
        for relative in REQUIRED_EXECUTION_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic unit-test fixture only\n", encoding="utf-8")
        schema = self.root / "src/railway_recon/resources/unit.schema.json"
        write_json(schema, {"title": "Synthetic fixture schema"})
        self.protocol = self.root / "benchmarks/protocols/p2-small.json"
        write_json(self.protocol, create_study_protocol(counts={"development": 1, "test": 1}))
        self.output = self.root / "research_freezes/p2.json"

    def _rehash(self, manifest: dict) -> None:
        manifest.pop("payload_sha256", None)
        manifest["payload_sha256"] = sha256_json(manifest)
        write_json(self.output, manifest)

    def _freeze(self) -> dict:
        freeze_study_protocol(self.protocol, self.output)
        return load_json(self.output)

    def test_freeze_binds_sources_schema_protocol_and_no_customer_data(self) -> None:
        private = self.root / "paper/customer-draft.md"
        private.parent.mkdir()
        private.write_text("Unit-test marker, not real customer information", encoding="utf-8")
        source = self.root / "src/railway_recon/synthetic_track_pilot.py"
        before = sha256_file(source)
        manifest = self._freeze()
        bound = {record["path"] for record in manifest["files"]}
        self.assertTrue(set(REQUIRED_EXECUTION_FILES) <= bound)
        self.assertIn("benchmarks/protocols/p2-small.json", bound)
        self.assertIn("src/railway_recon/resources/unit.schema.json", bound)
        self.assertNotIn("paper/customer-draft.md", bound)
        self.assertEqual(sha256_file(source), before)
        self.assertEqual(verify_study_freeze(self.protocol, self.output)["status"], "pass")
        with self.assertRaises(FileExistsError):
            freeze_study_protocol(self.protocol, self.output)

    def test_missing_execution_file_refuses_freeze(self) -> None:
        (self.root / "scripts/run_synthetic_track_study.py").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "required study execution"):
            self._freeze()
        self.assertFalse(self.output.exists())

    def test_mutated_source_protocol_and_schema_fail_verification(self) -> None:
        self._freeze()
        for relative in (
            "src/railway_recon/synthetic_track_study.py",
            "src/railway_recon/resources/unit.schema.json",
            "benchmarks/protocols/p2-small.json",
        ):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(relative=relative):
                result = verify_study_freeze(self.protocol, self.output)
                self.assertEqual(result["status"], "fail")
                self.assertTrue(any("SHA-256 mismatch" in issue for issue in result["issues"]))
            path.write_bytes(original)

    def test_valid_partial_manifest_is_not_test_authorization(self) -> None:
        original = self._freeze()
        for missing in (
            "benchmarks/protocols/p2-small.json",
            "src/railway_recon/synthetic_track_study.py",
            "src/railway_recon/synthetic_track_pilot.py",
            "src/railway_recon/resources/unit.schema.json",
        ):
            manifest = copy.deepcopy(original)
            manifest["files"] = [item for item in manifest["files"] if item["path"] != missing]
            manifest["file_count"] = len(manifest["files"])
            self._rehash(manifest)
            with self.subTest(missing=missing):
                result = verify_study_freeze(self.protocol, self.output)
                self.assertEqual(result["status"], "fail")
                self.assertTrue(any("Missing required hash binding" in i for i in result["issues"]))

    def test_malformed_or_forged_manifest_fails_closed(self) -> None:
        original = self._freeze()
        for key, replacement in (
            ("files", []), ("files", None), ("files", [None]),
            ("status", "draft"), ("freeze_id", "another-protocol"),
            ("file_count", 0), ("schema_version", "not-a-freeze"),
        ):
            manifest = copy.deepcopy(original)
            manifest[key] = replacement
            self._rehash(manifest)
            with self.subTest(key=key, replacement=replacement):
                self.assertEqual(verify_study_freeze(self.protocol, self.output)["status"], "fail")
        manifest = copy.deepcopy(original)
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        manifest["file_count"] += 1
        self._rehash(manifest)
        self.assertEqual(verify_study_freeze(self.protocol, self.output)["status"], "fail")
        self.output.write_text("not json", encoding="utf-8")
        self.assertEqual(verify_study_freeze(self.protocol, self.output)["status"], "fail")

    def test_new_transitive_source_requires_a_new_freeze(self) -> None:
        self._freeze()
        added = self.root / "src/railway_recon/new_dependency.py"
        added.write_text("# New synthetic fixture import\n", encoding="utf-8")
        result = verify_study_freeze(self.protocol, self.output)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("new_dependency.py" in issue for issue in result["issues"]))

    def test_missing_or_outside_manifest_fails_without_reading_cases(self) -> None:
        self.assertEqual(verify_study_freeze(self.protocol, self.output)["status"], "fail")
        outside = self.root.parent / "outside.json"
        write_json(outside, {})
        self.assertEqual(verify_study_freeze(self.protocol, outside)["status"], "fail")
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            freeze_study_protocol(self.protocol, outside)


if __name__ == "__main__":
    unittest.main()
