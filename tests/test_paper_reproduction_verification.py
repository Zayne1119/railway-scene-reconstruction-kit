from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_verify", ROOT / "scripts/verify_paper_reproduction.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class ReproductionVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway-reproduction-verify-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def bind_manifest(self, root, manifest):
        manifest = dict(manifest)
        manifest["files"] = [{"path": path.relative_to(root).as_posix(),
                               "bytes": path.stat().st_size, "sha256": VERIFY.sha256_file(path)}
                              for path in sorted(root.rglob("*"))
                              if path.is_file() and path.name != "manifest.json"]
        self.write_json(root / "manifest.json", manifest)

    def refresh(self, root):
        manifest = json.loads((root / "manifest.json").read_bytes())
        if manifest["schema_version"] == VERIFY.PUBLIC_SCHEMA:
            path = root / "prediction_lock.json"
            lock = json.loads(path.read_bytes())
            lock["files"] = {path.relative_to(root).as_posix(): VERIFY.sha256_file(path)
                             for path in sorted(root.rglob("*"))
                             if path.is_file() and "predictions" in path.relative_to(root).parts}
            self.write_json(root / "prediction_lock.json", lock)
        self.bind_manifest(root, manifest)

    def synthetic(self, name="synthetic-original"):
        root = self.directory / name
        root.mkdir()
        report = {"data_fingerprint_sha256": "a" * 64, "prediction_fingerprint_sha256": "b" * 64,
                  "summary_by_mode": {"guarded_evidence": {"tp": 2}}, "by_group": {"g": {"tp": 2}},
                  "by_condition": {"gap": {"tp": 2}}, "case_count": 2, "base_layout_count": 1}
        self.write_json(root / "report.json", report)
        self.write_json(root / "inputs/case.json", {"coordinates": [1, 2]})
        self.write_json(root / "predictions/guarded_evidence/case.json", {"findings": [1]})
        self.write_json(root / "timings.json", {"elapsed": 0.1})
        self.bind_manifest(root, {"schema_version": VERIFY.SYNTHETIC_SCHEMA,
            "source_unchanged_during_run": True, "split": "test",
            "data_fingerprint_sha256": "a" * 64, "prediction_fingerprint_sha256": "b" * 64,
            "protocol_sha256": "c" * 64, "policy_sha256": "d" * 64,
            "source_sha256": {"src/railway_recon/relationships.py": "e" * 64},
            "environment": {"python": "3.11", "packages": {"numpy": "2"}}})
        return root

    def public(self, name="public-original"):
        root = self.directory / name
        root.mkdir()
        self.write_json(root / "report.json", {"samples": [{"sample": "cloud"}],
            "by_parent_cloud": {"cloud": {"tp": 1}}, "pooled_descriptive_evaluation": {"tp": 1},
            "sample_count": 1, "parent_cloud_count": 1, "total_point_count": 3})
        self.write_json(root / "protocol_snapshot.json", {"policy": {"spacing": 1.9}})
        self.write_json(root / "samples/cloud/predictions/mode/candidates.json", {"lines": 2})
        np.savez_compressed(root / "samples/cloud/predictions/mode/mask.npz",
                            point_mask=np.array([True, False, True]))
        (root / "samples/cloud/evaluation").mkdir()
        np.savez_compressed(root / "samples/cloud/evaluation/error_indices.npz", fp=np.array([2]))
        self.write_json(root / "timings.json", {"elapsed": 0.1})
        self.write_json(root / "prediction_lock.json", {
            "status": "all_samples_all_modes_saved_before_any_reference_evaluation",
            "protocol_sha256": "c" * 64, "input_sha256": {"cloud.laz": "d" * 64}, "files": {}})
        manifest = {"schema_version": VERIFY.PUBLIC_SCHEMA, "status": "completed",
            "protocol_sha256": "c" * 64, "input_sha256": {"cloud.laz": "d" * 64},
            "receipt_sha256": {"receipt.json": "e" * 64},
            "source_sha256": {"src/railway_recon/public_rail_candidates.py": "f" * 64,
                              VERIFY.PUBLIC_NONALGORITHM_SOURCE: "0" * 64},
            "environment": {"python": "3.11", "packages": {"numpy": "2"}}}
        self.bind_manifest(root, manifest)
        self.refresh(root)
        return root

    def snapshot(self):
        root = self.directory / "snapshot"
        root.mkdir()
        source = root / "source.py"
        source.write_text("pass\n", encoding="utf-8")
        self.write_json(root / "snapshot_manifest.json", {
            "schema_version": VERIFY.SNAPSHOT_SCHEMA, "status": "completed", "file_count": 1,
            "files": [{"path": "source.py", "bytes": source.stat().st_size,
                       "sha256": VERIFY.sha256_file(source)}]})
        return root

    def replay(self, original):
        destination = original.with_name(original.name.replace("original", "replay"))
        shutil.copytree(original, destination)
        return destination

    def test_identical_synthetic_results_ignore_but_hash_verify_timing(self):
        original = self.synthetic()
        replay = self.replay(original)
        self.write_json(replay / "timings.json", {"elapsed": 99.0})
        self.refresh(replay)
        result = VERIFY.compare_runs(original, replay, "synthetic")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["excluded_from_between_run_result_equality"], ["timings.json"])
        self.assertTrue(result["excluded_files_still_integrity_verified"])
        self.assertFalse(result["replay_is_additional_held_out_test"])

    def test_timing_hash_corruption_is_not_ignored(self):
        original = self.synthetic()
        replay = self.replay(original)
        self.write_json(replay / "timings.json", {"elapsed": 99.0})
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            VERIFY.compare_runs(original, replay, "synthetic")

    def test_public_prediction_tamper_without_new_manifest_rejected(self):
        original = self.public()
        replay = self.replay(original)
        path = replay / "samples/cloud/predictions/mode/mask.npz"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            VERIFY.compare_runs(original, replay, "public")

    def test_public_mask_difference_fails_even_with_consistent_new_hashes(self):
        original = self.public()
        replay = self.replay(original)
        np.savez_compressed(replay / "samples/cloud/predictions/mode/mask.npz",
                            point_mask=np.array([True, False, False]))
        self.refresh(replay)
        result = VERIFY.compare_runs(original, replay, "public")
        self.assertEqual(result["status"], "fail")
        self.assertIn("samples/cloud/predictions/mode/mask.npz", result["mismatches"])

    def test_public_prediction_lock_omissions_rejected_even_if_manifest_updated(self):
        original = self.public()
        replay = self.replay(original)
        lock = json.loads((replay / "prediction_lock.json").read_bytes())
        lock["files"] = {}
        self.write_json(replay / "prediction_lock.json", lock)
        self.bind_manifest(replay, json.loads((replay / "manifest.json").read_bytes()))
        with self.assertRaisesRegex(ValueError, "every prediction"):
            VERIFY.compare_runs(original, replay, "public")

    def test_equal_arrays_with_different_container_bytes_are_reported(self):
        original = self.public()
        replay = self.replay(original)
        np.savez(replay / "samples/cloud/predictions/mode/mask.npz",
                 point_mask=np.array([True, False, True]))
        self.refresh(replay)
        result = VERIFY.compare_runs(original, replay, "public")
        self.assertEqual(result["status"], "pass")
        self.assertIn("samples/cloud/predictions/mode/mask.npz",
                      result["equal_content_different_serialized_bytes"])

    def test_known_public_nonalgo_difference_is_visible_not_silent(self):
        original = self.public()
        replay = self.replay(original)
        manifest = json.loads((replay / "manifest.json").read_bytes())
        manifest["source_sha256"][VERIFY.PUBLIC_NONALGORITHM_SOURCE] = "1" * 64
        self.write_json(replay / "manifest.json", manifest)
        result = VERIFY.compare_runs(original, replay, "public")
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["source_comparison"]["all_recorded_sources_identical"])
        self.assertEqual(result["source_comparison"]["differences"][0]["path"],
                         VERIFY.PUBLIC_NONALGORITHM_SOURCE)

    def test_algorithm_source_difference_fails(self):
        original = self.synthetic()
        replay = self.replay(original)
        manifest = json.loads((replay / "manifest.json").read_bytes())
        manifest["source_sha256"]["src/railway_recon/relationships.py"] = "1" * 64
        self.write_json(replay / "manifest.json", manifest)
        result = VERIFY.compare_runs(original, replay, "synthetic")
        self.assertEqual(result["status"], "fail")
        self.assertIn("algorithm_source_inventory", result["mismatches"])

    def test_report_by_group_mismatch_is_not_hidden_by_equal_fingerprints(self):
        original = self.synthetic()
        replay = self.replay(original)
        report = json.loads((replay / "report.json").read_bytes())
        report["by_group"]["g"]["tp"] = 1
        self.write_json(replay / "report.json", report)
        self.refresh(replay)
        result = VERIFY.compare_runs(original, replay, "synthetic")
        self.assertEqual(result["status"], "fail")
        self.assertIn("report.by_group", result["mismatches"])

    def test_snapshot_checks_all_bound_files_but_not_new_environment_files(self):
        root = self.snapshot()
        (root / "installed_environment.txt").write_text("not source", encoding="utf-8")
        result = VERIFY.verify_snapshot(root)
        self.assertEqual(result["files_verified"], 1)
        self.assertFalse(result["additional_environment_and_run_files_attested_by_snapshot"])
        (root / "source.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            VERIFY.verify_snapshot(root)

    def test_unbound_run_extra_is_rejected(self):
        root = self.synthetic()
        (root / "extra.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "every file"):
            VERIFY.verify_inventory(root, "manifest.json", VERIFY.SYNTHETIC_SCHEMA)

    def test_unsafe_paths_rejected(self):
        for name in ("../x", "/x", "x/../z", "x\\z", "C:/x", "x//z"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                VERIFY.checked_path(self.directory, name)

    def test_receipt_hash_links_without_absolute_paths_and_no_input_mutation(self):
        synthetic, public, snapshot = self.synthetic(), self.public(), self.snapshot()
        synthetic_replay, public_replay = self.replay(synthetic), self.replay(public)
        before = VERIFY.sha256_file(synthetic / "manifest.json")
        result = VERIFY.verify_reproduction(synthetic_original=synthetic, synthetic_replay=synthetic_replay,
                    public_original=public, public_replay=public_replay, snapshot=snapshot,
                    output=self.directory / "verified")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(before, VERIFY.sha256_file(synthetic / "manifest.json"))
        text = (self.directory / "verified/report.json").read_text(encoding="utf-8")
        self.assertNotIn(str(self.directory), text)
        self.assertNotIn(str(self.directory).replace("\\", "\\\\"), text)
        manifest = json.loads((self.directory / "verified/manifest.json").read_bytes())
        self.assertEqual(len(manifest["input_content_addresses"]), 5)
        self.assertTrue(all(value.startswith("sha256:") for value in manifest["input_content_addresses"].values()))
        record = manifest["files"][0]
        self.assertEqual(record["sha256"], VERIFY.sha256_file(self.directory / "verified/report.json"))

    def test_output_inside_run_or_existing_directory_rejected(self):
        synthetic, public, snapshot = self.synthetic(), self.public(), self.snapshot()
        args = {"synthetic_original": synthetic, "synthetic_replay": self.replay(synthetic),
                "public_original": public, "public_replay": self.replay(public), "snapshot": snapshot}
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            VERIFY.verify_reproduction(**args, output=synthetic / "bad")
        with self.assertRaises(FileExistsError):
            VERIFY.verify_reproduction(**args, output=self.directory)

    def test_original_directory_cannot_be_passed_as_its_own_replay(self):
        root = self.synthetic()
        with self.assertRaisesRegex(ValueError, "distinct run"):
            VERIFY.compare_runs(root, root, "synthetic")
