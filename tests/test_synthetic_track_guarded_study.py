from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from railway_recon.synthetic_track_adaptive_audit import ADAPTIVE_MODES, audit_adaptive_candidate
from railway_recon.synthetic_track_adaptive_stress import generate_adaptive_stress_cases
from railway_recon.synthetic_track_guarded_study import (
    render_guarded_case,
    run_guarded_split,
    run_guarded_stress,
)
from railway_recon.synthetic_track_pilot import _canonical_bytes, _write_json
from railway_recon.synthetic_track_study_protocol import create_study_protocol
from railway_recon.synthetic_track_study_scene import generate_study_layout

ROOT = Path(__file__).resolve().parents[1]


class GuardedRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway-guarded-runner-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def protocol(self, counts=None):
        from railway_recon.synthetic_track_guarded_protocol import create_guarded_protocol

        base = create_study_protocol(
            seed=9461, counts=counts or {"development": 1, "validation": 1, "test": 1}
        )
        path = self.directory / "protocol.json"
        _write_json(path, create_guarded_protocol(base_protocol=base))
        return path, base

    def assert_artifacts_bound(self, output):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        files = set()
        for entry in manifest["files"]:
            payload = (output / entry["path"]).read_bytes()
            self.assertEqual(len(payload), entry["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            files.add(entry["path"])
        self.assertEqual(files, {path.relative_to(output).as_posix()
                                 for path in output.rglob("*") if path.is_file()
                                 and path.name != "manifest.json"})
        self.assertIn("uv.lock", manifest["source_sha256"])
        self.assertIn("scripts/run_synthetic_track_guarded_study.py", manifest["source_sha256"])
        self.assertIn("numpy", manifest["environment"]["packages"])
        policy = json.loads((output / "policy_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["policy_sha256"], hashlib.sha256(_canonical_bytes(policy)).hexdigest())
        self.assertTrue(manifest["source_unchanged_during_run"])
        return manifest

    def test_complete_split_matches_v1_fingerprint_and_repeats_all_six_modes(self):
        source, base = self.protocol()
        observed = []

        def generate(seed, family_index):
            observed.append(seed)
            return generate_study_layout(seed, family_index)

        with patch("railway_recon.synthetic_track_study_scene.generate_study_layout", side_effect=generate):
            first = run_guarded_split(source, "development", self.directory / "first")
            repeated = run_guarded_split(source, "development", self.directory / "repeat")
        self.assertEqual(first, repeated)
        groups = [group for group in base["groups"] if group["split"] == "development"]
        self.assertEqual(observed, [group["seed"] for group in groups] * 2)
        fingerprint = hashlib.sha256()
        for group in groups:
            for case in sorted(generate_study_layout(group["seed"], group["family_index"]),
                               key=lambda item: item["case_id"]):
                fingerprint.update(_canonical_bytes(case))
        self.assertEqual(first["data_fingerprint_sha256"], fingerprint.hexdigest())
        self.assertEqual(first["case_count"], 12)
        self.assertEqual(first["detector_run_count"], 72)
        self.assertEqual(len(first["summary_by_mode"]), 6)
        self.assertFalse(first["held_out_test"])
        self.assertFalse(first["registered_test_generated"])
        self.assertFalse(first["confidence_labels_are_probabilities"])
        manifest = self.assert_artifacts_bound(self.directory / "first")
        self.assertEqual(manifest["protocol_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertIsNone(manifest["freeze_sha256"])
        for path in (self.directory / "first" / "inputs").glob("*.json"):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse({"truth", "condition", "case_id", "split"}.intersection(candidate))
        for path in (self.directory / "first" / "figures").glob("*.png"):
            with Image.open(path) as figure:
                self.assertEqual(figure.size, (1840, 1100))
        with self.assertRaises(FileExistsError):
            run_guarded_split(source, "development", self.directory / "first")

    def test_test_requires_freeze_before_generation_or_output(self):
        source, _ = self.protocol()
        with patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate:
            with self.assertRaisesRegex(ValueError, "freeze"):
                run_guarded_split(source, "test", self.directory / "test")
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_failed_freeze_does_not_authorize_test(self):
        source, _ = self.protocol()
        freeze = self.directory / "freeze.json"
        _write_json(freeze, {})
        with (
            patch("railway_recon.synthetic_track_study_protocol._repository_root", return_value=ROOT),
            patch("railway_recon.synthetic_track_guarded_protocol.verify_guarded_freeze", return_value={"status": "fail"}),
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                run_guarded_split(source, "test", self.directory / "test", freeze)
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_foreign_protocol_root_rejected_before_freeze_and_generation(self):
        source, _ = self.protocol()
        with (
            patch("railway_recon.synthetic_track_study_protocol._repository_root", return_value=self.directory),
            patch("railway_recon.synthetic_track_guarded_protocol.verify_guarded_freeze") as verify,
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "executing toolkit"):
                run_guarded_split(source, "test", self.directory / "test", self.directory / "freeze.json")
            verify.assert_not_called()
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_foreign_verified_root_is_not_accepted(self):
        source, _ = self.protocol()
        freeze = self.directory / "freeze.json"
        _write_json(freeze, {})
        with (
            patch("railway_recon.synthetic_track_study_protocol._repository_root", return_value=ROOT),
            patch("railway_recon.synthetic_track_guarded_protocol.verify_guarded_freeze",
                  return_value={"status": "pass", "repository_root": str(self.directory)}),
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "executing toolkit"):
                run_guarded_split(source, "test", self.directory / "test", freeze)
            generate.assert_not_called()

    def test_changed_protocol_during_freeze_verification_is_rejected(self):
        source, _ = self.protocol()
        freeze = self.directory / "freeze.json"
        _write_json(freeze, {})

        def mutate(*_):
            source.write_text("{}", encoding="utf-8")
            return {"status": "pass", "repository_root": str(ROOT)}

        with (
            patch("railway_recon.synthetic_track_study_protocol._repository_root", return_value=ROOT),
            patch("railway_recon.synthetic_track_guarded_protocol.verify_guarded_freeze", side_effect=mutate),
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                run_guarded_split(source, "test", self.directory / "test", freeze)
            generate.assert_not_called()

    def test_missing_or_unknown_split_never_substituted(self):
        source, _ = self.protocol(counts={"development": 1, "test": 1})
        for split in ("validation", "all"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                run_guarded_split(source, split, self.directory / split)
            self.assertFalse((self.directory / split).exists())

    def test_legacy_stress_preserves_all_original_cases_and_baseline_predictions(self):
        output = self.directory / "legacy"
        report = run_guarded_stress(output)
        self.assertEqual(report["suite"], "legacy")
        self.assertEqual(report["seed"], 20260907)
        self.assertEqual(report["case_count"], 18)
        self.assertEqual(report["detector_run_count"], 108)
        self.assertFalse(report["suite_results_pooled"])
        self.assertFalse(report["registered_test_generated"])
        expected = hashlib.sha256()
        for case in sorted(generate_adaptive_stress_cases(20260907), key=lambda item: item["case_id"]):
            expected.update(_canonical_bytes(case))
            for mode in ADAPTIVE_MODES:
                saved = json.loads((output / "predictions" / mode / f"{case['case_id']}.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, audit_adaptive_candidate(copy.deepcopy(case["input"]), mode, report["policy"]))
        self.assertEqual(report["data_fingerprint_sha256"], expected.hexdigest())
        self.assertEqual(report["by_condition"]["wrong_connection_bridge_clutter"]["adaptive_evidence"]["diagnosis"]["fn"], 6)
        failures = json.loads((output / "failure_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(item["mode"] == "adaptive_evidence" and
                             item["condition"] == "wrong_connection_bridge_clutter" for item in failures), 6)
        with (output / "case_results.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 108)
        self.assertEqual(len(json.loads((output / "case_index.json").read_text(encoding="utf-8"))), 18)
        manifest = self.assert_artifacts_bound(output)
        self.assertIsNone(manifest["protocol_sha256"])
        with self.assertRaises(FileExistsError):
            run_guarded_stress(output)

    def test_challenge_suite_is_separate_and_has_no_legacy_conditions(self):
        output = self.directory / "challenge"
        report = run_guarded_stress(output, suite="challenge")
        self.assertEqual(report["suite"], "challenge")
        self.assertEqual(report["seed"], 20260908)
        self.assertEqual(report["case_count"], 24)
        self.assertEqual(report["detector_run_count"], 144)
        self.assertEqual(report["base_layout_count"], 6)
        self.assertEqual(set(report["by_condition"]), {
            "normal_tight_curve", "gap_curved_continuation", "wrong_connection_smooth_fork",
            "wrong_connection_perturbed_bridge_clutter",
        })
        for value in report["summary_by_mode"].values():
            self.assertEqual(value["truth_defect_count"], 18)
            self.assertEqual(value["normal_case_count"], 6)
        self.assert_artifacts_bound(output)

    def test_invalid_stress_settings_create_no_output(self):
        for index, kwargs in enumerate(({"suite": "all"}, {"suite": "mixed"}, {"seed": -1},
                                        {"seed": True}, {"seed": 1.5}, {"policy": []},
                                        {"policy": {"not_a_policy_field": 1}})):
            output = self.directory / str(index)
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                run_guarded_stress(output, **kwargs)
            self.assertFalse(output.exists())

    def test_changed_detection_fails_and_retains_partial_predictions(self):
        from railway_recon.synthetic_track_guarded_audit import audit_guarded_candidate

        case = generate_adaptive_stress_cases(20260907)[0]

        def changed(value, mode, policy):
            prediction = audit_guarded_candidate(value, mode, policy)
            if mode == "guarded_evidence":
                prediction["findings"][0]["position"][0] += 0.1
            return prediction

        output = self.directory / "changed-detection"
        with (
            patch("railway_recon.synthetic_track_adaptive_stress.generate_adaptive_stress_cases", return_value=[case]),
            patch("railway_recon.synthetic_track_guarded_audit.audit_guarded_candidate", side_effect=changed),
            self.assertRaisesRegex(ValueError, "geometric detection set"),
        ):
            run_guarded_stress(output)
        failure = json.loads((output / "run_failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["started_case_ids"], [case["case_id"]])
        self.assertEqual(failure["completed_case_ids"], [])
        self.assertEqual(len(list((output / "predictions").rglob("*.json"))), 6)
        self.assertFalse((output / "manifest.json").exists())

    def test_source_change_during_run_prevents_success_manifest(self):
        case = generate_adaptive_stress_cases(20260907)[0]
        output = self.directory / "changed-source"
        with (
            patch("railway_recon.synthetic_track_adaptive_stress.generate_adaptive_stress_cases", return_value=[case]),
            patch("railway_recon.synthetic_track_guarded_study._source_bindings", side_effect=[{"x": "a"}, {"x": "b"}]),
            self.assertRaisesRegex(ValueError, "source changed during run"),
        ):
            run_guarded_stress(output)
        self.assertTrue((output / "run_failure.json").exists())
        self.assertTrue((output / "report.json").exists())
        self.assertFalse((output / "manifest.json").exists())

    def test_renderer_rejects_four_mode_layout(self):
        with self.assertRaisesRegex(ValueError, "six modes"):
            render_guarded_case({}, {}, {}, self.directory / "four.png", ADAPTIVE_MODES)

    def test_cli_help_and_mixed_suite_rejection(self):
        script = ROOT / "scripts" / "run_synthetic_track_guarded_study.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"],
                                     capture_output=True, text=True, check=False)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("stress", help_result.stdout)
        output = self.directory / "mixed"
        rejected = subprocess.run([sys.executable, str(script), "stress", "--suite", "all", "--output", str(output)],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(rejected.returncode, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
