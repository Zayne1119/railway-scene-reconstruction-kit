from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railway_recon.synthetic_track_pilot import _write_json
from railway_recon.synthetic_track_study import evaluate_study_prediction, run_study_split
from railway_recon.synthetic_track_study_protocol import create_study_protocol
from railway_recon.synthetic_track_study_scene import generate_study_layout


def evaluation_fixture():
    truth = {"defects": [{"kind": "gap", "entity_ids": ["edge"], "position": [0, 0, 0]}]}
    prediction = {
        "findings": [{"kind": "gap", "entity_ids": ["edge"], "position": [0, 0, 0],
                      "cause_status": "abstain"}],
        "checks": [{"kind": "gap", "entity_ids": ["edge"], "status": "flag",
                    "measurements": {"evidence_status": "abstain"}}],
    }
    return truth, prediction


class StudyEvaluationTests(unittest.TestCase):
    def test_abstention_is_detected_but_never_correct_gap_diagnosis(self):
        result = evaluate_study_prediction(*evaluation_fixture())
        self.assertEqual(result["detection"]["tp"], 1)
        self.assertEqual(result["diagnosis"]["tp"], 0)
        self.assertEqual(result["diagnosis"]["fn"], 1)
        self.assertEqual(result["cause_evaluation"]["confusion"], {"gap->abstain": 1})
        self.assertEqual(result["cause_evaluation"]["truth_with_assigned_diagnosis"], 0)

    def test_wrong_cause_is_diagnosis_fp_and_fn_not_geometric_false_alarm(self):
        truth, prediction = evaluation_fixture()
        prediction["findings"][0].update(kind="wrong_connection", cause_status="assigned")
        result = evaluate_study_prediction(truth, prediction)
        self.assertEqual(result["detection"]["fp"], 0)
        self.assertEqual(result["diagnosis"]["fp"], 1)
        self.assertEqual(result["diagnosis"]["fn"], 1)

    def test_repeated_alarm_and_abstention_cannot_corrupt_coverage(self):
        truth, prediction = evaluation_fixture()
        assigned = copy.deepcopy(prediction["findings"][0])
        assigned["cause_status"] = "assigned"
        prediction["findings"].append(assigned)
        result = evaluate_study_prediction(truth, prediction)
        self.assertEqual(result["detection"]["tp"], 1)
        self.assertEqual(result["detection"]["fp"], 1)
        self.assertEqual(result["cause_evaluation"]["truth_with_assigned_diagnosis"], 1)
        self.assertEqual(result["cause_evaluation"]["confusion"], {"gap->gap": 1})
        self.assertEqual(result["diagnosis"]["matches"][0]["prediction_index"], 1)

    def test_implicit_cause_status_is_rejected(self):
        truth, prediction = evaluation_fixture()
        del prediction["findings"][0]["cause_status"]
        with self.assertRaises(ValueError):
            evaluate_study_prediction(truth, prediction)


class StudyRunnerTests(unittest.TestCase):
    def test_one_split_isolated_repeatable_and_artifact_hashes_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = create_study_protocol(seed=8307, counts={"development": 1, "validation": 1, "test": 1})
            source = root / "protocol.json"
            _write_json(source, protocol)
            observed_seeds = []

            def generate(seed, family_index):
                observed_seeds.append(seed)
                return generate_study_layout(seed, family_index)

            with patch("railway_recon.synthetic_track_study_scene.generate_study_layout", side_effect=generate):
                first = run_study_split(source, "development", root / "first")
                second = run_study_split(source, "development", root / "second")
            expected = [group["seed"] for group in protocol["groups"] if group["split"] == "development"]
            self.assertEqual(observed_seeds, expected * 2)
            self.assertEqual(first, second)
            self.assertEqual(first["case_count"], 12)
            self.assertFalse(first["held_out_test"])
            manifest = json.loads((root / "first" / "manifest.json").read_text())
            for item in manifest["files"]:
                payload = (root / "first" / item["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            with self.assertRaises(FileExistsError):
                run_study_split(source, "development", root / "first")

    def test_test_requires_freeze_before_generation_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "protocol.json"
            _write_json(source, create_study_protocol(counts={"development": 1, "test": 1}))
            with patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate:
                with self.assertRaisesRegex(ValueError, "freeze"):
                    run_study_split(source, "test", root / "test")
                generate.assert_not_called()
                self.assertFalse((root / "test").exists())

    def test_missing_selected_split_and_failed_freeze_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "protocol.json"
            _write_json(source, create_study_protocol(counts={"development": 1, "test": 1}))
            with self.assertRaisesRegex(ValueError, "no registered"):
                run_study_split(source, "validation", root / "val")
            with (
                patch("railway_recon.synthetic_track_study_protocol.verify_study_freeze",
                      return_value={"status": "fail"}),
                patch("railway_recon.synthetic_track_study_protocol._repository_root",
                      return_value=Path(__file__).resolve().parents[1]),
                self.assertRaisesRegex(ValueError, "verification"),
            ):
                run_study_split(source, "test", root / "test", root / "freeze.json")
            self.assertFalse((root / "test").exists())

    def test_freeze_cannot_verify_another_repository_than_executing_toolkit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "protocol.json"
            _write_json(source, create_study_protocol(counts={"test": 1}))
            with (
                patch("railway_recon.synthetic_track_study_protocol._repository_root",
                      return_value=root),
                patch("railway_recon.synthetic_track_study_protocol.verify_study_freeze") as verify,
                patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
            ):
                with self.assertRaisesRegex(ValueError, "executing toolkit"):
                    run_study_split(source, "test", root / "test", root / "freeze.json")
                verify.assert_not_called()
                generate.assert_not_called()
            self.assertFalse((root / "test").exists())
