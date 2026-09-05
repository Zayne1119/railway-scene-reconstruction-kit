from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railway_recon.synthetic_track_adaptive_study import (
    evaluate_adaptive_prediction,
    run_adaptive_split,
)
from railway_recon.synthetic_track_pilot import _canonical_bytes, _write_json
from railway_recon.synthetic_track_study import evaluate_study_prediction
from railway_recon.synthetic_track_study_protocol import create_study_protocol
from railway_recon.synthetic_track_study_scene import generate_study_layout


def fixture():
    truth = {"defects": [{"kind": "gap", "entity_ids": ["edge"], "position": [0, 0, 0]}]}
    prediction = {
        "findings": [
            {
                "kind": "gap",
                "entity_ids": ["edge"],
                "position": [0, 0, 0],
                "cause_status": "assigned",
                "decision_basis": "geometry_fallback",
                "evidence_verified": False,
                "confidence_level": "inferred_geometry",
            }
        ],
        "checks": [{"kind": "gap", "entity_ids": ["edge"], "status": "flag", "measurements": {}}],
    }
    return truth, prediction


class AdaptiveEvaluationTests(unittest.TestCase):
    def test_fallback_is_assigned_but_not_evidence_verified(self):
        truth, prediction = fixture()
        result = evaluate_adaptive_prediction(truth, prediction)
        reference = evaluate_study_prediction(truth, prediction)
        self.assertEqual(result["diagnosis"], reference["diagnosis"])
        route = result["decision_routes"]["geometry_fallback"]
        self.assertEqual(route["diagnosis_tp"], 1)
        self.assertEqual(route["diagnosis_coverage"], 1)
        self.assertEqual(route["evidence_verified_findings"], 0)
        self.assertEqual(route["confidence_labels"], {"inferred_geometry": 1})

    def test_abstention_remains_fn_and_coverage_zero(self):
        truth, prediction = fixture()
        prediction["findings"][0].update(cause_status="abstain", decision_basis="evidence_conflict",
                                         confidence_level="unresolved_conflict")
        result = evaluate_adaptive_prediction(truth, prediction)
        route = result["decision_routes"]["evidence_conflict"]
        self.assertEqual(result["detection"]["tp"], 1)
        self.assertEqual(route["diagnosis_tp"], 0)
        self.assertEqual(route["diagnosis_fn"], 1)
        self.assertEqual(route["diagnosis_coverage"], 0)

    def test_misdiagnosis_adds_fp_fn_without_geometric_false_alarm(self):
        truth, prediction = fixture()
        prediction["findings"][0]["kind"] = "wrong_connection"
        result = evaluate_adaptive_prediction(truth, prediction)
        route = result["decision_routes"]["geometry_fallback"]
        self.assertEqual(result["detection"]["fp"], 0)
        self.assertEqual(route["diagnosis_fp"], 1)
        self.assertEqual(route["diagnosis_fn"], 1)
        self.assertEqual(route["wrong_assigned_cause_count"], 1)
        self.assertEqual(route["wrong_cause_among_assigned"], 1)

    def test_repeated_abstained_then_wrong_then_correct_indices_reconcile(self):
        truth, prediction = fixture()
        abstained = copy.deepcopy(prediction["findings"][0])
        abstained.update(cause_status="abstain", decision_basis="evidence_conflict",
                         confidence_level="unresolved_conflict")
        wrong = copy.deepcopy(prediction["findings"][0])
        wrong.update(kind="wrong_connection")
        correct = copy.deepcopy(prediction["findings"][0])
        correct.update(decision_basis="evidence_supported", evidence_verified=True,
                       confidence_level="observation_supported")
        prediction["findings"] = [abstained, wrong, correct]
        result = evaluate_adaptive_prediction(truth, prediction)
        self.assertEqual(result["diagnosis"]["matches"][0]["prediction_index"], 2)
        self.assertEqual(result["decision_routes"]["evidence_supported"]["diagnosis_tp"], 1)
        self.assertEqual(result["decision_routes"]["geometry_fallback"]["diagnosis_fp"], 1)
        self.assertEqual(
            sum(route["truth_count"] for route in result["decision_routes"].values()), 1
        )
        for key in ("tp", "fp", "fn"):
            self.assertEqual(
                sum(route[f"diagnosis_{key}"] for route in result["decision_routes"].values()),
                result["diagnosis"][key],
            )

    def test_missing_alarm_has_undetected_truth_and_no_fake_route_assignment(self):
        truth, prediction = fixture()
        prediction["findings"] = []
        prediction["checks"] = []
        result = evaluate_adaptive_prediction(truth, prediction)
        self.assertEqual(result["decision_routes"]["undetected"]["diagnosis_fn"], 1)
        self.assertEqual(result["decision_routes"]["undetected"]["truth_count"], 1)
        self.assertEqual(result["decision_routes"]["undetected"]["findings"], 0)

    def test_normal_false_alarm_does_not_create_truth_denominator(self):
        _, prediction = fixture()
        result = evaluate_adaptive_prediction({"defects": []}, prediction)
        route = result["decision_routes"]["geometry_fallback"]
        self.assertEqual(route["diagnosis_fp"], 1)
        self.assertEqual(route["truth_count"], 0)
        self.assertIsNone(route["diagnosis_coverage"])

    def test_fallback_verification_and_numeric_confidence_are_rejected(self):
        for fields in (
            {"evidence_verified": True},
            {"confidence_level": 0.99},
            {"confidence_level": "observation_supported"},
            {"cause_status": "abstain"},
            {"decision_basis": "unknown"},
        ):
            truth, prediction = fixture()
            prediction["findings"][0].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                evaluate_adaptive_prediction(truth, prediction)


class AdaptiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="railway-adaptive-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def protocol(self, counts=None):
        from railway_recon.synthetic_track_adaptive_protocol import create_adaptive_protocol

        base = create_study_protocol(
            seed=8361, counts=counts or {"development": 1, "validation": 1, "test": 1}
        )
        protocol = create_adaptive_protocol(base_protocol=base)
        source = self.directory / "protocol.json"
        _write_json(source, protocol)
        return source, base

    def test_complete_split_uses_v1_fingerprint_all_four_modes_and_repeats(self):
        source, base = self.protocol()
        observed = []

        def generate(seed, family_index):
            observed.append(seed)
            return generate_study_layout(seed, family_index)

        with patch(
            "railway_recon.synthetic_track_study_scene.generate_study_layout", side_effect=generate
        ):
            first = run_adaptive_split(source, "development", self.directory / "first")
            repeated = run_adaptive_split(source, "development", self.directory / "second")
        groups = [group for group in base["groups"] if group["split"] == "development"]
        self.assertEqual(observed, [group["seed"] for group in groups] * 2)
        expected_fingerprint = hashlib.sha256()
        for group in groups:
            for case in sorted(
                generate_study_layout(group["seed"], group["family_index"]),
                key=lambda case: case["case_id"],
            ):
                expected_fingerprint.update(_canonical_bytes(case))
        self.assertEqual(first["data_fingerprint_sha256"], expected_fingerprint.hexdigest())
        self.assertEqual(first, repeated)
        self.assertEqual(first["case_count"], 12)
        self.assertEqual(first["detector_run_count"], 48)
        self.assertEqual(len(first["summary_by_mode"]), 4)
        self.assertFalse(first["held_out_test"])
        self.assertFalse(first["confidence_labels_are_probabilities"])
        manifest = json.loads((self.directory / "first" / "manifest.json").read_text())
        for entry in manifest["files"]:
            payload = (self.directory / "first" / entry["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
        for path in (self.directory / "first" / "inputs").glob("*.json"):
            candidate = json.loads(path.read_text())
            self.assertFalse({"truth", "condition", "case_id", "split"}.intersection(candidate))
        for result in first["summary_by_mode"].values():
            for key in ("tp", "fp", "fn"):
                self.assertEqual(
                    sum(route[f"diagnosis_{key}"] for route in result["decision_routes"].values()),
                    result["diagnosis"][key],
                )
        with self.assertRaises(FileExistsError):
            run_adaptive_split(source, "development", self.directory / "first")

    def test_test_requires_freeze_before_any_generation_or_output(self):
        source, _ = self.protocol()
        with patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate:
            with self.assertRaisesRegex(ValueError, "freeze"):
                run_adaptive_split(source, "test", self.directory / "test")
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_failed_freeze_is_not_treated_as_test_authorization(self):
        source, _ = self.protocol()
        with (
            patch(
                "railway_recon.synthetic_track_study_protocol._repository_root",
                return_value=Path(__file__).resolve().parents[1],
            ),
            patch(
                "railway_recon.synthetic_track_adaptive_protocol.verify_adaptive_freeze",
                return_value={"status": "fail"},
            ),
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                run_adaptive_split(
                    source, "test", self.directory / "test", self.directory / "freeze.json"
                )
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_freeze_must_belong_to_executing_repository(self):
        source, _ = self.protocol()
        with (
            patch(
                "railway_recon.synthetic_track_study_protocol._repository_root",
                return_value=self.directory,
            ),
            patch(
                "railway_recon.synthetic_track_adaptive_protocol.verify_adaptive_freeze"
            ) as verify,
            patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate,
        ):
            with self.assertRaisesRegex(ValueError, "executing toolkit"):
                run_adaptive_split(
                    source, "test", self.directory / "test", self.directory / "freeze.json"
                )
            verify.assert_not_called()
            generate.assert_not_called()
        self.assertFalse((self.directory / "test").exists())

    def test_missing_split_is_not_implicitly_replaced(self):
        source, _ = self.protocol(counts={"development": 1, "test": 1})
        with self.assertRaisesRegex(ValueError, "no registered groups"):
            run_adaptive_split(source, "validation", self.directory / "validation")
        self.assertFalse((self.directory / "validation").exists())


if __name__ == "__main__":
    unittest.main()
