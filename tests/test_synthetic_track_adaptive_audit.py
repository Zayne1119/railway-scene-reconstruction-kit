from __future__ import annotations

import copy
import unittest

import numpy as np

from railway_recon.synthetic_track_adaptive_audit import ADAPTIVE_MODES, audit_adaptive_candidate
from railway_recon.synthetic_track_study import evaluate_study_prediction
from railway_recon.synthetic_track_study_audit import audit_study_candidate
from tests.test_synthetic_track_audit import sample_input


class AdaptiveAuditTests(unittest.TestCase):
    def wrong_connection(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        return value

    def finding(self, output):
        return next(item for item in output["findings"] if item["entity_ids"] == ["ab"])

    def test_missing_observations_keep_inferred_cause_not_verified(self):
        value = self.wrong_connection()
        value["observations"] = []
        finding = self.finding(audit_adaptive_candidate(value))
        self.assertEqual((finding["kind"], finding["cause_status"]), ("wrong_connection", "assigned"))
        self.assertEqual(finding["decision_basis"], "geometry_fallback")
        self.assertFalse(finding["evidence_verified"])
        self.assertEqual(finding["confidence_level"], "inferred_geometry")
        strict = self.finding(audit_adaptive_candidate(value, "topology_evidence"))
        self.assertEqual(strict["cause_status"], "abstain")
        self.assertEqual(strict["decision_basis"], "evidence_abstention")

    def test_sparse_observations_are_not_promoted_to_verified(self):
        value = self.wrong_connection()
        value["observations"] = [[10.0, 0, 0], [10.0, 4, 0]]
        finding = self.finding(audit_adaptive_candidate(value))
        self.assertEqual(finding["decision_basis"], "geometry_fallback")
        self.assertFalse(finding["evidence_verified"])

    def test_dense_observations_can_support_wrong_connection(self):
        finding = self.finding(audit_adaptive_candidate(self.wrong_connection()))
        self.assertEqual(finding["kind"], "wrong_connection")
        self.assertTrue(finding["evidence_verified"])
        self.assertEqual(finding["decision_reason"], "alternative_path_supported")

    def test_missing_alternative_context_does_not_turn_wrong_target_into_gap(self):
        value = self.wrong_connection()
        value["observations"] = [point for point in value["observations"]
                                 if point[1] == 4 or point[0] < 10]
        finding = self.finding(audit_adaptive_candidate(value))
        self.assertEqual(finding["kind"], "wrong_connection")
        self.assertEqual(finding["decision_reason"], "alternative_neighborhood_missing_or_sparse")
        self.assertFalse(finding["evidence_verified"])

    def test_covered_neighborhoods_without_either_path_support_expose_conflict(self):
        value = self.wrong_connection()
        value["observations"] = [point for point in value["observations"]
                                 if point[0] < 9.75 or point[0] > 10.25]
        output = audit_adaptive_candidate(value)
        finding = self.finding(output)
        self.assertEqual(finding["cause_status"], "abstain")
        self.assertEqual(finding["decision_basis"], "evidence_conflict")
        truth = {"defects": [{"kind": "wrong_connection", "entity_ids": ["ab"],
                               "position": [10.005, 2, 0]}]}
        evaluation = evaluate_study_prediction(truth, output)
        self.assertEqual(evaluation["detection"]["tp"], 1)
        self.assertEqual(evaluation["diagnosis"]["fn"], 1)

    def test_positive_clutter_is_not_guaranteed_correct_evidence(self):
        value = self.wrong_connection()
        value["observations"].extend([[10.005, float(y), 0] for y in np.linspace(0, 4, 100)])
        finding = self.finding(audit_adaptive_candidate(value))
        self.assertEqual(finding["kind"], "gap")
        self.assertTrue(finding["evidence_verified"])
        # This deliberately wrong cause demonstrates that internal support is
        # not a truth/correctness certificate. It remains a measured failure.
        truth = {"defects": [{"kind": "wrong_connection", "entity_ids": ["ab"]}]}
        self.assertEqual(evaluate_study_prediction(truth, audit_adaptive_candidate(value))["diagnosis"]["fn"], 1)

    def test_shared_alarm_entities_positions_and_old_classifications_are_unchanged(self):
        value = self.wrong_connection()
        original = copy.deepcopy(value)
        reference = audit_study_candidate(value, "local_geometry")
        alarms = [(item["entity_ids"], item["position"]) for item in reference["findings"]]
        for mode in ADAPTIVE_MODES:
            output = audit_adaptive_candidate(value, mode)
            self.assertEqual([(item["entity_ids"], item["position"]) for item in output["findings"]], alarms)
            if mode != "adaptive_evidence":
                old = audit_study_candidate(value, mode)
                self.assertEqual(output["checks"], old["checks"])
                self.assertEqual([(item["kind"], item["cause_status"]) for item in output["findings"]],
                                 [(item["kind"], item["cause_status"]) for item in old["findings"]])
        self.assertEqual(value, original)

    def test_normal_missing_evidence_does_not_create_alarms(self):
        value = sample_input()
        value["observations"] = []
        self.assertEqual(audit_adaptive_candidate(value)["findings"], [])

    def test_rigid_transform_and_opaque_id_order_preserve_decision_provenance(self):
        value = self.wrong_connection()
        expected = self.finding(audit_adaptive_candidate(value))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        shift = np.array([21.0, -7.0, 4.0])
        names = {"a": "z-source", "b": "y-alternative", "c": "x-second", "d": "w-target"}
        for segment in value["segments"]:
            segment["points"] = (np.asarray(segment["points"]) @ rotation.T + shift).tolist()
            segment["id"] = names[segment["id"]]
        value["segments"].reverse()
        for edge in value["connections"]:
            edge["source"], edge["target"] = names[edge["source"]], names[edge["target"]]
        value["observations"] = (np.asarray(value["observations"]) @ rotation.T + shift).tolist()
        actual = self.finding(audit_adaptive_candidate(value))
        for key in ("kind", "cause_status", "decision_basis", "evidence_verified", "confidence_level"):
            self.assertEqual(actual[key], expected[key])
        np.testing.assert_allclose(actual["position"], np.asarray(expected["position"]) @ rotation.T + shift)

    def test_truth_and_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            audit_adaptive_candidate(sample_input(), "bad")
        value = sample_input()
        value["truth"] = {"kind": "gap"}
        with self.assertRaises(ValueError):
            audit_adaptive_candidate(value)
