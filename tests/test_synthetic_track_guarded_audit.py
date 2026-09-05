"""Regression and limitation checks, without registered held-out generation."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from railway_recon.synthetic_track_adaptive_audit import ADAPTIVE_MODES, audit_adaptive_candidate
from railway_recon.synthetic_track_adaptive_stress import generate_adaptive_stress_cases
from railway_recon.synthetic_track_adaptive_study import evaluate_adaptive_prediction
from railway_recon.synthetic_track_guarded_audit import (
    GUARDED_MODES,
    _direction_measurements,
    audit_guarded_candidate,
)
from railway_recon.synthetic_track_study_audit import StudyAuditPolicy


class GuardedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_adaptive_stress_cases()

    def test_old_four_modes_are_exactly_unchanged(self):
        for case in self.cases[:3]:
            for mode in ADAPTIVE_MODES:
                self.assertEqual(audit_guarded_candidate(case["input"], mode),
                                 audit_adaptive_candidate(case["input"], mode))

    def test_guard_rejects_directionally_inconsistent_false_bridge(self):
        cases = [case for case in self.cases
                 if case["truth"]["condition"] == "wrong_connection_bridge_clutter"]
        for case in cases:
            old = audit_adaptive_candidate(case["input"])
            result = audit_guarded_candidate(case["input"])
            self.assertEqual(old["findings"][0]["kind"], "gap")
            self.assertTrue(old["findings"][0]["evidence_verified"])
            finding = result["findings"][0]
            self.assertEqual(finding["kind"], "wrong_connection")
            self.assertEqual(finding["decision_basis"], "geometry_fallback")
            self.assertFalse(finding["evidence_verified"])
            self.assertTrue(finding["measurements"]["observation_support_rejected_by_direction"])
            self.assertEqual(evaluate_adaptive_prediction(case["truth"], result)["diagnosis"]["tp"], 1)

    def test_missing_observation_real_gap_uses_direction_not_closest_route(self):
        for case in self.cases:
            if case["truth"]["condition"] != "gap_nearby_alternative_no_observations":
                continue
            result = audit_guarded_candidate(case["input"])
            finding = result["findings"][0]
            self.assertEqual(finding["kind"], "gap")
            self.assertEqual(finding["cause_status"], "assigned")
            self.assertFalse(finding["evidence_verified"])

    def test_geometry_ignores_observation_values(self):
        original = self.cases[0]["input"]
        changed = copy.deepcopy(original)
        changed["observations"] = [[1000.0, 1000.0, 1000.0]]
        before = audit_guarded_candidate(original, "direction_geometry")
        after = audit_guarded_candidate(changed, "direction_geometry")
        self.assertEqual(before["findings"], after["findings"])
        self.assertEqual(before["checks"], after["checks"])

    def test_direction_alone_does_not_invent_an_alternative(self):
        source = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = np.array([[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
        scene = {
            "schema_version": "railway.synthetic-track-input.v1",
            "coordinate_system": "synthetic_local_m", "gauge_m": 1.435,
            "segments": [{"id": "s", "points": source.tolist()},
                         {"id": "t", "points": target.tolist()}],
            "connections": [{"id": "e", "source": "s", "target": "t"}],
            "observations": np.concatenate((np.linspace(*source, 61),
                                             np.linspace(*target, 61),
                                             np.linspace(source[-1], target[0], 61))).tolist(),
        }
        finding = audit_guarded_candidate(scene)["findings"][0]
        self.assertFalse(finding["measurements"]["current_chord_direction_compatible"])
        self.assertEqual(finding["measurements"]["alternative_candidates"], [])
        self.assertEqual(finding["kind"], "gap")
        self.assertTrue(finding["evidence_verified"])
        self.assertFalse(finding["measurements"]["observation_support_rejected_by_direction"])

    def test_new_modes_preserve_numeric_policy(self):
        for mode in GUARDED_MODES[-2:]:
            result = audit_guarded_candidate(self.cases[0]["input"], mode)
            old = audit_adaptive_candidate(self.cases[0]["input"])
            self.assertEqual(result["policy"], old["policy"])

    def test_all_modes_preserve_detection_and_do_not_mutate_input(self):
        for case in self.cases[:6]:
            snapshot = copy.deepcopy(case["input"])
            expected = audit_adaptive_candidate(snapshot)
            alarms = [(x["entity_ids"], x["position"]) for x in expected["findings"]]
            for mode in GUARDED_MODES:
                prediction = audit_guarded_candidate(case["input"], mode)
                self.assertEqual(alarms, [(x["entity_ids"], x["position"])
                                         for x in prediction["findings"]])
                evaluate_adaptive_prediction(case["truth"], prediction)
            self.assertEqual(snapshot, case["input"])

    def test_direction_measure_is_not_a_curvature_or_identity_oracle(self):
        policy = StudyAuditPolicy()
        # A quarter-circle gap can be real, despite failing this chord check.
        source = np.array([[1.0, -0.1, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0], [-0.1, 1.0, 0.0]])
        self.assertFalse(_direction_measurements(source, target, policy)["current_chord_direction_compatible"])
        # Conversely, collinear coordinates cannot certify correct route identity.
        target = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
        self.assertTrue(_direction_measurements(source, target, policy)["current_chord_direction_compatible"])

    def test_rigid_transform_and_entity_order_preserve_decisions(self):
        rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
        shift = np.array([11.0, -4.0, 3.0])
        for case in self.cases[:6]:
            changed = copy.deepcopy(case["input"])
            for segment in changed["segments"]:
                segment["points"] = (np.asarray(segment["points"]) @ rotation.T + shift).tolist()
            if changed["observations"]:
                changed["observations"] = (np.asarray(changed["observations"]) @ rotation.T + shift).tolist()
            changed["segments"].reverse()
            changed["connections"].reverse()
            for mode in GUARDED_MODES[-2:]:
                before = audit_guarded_candidate(case["input"], mode)
                after = audit_guarded_candidate(changed, mode)
                for a, b in zip(before["findings"], after["findings"]):
                    self.assertEqual((a["kind"], a["decision_basis"], a["decision_reason"]),
                                     (b["kind"], b["decision_basis"], b["decision_reason"]))
                    np.testing.assert_allclose(np.asarray(a["position"]) @ rotation.T + shift,
                                               b["position"], atol=1e-10)

    def test_unknown_mode_truth_and_bad_policy_rejected(self):
        scene = copy.deepcopy(self.cases[0]["input"])
        with self.assertRaises(ValueError):
            audit_guarded_candidate(scene, "oracle")
        scene["truth"] = self.cases[0]["truth"]
        with self.assertRaises(ValueError):
            audit_guarded_candidate(scene)
        with self.assertRaises(ValueError):
            audit_guarded_candidate(self.cases[0]["input"], policy={"minimum_direction_cosine": 1.1})


if __name__ == "__main__":
    unittest.main()
