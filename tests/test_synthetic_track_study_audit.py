from __future__ import annotations

import copy
import unittest

from railway_recon.synthetic_track_study_audit import STUDY_MODES, audit_study_candidate
from tests.test_synthetic_track_audit import sample_input


class StudyAuditTests(unittest.TestCase):
    def test_three_modes_have_identical_alarm_entities_and_locations(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        outputs = [audit_study_candidate(value, mode) for mode in STUDY_MODES]
        self.assertEqual([item["findings"][0]["kind"] for item in outputs],
                         ["gap", "wrong_connection", "wrong_connection"])
        baseline = [(item["entity_ids"], item["position"]) for item in outputs[0]["findings"]]
        for output in outputs[1:]:
            self.assertEqual([(item["entity_ids"], item["position"])
                              for item in output["findings"]], baseline)

    def test_empty_evidence_preserves_alarms_without_evidence_diagnosis(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        value["observations"] = []
        output = audit_study_candidate(value, "topology_evidence")
        self.assertEqual(output["findings"][0]["kind"], "gap")
        self.assertEqual(output["summary"]["evidence_abstain_count"], 2)
        self.assertEqual(output["findings"][0]["cause_status"], "abstain")
        self.assertEqual(audit_study_candidate(value, "topology_only")["findings"][0]["kind"],
                         "wrong_connection")

    def test_sparse_neighborhood_does_not_treat_endpoint_returns_as_full_coverage(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        value["observations"] = [[10, 0, 0], [10, 4, 0]]
        output = audit_study_candidate(value, "topology_evidence")
        self.assertEqual(output["findings"][0]["kind"], "gap")
        self.assertGreater(output["summary"]["evidence_abstain_count"], 0)

    def test_normal_has_no_alarm_including_missing_observations(self):
        value = sample_input()
        for observations in (value["observations"], []):
            value["observations"] = observations
            for mode in STUDY_MODES:
                self.assertEqual(audit_study_candidate(value, mode)["findings"], [])

    def test_topology_only_is_observation_invariant_and_input_is_unchanged(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        original = copy.deepcopy(value)
        output = audit_study_candidate(value, "topology_only")
        self.assertEqual(value, original)
        value["observations"] = []
        without = audit_study_candidate(value, "topology_only")
        self.assertEqual(output["findings"], without["findings"])
        self.assertEqual(output["checks"], without["checks"])

    def test_invalid_policy_mode_and_truth_are_rejected(self):
        with self.assertRaises(ValueError):
            audit_study_candidate(sample_input(), "unknown")
        with self.assertRaises(ValueError):
            audit_study_candidate(sample_input(), policy={"minimum_neighborhood_support": 1.1})
        value = sample_input()
        value["truth"] = {"kind": "wrong_connection"}
        with self.assertRaises(ValueError):
            audit_study_candidate(value)
