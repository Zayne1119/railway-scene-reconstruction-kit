from __future__ import annotations

import copy
import unittest

import numpy as np

from railway_recon.synthetic_track_audit import audit_track_candidate


def sample_input() -> dict:
    def points(start, end, y):
        return [[float(x), float(y), 0.0] for x in np.linspace(start, end, 41)]
    return {
        "schema_version": "railway.synthetic-track-input.v1",
        "coordinate_system": "synthetic_local_m", "gauge_m": 1.435,
        "segments": [
            {"id": "a", "points": points(0, 10, 0)},
            {"id": "b", "points": points(10.01, 20, 0)},
            {"id": "c", "points": points(0, 10, 4)},
            {"id": "d", "points": points(10.01, 20, 4)},
        ],
        "connections": [
            {"id": "ab", "source": "a", "target": "b"},
            {"id": "cd", "source": "c", "target": "d"},
        ],
        "observations": [
            [float(x), float(y), 0.0] for y in (0, 4) for x in np.linspace(0, 20, 401)
        ],
    }


class SyntheticTrackAuditTests(unittest.TestCase):
    def test_normal_nearby_fragments_are_not_duplicate(self):
        value = sample_input()
        for mode in ("local_geometry", "topology_evidence"):
            result = audit_track_candidate(value, mode)
            self.assertEqual(result["findings"], [])
            self.assertGreater(result["summary"]["check_count"], 0)

    def test_geometric_mutation_remeasures_gap(self):
        value = sample_input()
        value["segments"][0]["points"] = value["segments"][0]["points"][:-2]
        for mode in ("local_geometry", "topology_evidence"):
            result = audit_track_candidate(value, mode)
            self.assertEqual(len(result["findings"]), 1)
            self.assertEqual(result["findings"][0]["kind"], "gap")
            self.assertAlmostEqual(
                result["findings"][0]["measurements"]["endpoint_separation_m"], 0.51
            )

    def test_same_error_detected_by_both_but_type_is_distinct(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        local = audit_track_candidate(value)
        enhanced = audit_track_candidate(value, "topology_evidence")
        self.assertEqual(local["findings"][0]["entity_ids"], ["ab"])
        self.assertEqual(enhanced["findings"][0]["entity_ids"], ["ab"])
        self.assertEqual(local["findings"][0]["kind"], "gap")
        self.assertEqual(enhanced["findings"][0]["kind"], "wrong_connection")

    def test_empty_evidence_does_not_clear_a_broken_connection(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        value["observations"] = []
        result = audit_track_candidate(value, "topology_evidence")
        self.assertEqual(result["findings"][0]["kind"], "gap")
        self.assertEqual(result["summary"]["evidence_abstain_count"], 2)

    def test_duplicate_geometry_detected_without_identity_labels(self):
        value = sample_input()
        duplicate = copy.deepcopy(value["segments"][0])
        duplicate["id"] = "e"
        for point in duplicate["points"]:
            point[1] += 0.015
        value["segments"].append(duplicate)
        result = audit_track_candidate(value)
        overlaps = [row for row in result["findings"] if row["kind"] == "duplicate"]
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(set(overlaps[0]["entity_ids"]), {"a", "e"})

    def test_crossing_is_not_longitudinal_duplication(self):
        value = sample_input()
        value["segments"].append({"id": "e", "points": [[5, -3, 0], [5, 7, 0]]})
        self.assertEqual(audit_track_candidate(value)["findings"], [])

    def test_rigid_transform_and_id_changes_preserve_measurements(self):
        value = sample_input()
        value["connections"][0]["target"] = "d"
        before = copy.deepcopy(value)
        reference = audit_track_candidate(value, "topology_evidence")
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for segment in value["segments"]:
            segment["points"] = (np.asarray(segment["points"]) @ rotation.T + 31.7).tolist()
            segment["id"] += "-renamed"
        for edge in value["connections"]:
            edge["id"] += "-renamed"
            edge["source"] += "-renamed"
            edge["target"] += "-renamed"
        value["observations"] = (np.asarray(value["observations"]) @ rotation.T + 31.7).tolist()
        result = audit_track_candidate(value, "topology_evidence")
        self.assertEqual(result["findings"][0]["kind"], reference["findings"][0]["kind"])
        self.assertAlmostEqual(
            result["findings"][0]["measurements"]["endpoint_separation_m"],
            reference["findings"][0]["measurements"]["endpoint_separation_m"],
        )
        self.assertEqual(audit_track_candidate(before, "topology_evidence"), reference)

    def test_rejects_truth_cached_residuals_nonfinite_and_bad_references(self):
        mutations = (
            lambda v: v.update(truth={"variant": "gap"}),
            lambda v: v["segments"][0].update(endpoint_separation_m=0.0),
            lambda v: v["segments"][0]["points"][0].__setitem__(0, float("nan")),
            lambda v: v["connections"][0].update(target="unknown"),
        )
        for mutate in mutations:
            value = sample_input()
            mutate(value)
            with self.assertRaises(ValueError):
                audit_track_candidate(value)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_track_candidate(sample_input(), policy={"maximum_connection_gap_m": -1})
        with self.assertRaises(ValueError):
            audit_track_candidate(sample_input(), mode="not-a-method")

    def test_equal_distance_alternatives_do_not_choose_diagnosis_by_id(self):
        value = sample_input()
        value["segments"] = [
            {"id": "source", "points": [[-1, 0, 0], [0, 0, 0]]},
            {"id": "target", "points": [[0, 4, 0], [1, 4, 0]]},
            {"id": "first", "points": [[-0.1, 0, 0], [0.7, 0, 0]]},
            {"id": "last", "points": [[0.1, 0, 0], [0.9, 0, 0]]},
        ]
        value["connections"] = [{"id": "edge", "source": "source", "target": "target"}]
        value["observations"] = [[0.1, 0, 0]]
        first = audit_track_candidate(value, "topology_evidence")
        value["segments"][2]["id"], value["segments"][3]["id"] = "last", "first"
        second = audit_track_candidate(value, "topology_evidence")
        self.assertEqual(first["findings"][0]["kind"], "wrong_connection")
        self.assertEqual(second["findings"][0]["kind"], "wrong_connection")
        self.assertEqual(first["findings"][0]["measurements"]["supported_alternative_count"], 1)

    def test_duplicate_location_is_symmetric_under_renaming(self):
        value = sample_input()
        duplicate = copy.deepcopy(value["segments"][0])
        duplicate["id"] = "z"
        for point in duplicate["points"]:
            point[1] += 0.015
        value["segments"].append(duplicate)
        first = audit_track_candidate(value)["findings"][0]["position"]
        value["segments"][-1]["id"] = "0"
        second = audit_track_candidate(value)["findings"][0]["position"]
        np.testing.assert_allclose(first, second, atol=1e-12)
        self.assertAlmostEqual(first[1], 0.0075)


if __name__ == "__main__":
    unittest.main()
