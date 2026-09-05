"""Pressure-probe geometry contracts, without any detector or policy import."""

from __future__ import annotations

import json
import math
import random
import unittest
from collections import Counter, defaultdict

from railway_recon.synthetic_track_adaptive_stress import (
    ADAPTIVE_STRESS_CONDITIONS,
    _affected_endpoints,
    _sample_end_region,
    generate_adaptive_stress_cases,
)
from railway_recon.synthetic_track_study_scene import _reidentify, generate_study_layout


def _geometry(scene):
    return Counter(
        tuple(tuple(point) for point in segment["points"]) for segment in scene["segments"]
    )


def _relations(scene):
    points = {
        segment["id"]: tuple(tuple(p) for p in segment["points"]) for segment in scene["segments"]
    }
    return Counter(
        (points[edge["source"]], points[edge["target"]]) for edge in scene["connections"]
    )


class SyntheticTrackAdaptiveStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_adaptive_stress_cases()
        cls.groups = defaultdict(dict)
        for case in cls.cases:
            cls.groups[case["layout_id"]][case["truth"]["condition"]] = case

    def test_exactly_eighteen_cases_with_disjoint_unregistered_seed_domain(self):
        self.assertEqual(len(self.cases), 18)
        self.assertEqual(len(self.groups), 6)
        self.assertEqual(len({case["case_id"] for case in self.cases}), 18)
        for conditions in self.groups.values():
            self.assertEqual(set(conditions), set(ADAPTIVE_STRESS_CONDITIONS))
            seeds = {
                case["truth"]["parameters"]["stress_source_seed"] for case in conditions.values()
            }
            self.assertEqual(len(seeds), 1)
            self.assertLess(next(iter(seeds)), 0)
        self.assertEqual(
            len({case["truth"]["parameters"]["stress_source_seed"] for case in self.cases}), 6
        )

    def test_deterministic_json_and_input_validation(self):
        self.assertEqual(self.cases, generate_adaptive_stress_cases())
        self.assertNotEqual(self.cases, generate_adaptive_stress_cases(20260908))
        json.dumps(self.cases, allow_nan=False)
        for invalid in (True, -1, 1.5, "20260907"):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                generate_adaptive_stress_cases(invalid)

    def test_labels_and_parameter_metadata_never_reach_detector_input(self):
        all_ids = []
        for case in self.cases:
            scene = case["input"]
            self.assertEqual(
                set(scene),
                {
                    "schema_version",
                    "coordinate_system",
                    "gauge_m",
                    "segments",
                    "connections",
                    "observations",
                },
            )
            ids = {item["id"] for item in [*scene["segments"], *scene["connections"]]}
            all_ids.extend(ids)
            for defect in case["truth"]["defects"]:
                self.assertTrue(set(defect["entity_ids"]) <= ids)
            self.assertEqual(len(case["truth"]["defects"]), 1)
            serialized = json.dumps(scene)
            for marker in (
                "condition",
                "truth",
                "variant",
                "stress_source_seed",
                case["layout_id"],
            ):
                self.assertNotIn(marker, serialized)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_observation_changes_preserve_original_geometry_relations_and_labels(self):
        for conditions in self.groups.values():
            representative = next(iter(conditions.values()))
            parameters = representative["truth"]["parameters"]
            originals = {
                case["truth"]["condition"]: case
                for case in generate_study_layout(
                    parameters["stress_source_seed"], parameters["family_index"]
                )
            }
            for condition, case in conditions.items():
                source = (
                    "gap_with_nearby_alternative"
                    if condition == ADAPTIVE_STRESS_CONDITIONS[0]
                    else "wrong_connection"
                )
                baseline = originals[source]
                self.assertEqual(_geometry(case["input"]), _geometry(baseline["input"]))
                self.assertEqual(_relations(case["input"]), _relations(baseline["input"]))
                self.assertEqual(case["truth"]["variant"], baseline["truth"]["variant"])
                self.assertEqual(
                    case["truth"]["defects"][0]["position"],
                    baseline["truth"]["defects"][0]["position"],
                )
                if condition == ADAPTIVE_STRESS_CONDITIONS[0]:
                    self.assertEqual(case["input"]["observations"], [])
                elif condition == ADAPTIVE_STRESS_CONDITIONS[1]:
                    baseline_points = Counter(tuple(p) for p in baseline["input"]["observations"])
                    current = Counter(tuple(p) for p in case["input"]["observations"])
                    self.assertFalse(baseline_points - current)
                    added = list((current - baseline_points).elements())
                    self.assertEqual(
                        len(added), case["truth"]["parameters"]["clutter_added_point_count"]
                    )
                    start, end = [
                        points[index] for points, index in zip(_affected_endpoints(case), (-1, 0))
                    ]
                    midpoint = [(a + b) / 2 for a, b in zip(start, end)]
                    self.assertLess(min(math.dist(point, midpoint) for point in added), 0.15)

    def test_occluded_case_observes_only_nominated_source_and_target_contexts(self):
        for conditions in self.groups.values():
            case = conditions[ADAPTIVE_STRESS_CONDITIONS[2]]
            source, target = _affected_endpoints(case)
            parameters = case["truth"]["parameters"]
            extent = parameters["visible_endpoint_extent_m"]
            visible = [
                *_sample_end_region(
                    source, extent, parameters["observation_spacing_m"], at_end=True
                ),
                *_sample_end_region(
                    target, extent, parameters["observation_spacing_m"], at_end=False
                ),
            ]
            observations = case["input"]["observations"]
            self.assertEqual(len(observations), len(visible))
            self.assertGreater(len(observations), 10)
            for point in observations:
                self.assertLess(min(math.dist(point, sample) for sample in visible), 0.05)

    def test_sampling_is_rigid_transform_equivariant_and_identity_relabeling_preserves_input(self):
        case = self.cases[0]
        original = case["input"]
        renamed = _reidentify(case, random.Random(4107))["input"]
        self.assertEqual(_geometry(original), _geometry(renamed))
        self.assertEqual(_relations(original), _relations(renamed))
        self.assertEqual(original["observations"], renamed["observations"])

        def transform(point):
            return [3.0 - point[1], 7.0 + point[0], -2.0 + point[2]]

        points = original["segments"][0]["points"]
        for at_end in (False, True):
            expected = [
                transform(point) for point in _sample_end_region(points, 4.3, 0.17, at_end=at_end)
            ]
            actual = _sample_end_region(
                [transform(point) for point in points], 4.3, 0.17, at_end=at_end
            )
            self.assertEqual(len(expected), len(actual))
            self.assertTrue(
                all(math.dist(first, second) < 1e-9 for first, second in zip(expected, actual))
            )


if __name__ == "__main__":
    unittest.main()
