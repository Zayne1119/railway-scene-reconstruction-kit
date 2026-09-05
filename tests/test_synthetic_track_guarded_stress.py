"""Construction contracts for independent guarded challenges; no detector oracle."""

from __future__ import annotations

import ast
import inspect
import json
import math
import unittest
from collections import Counter, defaultdict

from railway_recon import synthetic_track_guarded_stress as module
from railway_recon.synthetic_track_guarded_stress import (
    GUARDED_STRESS_CONDITIONS,
    GUARDED_STRESS_NAMESPACE,
    _main_point,
    generate_guarded_stress_cases,
)


def _geometry(scene):
    return Counter(tuple(tuple(point) for point in segment["points"])
                   for segment in scene["segments"])


def _endpoints(case):
    scene = case["input"]
    edge = scene["connections"][0]
    segments = {segment["id"]: segment["points"] for segment in scene["segments"]}
    return segments[edge["source"]], segments[edge["target"]]


def _relations(scene):
    points = {item["id"]: tuple(tuple(point) for point in item["points"])
              for item in scene["segments"]}
    return Counter((points[item["source"]], points[item["target"]])
                   for item in scene["connections"])


def _angle(first, second):
    cosine = sum(a * b for a, b in zip(first, second)) / (
        math.sqrt(sum(value**2 for value in first))
        * math.sqrt(sum(value**2 for value in second))
    )
    return math.acos(max(-1.0, min(1.0, cosine)))


class SyntheticTrackGuardedStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_guarded_stress_cases()
        cls.groups = defaultdict(dict)
        for case in cls.cases:
            cls.groups[case["layout_id"]][case["truth"]["condition"]] = case

    def test_twenty_four_cases_are_six_complete_independent_analytic_groups(self):
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(len(self.groups), 6)
        self.assertEqual(len({case["case_id"] for case in self.cases}), 24)
        self.assertEqual(Counter(case["truth"]["variant"] for case in self.cases),
                         {"normal": 6, "gap": 6, "wrong_connection": 12})
        for conditions in self.groups.values():
            self.assertEqual(set(conditions), set(GUARDED_STRESS_CONDITIONS))
            for case in conditions.values():
                parameters = case["truth"]["parameters"]
                self.assertEqual(parameters["development_probe_namespace"],
                                 GUARDED_STRESS_NAMESPACE)
                self.assertIn("not_held_out_test", parameters["evaluation_role"])
                self.assertNotIn("stress_source_seed", parameters)

    def test_deterministic_finite_json_and_seed_validation(self):
        self.assertEqual(self.cases, generate_guarded_stress_cases())
        self.assertNotEqual(self.cases, generate_guarded_stress_cases(20260909))
        json.dumps(self.cases, allow_nan=False)
        for invalid in (True, -1, 1.5, "20260908", None):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                generate_guarded_stress_cases(invalid)

    def test_input_contains_only_geometry_and_opaque_entities(self):
        all_ids = []
        for case in self.cases:
            scene = case["input"]
            self.assertEqual(set(scene), {
                "schema_version", "coordinate_system", "gauge_m", "segments",
                "connections", "observations",
            })
            ids = {item["id"] for item in [*scene["segments"], *scene["connections"]]}
            all_ids.extend(ids)
            for identity in ids:
                self.assertRegex(identity, r"^[0-9a-f]{32}$")
            for edge in scene["connections"]:
                self.assertIn(edge["source"], ids)
                self.assertIn(edge["target"], ids)
                self.assertEqual(set(edge), {"id", "source", "target"})
            for segment in scene["segments"]:
                self.assertEqual(set(segment), {"id", "points"})
                self.assertGreater(len(segment["points"]), 2)
                self.assertTrue(all(math.dist(a, b) > 0 for a, b in
                                    zip(segment["points"], segment["points"][1:])))
            for defect in case["truth"]["defects"]:
                self.assertTrue(set(defect["entity_ids"]) <= ids)
                source, target = _endpoints(case)
                self.assertEqual(defect["position"],
                                 [(a + b) / 2 for a, b in zip(source[-1], target[0])])
            serialized = json.dumps(scene)
            for marker in ("condition", "variant", "parameters", "route_identity",
                           GUARDED_STRESS_NAMESPACE, case["layout_id"]):
                self.assertNotIn(marker, serialized)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_normal_control_really_has_unmodified_touching_route(self):
        for conditions in self.groups.values():
            case = conditions[GUARDED_STRESS_CONDITIONS[0]]
            self.assertEqual(case["truth"]["defects"], [])
            source, target = _endpoints(case)
            self.assertEqual(source[-1], target[0])
            self.assertEqual(case["truth"]["parameters"]["geometry_modification"], "none")
            self.assertEqual(case["truth"]["parameters"]["relation_modification"], "none")

    def test_curved_gap_is_removed_known_arc_not_a_wrong_target_label(self):
        for conditions in self.groups.values():
            normal = conditions[GUARDED_STRESS_CONDITIONS[0]]
            gap = conditions[GUARDED_STRESS_CONDITIONS[1]]
            normal_source, normal_target = _endpoints(normal)
            source, target = _endpoints(gap)
            parameters = gap["truth"]["parameters"]
            removed = parameters["removed_planar_arc_length_m"]
            self.assertEqual(source[0], normal_source[0])
            self.assertEqual(target, normal_target)
            self.assertEqual(source[-1], _main_point(-removed, parameters))
            self.assertEqual(parameters["original_source_endpoint_m"], normal_source[-1])
            self.assertEqual(gap["input"]["observations"], normal["input"]["observations"])
            self.assertGreater(math.dist(source[-1], target[0]), 5.0)
            chord = [b - a for a, b in zip(source[-1], target[0])]
            tangent = [b - a for a, b in zip(source[-2], source[-1])]
            # A construction property, not an audit threshold or success assertion.
            self.assertGreater(_angle(chord, tangent), math.radians(8.0))
            arc_midpoint = _main_point(-removed / 2, parameters)
            chord_midpoint = [(a + b) / 2 for a, b in zip(source[-1], target[0])]
            self.assertGreater(math.dist(arc_midpoint, chord_midpoint), 0.2)

    def test_wrong_target_is_actual_relation_replacement_on_identical_geometry(self):
        for conditions in self.groups.values():
            normal = conditions[GUARDED_STRESS_CONDITIONS[0]]
            wrong = conditions[GUARDED_STRESS_CONDITIONS[2]]
            self.assertEqual(_geometry(normal["input"]), _geometry(wrong["input"]))
            self.assertNotEqual(_relations(normal["input"]), _relations(wrong["input"]))
            self.assertEqual(normal["input"]["observations"], wrong["input"]["observations"])
            source, target = _endpoints(wrong)
            parameters = wrong["truth"]["parameters"]
            self.assertEqual(source[-1], parameters["original_target_position_m"])
            self.assertEqual(target[0], parameters["replacement_target_position_m"])
            self.assertTrue(parameters["route_identity_is_latent"])
            self.assertIn("need not uniquely determine identity", parameters["identity_limitation"])
            chord = [b - a for a, b in zip(source[-1], target[0])]
            for tangent in ([b - a for a, b in zip(source[-2], source[-1])],
                            [b - a for a, b in zip(target[0], target[1])]):
                self.assertLess(_angle(chord, tangent), math.radians(12.0))

    def test_clutter_only_adds_biased_wavy_noisy_observations(self):
        for conditions in self.groups.values():
            wrong = conditions[GUARDED_STRESS_CONDITIONS[2]]
            clutter = conditions[GUARDED_STRESS_CONDITIONS[3]]
            self.assertEqual(_geometry(wrong["input"]), _geometry(clutter["input"]))
            self.assertEqual(_relations(wrong["input"]), _relations(clutter["input"]))
            baseline = Counter(tuple(point) for point in wrong["input"]["observations"])
            current = Counter(tuple(point) for point in clutter["input"]["observations"])
            self.assertFalse(baseline - current)
            added = list((current - baseline).elements())
            parameters = clutter["truth"]["parameters"]
            self.assertEqual(len(added), parameters["clutter_added_point_count"])
            self.assertGreater(abs(parameters["clutter_lateral_bias_m"]), 0)
            self.assertGreater(parameters["clutter_lateral_wobble_amplitude_m"], 0)
            self.assertGreater(parameters["clutter_noise_std_m"], 0)
            source, target = _endpoints(clutter)
            start, end = source[-1], target[0]
            direction = [b - a for a, b in zip(start, end)]
            square_length = sum(value**2 for value in direction)
            distances = []
            for point in added:
                fraction = sum((p - a) * delta for p, a, delta in
                               zip(point, start, direction)) / square_length
                closest = [a + fraction * delta for a, delta in zip(start, direction)]
                distances.append(math.dist(point, closest))
            self.assertTrue(all(distance > 0 for distance in distances))
            self.assertGreater(max(distances), 0.02)
            self.assertLess(max(distances), 0.2)

    def test_generator_depends_only_on_standard_library_not_study_or_detector(self):
        syntax = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"__future__", "copy", "math", "random", "typing"})


if __name__ == "__main__":
    unittest.main()
