"""Study generation contracts, without importing any detector or audit policy."""

from __future__ import annotations

import json
import math
import re
import unittest
from collections import Counter

from railway_recon.synthetic_track_study_scene import STUDY_VARIANTS, generate_study_layout


def _geometry(scene):
    return Counter(tuple(tuple(point) for point in item["points"]) for item in scene["segments"])


def _relations(scene):
    segments = {item["id"]: tuple(tuple(point) for point in item["points"]) for item in scene["segments"]}
    return Counter((segments[edge["source"]], segments[edge["target"]]) for edge in scene["connections"])


class SyntheticTrackStudySceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_study_layout(7291, family_index=2)
        cls.conditions = {case["truth"]["condition"]: case for case in cls.cases}

    def test_twelve_conditions_one_group_and_opaque_distinct_ids(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(set(self.conditions), set(STUDY_VARIANTS))
        self.assertEqual(len({case["layout_id"] for case in self.cases}), 1)
        identities = []
        for case in self.cases:
            self.assertRegex(case["case_id"], r"^[0-9a-f]{32}$")
            identities.append(case["case_id"])
            scene = case["input"]
            segment_ids = {item["id"] for item in scene["segments"]}
            edge_ids = {item["id"] for item in scene["connections"]}
            self.assertFalse(segment_ids & edge_ids)
            for edge in scene["connections"]:
                self.assertIn(edge["source"], segment_ids)
                self.assertIn(edge["target"], segment_ids)
            for defect in case["truth"]["defects"]:
                identities.append(defect["id"])
                expected = segment_ids if defect["kind"] == "duplicate" else edge_ids
                self.assertTrue(set(defect["entity_ids"]) <= expected)
            identities.extend(segment_ids | edge_ids)
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{32}", value) for value in identities))

    def test_deterministic_and_seed_family_changes_are_independent(self):
        self.assertEqual(self.cases, generate_study_layout(7291, family_index=2))
        self.assertNotEqual(self.cases, generate_study_layout(7292, family_index=2))
        self.assertNotEqual(self.cases, generate_study_layout(7291, family_index=3))
        json.dumps(self.cases, allow_nan=False)
        for invalid in (True, 2.5, "1"):
            with self.subTest(seed=invalid), self.assertRaises(TypeError):
                generate_study_layout(invalid)
        for invalid in (True, -1, "1"):
            with self.subTest(family=invalid), self.assertRaises(ValueError):
                generate_study_layout(1, family_index=invalid)

    def test_no_condition_or_truth_fields_reach_input(self):
        for case in self.cases:
            scene = case["input"]
            self.assertEqual(set(scene), {
                "schema_version", "coordinate_system", "gauge_m", "segments", "connections",
                "observations",
            })
            for segment in scene["segments"]:
                self.assertEqual(set(segment), {"id", "points"})
            serialized = json.dumps(scene)
            for marker in ("truth", "variant", "condition", "family", case["layout_id"], case["case_id"]):
                self.assertNotIn(marker, serialized)

    def test_observation_conditions_preserve_geometry_and_true_fault_entities(self):
        for name in (
            "normal_no_observations", "gap_no_observations", "wrong_connection_no_observations",
            "wrong_connection_sparse_observations",
        ):
            changed = self.conditions[name]
            base = self.conditions[changed["truth"]["variant"]]
            self.assertEqual(_geometry(base["input"]), _geometry(changed["input"]))
            self.assertEqual(_relations(base["input"]), _relations(changed["input"]))
            if name.endswith("no_observations"):
                self.assertEqual(changed["input"]["observations"], [])
            else:
                source_points = {tuple(point) for point in base["input"]["observations"]}
                retained = changed["input"]["observations"]
                self.assertGreater(len(retained), 0)
                self.assertLess(len(retained), len(source_points))
                self.assertTrue({tuple(point) for point in retained} <= source_points)
            self.assertEqual(
                [item["position"] for item in base["truth"]["defects"]],
                [item["position"] for item in changed["truth"]["defects"]],
            )

    def test_normal_controls_have_empty_truth_and_crossings_have_actual_geometry(self):
        base = self.conditions["normal"]
        for condition, case in self.conditions.items():
            if condition.startswith("normal"):
                self.assertEqual(case["truth"]["variant"], "normal")
                self.assertEqual(case["truth"]["defects"], [])
        for condition in ("normal_shallow_crossing", "normal_grade_separated_crossing"):
            case = self.conditions[condition]
            added = list((_geometry(case["input"]) - _geometry(base["input"])).elements())
            self.assertEqual(len(added), 1)
            params = case["truth"]["parameters"]
            center = params["crossing_position_m"]
            self.assertTrue(any(math.dist(point, center) < 1e-9 for point in added[0]))
            normal_points = [point for segment in base["input"]["segments"] for point in segment["points"]]
            closest_xy = min(normal_points, key=lambda p: math.hypot(p[0]-center[0], p[1]-center[1]))
            self.assertLess(math.hypot(closest_xy[0]-center[0], closest_xy[1]-center[1]), 1e-8)
            self.assertAlmostEqual(center[2]-closest_xy[2], params["crossing_clearance_m"])
            if condition == "normal_shallow_crossing":
                self.assertEqual(params["crossing_clearance_m"], 0.0)
            else:
                self.assertGreater(params["crossing_clearance_m"], 4.0)

    def test_jitter_is_small_actual_endpoint_change_with_interior_preserved(self):
        baseline = self.conditions["normal"]["input"]
        jittered = self.conditions["normal_endpoint_jitter"]
        by_interior = {tuple(tuple(p) for p in s["points"][1:-1]): s for s in baseline["segments"]}
        bound = jittered["truth"]["parameters"]["extra_endpoint_jitter_bound_m"]
        for segment in jittered["input"]["segments"]:
            original = by_interior[tuple(tuple(p) for p in segment["points"][1:-1])]
            for index in (0, -1):
                displacement = math.dist(segment["points"][index], original["points"][index])
                self.assertGreater(displacement, 0)
                self.assertLessEqual(displacement, math.sqrt(3)*bound)
        self.assertEqual(jittered["input"]["observations"], baseline["observations"])

    def test_nearby_alternative_preserves_real_gap_and_adds_crossing_route(self):
        base = self.conditions["gap"]
        confounded = self.conditions["gap_with_nearby_alternative"]
        self.assertEqual(confounded["truth"]["variant"], "gap")
        self.assertEqual(len(confounded["truth"]["defects"]), 1)
        self.assertFalse(_geometry(base["input"]) - _geometry(confounded["input"]))
        self.assertEqual(sum((_geometry(confounded["input"]) - _geometry(base["input"])).values()), 2)
        self.assertEqual(len(confounded["input"]["connections"]), len(base["input"]["connections"])+1)
        self.assertEqual(base["truth"]["defects"][0]["position"], confounded["truth"]["defects"][0]["position"])
        scene = confounded["input"]
        edge_id = confounded["truth"]["defects"][0]["entity_ids"][0]
        edge = next(item for item in scene["connections"] if item["id"] == edge_id)
        by_id = {item["id"]: item for item in scene["segments"]}
        self.assertGreater(math.dist(by_id[edge["source"]]["points"][-1], by_id[edge["target"]]["points"][0]), 0.20)
        self.assertNotEqual(confounded["truth"]["parameters"]["crossing_angle_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
