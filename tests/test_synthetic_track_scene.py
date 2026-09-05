"""Generator contract tests; deliberately independent of the pilot detector."""

from __future__ import annotations

import json
import math
import re
import unittest
from collections import Counter, defaultdict
from itertools import pairwise

from railway_recon.synthetic_track_scene import generate_pilot_cases


def _groups(cases):
    groups = defaultdict(dict)
    for case in cases:
        groups[case["layout_id"]][case["truth"]["variant"]] = case
    return groups


def _connection_spans(case):
    segments = {segment["id"]: segment for segment in case["input"]["segments"]}
    return {
        connection["id"]: math.dist(
            segments[connection["source"]]["points"][-1],
            segments[connection["target"]]["points"][0],
        )
        for connection in case["input"]["connections"]
    }


def _geometry(case):
    return Counter(
        tuple(tuple(point) for point in segment["points"])
        for segment in case["input"]["segments"]
    )


def _check_exactly_six_paired_layouts_and_24_cases(cases):
    assert len(cases) == 24
    assert len({case["case_id"] for case in cases}) == 24
    groups = _groups(cases)
    assert len(groups) == 6
    for variants in groups.values():
        assert set(variants) == {"normal", "gap", "wrong_connection", "duplicate"}
        normal = variants["normal"]
        parameters = normal["truth"]["parameters"]
        assert parameters["track_count"] in {2, 3}
        assert parameters["segments_per_track"] in {2, 3}
        assert len(normal["input"]["segments"]) == (
            parameters["track_count"] * parameters["segments_per_track"]
        )
        assert normal["truth"]["defects"] == []
        for variant in variants.values():
            assert variant["input"]["observations"] == normal["input"]["observations"]


def _check_seed_is_deterministic_json_serializable_and_changes_layouts(cases):
    assert json.dumps(cases, allow_nan=False) == json.dumps(generate_pilot_cases(), allow_nan=False)
    different = generate_pilot_cases(seed=812)
    assert cases != different
    assert cases[0]["input"]["observations"] != different[0]["input"]["observations"]


def _check_ids_and_connections_resolve_without_truth_leakage(cases):
    all_entity_ids = []
    for case in cases:
        assert re.fullmatch(r"[0-9a-f]{32}", case["case_id"])
        assert re.fullmatch(r"[0-9a-f]{32}", case["layout_id"])
        scene = case["input"]
        assert set(scene) == {
            "schema_version", "coordinate_system", "gauge_m", "segments", "connections",
            "observations",
        }
        assert scene["coordinate_system"] == "synthetic_local_m"
        assert scene["gauge_m"] == 1.435
        serialized = json.dumps(scene)
        assert case["case_id"] not in serialized
        assert case["layout_id"] not in serialized
        for marker in ("normal", "gap", "wrong_connection", "duplicate", "truth", "track_id"):
            assert marker not in serialized
        segment_ids = {segment["id"] for segment in scene["segments"]}
        connection_ids = {connection["id"] for connection in scene["connections"]}
        assert len(segment_ids) == len(scene["segments"])
        assert len(connection_ids) == len(scene["connections"])
        assert not segment_ids & connection_ids
        for segment in scene["segments"]:
            assert set(segment) == {"id", "points"}
            assert len(segment["points"]) > 2
            for point in segment["points"]:
                assert len(point) == 3
                assert all(math.isfinite(value) for value in point)
        for connection in scene["connections"]:
            assert set(connection) == {"id", "source", "target"}
            assert connection["source"] in segment_ids
            assert connection["target"] in segment_ids
            assert connection["source"] != connection["target"]
        for defect in case["truth"]["defects"]:
            expected_ids = segment_ids if defect["kind"] == "duplicate" else connection_ids
            assert set(defect["entity_ids"]) <= expected_ids
            assert len(defect["position"]) == 3
        all_entity_ids.extend(segment_ids | connection_ids)
        assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in segment_ids | connection_ids)
    assert len(all_entity_ids) == len(set(all_entity_ids))


def _check_normal_connections_and_direction_are_geometrically_consistent(cases):
    for variants in _groups(cases).values():
        normal = variants["normal"]
        assert all(distance < 0.02 for distance in _connection_spans(normal).values())
        angle = normal["truth"]["parameters"]["rotation_rad"]
        along = [math.cos(angle), math.sin(angle)]
        for segment in normal["input"]["segments"]:
            points = segment["points"]
            assert all(
                sum((b[i] - a[i]) * along[i] for i in range(2)) > 0
                for a, b in pairwise(points)
            )


def _check_gap_changes_actual_geometry_and_preserves_other_segments(cases):
    for variants in _groups(cases).values():
        normal, gap = variants["normal"], variants["gap"]
        assert len(gap["truth"]["defects"]) == 1
        assert len(gap["input"]["segments"]) == len(normal["input"]["segments"])
        normal_geometry, gap_geometry = _geometry(normal), _geometry(gap)
        assert sum((normal_geometry - gap_geometry).values()) == 1
        assert sum((gap_geometry - normal_geometry).values()) == 1
        defect = gap["truth"]["defects"][0]
        spans = _connection_spans(gap)
        affected_id = defect["entity_ids"][0]
        removed = gap["truth"]["parameters"]["removed_length_m"]
        assert 0.25 <= removed <= 0.80
        assert abs(spans[affected_id] - removed) < 0.02
        assert all(span < 0.02 for key, span in spans.items() if key != affected_id)


def _check_wrong_connection_replaces_relation_but_leaves_geometry_intact(cases):
    for variants in _groups(cases).values():
        normal, wrong = variants["normal"], variants["wrong_connection"]
        assert _geometry(normal) == _geometry(wrong)
        assert len(normal["input"]["connections"]) == len(wrong["input"]["connections"])
        assert len(wrong["truth"]["defects"]) == 1
        defect = wrong["truth"]["defects"][0]
        affected_id = defect["entity_ids"][0]
        spans = _connection_spans(wrong)
        assert spans[affected_id] > 3.0
        assert all(span < 0.02 for key, span in spans.items() if key != affected_id)


def _check_duplicate_is_real_long_overlapping_geometry(cases):
    for variants in _groups(cases).values():
        normal, duplicate = variants["normal"], variants["duplicate"]
        assert len(duplicate["input"]["segments"]) == len(normal["input"]["segments"]) + 1
        assert not (_geometry(normal) - _geometry(duplicate))
        assert len(duplicate["truth"]["defects"]) == 1
        defect = duplicate["truth"]["defects"][0]
        assert len(defect["entity_ids"]) == 2
        segments = {segment["id"]: segment["points"] for segment in duplicate["input"]["segments"]}
        original, copied = [segments[entity_id] for entity_id in defect["entity_ids"]]
        assert len(original) == len(copied)
        expected = duplicate["truth"]["parameters"]["duplicate_offset_m"]
        assert 0.01 <= expected <= 0.02
        assert all(abs(math.dist(a, b) - expected) < 1e-10 for a, b in zip(original, copied))
        assert sum(math.dist(a, b) for a, b in pairwise(original)) > 15.0


def _check_variants_have_no_shared_mutable_payloads(cases):
    regenerated = generate_pilot_cases()
    variants = next(iter(_groups(regenerated).values()))
    normal = variants["normal"]
    gap = variants["gap"]
    normal["input"]["observations"][0][0] += 999
    normal["input"]["segments"][0]["points"][0][0] += 999
    normal["truth"]["parameters"]["translation_m"][0] += 999
    original_gap = next(case for case in cases if case["case_id"] == gap["case_id"])
    assert gap == original_gap


class SyntheticTrackSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = generate_pilot_cases()

    def test_exactly_six_paired_layouts_and_24_cases(self):
        _check_exactly_six_paired_layouts_and_24_cases(self.cases)

    def test_seed_is_deterministic_json_serializable_and_changes_layouts(self):
        _check_seed_is_deterministic_json_serializable_and_changes_layouts(self.cases)
        for invalid in (True, "20260905", 1.5):
            with self.subTest(seed=invalid), self.assertRaises(TypeError):
                generate_pilot_cases(seed=invalid)

    def test_ids_and_connections_resolve_without_truth_leakage(self):
        _check_ids_and_connections_resolve_without_truth_leakage(self.cases)

    def test_normal_connections_and_direction_are_geometrically_consistent(self):
        _check_normal_connections_and_direction_are_geometrically_consistent(self.cases)

    def test_gap_changes_actual_geometry_and_preserves_other_segments(self):
        _check_gap_changes_actual_geometry_and_preserves_other_segments(self.cases)

    def test_wrong_connection_replaces_relation_but_leaves_geometry_intact(self):
        _check_wrong_connection_replaces_relation_but_leaves_geometry_intact(self.cases)

    def test_duplicate_is_real_long_overlapping_geometry(self):
        _check_duplicate_is_real_long_overlapping_geometry(self.cases)

    def test_variants_have_no_shared_mutable_payloads(self):
        _check_variants_have_no_shared_mutable_payloads(self.cases)


if __name__ == "__main__":
    unittest.main()
