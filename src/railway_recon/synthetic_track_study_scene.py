"""Customer-independent analytic conditions for a grouped track audit study.

All shapes are centreline fragments; observations are noisy mathematical proxies,
not LiDAR. Condition labels specify the independently applied modification and
never depend on detector results or thresholds. Crossings are separate through
routes, not turnouts or coincident parallel railway tracks. Generation does not
read any files. Importing this module does not generate study or test cases.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from .synthetic_track_scene import (
    _layout_parameters,
    _make_layout,
    _make_variant,
    _opaque_id,
    _point,
    _sample_interval,
)

STUDY_VARIANTS = (
    "normal",
    "gap",
    "wrong_connection",
    "duplicate",
    "normal_shallow_crossing",
    "normal_grade_separated_crossing",
    "normal_endpoint_jitter",
    "normal_no_observations",
    "gap_no_observations",
    "wrong_connection_no_observations",
    "wrong_connection_sparse_observations",
    "gap_with_nearby_alternative",
)


def _reidentify(case: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Make all entity identities unrelated, preserving geometry and references."""
    result = copy.deepcopy(case)
    scene = result["input"]
    identities = {
        item["id"]: _opaque_id(rng) for item in [*scene["segments"], *scene["connections"]]
    }
    for segment in scene["segments"]:
        segment["id"] = identities[segment["id"]]
    for connection in scene["connections"]:
        connection["id"] = identities[connection["id"]]
        connection["source"] = identities[connection["source"]]
        connection["target"] = identities[connection["target"]]
    for defect in result["truth"]["defects"]:
        defect["id"] = _opaque_id(rng)
        defect["entity_ids"] = [identities[value] for value in defect["entity_ids"]]
    result["case_id"] = _opaque_id(rng)
    rng.shuffle(scene["segments"])
    rng.shuffle(scene["connections"])
    return result


def _local_coordinates(point: list[float], parameters: dict[str, Any]) -> tuple[float, float]:
    angle = parameters["rotation_rad"]
    x = point[0] - parameters["translation_m"][0]
    y = point[1] - parameters["translation_m"][1]
    return math.cos(angle) * x + math.sin(angle) * y, -math.sin(angle) * x + math.cos(angle) * y


def _track_index(point: list[float], parameters: dict[str, Any]) -> int:
    along, across = _local_coordinates(point, parameters)
    first_across = _local_coordinates(_point(along, 0, parameters), parameters)[1]
    return round((across - first_across) / parameters["track_spacing_m"])


def _add_crossing(
    case: dict[str, Any], rng: random.Random, *, raised: bool = False, near_gap: bool = False
) -> None:
    """Add a through-route intersecting the analytic main alignment in XY.

    The near-gap condition splits this route close to the gap's source endpoint,
    so a geometry-only continuation search may encounter another plausible
    fragment. It remains a crossing route, with its own directed connection.
    Neither a detector decision nor a required diagnosis gain defines the case.
    """
    scene = case["input"]
    parameters = case["truth"]["parameters"]
    if near_gap:
        affected_id = case["truth"]["defects"][0]["entity_ids"][0]
        connection = next(item for item in scene["connections"] if item["id"] == affected_id)
        source = next(item for item in scene["segments"] if item["id"] == connection["source"])
        anchor = source["points"][-1]
        start_along, _ = _local_coordinates(anchor, parameters)
        # A distribution of fragment cut positions, not a search radius setting.
        cut_offset = rng.uniform(0.02, 0.22)
        center = start_along + cut_offset
        parameters["crossing_cut_offset_m"] = cut_offset
        angle = math.radians(rng.uniform(4.0, 14.0))
    else:
        segment = rng.choice(scene["segments"])
        anchor = segment["points"][len(segment["points"]) // 2]
        center, _ = _local_coordinates(anchor, parameters)
        angle = math.radians(rng.uniform(35.0, 65.0) if raised else rng.uniform(5.0, 11.0))
    track = _track_index(anchor, parameters)
    half_length = rng.uniform(11.0, 18.0)
    clearance = rng.uniform(4.5, 7.0) if raised else 0.0
    angle *= rng.choice((-1.0, 1.0))
    rotation = parameters["rotation_rad"]
    cross_direction = [-math.sin(rotation), math.cos(rotation), 0.0]

    def crossing_point(along: float) -> list[float]:
        point = _point(along, track, parameters)
        transverse = (along - center) * math.tan(angle)
        return [
            point[0] + cross_direction[0] * transverse,
            point[1] + cross_direction[1] * transverse,
            point[2] + clearance,
        ]

    # Explicitly include the crossing point; no nearest-vertex approximation.
    left = _sample_interval(center - half_length, center, parameters["candidate_spacing_m"])
    right = _sample_interval(center, center + half_length, parameters["candidate_spacing_m"])
    if near_gap:
        incoming_id, outgoing_id = _opaque_id(rng), _opaque_id(rng)
        scene["segments"].extend([
            {"id": incoming_id, "points": [crossing_point(value) for value in left]},
            {"id": outgoing_id, "points": [crossing_point(value) for value in right]},
        ])
        scene["connections"].append({
            "id": _opaque_id(rng), "source": incoming_id, "target": outgoing_id,
        })
    else:
        scene["segments"].append({
            "id": _opaque_id(rng),
            "points": [crossing_point(value) for value in [*left, *right[1:]]],
        })
    for value in _sample_interval(
        center - half_length, center + half_length, parameters["observation_spacing_m"]
    ):
        scene["observations"].append([
            coordinate + rng.gauss(0.0, parameters["observation_noise_std_m"])
            for coordinate in crossing_point(value)
        ])
    rng.shuffle(scene["observations"])
    parameters.update(
        crossing_angle_rad=angle,
        crossing_center_along_m=center,
        crossing_half_length_m=half_length,
        crossing_clearance_m=clearance,
        crossing_position_m=crossing_point(center),
    )


def generate_study_layout(seed: int, family_index: int = 0) -> list[dict[str, Any]]:
    """Generate one independent layout group with the 12 frozen condition names.

    ``truth.variant`` is normal/gap/wrong_connection/duplicate; ``truth.condition``
    is one of STUDY_VARIANTS. Only the unmodified input schema goes to a detector.
    Missing/sparse observation cases preserve exactly the corresponding baseline
    geometry and connections after identity relabeling. This function does not
    decide whether a seed belongs to development, validation, or a frozen test.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(family_index, bool) or not isinstance(family_index, int) or family_index < 0:
        raise ValueError("family_index must be a nonnegative integer")
    rng = random.Random(f"railway-track-study-v1:{seed}:{family_index}")
    parameters = _layout_parameters(family_index % 6, rng)
    parameters["family_index"] = family_index % 6
    segments, connections, observations = _make_layout(parameters, rng)
    layout_id = _opaque_id(rng)
    base = {
        variant: _make_variant(
            segments, connections, observations, parameters, variant, layout_id, rng
        )
        for variant in ("normal", "gap", "wrong_connection", "duplicate")
    }
    cases = []
    for condition in STUDY_VARIANTS:
        if condition.startswith("wrong_connection"):
            variant = "wrong_connection"
        elif condition.startswith("gap"):
            variant = "gap"
        elif condition == "duplicate":
            variant = "duplicate"
        else:
            variant = "normal"
        case = copy.deepcopy(base[variant])
        scene = case["input"]
        case["truth"]["condition"] = condition
        if condition.endswith("no_observations"):
            scene["observations"] = []
            case["truth"]["parameters"]["retained_observation_fraction"] = 0.0
        elif condition.endswith("sparse_observations"):
            fraction = rng.uniform(0.04, 0.12)
            retained = max(1, int(len(scene["observations"]) * fraction))
            scene["observations"] = rng.sample(scene["observations"], retained)
            case["truth"]["parameters"]["retained_observation_fraction"] = (
                retained / len(observations)
            )
        elif condition == "normal_endpoint_jitter":
            amplitude = rng.uniform(0.006, 0.012)
            for segment in scene["segments"]:
                for endpoint in (segment["points"][0], segment["points"][-1]):
                    for axis in range(3):
                        endpoint[axis] += rng.uniform(-amplitude, amplitude)
            case["truth"]["parameters"]["extra_endpoint_jitter_bound_m"] = amplitude
        elif condition == "normal_shallow_crossing":
            _add_crossing(case, rng)
        elif condition == "normal_grade_separated_crossing":
            _add_crossing(case, rng, raised=True)
        elif condition == "gap_with_nearby_alternative":
            _add_crossing(case, rng, near_gap=True)
        cases.append(_reidentify(case, rng))
    rng.shuffle(cases)
    return cases
