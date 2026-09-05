"""Adversarial development probes for connection-cause fallback assumptions.

These are constructed noisy centreline proxies, not customer or field data,
and not an additional independent held-out test. Labels inherit actual geometry
and relation modifications from the study generator. No detector or detector
policy is imported. A method may fail any probe; no prediction is made here.

Internal layout seeds are strictly negative, disjoint from the study protocol's
nonnegative seed domain. Registered development/validation/test seeds are not
read, generated, or inspected by this module.
"""

from __future__ import annotations

import copy
import math
import random
from itertools import pairwise
from typing import Any

from .synthetic_track_study_scene import _reidentify, generate_study_layout

ADAPTIVE_STRESS_CONDITIONS = (
    "gap_nearby_alternative_no_observations",
    "wrong_connection_bridge_clutter",
    "wrong_connection_alternative_context_occluded",
)


def _affected_endpoints(case: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    scene = case["input"]
    edge_id = case["truth"]["defects"][0]["entity_ids"][0]
    edge = next(item for item in scene["connections"] if item["id"] == edge_id)
    segments = {item["id"]: item["points"] for item in scene["segments"]}
    return segments[edge["source"]], segments[edge["target"]]


def _sample_end_region(
    points: list[list[float]], region_m: float, spacing_m: float, *, at_end: bool
) -> list[list[float]]:
    """Sample a fixed arc-length region independently of orientation and density."""
    ordered = list(reversed(points)) if at_end else points
    lengths = [0.0]
    for first, second in pairwise(ordered):
        lengths.append(lengths[-1] + math.dist(first, second))
    extent = min(region_m, lengths[-1])
    count = max(1, math.ceil(extent / spacing_m))
    result = []
    edge = 0
    for index in range(count + 1):
        distance = extent * index / count
        while edge + 1 < len(lengths) - 1 and lengths[edge + 1] < distance:
            edge += 1
        span = lengths[edge + 1] - lengths[edge]
        fraction = (distance - lengths[edge]) / span
        result.append([a + fraction * (b - a) for a, b in zip(ordered[edge], ordered[edge + 1])])
    return result


def _add_bridge_clutter(case: dict[str, Any], rng: random.Random) -> None:
    source, target = _affected_endpoints(case)
    start, end = source[-1], target[0]
    parameters = case["truth"]["parameters"]
    spacing = parameters["observation_spacing_m"] / 3.0
    sigma = parameters["observation_noise_std_m"] * rng.uniform(1.0, 2.0)
    count = max(2, math.ceil(math.dist(start, end) / spacing))
    # Deliberately false returns along the erroneous proposed connection. The
    # true track geometry and the incorrect relation are not changed.
    added = [
        [a + (b - a) * index / count + rng.gauss(0.0, sigma) for a, b in zip(start, end)]
        for index in range(count + 1)
    ]
    case["input"]["observations"].extend(added)
    rng.shuffle(case["input"]["observations"])
    parameters.update(
        clutter_added_point_count=len(added),
        clutter_spacing_m=spacing,
        clutter_noise_std_m=sigma,
    )


def _retain_endpoint_contexts(case: dict[str, Any], rng: random.Random) -> None:
    source, target = _affected_endpoints(case)
    parameters = case["truth"]["parameters"]
    extent = rng.uniform(3.0, 5.0)
    spacing = parameters["observation_spacing_m"]
    sigma = parameters["observation_noise_std_m"]
    proxy = [
        *_sample_end_region(source, extent, spacing, at_end=True),
        *_sample_end_region(target, extent, spacing, at_end=False),
    ]
    case["input"]["observations"] = [
        [coordinate + rng.gauss(0.0, sigma) for coordinate in point] for point in proxy
    ]
    rng.shuffle(case["input"]["observations"])
    parameters.update(
        visible_endpoint_extent_m=extent,
        endpoint_context_point_count=len(proxy),
    )


def generate_adaptive_stress_cases(seed: int = 20260907) -> list[dict[str, Any]]:
    """Return six layout groups with three adverse observation conditions each.

    1. A real gap near a crossing-route alternative, with all observations absent.
    2. A wrong connection with added artificial returns along its invalid bridge.
    3. A wrong connection where only its source-end and target-start regions are
       observed; the geometrically valid alternative continuation is occluded.

    Conditions are stored only in ``truth.condition``; detectors receive only
    ``input``. Every case has independently relabeled IDs and shuffled entities.
    The two wrong-connection probes retain exactly the same candidate geometry.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = []
    for family in range(6):
        layout_seed = -((seed + 1) * 6 + family + 1)
        rng = random.Random(f"adaptive-development-stress-v1:{seed}:{family}")
        generated = generate_study_layout(layout_seed, family)
        baseline = {case["truth"]["condition"]: case for case in generated}
        for condition in ADAPTIVE_STRESS_CONDITIONS:
            source_condition = (
                "gap_with_nearby_alternative"
                if condition == ADAPTIVE_STRESS_CONDITIONS[0]
                else "wrong_connection"
            )
            case = copy.deepcopy(baseline[source_condition])
            case["truth"]["condition"] = condition
            case["truth"]["parameters"]["stress_source_seed"] = layout_seed
            if condition == ADAPTIVE_STRESS_CONDITIONS[0]:
                case["input"]["observations"] = []
            elif condition == ADAPTIVE_STRESS_CONDITIONS[1]:
                _add_bridge_clutter(case, rng)
            else:
                _retain_endpoint_contexts(case, rng)
            cases.append(_reidentify(case, rng))
    random.Random(f"adaptive-development-stress-order-v1:{seed}").shuffle(cases)
    return cases
