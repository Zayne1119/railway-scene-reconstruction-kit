"""Independent analytic centreline fixtures for the P1 track-validation pilot.

These fixtures are short track-centreline polylines, not rail meshes. Observations
are noisy samples of the same mathematical centrelines: they are a measurement
*proxy*, not LiDAR, a sensor simulation, or independently acquired ground truth.
No customer scene, file, detector, or detector threshold is read by this module.
The six layouts are development fixtures, not an independent generalization test.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

_VARIANTS = ("normal", "gap", "wrong_connection", "duplicate")


def _opaque_id(rng: random.Random) -> str:
    return f"{rng.getrandbits(128):032x}"


def _distance(a: list[float], b: list[float]) -> float:
    return math.dist(a, b)


def _midpoint(a: list[float], b: list[float]) -> list[float]:
    return [(x + y) / 2.0 for x, y in zip(a, b)]


def _trim_end(points: list[list[float]], distance_m: float) -> list[list[float]]:
    """Remove real polyline length, interpolating within the final retained edge."""
    result = copy.deepcopy(points)
    remaining = distance_m
    while len(result) > 1:
        previous, end = result[-2:]
        length = _distance(previous, end)
        if remaining < length:
            fraction = (length - remaining) / length
            result[-1] = [a + fraction * (b - a) for a, b in zip(previous, end)]
            return result
        remaining -= length
        result.pop()
    raise ValueError("Requested gap removes the entire segment")


def _layout_parameters(index: int, rng: random.Random) -> dict[str, Any]:
    # Distinct analytic shapes, with jitter independent of any detector settings.
    length = (60.0, 84.0, 108.0, 132.0, 96.0, 144.0)[index] + rng.uniform(-5.0, 5.0)
    return {
        "track_count": 2 + index % 2,
        "segments_per_track": 2 + (index // 2) % 2,
        "length_m": length,
        "track_spacing_m": rng.uniform(3.8, 5.2),
        "quadratic_amplitude_m": (0.0, 2.5, -3.5, 0.0, 4.5, -2.0)[index],
        "sine_amplitude_m": (0.0, 0.0, 0.0, 1.8, 0.8, -1.5)[index],
        "grade": (0.0, 0.008, -0.012, 0.016, -0.006, 0.010)[index],
        "vertical_amplitude_m": (0.0, 0.0, 0.12, 0.0, 0.18, -0.15)[index],
        "rotation_rad": rng.uniform(-math.pi, math.pi),
        "translation_m": [rng.uniform(-150.0, 150.0) for _ in range(2)]
        + [rng.uniform(-10.0, 10.0)],
        "candidate_spacing_m": rng.uniform(0.40, 0.65),
        "observation_spacing_m": rng.uniform(0.18, 0.25),
        "observation_noise_std_m": rng.uniform(0.003, 0.006),
        "endpoint_jitter_bound_m": 0.003,
        # A tiny interval is omitted from one proxy track in alternate layouts.
        "occlusion_length_m": rng.uniform(0.3, 0.6) if index % 2 else 0.0,
        "occlusion_fraction": rng.uniform(0.20, 0.80),
    }


def _point(s: float, track: int, parameters: dict[str, Any]) -> list[float]:
    fraction = s / parameters["length_m"]
    x = s
    y = (track - (parameters["track_count"] - 1) / 2) * parameters["track_spacing_m"]
    y += parameters["quadratic_amplitude_m"] * fraction**2
    y += parameters["sine_amplitude_m"] * math.sin(2.0 * math.pi * fraction)
    z = parameters["grade"] * s
    z += parameters["vertical_amplitude_m"] * math.sin(math.pi * fraction)
    angle = parameters["rotation_rad"]
    tx, ty, tz = parameters["translation_m"]
    return [
        tx + math.cos(angle) * x - math.sin(angle) * y,
        ty + math.sin(angle) * x + math.cos(angle) * y,
        tz + z,
    ]


def _sample_interval(start: float, end: float, spacing: float) -> list[float]:
    count = max(1, math.ceil((end - start) / spacing))
    return [start + (end - start) * i / count for i in range(count + 1)]


def _make_layout(
    parameters: dict[str, Any], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, int]], list[list[float]]]:
    segments: list[dict[str, Any]] = []
    connections: list[dict[str, int]] = []
    observations: list[list[float]] = []
    parts = parameters["segments_per_track"]
    length = parameters["length_m"]
    boundaries = [0.0]
    boundaries.extend(
        length * (i + rng.uniform(-0.06, 0.06)) / parts for i in range(1, parts)
    )
    boundaries.append(length)
    parameters["segment_boundaries_m"] = boundaries
    occlusion_center = parameters["occlusion_fraction"] * length
    occlusion_half = parameters["occlusion_length_m"] / 2.0

    for track in range(parameters["track_count"]):
        for part in range(parts):
            samples = _sample_interval(
                boundaries[part], boundaries[part + 1], parameters["candidate_spacing_m"]
            )
            points = [_point(s, track, parameters) for s in samples]
            jitter = parameters["endpoint_jitter_bound_m"]
            for endpoint in (points[0], points[-1]):
                for axis in range(3):
                    endpoint[axis] += rng.uniform(-jitter, jitter)
            segment_index = len(segments)
            segments.append({"points": points})
            if part:
                connections.append({"source": segment_index - 1, "target": segment_index})
        for s in _sample_interval(0.0, length, parameters["observation_spacing_m"]):
            if track == 0 and occlusion_half > 0 and abs(s - occlusion_center) < occlusion_half:
                continue
            observations.append(
                [
                    value + rng.gauss(0.0, parameters["observation_noise_std_m"])
                    for value in _point(s, track, parameters)
                ]
            )
    rng.shuffle(observations)
    return segments, connections, observations


def _make_variant(
    base_segments: list[dict[str, Any]],
    base_connections: list[dict[str, int]],
    observations: list[list[float]],
    parameters: dict[str, Any],
    variant: str,
    layout_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    segments = copy.deepcopy(base_segments)
    connections = copy.deepcopy(base_connections)
    case_parameters = copy.deepcopy(parameters)
    defect: dict[str, Any] | None = None
    affected_connection: int | None = None
    affected_segments: list[int] = []

    if variant == "gap":
        affected_connection = rng.randrange(len(connections))
        connection = connections[affected_connection]
        removed_length = rng.uniform(0.25, 0.80)
        segment = segments[connection["source"]]
        segment["points"] = _trim_end(segment["points"], removed_length)
        case_parameters["removed_length_m"] = removed_length
        position = _midpoint(
            segment["points"][-1], segments[connection["target"]]["points"][0]
        )
        defect = {"kind": variant, "position": position}
    elif variant == "wrong_connection":
        affected_connection = rng.randrange(len(connections))
        connection = connections[affected_connection]
        parts = parameters["segments_per_track"]
        track, part = divmod(connection["source"], parts)
        neighbor = track + 1 if track + 1 < parameters["track_count"] else track - 1
        # Replace the actual target; the original valid relation is gone.
        connection["target"] = neighbor * parts + part + 1
        position = _midpoint(
            segments[connection["source"]]["points"][-1],
            segments[connection["target"]]["points"][0],
        )
        defect = {"kind": variant, "position": position}
    elif variant == "duplicate":
        original_index = rng.randrange(len(segments))
        original = segments[original_index]["points"]
        offset = rng.uniform(0.01, 0.02)
        angle = parameters["rotation_rad"]
        shift = [-math.sin(angle) * offset, math.cos(angle) * offset, 0.0]
        copied = [[value + delta for value, delta in zip(point, shift)] for point in original]
        affected_segments = [original_index, len(segments)]
        segments.append({"points": copied})
        case_parameters["duplicate_offset_m"] = offset
        middle = len(original) // 2
        defect = {"kind": variant, "position": _midpoint(original[middle], copied[middle])}

    segment_ids = [_opaque_id(rng) for _ in segments]
    connection_ids = [_opaque_id(rng) for _ in connections]
    input_segments = [
        {"id": segment_ids[i], "points": segment["points"]}
        for i, segment in enumerate(segments)
    ]
    input_connections = [
        {
            "id": connection_ids[i],
            "source": segment_ids[connection["source"]],
            "target": segment_ids[connection["target"]],
        }
        for i, connection in enumerate(connections)
    ]
    rng.shuffle(input_segments)
    rng.shuffle(input_connections)
    defects = []
    if defect is not None:
        entity_ids = (
            [connection_ids[affected_connection]]
            if affected_connection is not None
            else [segment_ids[i] for i in affected_segments]
        )
        defects.append({"id": _opaque_id(rng), **defect, "entity_ids": entity_ids})

    return {
        "case_id": _opaque_id(rng),
        "layout_id": layout_id,
        "input": {
            "schema_version": "railway.synthetic-track-input.v1",
            "coordinate_system": "synthetic_local_m",
            "gauge_m": 1.435,
            "segments": input_segments,
            "connections": input_connections,
            "observations": copy.deepcopy(observations),
        },
        "truth": {
            "schema_version": "railway.synthetic-track-truth.v1",
            "variant": variant,
            "defects": defects,
            "parameters": case_parameters,
        },
    }


def generate_pilot_cases(seed: int = 20260905) -> list[dict[str, Any]]:
    """Return six independent mathematical layouts with four variants each.

    Only pass a case's ``input`` to a detector. ``truth`` and ``layout_id`` are for
    evaluation/grouping. Every variant gets unrelated opaque entity IDs and a
    shuffled list order; observations are identical across its paired layout.
    Coordinates are generated local metres, with no physical location or CRS.
    The same integer seed reproduces the complete JSON-serializable result.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    cases = []
    for index in range(6):
        parameters = _layout_parameters(index, rng)
        segments, connections, observations = _make_layout(parameters, rng)
        layout_id = _opaque_id(rng)
        for variant in _VARIANTS:
            cases.append(
                _make_variant(
                    segments, connections, observations, parameters, variant, layout_id, rng
                )
            )
    rng.shuffle(cases)
    return cases
