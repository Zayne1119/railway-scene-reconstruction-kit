"""Constructed development challenges to geometry and observation assumptions.

These are independent mathematical centreline/observation proxies, not customer
data, physical railway curvature standards, LiDAR, or a held-out evaluation. The
generator uses its own analytic namespace and never reads study protocols, seeds,
files, predictions, or detector policies. Legacy stress cases are a separate suite
and are not regenerated or included here.

The latent layout has a circular through-route and a separate, smooth auxiliary
route fragment. A gap removes a known arc interval; a wrong connection replaces
the through-route's actual target with the auxiliary fragment. Route identity is
known by construction but deliberately absent from detector input. Smooth forks
therefore challenge claims that geometry and unordered observations can establish
unique route membership; their labels are not defined by detector output.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

GUARDED_STRESS_NAMESPACE = "guarded-analytic-v1"
GUARDED_STRESS_CONDITIONS = (
    "normal_tight_curve",
    "gap_curved_continuation",
    "wrong_connection_smooth_fork",
    "wrong_connection_perturbed_bridge_clutter",
)


def _opaque_id(rng: random.Random) -> str:
    return f"{rng.getrandbits(128):032x}"


def _samples(start: float, end: float, spacing: float) -> list[float]:
    count = max(1, math.ceil((end - start) / spacing))
    return [start + (end - start) * index / count for index in range(count + 1)]


def _world(point: list[float], parameters: dict[str, Any]) -> list[float]:
    x, y, z = point
    angle = parameters["rotation_rad"]
    tx, ty, tz = parameters["translation_m"]
    return [
        tx + math.cos(angle) * x - math.sin(angle) * y,
        ty + math.sin(angle) * x + math.cos(angle) * y,
        tz + z,
    ]


def _main_point(distance: float, parameters: dict[str, Any]) -> list[float]:
    radius = parameters["curve_radius_m"]
    angle = distance / radius
    return _world(
        [radius * math.sin(angle), radius * (1.0 - math.cos(angle)),
         parameters["grade"] * distance],
        parameters,
    )


def _auxiliary_point(distance: float, parameters: dict[str, Any]) -> list[float]:
    # A separately defined quadratic route, not a connection bridging the main
    # route. Its visible start and tangent happen to offer a smooth-looking link.
    start_x, start_y = parameters["auxiliary_start_local_xy_m"]
    heading = parameters["auxiliary_heading_rad"]
    lateral = parameters["auxiliary_quadratic_coefficient_per_m"] * distance**2
    return _world(
        [
            start_x + math.cos(heading) * distance - math.sin(heading) * lateral,
            start_y + math.sin(heading) * distance + math.cos(heading) * lateral,
            parameters["grade"] * (start_x + distance),
        ],
        parameters,
    )


def _make_case(
    segments: list[list[list[float]]],
    target_index: int,
    observations: list[list[float]],
    parameters: dict[str, Any],
    condition: str,
    layout_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    segment_ids = [_opaque_id(rng) for _ in segments]
    connection_id = _opaque_id(rng)
    input_segments = [
        {"id": identity, "points": copy.deepcopy(points)}
        for identity, points in zip(segment_ids, segments)
    ]
    rng.shuffle(input_segments)
    variant = (
        "normal" if condition == GUARDED_STRESS_CONDITIONS[0]
        else "gap" if condition == GUARDED_STRESS_CONDITIONS[1]
        else "wrong_connection"
    )
    defects = []
    if variant != "normal":
        defects.append({
            "id": _opaque_id(rng),
            "kind": variant,
            "position": [(a + b) / 2 for a, b in zip(segments[0][-1],
                                                    segments[target_index][0])],
            "entity_ids": [connection_id],
        })
    return {
        "case_id": _opaque_id(rng),
        "layout_id": layout_id,
        "input": {
            "schema_version": "railway.synthetic-track-input.v1",
            "coordinate_system": "synthetic_local_m",
            "gauge_m": 1.435,
            "segments": input_segments,
            "connections": [{
                "id": connection_id, "source": segment_ids[0],
                "target": segment_ids[target_index],
            }],
            "observations": copy.deepcopy(observations),
        },
        "truth": {
            "schema_version": "railway.synthetic-track-truth.v1",
            "variant": variant,
            "condition": condition,
            "defects": defects,
            "parameters": copy.deepcopy(parameters),
        },
    }


def _bridge_clutter(
    start: list[float], end: list[float], parameters: dict[str, Any], rng: random.Random
) -> list[list[float]]:
    length = math.dist(start, end)
    delta = [b - a for a, b in zip(start, end)]
    horizontal = math.hypot(delta[0], delta[1])
    sideways = [-delta[1] / horizontal, delta[0] / horizontal, 0.0]
    spacing = rng.uniform(0.06, 0.09)
    bias = rng.uniform(0.012, 0.025) * rng.choice((-1.0, 1.0))
    wobble = rng.uniform(0.015, 0.035)
    sigma = rng.uniform(0.004, 0.012)
    phase = rng.uniform(-math.pi, math.pi)
    added = []
    for along in _samples(0.0, length, spacing):
        fraction = along / length
        lateral = bias + wobble * math.sin(4.0 * math.pi * fraction + phase)
        added.append([
            coordinate + shift * fraction + normal * lateral + rng.gauss(0.0, sigma)
            for coordinate, shift, normal in zip(start, delta, sideways)
        ])
    parameters.update(
        clutter_added_point_count=len(added),
        clutter_spacing_m=spacing,
        clutter_lateral_bias_m=bias,
        clutter_lateral_wobble_amplitude_m=wobble,
        clutter_lateral_wobble_phase_rad=phase,
        clutter_noise_std_m=sigma,
        observation_modification="add_perturbed_false_bridge_returns_only",
    )
    return added


def generate_guarded_stress_cases(seed: int = 20260908) -> list[dict[str, Any]]:
    """Return 24 new challenges (six analytic groups by four conditions).

    Conditions: untouched circular route; removal of a true circular interval;
    smooth but incorrect replacement target; that same incorrect relation with
    added biased, wavy, noisy observations along its false bridge. Observations
    otherwise remain identical within a group. All labels record constructions,
    not predictions. A challenge is allowed to defeat any proposed method.

    The namespace is independent from both the legacy stress generator and the
    registered study generator. No claim of globally disjoint integer seed ranges
    is needed: neither generator is called and their seeds are never inspected.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = []
    for family in range(6):
        rng = random.Random(f"{GUARDED_STRESS_NAMESPACE}:{seed}:{family}")
        radius = (24.0, 32.0, 42.0, 55.0, 68.0, 80.0)[family] * rng.uniform(0.94, 1.06)
        start_x, start_y = rng.uniform(6.0, 10.0), rng.uniform(0.25, 0.85)
        parameters = {
            "development_probe_namespace": GUARDED_STRESS_NAMESPACE,
            "development_probe_seed": seed,
            "analytic_family_index": family,
            "evaluation_role": "constructed_adversarial_development_probe_not_held_out_test",
            "curve_radius_m": radius,
            "source_arc_extent_rad": rng.uniform(1.05, 1.20),
            "target_arc_extent_rad": rng.uniform(0.65, 0.85),
            "removed_arc_angle_rad": rng.uniform(0.35, 0.75),
            "auxiliary_start_local_xy_m": [start_x, start_y],
            "auxiliary_heading_rad": math.atan2(start_y, start_x) + rng.uniform(-0.018, 0.018),
            "auxiliary_quadratic_coefficient_per_m": rng.uniform(-0.002, 0.002),
            "auxiliary_visible_length_m": rng.uniform(17.0, 23.0),
            "grade": rng.uniform(-0.015, 0.015),
            "rotation_rad": rng.uniform(-math.pi, math.pi),
            "translation_m": [rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0),
                              rng.uniform(-5.0, 5.0)],
            "candidate_spacing_m": rng.uniform(0.22, 0.35),
            "observation_spacing_m": rng.uniform(0.08, 0.14),
            "observation_noise_std_m": rng.uniform(0.003, 0.007),
            "route_identity_is_latent": True,
            "main_route_definition": "circular_through_route_parameterized_by_signed_arc_length",
            "auxiliary_route_definition": "independent_quadratic_fragment_not_a_main_route_branch",
            "identity_limitation": (
                "Route identities are known by analytic construction but are absent from input; "
                "smooth geometry and unordered observations need not uniquely determine identity."
            ),
        }
        left = -radius * parameters["source_arc_extent_rad"]
        right = radius * parameters["target_arc_extent_rad"]
        candidate_spacing = parameters["candidate_spacing_m"]
        auxiliary_length = parameters["auxiliary_visible_length_m"]
        segments = [
            [_main_point(value, parameters) for value in _samples(left, 0.0, candidate_spacing)],
            [_main_point(value, parameters) for value in _samples(0.0, right, candidate_spacing)],
            [_auxiliary_point(value, parameters)
             for value in _samples(0.0, auxiliary_length, candidate_spacing)],
        ]
        observation_spacing = parameters["observation_spacing_m"]
        noiseless = [
            *[_main_point(value, parameters)
              for value in _samples(left, right, observation_spacing)],
            *[_auxiliary_point(value, parameters)
              for value in _samples(0.0, auxiliary_length, observation_spacing)],
        ]
        observations = [
            [value + rng.gauss(0.0, parameters["observation_noise_std_m"]) for value in point]
            for point in noiseless
        ]
        rng.shuffle(observations)
        layout_id = _opaque_id(rng)
        for condition in GUARDED_STRESS_CONDITIONS:
            candidate = copy.deepcopy(segments)
            case_parameters = copy.deepcopy(parameters)
            case_observations = copy.deepcopy(observations)
            target = 1
            case_parameters["geometry_modification"] = "none"
            case_parameters["relation_modification"] = "none"
            case_parameters["observation_modification"] = "none"
            if condition == GUARDED_STRESS_CONDITIONS[1]:
                removed_arc = radius * parameters["removed_arc_angle_rad"]
                candidate[0] = [
                    _main_point(value, parameters)
                    for value in _samples(left, -removed_arc, candidate_spacing)
                ]
                case_parameters.update(
                    geometry_modification="remove_terminal_interval_of_circular_source_route",
                    removed_planar_arc_length_m=removed_arc,
                    removed_length_m=removed_arc * math.sqrt(1.0 + parameters["grade"]**2),
                    original_source_endpoint_m=copy.deepcopy(segments[0][-1]),
                )
            elif condition in GUARDED_STRESS_CONDITIONS[2:]:
                target = 2
                case_parameters.update(
                    relation_modification="replace_main_route_target_with_independent_auxiliary_route",
                    original_target_position_m=copy.deepcopy(segments[1][0]),
                    replacement_target_position_m=copy.deepcopy(segments[2][0]),
                )
                if condition == GUARDED_STRESS_CONDITIONS[3]:
                    case_observations.extend(_bridge_clutter(
                        candidate[0][-1], candidate[2][0], case_parameters, rng
                    ))
                    rng.shuffle(case_observations)
            cases.append(_make_case(
                candidate, target, case_observations, case_parameters, condition, layout_id, rng
            ))
    random.Random(f"{GUARDED_STRESS_NAMESPACE}:order:{seed}").shuffle(cases)
    return cases
