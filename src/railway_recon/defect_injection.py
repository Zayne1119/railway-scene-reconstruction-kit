from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, sha256_json, write_json
from .track_graph import audit_track_graph


def _normalise_seed(graph: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Create a passing synthetic control while preserving the input topology."""

    seed = copy.deepcopy(graph)
    seed["status"] = "candidate"
    quality = settings["quality"]
    nominal_gauge = float(settings["nominal_gauge_m"])
    for source in seed.get("sources", []):
        source["review_status"] = "accepted"
        source["review_mode"] = "independent_review"
    for observation in seed.get("observations", []):
        observation["direction_dot"] = 1.0
        observation["frame_determinant"] = 1.0
        observation["gauge_m"] = nominal_gauge
        observation["rail_top_crosslevel_m"] = 0.0
        observation["rail_position_correction_m"] = 0.0
        observation["support_ratio"] = max(
            1.0, float(quality["minimum_observation_bin_coverage"])
        )
        observation["evidence_level"] = "observed"
        observation["review_status"] = "accepted"
    for edge in seed.get("edges", []):
        edge["evidence_level"] = "observed"
    for seam in seed.get("seams", []):
        seam["signed_chainage_gap_m"] = 0.0
        seam["lateral_difference_m"] = 0.0
        seam["vertical_difference_m"] = 0.0
        seam["endpoint_3d_difference_m"] = 0.0
        seam["tangent_dot"] = 1.0
        seam["required_correction_m"] = 0.0
    for event in seed.get("identity_events", []):
        event["order_crossing"] = False

    global_start = float(seed["route"]["source_chainage_start_m"])
    global_end = float(seed["route"]["source_chainage_end_m"])
    observations = list(seed.get("observations", []))
    for track in seed.get("tracks", []):
        members = sorted(
            [item for item in observations if item["global_track_id"] == track["id"]],
            key=lambda item: item["chainage_start_m"],
        )
        if not members:
            continue
        if float(members[0]["chainage_start_m"]) > global_start:
            members[0]["start_boundary_reason"] = "confirmed_turnout"
        if float(members[-1]["chainage_end_m"]) < global_end:
            members[-1]["end_boundary_reason"] = "confirmed_buffer_stop"
    return seed


def _first_interior_start(graph: dict[str, Any]) -> dict[str, Any]:
    global_start = float(graph["route"]["source_chainage_start_m"])
    first_by_track: dict[str, dict[str, Any]] = {}
    for item in graph["observations"]:
        track_id = str(item["global_track_id"])
        current = first_by_track.get(track_id)
        if current is None or float(item["chainage_start_m"]) < float(
            current["chainage_start_m"]
        ):
            first_by_track[track_id] = item
    candidates = [
        item
        for item in first_by_track.values()
        if float(item["chainage_start_m"]) > global_start
    ]
    if not candidates:
        raise ValueError("Defect seed has no interior track start")
    return min(candidates, key=lambda item: item["chainage_start_m"])


def _inject_duplicate_track(graph: dict[str, Any], _: dict[str, Any]) -> None:
    clone = copy.deepcopy(graph["tracks"][0])
    clone["id"] = "INJECTED-DUPLICATE-TRACK"
    clone["observation_ids"] = []
    clone["median_lateral_offset_m"] = float(clone["median_lateral_offset_m"]) + 0.05
    graph["tracks"].append(clone)


def _injections() -> list[tuple[str, str, Callable[[dict[str, Any], dict[str, Any]], None]]]:
    return [
        (
            "reversed_frame",
            "canonical_direction",
            lambda graph, settings: graph["observations"][0].update(
                direction_dot=float(settings["quality"]["minimum_direction_dot"]) - 0.1
            ),
        ),
        (
            "gauge_drift",
            "rail_gauge",
            lambda graph, settings: graph["observations"][0].update(
                gauge_m=float(settings["nominal_gauge_m"])
                + float(settings["quality"]["maximum_gauge_error_m"])
                + 0.01
            ),
        ),
        (
            "rail_top_crosslevel",
            "rail_top_crosslevel",
            lambda graph, settings: graph["observations"][0].update(
                rail_top_crosslevel_m=float(
                    settings["quality"]["maximum_rail_top_crosslevel_m"]
                )
                + 0.01
            ),
        ),
        (
            "low_longitudinal_support",
            "observation_support",
            lambda graph, settings: graph["observations"][0].update(
                support_ratio=max(
                    0.0,
                    float(settings["quality"]["minimum_observation_bin_coverage"])
                    - 0.01,
                )
            ),
        ),
        (
            "broken_segment_seam",
            "segment_seams",
            lambda graph, settings: graph["seams"][0].update(
                lateral_difference_m=float(
                    settings["quality"]["maximum_seam_lateral_error_m"]
                )
                + 0.01
            ),
        ),
        (
            "unconfirmed_track_birth",
            "interior_track_termination",
            lambda graph, settings: _first_interior_start(graph).update(
                start_boundary_reason=None
            ),
        ),
        ("overlapping_duplicate_track", "duplicate_tracks", _inject_duplicate_track),
    ]


def evaluate_defect_injection(
    graph_path: str | Path,
    settings_path: str | Path,
    output_path: str | Path,
) -> Path:
    graph_source = Path(graph_path).resolve()
    settings_source = Path(settings_path).resolve()
    graph = load_json(graph_source)
    settings = load_json(settings_source)
    if graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError("Unsupported TrackGraph")
    if settings.get("schema_version") != "railway.track-graph-settings.v1":
        raise ValueError("Unsupported TrackGraph settings")
    seed = _normalise_seed(graph, settings)
    control = audit_track_graph(seed, settings)
    if control["status"] != "pass":
        failures = [
            item["id"] for item in control["checks"] if item["status"] != "pass"
        ]
        raise ValueError(f"Normalised defect-injection control does not pass: {failures}")

    variants: list[dict[str, Any]] = []
    true_positive_count = 0
    total_failed_checks = 0
    for defect_id, expected_check, mutate in _injections():
        candidate = copy.deepcopy(seed)
        mutate(candidate, settings)
        result = audit_track_graph(candidate, settings)
        failed_checks = [
            str(item["id"]) for item in result["checks"] if item["status"] == "fail"
        ]
        detected = expected_check in failed_checks
        true_positive_count += int(detected)
        total_failed_checks += len(failed_checks)
        variants.append(
            {
                "defect_id": defect_id,
                "expected_check_id": expected_check,
                "detected": detected,
                "audit_status": result["status"],
                "failed_check_ids": failed_checks,
                "unexpected_failed_check_ids": [
                    item for item in failed_checks if item != expected_check
                ],
            }
        )
    count = len(variants)
    report = {
        "schema_version": "railway.track-defect-injection-evaluation.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "synthetic_known_defect_quality_gate_validation",
        "inputs": {
            "source_track_graph": str(graph_source),
            "source_track_graph_sha256": sha256_file(graph_source),
            "settings": str(settings_source),
            "settings_sha256": sha256_file(settings_source),
        },
        "control": {
            "derivation": "input_topology_with_audit_quantities_normalised_to_passing_values",
            "seed_sha256": sha256_json(seed),
            "status": control["status"],
            "passed": control["passed"],
        },
        "summary": {
            "injected_defect_count": count,
            "detected_defect_count": true_positive_count,
            "missed_defect_count": count - true_positive_count,
            "defect_recall": true_positive_count / count if count else None,
            "expected_check_precision": (
                true_positive_count / total_failed_checks if total_failed_checks else None
            ),
            "unexpected_failed_check_count": total_failed_checks - true_positive_count,
        },
        "variants": variants,
        "limitations": [
            "This validates deterministic gate behavior on injected faults, not field defect prevalence.",
            "The control is synthetic and is not evidence that the source TrackGraph passes.",
            "Thresholds are frozen project defaults, not statutory railway tolerances.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite defect injection report: {output}")
    write_json(output, report)
    return output
