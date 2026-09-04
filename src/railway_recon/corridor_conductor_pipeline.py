from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import UTC, datetime
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms.mesh import ObjWriter, sweep_mesh
from .algorithms.track_graph_mesh import (
    _sample_centerline,
    _sample_tangent,
    _track_centerline,
)
from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .multi_source import effective_camera_csv_path
from .registry import new_registry, summarize_registry, validate_registry_value
from .track_graph import (
    RouteSampler,
    load_track_graph_settings,
    validate_track_graph_bindings,
)


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "corridor-conductor-pipeline.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def select_conductor_sequence_options(
    options: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select evidence-valid cable candidates jointly along each track and wire."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for option in options:
        grouped[(str(option["track_id"]), str(option["wire_type"]))].append(option)
    selected: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    changed_from_local_greedy = 0
    for (track_id, wire_type), members in sorted(grouped.items()):
        stages_by_interval: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
        for member in members:
            interval = (
                float(member["chainage_start_m"]),
                float(member["chainage_end_m"]),
            )
            stages_by_interval[interval].append(member)
        stages = [
            sorted(
                stage,
                key=lambda item: (
                    float(item["selection_score"]),
                    str(item["source_candidate_id"]),
                ),
            )
            for _, stage in sorted(stages_by_interval.items())
        ]
        contiguous_runs: list[list[list[dict[str, Any]]]] = []
        for stage in stages:
            if not contiguous_runs:
                contiguous_runs.append([stage])
                continue
            previous = contiguous_runs[-1][-1][0]
            gap = float(stage[0]["chainage_start_m"]) - float(
                previous["chainage_end_m"]
            )
            if abs(gap) <= float(settings["maximum_adjacent_chainage_gap_m"]):
                contiguous_runs[-1].append(stage)
            else:
                contiguous_runs.append([stage])
        for run_index, run in enumerate(contiguous_runs, start=1):
            costs: list[list[float]] = []
            parents: list[list[int | None]] = []
            first_costs = [float(item["selection_score"]) for item in run[0]]
            costs.append(first_costs)
            parents.append([None] * len(run[0]))
            for stage_index in range(1, len(run)):
                previous_stage = run[stage_index - 1]
                stage = run[stage_index]
                stage_costs: list[float] = []
                stage_parents: list[int | None] = []
                for option in stage:
                    transitions = []
                    for previous_index, previous in enumerate(previous_stage):
                        cross_delta = abs(
                            float(option["wire_track_offset_m"])
                            - float(previous["wire_track_offset_m"])
                        )
                        height_delta = abs(
                            float(option["height_above_rail_m"])
                            - float(previous["height_above_rail_m"])
                        )
                        transition_cost = (
                            float(settings["sequence_continuity_cross_weight"])
                            * cross_delta
                            + float(settings["sequence_continuity_height_weight"])
                            * height_delta
                        )
                        transitions.append(
                            (
                                costs[-1][previous_index]
                                + float(option["selection_score"])
                                + transition_cost,
                                str(previous["source_candidate_id"]),
                                previous_index,
                            )
                        )
                    best = min(transitions)
                    stage_costs.append(float(best[0]))
                    stage_parents.append(int(best[2]))
                costs.append(stage_costs)
                parents.append(stage_parents)
            final_index = min(
                range(len(run[-1])),
                key=lambda index: (
                    costs[-1][index],
                    str(run[-1][index]["source_candidate_id"]),
                ),
            )
            chosen_indexes = [final_index]
            for stage_index in range(len(run) - 1, 0, -1):
                parent = parents[stage_index][chosen_indexes[-1]]
                if parent is None:
                    raise ValueError("Conductor sequence optimizer lost a parent state")
                chosen_indexes.append(parent)
            chosen_indexes.reverse()
            chosen = [
                stage[index] for stage, index in zip(run, chosen_indexes)
            ]
            for stage, item in zip(run, chosen):
                greedy = min(
                    stage,
                    key=lambda option: (
                        float(option["selection_score"]),
                        str(option["source_candidate_id"]),
                    ),
                )
                if item["source_candidate_id"] != greedy["source_candidate_id"]:
                    changed_from_local_greedy += 1
            selected.extend(chosen)
            run_records.append(
                {
                    "track_id": track_id,
                    "wire_type": wire_type,
                    "run_index": run_index,
                    "segment_count": len(run),
                    "chainage_start_m": float(run[0][0]["chainage_start_m"]),
                    "chainage_end_m": float(run[-1][0]["chainage_end_m"]),
                    "candidate_option_count": sum(len(stage) for stage in run),
                    "selected_cost": float(costs[-1][final_index]),
                }
            )
    selected.sort(
        key=lambda item: (
            str(item["track_id"]),
            str(item["wire_type"]),
            float(item["chainage_start_m"]),
        )
    )
    assignments: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in selected:
        assignments[
            (str(item["segment_id"]), str(item["source_candidate_id"]))
        ].append(str(item["id"]))
    duplicates = {
        f"{segment_id}:{candidate_id}": span_ids
        for (segment_id, candidate_id), span_ids in assignments.items()
        if len(span_ids) > 1
    }
    if duplicates:
        raise ValueError(
            "Conductor sequence optimizer reused source candidates: "
            + json.dumps(duplicates, sort_keys=True)
        )
    return selected, {
        "method": "dynamic_programming_local_evidence_plus_sequence_continuity",
        "option_count": len(options),
        "selected_span_count": len(selected),
        "contiguous_run_count": len(run_records),
        "changed_from_local_greedy_count": changed_from_local_greedy,
        "continuity_cross_weight": float(
            settings["sequence_continuity_cross_weight"]
        ),
        "continuity_height_weight": float(
            settings["sequence_continuity_height_weight"]
        ),
        "runs": run_records,
        "status": "joint_sequence_selection_complete",
    }


def select_conductor_spans(
    manifest: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    report_paths: dict[str, str],
    graph: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    segments = {str(item["id"]): item for item in manifest.get("segments", [])}
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graph.get("observations", []):
        observations[str(item["segment_id"])].append(item)
    wire_types = settings["wire_types"]
    decisions: list[dict[str, Any]] = []
    span_options: list[dict[str, Any]] = []
    for segment_id, report in reports.items():
        if segment_id not in segments:
            raise ValueError(f"Linear report is not in the segment manifest: {segment_id}")
        segment = segments[segment_id]
        length = float(segment["chainage_end_m"]) - float(segment["chainage_start_m"])
        frame = report["frame"]
        origin = np.asarray(frame["origin_xy"], dtype=np.float64)
        along = np.asarray(frame["along_xy"], dtype=np.float64)
        cross = np.asarray(frame["cross_xy"], dtype=np.float64)
        candidates = list(report.get("cable_candidates", []))
        for observation in observations.get(segment_id, []):
            track_id = str(observation["global_track_id"])
            rail_cross = float(observation["lateral_offset_m"])
            rail_z = float(observation["rail_top_z_m"])
            for wire_type, policy in wire_types.items():
                eligible: list[tuple[float, dict[str, Any], dict[str, float]]] = []
                for candidate in candidates:
                    cross_error = abs(float(candidate["cross_position_m"]) - rail_cross)
                    height = float(candidate["elevation_z"]) - rail_z
                    coverage = float(candidate.get("longitudinal_coverage", 0.0))
                    metrics = {
                        "cross_error_m": cross_error,
                        "height_above_rail_m": height,
                        "longitudinal_coverage": coverage,
                    }
                    passed = (
                        cross_error
                        <= float(settings["maximum_track_cross_difference_m"])
                        and float(policy["minimum_height_above_rail_m"])
                        <= height
                        <= float(policy["maximum_height_above_rail_m"])
                        and coverage >= float(settings["minimum_longitudinal_coverage"])
                    )
                    score = (
                        abs(height - float(policy["target_height_above_rail_m"]))
                        + float(settings["cross_error_weight"]) * cross_error
                        - float(settings["coverage_reward_weight"]) * coverage
                    )
                    decisions.append(
                        {
                            "segment_id": segment_id,
                            "track_id": track_id,
                            "wire_type": wire_type,
                            "candidate_id": str(candidate["id"]),
                            "metrics": metrics,
                            "passed_pair_gate": passed,
                            "local_selection_score": float(score) if passed else None,
                        }
                    )
                    if passed:
                        eligible.append((score, candidate, metrics))
                if not eligible:
                    continue
                for score, candidate, metrics in eligible:
                    wire_cross = float(candidate["cross_position_m"])
                    start_xy = origin + cross * wire_cross
                    end_xy = origin + along * length + cross * wire_cross
                    elevation = float(candidate["elevation_z"])
                    span_options.append(
                        {
                            "id": (
                                f"{wire_type.upper()}-{track_id}-{segment_id.upper()}"
                            ),
                            "segment_id": segment_id,
                            "track_id": track_id,
                            "wire_type": wire_type,
                            "source_candidate_id": str(candidate["id"]),
                            "source_report": report_paths[segment_id],
                            "chainage_start_m": float(segment["chainage_start_m"]),
                            "chainage_end_m": float(segment["chainage_end_m"]),
                            "start_xyz": [
                                float(start_xy[0]),
                                float(start_xy[1]),
                                elevation,
                            ],
                            "end_xyz": [
                                float(end_xy[0]),
                                float(end_xy[1]),
                                elevation,
                            ],
                            "cross_position_m": wire_cross,
                            "track_lateral_offset_m": rail_cross,
                            "wire_track_offset_m": wire_cross - rail_cross,
                            "height_above_rail_m": metrics["height_above_rail_m"],
                            "longitudinal_coverage": metrics[
                                "longitudinal_coverage"
                            ],
                            "selection_score": float(score),
                        }
                    )
    spans, sequence_selection = select_conductor_sequence_options(
        span_options, settings
    )
    selected_keys = {
        (
            str(span["segment_id"]),
            str(span["track_id"]),
            str(span["wire_type"]),
            str(span["source_candidate_id"]),
        )
        for span in spans
    }
    for decision in decisions:
        decision["selected_for_span"] = (
            str(decision["segment_id"]),
            str(decision["track_id"]),
            str(decision["wire_type"]),
            str(decision["candidate_id"]),
        ) in selected_keys
    return {
        "schema_version": "railway.corridor-conductor-selection.v1",
        "segment_count": len(segments),
        "linear_report_count": len(reports),
        "track_observation_segment_count": len(observations),
        "missing_linear_report_segment_ids": sorted(set(segments) - set(reports)),
        "decision_count": len(decisions),
        "span_option_count": len(span_options),
        "span_count": len(spans),
        "sequence_selection": sequence_selection,
        "decisions": decisions,
        "spans": spans,
        "status": "track_height_bound_conductor_fragments_selected",
        "limitations": [
            "Each span is an observed 50 m-class fragment; missing segments remain empty.",
            "Contact-wire and messenger-wire labels are inferred from lateral alignment and height above the recovered rail top.",
            "Fragments are not connected until their endpoint seam audit passes.",
        ],
    }


def bind_conductor_spans_to_track_centerlines(
    spans: list[dict[str, Any]],
    track_centerlines: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind evidence-selected wire fragments to canonical TrackGraph geometry.

    The linear-candidate reports use a local frame for each segment.  Those frames
    are appropriate for selecting a cable relative to a rail, but independently
    projecting every fragment through them creates avoidable seams.  Geometry
    therefore follows one global 3D centerline per track while retaining the
    selected lateral offset, height above rail and original evidence endpoints.
    """

    bound: list[dict[str, Any]] = []
    endpoint_adjustments: list[float] = []
    endpoint_z_adjustments: list[float] = []
    for source_span in spans:
        span = dict(source_span)
        track_id = str(span["track_id"])
        if track_id not in track_centerlines:
            raise ValueError(f"Conductor span references unknown track: {track_id}")
        chainages, centerline = track_centerlines[track_id]
        values = np.asarray(
            [float(span["chainage_start_m"]), float(span["chainage_end_m"])],
            dtype=np.float64,
        )
        tolerance = 1e-6
        if (
            float(values[0]) < float(chainages[0]) - tolerance
            or float(values[-1]) > float(chainages[-1]) + tolerance
        ):
            raise ValueError(
                f"Conductor span {span['id']} lies outside TrackGraph geometry"
            )
        sampled = _sample_centerline(chainages, centerline, values)
        lateral_offset = float(span["wire_track_offset_m"])
        for index, chainage in enumerate(values):
            tangent = _sample_tangent(
                chainages, centerline, float(chainage)
            )[:2]
            tangent /= np.linalg.norm(tangent)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            sampled[index, :2] += lateral_offset * normal
        source_start = np.asarray(span["start_xyz"], dtype=np.float64)
        source_end = np.asarray(span["end_xyz"], dtype=np.float64)
        height_above_rail = float(span["height_above_rail_m"])
        bound_start = [
            float(sampled[0, 0]),
            float(sampled[0, 1]),
            float(sampled[0, 2] + height_above_rail),
        ]
        bound_end = [
            float(sampled[1, 0]),
            float(sampled[1, 1]),
            float(sampled[1, 2] + height_above_rail),
        ]
        start_adjustment = float(
            np.linalg.norm(source_start[:2] - np.asarray(bound_start[:2]))
        )
        end_adjustment = float(
            np.linalg.norm(source_end[:2] - np.asarray(bound_end[:2]))
        )
        endpoint_adjustments.extend((start_adjustment, end_adjustment))
        start_z_adjustment = abs(float(source_start[2] - bound_start[2]))
        end_z_adjustment = abs(float(source_end[2] - bound_end[2]))
        endpoint_z_adjustments.extend((start_z_adjustment, end_z_adjustment))
        span["evidence_start_xyz"] = span["start_xyz"]
        span["evidence_end_xyz"] = span["end_xyz"]
        span["start_xyz"] = bound_start
        span["end_xyz"] = bound_end
        span["geometry_binding"] = "global_track_graph_centerline"
        span["geometry_binding_adjustment_m"] = {
            "start_xy": start_adjustment,
            "end_xy": end_adjustment,
            "start_z": start_z_adjustment,
            "end_z": end_z_adjustment,
        }
        bound.append(span)
    return bound, {
        "method": "global_3d_track_graph_plus_track_relative_wire_offset_and_height",
        "span_count": len(bound),
        "maximum_endpoint_xy_adjustment_m": (
            max(endpoint_adjustments) if endpoint_adjustments else 0.0
        ),
        "mean_endpoint_xy_adjustment_m": (
            float(np.mean(endpoint_adjustments)) if endpoint_adjustments else 0.0
        ),
        "maximum_endpoint_z_adjustment_m": (
            max(endpoint_z_adjustments) if endpoint_z_adjustments else 0.0
        ),
        "mean_endpoint_z_adjustment_m": (
            float(np.mean(endpoint_z_adjustments))
            if endpoint_z_adjustments
            else 0.0
        ),
        "source_evidence_endpoints_retained": True,
    }


def _circle_profile(radius: float, sides: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _audit_span_seams(
    spans: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        groups[(str(span["track_id"]), str(span["wire_type"]))].append(span)
    records: list[dict[str, Any]] = []
    for (track_id, wire_type), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: float(item["chainage_start_m"]))
        for left, right in pairwise(ordered):
            chainage_gap = float(right["chainage_start_m"]) - float(
                left["chainage_end_m"]
            )
            left_end = np.asarray(left["end_xyz"], dtype=np.float64)
            right_start = np.asarray(right["start_xyz"], dtype=np.float64)
            xy_error = float(np.linalg.norm(left_end[:2] - right_start[:2]))
            z_error = abs(float(left_end[2] - right_start[2]))
            adjacent = abs(chainage_gap) <= float(
                settings["maximum_adjacent_chainage_gap_m"]
            )
            passed = (
                adjacent
                and xy_error <= float(settings["maximum_seam_xy_error_m"])
                and z_error <= float(settings["maximum_seam_z_error_m"])
            )
            records.append(
                {
                    "track_id": track_id,
                    "wire_type": wire_type,
                    "left_span_id": left["id"],
                    "right_span_id": right["id"],
                    "chainage_gap_m": chainage_gap,
                    "endpoint_xy_error_m": xy_error,
                    "endpoint_z_error_m": z_error,
                    "adjacent": adjacent,
                    "passed": passed,
                    "action": (
                        "eligible_for_future_weld"
                        if passed
                        else "keep_fragments_separate"
                    ),
                }
            )
    return records


def snap_passing_conductor_seams(
    spans: list[dict[str, Any]], seams: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Close only seams that already satisfy the evidence-based audit gates."""

    repaired = [dict(span) for span in spans]
    by_id = {str(span["id"]): span for span in repaired}
    records: list[dict[str, Any]] = []
    for seam in seams:
        if not seam.get("passed"):
            continue
        left = by_id[str(seam["left_span_id"])]
        right = by_id[str(seam["right_span_id"])]
        left_end = np.asarray(left["end_xyz"], dtype=np.float64)
        right_start = np.asarray(right["start_xyz"], dtype=np.float64)
        joint = (left_end + right_start) / 2.0
        left.setdefault("pre_seam_repair_end_xyz", left["end_xyz"])
        right.setdefault("pre_seam_repair_start_xyz", right["start_xyz"])
        left["end_xyz"] = joint.tolist()
        right["start_xyz"] = joint.tolist()
        left["end_seam_repair"] = "snap_within_passed_audit_gate"
        right["start_seam_repair"] = "snap_within_passed_audit_gate"
        records.append(
            {
                "left_span_id": str(left["id"]),
                "right_span_id": str(right["id"]),
                "joint_xyz": joint.tolist(),
                "left_endpoint_adjustment_m": float(
                    np.linalg.norm(joint - left_end)
                ),
                "right_endpoint_adjustment_m": float(
                    np.linalg.norm(joint - right_start)
                ),
                "source_seam_xy_error_m": float(seam["endpoint_xy_error_m"]),
                "source_seam_z_error_m": float(seam["endpoint_z_error_m"]),
                "policy": "snap_only_if_original_seam_passed",
            }
        )
    adjustments = [
        value
        for record in records
        for value in (
            float(record["left_endpoint_adjustment_m"]),
            float(record["right_endpoint_adjustment_m"]),
        )
    ]
    return repaired, {
        "method": "midpoint_snap_after_xy_z_gate",
        "repaired_seam_count": len(records),
        "maximum_endpoint_adjustment_m": max(adjustments) if adjustments else 0.0,
        "records": records,
        "status": "only_pre_audit_passing_seams_closed",
    }


def _write_materials(path: Path) -> None:
    content = """# Corridor conductor candidate materials
newmtl ContactWireCandidate
Kd 0.70 0.95 0.15
Ke 0.04 0.06 0.01
Ns 45

newmtl MessengerWireCandidate
Kd 0.95 0.55 0.08
Ke 0.06 0.03 0.01
Ns 45
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run_corridor_conductor_pipeline(
    project: ProjectConfig,
    graph_value: str | Path,
    output_dir_value: str | Path,
    output_name: str,
    *,
    settings_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one safe path component")
    graph_path = project.resolve(graph_value)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    graph = load_json(graph_path)
    binding_failures = validate_track_graph_bindings(project, graph)
    if binding_failures:
        raise ValueError(f"TrackGraph project bindings are stale: {binding_failures}")
    settings_path = project.resolve(settings_value) if settings_value else None
    settings = load_json(settings_path) if settings_path else _resource_settings()
    if settings.get("schema_version") != "railway.corridor-conductor-settings.v1":
        raise ValueError("Unsupported corridor-conductor settings")
    manifest_path = project.workspace_path("segment_manifest")
    manifest = load_json(manifest_path)
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for segment in manifest.get("segments", []):
        segment_id = str(segment["id"])
        path = project.workspace_path("reports") / f"{segment_id}_linear_candidates.json"
        if path.is_file():
            reports[segment_id] = load_json(path)
            report_paths[segment_id] = str(path)
    selection = select_conductor_spans(
        manifest, reports, report_paths, graph, settings
    )
    spans = selection["spans"]
    if not spans:
        raise ValueError("No conductor fragments passed the track-height binding gate")
    camera_path = effective_camera_csv_path(project)
    track_graph_settings = load_track_graph_settings(project)
    route = RouteSampler(
        camera_trajectory(load_camera_rows(camera_path)),
        float(track_graph_settings.get("route_frame_tangent_window_m", 5.0)),
    )
    track_build_config = load_json(
        project.resolve(project.value["algorithms"]["track_build"])
    )
    step = float(track_build_config["polyline_step_m"])
    required_track_ids = {str(span["track_id"]) for span in spans}
    observations_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in graph.get("observations", []):
        track_id = str(observation["global_track_id"])
        if track_id in required_track_ids:
            observations_by_track[track_id].append(observation)
    missing_track_ids = sorted(required_track_ids - observations_by_track.keys())
    if missing_track_ids:
        raise ValueError(
            "Conductor candidates reference tracks without observations: "
            + ", ".join(missing_track_ids)
        )
    track_centerlines = {
        track_id: _track_centerline(route, members, step)
        for track_id, members in observations_by_track.items()
    }
    spans, geometry_binding = bind_conductor_spans_to_track_centerlines(
        spans, track_centerlines
    )
    selection["spans"] = spans
    selection["geometry_binding"] = geometry_binding
    selection["limitations"].append(
        "Rendered XYZ geometry follows global TrackGraph position and grade plus measured wire offsets; original local-frame evidence endpoints remain in each span record."
    )
    pre_repair_seams = _audit_span_seams(spans, settings)
    spans, seam_repair = snap_passing_conductor_seams(
        spans, pre_repair_seams
    )
    selection["spans"] = spans
    selection["seam_repair"] = seam_repair
    selection["limitations"].append(
        "Only adjacent joints already passing both XY and Z seam gates are midpoint-snapped; failed and nonadjacent joints remain open."
    )
    output_dir = project.resolve(output_dir_value)
    obj_path = output_dir / f"{output_name}.obj"
    mtl_path = output_dir / f"{output_name}.mtl"
    origin_path = output_dir / "model_origin.json"
    audit_path = output_dir / "mesh_audit.json"
    registry_path = output_dir / "asset_registry.json"
    selection_path = output_dir / "selection_report.json"
    seam_path = output_dir / "seam_audit.json"
    report_path = output_dir / "pipeline_report.json"
    outputs = (
        obj_path,
        mtl_path,
        origin_path,
        audit_path,
        registry_path,
        selection_path,
        seam_path,
        report_path,
    )
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite corridor conductor outputs: {existing}")
    first = spans[0]
    origin = np.asarray(
        [float(first["start_xyz"][0]), float(first["start_xyz"][1]), 0.0],
        dtype=np.float64,
    )
    writer = ObjWriter(origin, material_library=mtl_path.name)
    registry = new_registry(project.project_id)
    for span in spans:
        policy = settings["wire_types"][span["wire_type"]]
        profile = _circle_profile(
            float(policy["display_radius_m"]), int(settings["profile_sides"])
        )
        centerline = np.asarray(
            [span["start_xyz"], span["end_xyz"]], dtype=np.float64
        )
        vertices, faces = sweep_mesh(centerline, profile)
        material = (
            "ContactWireCandidate"
            if span["wire_type"] == "contact_wire"
            else "MessengerWireCandidate"
        )
        writer.add_mesh(str(span["id"]), vertices, faces, material)
        registry["assets"].append(
            {
                "id": str(span["id"]),
                "type": str(span["wire_type"]),
                "status": "candidate",
                "evidence_level": "rule_inferred",
                "confidence": float(settings["automatic_candidate_confidence"]),
                "chainage_m": (
                    float(span["chainage_start_m"])
                    + float(span["chainage_end_m"])
                )
                / 2.0,
                "sources": [
                    {
                        "kind": "point_cloud",
                        "reference": (
                            f"{span['source_report']}#{span['source_candidate_id']}"
                        ),
                    },
                    {
                        "kind": "rule",
                        "reference": f"{selection_path}#{span['id']}",
                        "note": "Bound to recovered track by lateral offset and rail-top height",
                    },
                ],
                "parameters": {
                    key: span[key]
                    for key in (
                        "segment_id",
                        "track_id",
                        "chainage_start_m",
                        "chainage_end_m",
                        "cross_position_m",
                        "track_lateral_offset_m",
                        "wire_track_offset_m",
                        "height_above_rail_m",
                        "longitudinal_coverage",
                    )
                },
                "geometry": {"file": str(obj_path), "node": str(span["id"])},
                "limitations": [
                    "Semantic wire type is height-rule interpreted, not photo reviewed.",
                    "Only evidence-consistent segment joints are snapped; discrepant joints remain open for review.",
                ],
            }
        )
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(
        origin_path,
        {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"},
    )
    write_json(selection_path, selection)
    seams = _audit_span_seams(spans, settings)
    adjacent_seams = [item for item in seams if item["adjacent"]]
    nonadjacent_seams = [item for item in seams if not item["adjacent"]]
    passing_seams = [item for item in adjacent_seams if item["passed"]]
    failing_adjacent_seams = [item for item in adjacent_seams if not item["passed"]]
    write_json(
        seam_path,
        {
            "schema_version": "railway.corridor-conductor-seam-audit.v1",
            "seam_count": len(seams),
            "adjacent_seam_count": len(adjacent_seams),
            "passing_seam_count": len(passing_seams),
            "failing_adjacent_seam_count": len(failing_adjacent_seams),
            "nonadjacent_fragment_pair_count": len(nonadjacent_seams),
            "pre_repair_seams": pre_repair_seams,
            "seam_repair": seam_repair,
            "seams": seams,
            "status": "passing_fragment_seams_closed_failures_retained",
        },
    )
    audit = audit_obj(obj_path)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(audit_path, audit)
    if not audit["passed"]:
        raise ValueError("Corridor conductor candidate OBJ failed mesh audit")
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Corridor conductor registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)
    by_type = {
        wire_type: sum(span["wire_type"] == wire_type for span in spans)
        for wire_type in settings["wire_types"]
    }
    result = {
        "schema_version": "railway.corridor-conductor-pipeline.v1",
        "project_id": project.project_id,
        "status": "candidate_fragments_written_review_required_not_formal_release",
        "formal_release": False,
        "canonical_registry_updated": False,
        "source_track_graph": str(graph_path),
        "source_track_graph_sha256": sha256_file(graph_path),
        "segment_manifest": str(manifest_path),
        "settings_source": str(settings_path) if settings_path else "bundled_default",
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "output_origin": str(origin_path),
        "output_mesh_audit": str(audit_path),
        "output_asset_registry": str(registry_path),
        "selection_report": str(selection_path),
        "seam_audit": str(seam_path),
        "output_obj_sha256": sha256_file(obj_path),
        "span_count": len(spans),
        "span_count_by_type": by_type,
        "sequence_selection": {
            key: value
            for key, value in selection["sequence_selection"].items()
            if key != "runs"
        },
        "geometry_binding": geometry_binding,
        "seam_repair": {
            key: value for key, value in seam_repair.items() if key != "records"
        },
        "adjacent_seam_count": len(adjacent_seams),
        "passing_seam_count": len(passing_seams),
        "failing_adjacent_seam_count": len(failing_adjacent_seams),
        "nonadjacent_fragment_pair_count": len(nonadjacent_seams),
        "limitations": selection["limitations"],
    }
    write_json(report_path, result)
    return result
