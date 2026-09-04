from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import summarize_registry
from .supplemental_mesh_refinement import write_refined_obj

RIGHT_MAIN_ROOF_OBJECTS = (
    "SEG2050-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-01-RUN-01",
    "SEG2100-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-01-RUN-01",
    "SEG2150-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-01-RUN-01",
)


def audit_canopy_group_seams(
    *,
    obj_path: str | Path,
    model_origin_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
    object_names: tuple[str, ...] = RIGHT_MAIN_ROOF_OBJECTS,
    boundary_tolerance_m: float = 0.003,
    maximum_station_gap_m: float = 0.006,
    maximum_cross_z_p90_m: float = 0.02,
) -> Path:
    """Audit adjacent roof-group boundaries in a shared corridor frame."""
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)

    members: list[dict[str, Any]] = []
    for name in object_names:
        if name not in model.faces_by_object:
            raise ValueError(f"Roof-group object is absent from model: {name}")
        vertices = model.vertices[object_vertex_indices(model, name)] + origin
        delta = vertices[:, :2] - frame_origin
        station = delta @ along
        lateral = delta @ cross
        members.append(
            {
                "object_name": name,
                "station": station,
                "cross_z": np.column_stack((lateral, vertices[:, 2])),
                "station_min_m": float(np.min(station)),
                "station_max_m": float(np.max(station)),
            }
        )
    members.sort(key=lambda item: float(item["station_min_m"]))

    seams: list[dict[str, Any]] = []
    for left, right in pairwise(members):
        left_boundary = left["cross_z"][
            np.abs(left["station"] - left["station_max_m"]) <= boundary_tolerance_m
        ]
        right_boundary = right["cross_z"][
            np.abs(right["station"] - right["station_min_m"]) <= boundary_tolerance_m
        ]
        if not len(left_boundary) or not len(right_boundary):
            raise ValueError("Roof-group boundary vertices are absent")
        pairwise_distances = np.linalg.norm(
            left_boundary[:, None, :] - right_boundary[None, :, :], axis=2
        )
        symmetric = np.concatenate(
            (
                np.min(pairwise_distances, axis=1),
                np.min(pairwise_distances, axis=0),
            )
        )
        station_gap = float(right["station_min_m"] - left["station_max_m"])
        p90 = float(np.percentile(symmetric, 90))
        passed = abs(station_gap) <= maximum_station_gap_m and p90 <= maximum_cross_z_p90_m
        seams.append(
            {
                "left_object": left["object_name"],
                "right_object": right["object_name"],
                "station_m": float(
                    (left["station_max_m"] + right["station_min_m"]) / 2.0
                ),
                "station_gap_m": station_gap,
                "left_boundary_vertex_count": len(left_boundary),
                "right_boundary_vertex_count": len(right_boundary),
                "cross_z_nearest_distance_p90_m": p90,
                "cross_z_nearest_distance_maximum_m": float(np.max(symmetric)),
                "left_boundary_cross_z_m": left_boundary[
                    np.lexsort((left_boundary[:, 1], left_boundary[:, 0]))
                ].tolist(),
                "right_boundary_cross_z_m": right_boundary[
                    np.lexsort((right_boundary[:, 1], right_boundary[:, 0]))
                ].tolist(),
                "passed": passed,
            }
        )

    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.canopy-group-seam-audit.v1",
            "obj": str(Path(obj_path).resolve()),
            "object_names": [item["object_name"] for item in members],
            "boundary_tolerance_m": boundary_tolerance_m,
            "maximum_station_gap_m": maximum_station_gap_m,
            "maximum_cross_z_p90_m": maximum_cross_z_p90_m,
            "seams": seams,
            "passed": all(item["passed"] for item in seams),
        },
    )
    return destination


def audit_right_canopy_union_seams(
    *,
    obj_path: str | Path,
    model_origin_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
    boundary_tolerance_m: float = 0.003,
    maximum_station_gap_m: float = 0.006,
    maximum_cross_z_p90_m: float = 0.02,
) -> Path:
    """Audit the union of all right-canopy roof pieces at ownership seams."""
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    prefixes = ("SEG2050-CANOPY--", "SEG2100-CANOPY--", "SEG2150-CANOPY--")
    members: list[dict[str, Any]] = []
    for prefix in prefixes:
        names = sorted(
            name
            for name in model.faces_by_object
            if name.startswith(prefix) and "RIGHT-CANOPY-ROOF-CANDIDATE" in name
        )
        if not names:
            raise ValueError(f"No right-canopy roof objects found for {prefix}")
        vertex_indices = np.unique(
            np.concatenate([object_vertex_indices(model, name) for name in names])
        )
        vertices = model.vertices[vertex_indices] + origin
        delta = vertices[:, :2] - frame_origin
        station = delta @ along
        lateral = delta @ cross
        members.append(
            {
                "segment_prefix": prefix,
                "object_names": names,
                "station": station,
                "cross_z": np.column_stack((lateral, vertices[:, 2])),
                "station_min_m": float(np.min(station)),
                "station_max_m": float(np.max(station)),
            }
        )
    members.sort(key=lambda item: float(item["station_min_m"]))

    seams: list[dict[str, Any]] = []
    for left, right in pairwise(members):
        left_boundary = left["cross_z"][
            np.abs(left["station"] - left["station_max_m"]) <= boundary_tolerance_m
        ]
        right_boundary = right["cross_z"][
            np.abs(right["station"] - right["station_min_m"]) <= boundary_tolerance_m
        ]
        if not len(left_boundary) or not len(right_boundary):
            raise ValueError("Right-canopy union boundary vertices are absent")
        distances = np.linalg.norm(
            left_boundary[:, None, :] - right_boundary[None, :, :], axis=2
        )
        symmetric = np.concatenate(
            (np.min(distances, axis=1), np.min(distances, axis=0))
        )
        station_gap = float(right["station_min_m"] - left["station_max_m"])
        p90 = float(np.percentile(symmetric, 90))
        passed = abs(station_gap) <= maximum_station_gap_m and p90 <= maximum_cross_z_p90_m
        seams.append(
            {
                "left_segment": left["segment_prefix"].split("-")[0],
                "right_segment": right["segment_prefix"].split("-")[0],
                "station_m": float(
                    (left["station_max_m"] + right["station_min_m"]) / 2.0
                ),
                "station_gap_m": station_gap,
                "left_boundary_vertex_count": len(left_boundary),
                "right_boundary_vertex_count": len(right_boundary),
                "cross_z_nearest_distance_p90_m": p90,
                "cross_z_nearest_distance_maximum_m": float(np.max(symmetric)),
                "left_boundary_cross_z_m": left_boundary[
                    np.lexsort((left_boundary[:, 1], left_boundary[:, 0]))
                ].tolist(),
                "right_boundary_cross_z_m": right_boundary[
                    np.lexsort((right_boundary[:, 1], right_boundary[:, 0]))
                ].tolist(),
                "passed": passed,
            }
        )

    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.right-canopy-union-seam-audit.v1",
            "obj": str(Path(obj_path).resolve()),
            "segments": [
                {
                    "segment_prefix": item["segment_prefix"],
                    "object_count": len(item["object_names"]),
                    "object_names": item["object_names"],
                }
                for item in members
            ],
            "boundary_tolerance_m": boundary_tolerance_m,
            "maximum_station_gap_m": maximum_station_gap_m,
            "maximum_cross_z_p90_m": maximum_cross_z_p90_m,
            "seams": seams,
            "passed": all(item["passed"] for item in seams),
            "metric_note": (
                "The seam is evaluated on the union of all segmented roof surfaces, "
                "because the same physical width may be partitioned between different "
                "surface objects on adjacent segments."
            ),
        },
    )
    return destination


def _merged_vertical_intervals(
    sections: list[dict[str, Any]],
    cross_m: float,
    *,
    vertical_merge_tolerance_m: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for section in sections:
        cross_min = float(section["cross_min_m"])
        cross_max = float(section["cross_max_m"])
        if cross_m < cross_min - 1.0e-9 or cross_m > cross_max + 1.0e-9:
            continue
        fraction = (cross_m - cross_min) / (cross_max - cross_min)
        bottom = float(section["bottom_z_at_min_cross_m"]) + fraction * (
            float(section["bottom_z_at_max_cross_m"])
            - float(section["bottom_z_at_min_cross_m"])
        )
        top = float(section["top_z_at_min_cross_m"]) + fraction * (
            float(section["top_z_at_max_cross_m"])
            - float(section["top_z_at_min_cross_m"])
        )
        intervals.append((min(bottom, top), max(bottom, top)))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lower, upper in intervals:
        if not merged or lower > merged[-1][1] + vertical_merge_tolerance_m:
            merged.append((lower, upper))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
    return merged


def compare_roof_cross_section_unions(
    left_sections: list[dict[str, Any]],
    right_sections: list[dict[str, Any]],
    *,
    sample_count: int = 401,
    vertical_merge_tolerance_m: float = 0.01,
    maximum_endpoint_p90_m: float = 0.03,
    maximum_coverage_mismatch_ratio: float = 0.01,
) -> dict[str, Any]:
    """Compare physical cross-section unions without penalising mesh partitions."""
    if sample_count < 3:
        raise ValueError("Cross-section comparison requires at least three samples")
    if not left_sections or not right_sections:
        raise ValueError("Both seam sides require cross-section members")
    cross_min = min(
        min(float(item["cross_min_m"]) for item in left_sections),
        min(float(item["cross_min_m"]) for item in right_sections),
    )
    cross_max = max(
        max(float(item["cross_max_m"]) for item in left_sections),
        max(float(item["cross_max_m"]) for item in right_sections),
    )
    samples = np.linspace(cross_min, cross_max, sample_count)
    endpoint_distances: list[float] = []
    coverage_mismatch_count = 0
    interval_count_mismatch_count = 0
    common_sample_count = 0
    for cross_m in samples:
        left = _merged_vertical_intervals(
            left_sections,
            float(cross_m),
            vertical_merge_tolerance_m=vertical_merge_tolerance_m,
        )
        right = _merged_vertical_intervals(
            right_sections,
            float(cross_m),
            vertical_merge_tolerance_m=vertical_merge_tolerance_m,
        )
        if bool(left) != bool(right):
            coverage_mismatch_count += 1
            continue
        if not left:
            continue
        common_sample_count += 1
        if len(left) != len(right):
            interval_count_mismatch_count += 1
        left_endpoints = np.asarray([value for item in left for value in item])
        right_endpoints = np.asarray([value for item in right for value in item])
        distances = np.abs(left_endpoints[:, None] - right_endpoints[None, :])
        endpoint_distances.extend(np.min(distances, axis=1).tolist())
        endpoint_distances.extend(np.min(distances, axis=0).tolist())
    if not endpoint_distances:
        raise ValueError("Seam sides have no common physical roof cross-section")
    values = np.asarray(endpoint_distances, dtype=np.float64)
    coverage_mismatch_ratio = coverage_mismatch_count / sample_count
    interval_count_mismatch_ratio = (
        interval_count_mismatch_count / common_sample_count
        if common_sample_count
        else 1.0
    )
    endpoint_p90 = float(np.percentile(values, 90))
    passed = (
        endpoint_p90 <= maximum_endpoint_p90_m
        and coverage_mismatch_ratio <= maximum_coverage_mismatch_ratio
    )
    return {
        "sample_count": sample_count,
        "common_sample_count": common_sample_count,
        "coverage_mismatch_count": coverage_mismatch_count,
        "coverage_mismatch_ratio": coverage_mismatch_ratio,
        "interval_count_mismatch_count": interval_count_mismatch_count,
        "interval_count_mismatch_ratio": interval_count_mismatch_ratio,
        "endpoint_nearest_distance_p50_m": float(np.percentile(values, 50)),
        "endpoint_nearest_distance_p90_m": endpoint_p90,
        "endpoint_nearest_distance_maximum_m": float(np.max(values)),
        "maximum_endpoint_p90_m": maximum_endpoint_p90_m,
        "maximum_coverage_mismatch_ratio": maximum_coverage_mismatch_ratio,
        "passed": passed,
    }


def _boundary_sections(
    *,
    model: Any,
    names: list[str],
    origin: np.ndarray,
    frame_origin: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
    boundary_station_m: float,
    boundary_tolerance_m: float,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for name in names:
        vertices = model.vertices[object_vertex_indices(model, name)] + origin
        delta = vertices[:, :2] - frame_origin
        station = delta @ along
        boundary = vertices[np.abs(station - boundary_station_m) <= boundary_tolerance_m]
        if not len(boundary):
            continue
        lateral = (boundary[:, :2] - frame_origin) @ cross
        cross_min = float(np.min(lateral))
        cross_max = float(np.max(lateral))
        width = cross_max - cross_min
        if width <= 1.0e-6:
            continue
        endpoint_tolerance = min(0.002, width * 0.05)
        at_min = boundary[np.abs(lateral - cross_min) <= endpoint_tolerance, 2]
        at_max = boundary[np.abs(lateral - cross_max) <= endpoint_tolerance, 2]
        if not len(at_min) or not len(at_max):
            continue
        sections.append(
            {
                "object_name": name,
                "cross_min_m": cross_min,
                "cross_max_m": cross_max,
                "bottom_z_at_min_cross_m": float(np.min(at_min)),
                "top_z_at_min_cross_m": float(np.max(at_min)),
                "bottom_z_at_max_cross_m": float(np.min(at_max)),
                "top_z_at_max_cross_m": float(np.max(at_max)),
            }
        )
    return sections


def audit_right_canopy_physical_seams(
    *,
    obj_path: str | Path,
    model_origin_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
    boundary_tolerance_m: float = 0.003,
) -> Path:
    """Audit physical roof unions while ignoring harmless internal partitions."""
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    prefixes = ("SEG2050-CANOPY--", "SEG2100-CANOPY--", "SEG2150-CANOPY--")
    members: list[dict[str, Any]] = []
    for prefix in prefixes:
        names = sorted(
            name
            for name in model.faces_by_object
            if name.startswith(prefix) and "RIGHT-CANOPY-ROOF-CANDIDATE" in name
        )
        indices = np.unique(
            np.concatenate([object_vertex_indices(model, name) for name in names])
        )
        vertices = model.vertices[indices] + origin
        station = (vertices[:, :2] - frame_origin) @ along
        members.append(
            {
                "segment": prefix.split("-")[0],
                "names": names,
                "station_min_m": float(np.min(station)),
                "station_max_m": float(np.max(station)),
            }
        )
    members.sort(key=lambda item: float(item["station_min_m"]))

    seams: list[dict[str, Any]] = []
    for left, right in pairwise(members):
        left_sections = _boundary_sections(
            model=model,
            names=left["names"],
            origin=origin,
            frame_origin=frame_origin,
            along=along,
            cross=cross,
            boundary_station_m=float(left["station_max_m"]),
            boundary_tolerance_m=boundary_tolerance_m,
        )
        right_sections = _boundary_sections(
            model=model,
            names=right["names"],
            origin=origin,
            frame_origin=frame_origin,
            along=along,
            cross=cross,
            boundary_station_m=float(right["station_min_m"]),
            boundary_tolerance_m=boundary_tolerance_m,
        )
        comparison = compare_roof_cross_section_unions(left_sections, right_sections)
        seams.append(
            {
                "left_segment": left["segment"],
                "right_segment": right["segment"],
                "station_gap_m": float(
                    right["station_min_m"] - left["station_max_m"]
                ),
                "left_sections": left_sections,
                "right_sections": right_sections,
                "comparison": comparison,
                "passed": comparison["passed"],
            }
        )
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.right-canopy-physical-seam-audit.v1",
            "obj": str(Path(obj_path).resolve()),
            "boundary_tolerance_m": boundary_tolerance_m,
            "seams": seams,
            "passed": all(item["passed"] for item in seams),
            "metric_note": (
                "Cross-section solids are compared after merging overlapping vertical "
                "intervals, so harmless internal surface partitions do not create a false gap."
            ),
        },
    )
    return destination


def evaluate_canopy_seam_candidate_gate(
    *,
    baseline_support: dict[str, Any],
    candidate_support: dict[str, Any],
    baseline_plane_audit: dict[str, Any],
    candidate_plane_audit: dict[str, Any],
    weld_report: dict[str, Any],
    physical_seam_audit: dict[str, Any],
    maximum_vertex_displacement_m: float = 0.2,
    maximum_coverage_drop_at_0_10m: float = 0.01,
) -> dict[str, Any]:
    """Fail closed when a cleaner seam sacrifices point-cloud evidence."""
    changed_objects = [str(value) for value in weld_report["changed_objects"]]
    baseline_by_name = {
        str(item["object_name"]): item for item in baseline_support["objects"]
    }
    candidate_by_name = {
        str(item["object_name"]): item for item in candidate_support["objects"]
    }
    missing_support = sorted(
        name
        for name in changed_objects
        if name not in baseline_by_name or name not in candidate_by_name
    )
    if missing_support:
        raise ValueError(f"Changed roof objects lack point-support records: {missing_support}")
    coverage_changes = []
    for name in changed_objects:
        baseline = float(baseline_by_name[name]["coverage_at_0_10m"])
        candidate = float(candidate_by_name[name]["coverage_at_0_10m"])
        coverage_changes.append(
            {
                "object_name": name,
                "baseline_coverage_at_0_10m": baseline,
                "candidate_coverage_at_0_10m": candidate,
                "coverage_change": candidate - baseline,
                "coverage_drop": max(0.0, baseline - candidate),
            }
        )
    maximum_coverage_drop = max(
        float(item["coverage_drop"]) for item in coverage_changes
    )
    baseline_failed = {
        str(item["object_name"])
        for item in baseline_plane_audit["records"]
        if not item["passed"]
    }
    candidate_failed = {
        str(item["object_name"])
        for item in candidate_plane_audit["records"]
        if not item["passed"]
    }
    new_plane_failures = sorted(candidate_failed - baseline_failed)
    measured_displacement = float(weld_report["maximum_vertex_displacement_m"])
    gates = {
        "physical_seams_passed": bool(physical_seam_audit["passed"]),
        "maximum_vertex_displacement_within_limit": (
            measured_displacement <= maximum_vertex_displacement_m
        ),
        "point_coverage_drop_within_limit": (
            maximum_coverage_drop <= maximum_coverage_drop_at_0_10m
        ),
        "no_new_plane_failures": not new_plane_failures,
    }
    accepted = all(gates.values())
    return {
        "schema_version": "railway.canopy-seam-candidate-gate.v1",
        "status": "accepted" if accepted else "rejected_fail_closed",
        "accepted": accepted,
        "changed_object_count": len(changed_objects),
        "maximum_vertex_displacement_m": measured_displacement,
        "maximum_vertex_displacement_limit_m": maximum_vertex_displacement_m,
        "maximum_coverage_drop_at_0_10m": maximum_coverage_drop,
        "maximum_coverage_drop_limit_at_0_10m": maximum_coverage_drop_at_0_10m,
        "new_plane_failures": new_plane_failures,
        "baseline_plane_pass_count": int(baseline_plane_audit["passed_count"]),
        "candidate_plane_pass_count": int(candidate_plane_audit["passed_count"]),
        "coverage_changes": sorted(
            coverage_changes,
            key=lambda item: float(item["coverage_drop"]),
            reverse=True,
        ),
        "gates": gates,
        "decision": (
            "promote_only_after_fixed_view_and_glb_review"
            if accepted
            else "stop_before_fixed_view_glb_and_web_promotion"
        ),
    }


def select_uniform_group_correction(
    plane_report: dict[str, Any],
    *,
    object_names: tuple[str, ...] = RIGHT_MAIN_ROOF_OBJECTS,
    maximum_correction_spread_m: float = 0.025,
    maximum_post_group_center_residual_m: float = 0.015,
) -> dict[str, Any]:
    """Select one vertical translation without applying unstable per-piece slope changes."""
    records = {str(item["object_name"]): item for item in plane_report["records"]}
    missing = sorted(set(object_names) - set(records))
    if missing:
        raise ValueError(f"Roof-group records are absent: {missing}")

    members: list[dict[str, Any]] = []
    corrections: list[float] = []
    weights: list[int] = []
    required_gates = (
        "minimum_inlier_count",
        "minimum_inlier_fraction",
        "residual_p90_within_80mm",
        "vertical_correction_within_200mm",
    )
    for name in object_names:
        record = records[name]
        failed_required = [gate for gate in required_gates if not record["gates"][gate]]
        if failed_required:
            raise ValueError(f"Roof-group member failed required gates: {name}: {failed_required}")
        correction = float(record["vertical_correction_at_center_m"])
        weight = int(record["inlier_point_count"])
        corrections.append(correction)
        weights.append(weight)
        members.append(
            {
                "object_name": name,
                "individual_plane_gate_passed": bool(record["passed"]),
                "individual_vertical_correction_m": correction,
                "fit_residual_p90_m": float(record["residual_p90_m"]),
                "fit_angle_change_deg": float(record["plane_angle_change_deg"]),
                "inlier_point_count": weight,
            }
        )

    correction_array = np.asarray(corrections, dtype=np.float64)
    uniform = float(np.average(correction_array, weights=np.asarray(weights, dtype=np.float64)))
    residuals = correction_array - uniform
    spread = float(np.ptp(correction_array))
    maximum_residual = float(np.max(np.abs(residuals)))
    if spread > maximum_correction_spread_m:
        raise ValueError(f"Roof-group correction spread is unsafe: {spread:.6f} m")
    if maximum_residual > maximum_post_group_center_residual_m:
        raise ValueError(
            f"Uniform roof-group correction leaves an unsafe center residual: {maximum_residual:.6f} m"
        )
    for member, residual in zip(members, residuals, strict=True):
        member["post_group_center_residual_m"] = float(residual)
    return {
        "object_names": list(object_names),
        "members": members,
        "uniform_vertical_correction_m": uniform,
        "correction_spread_m": spread,
        "maximum_post_group_center_residual_m": maximum_residual,
        "gates": {
            "required_point_and_residual_gates_passed": True,
            "correction_spread_within_limit": True,
            "post_group_center_residual_within_limit": True,
            "single_translation_preserves_segment_seam_offsets": True,
        },
        "status": "passed_uniform_translation_gate",
    }


def _update_registry(
    source_path: Path,
    output_path: Path,
    output_obj: Path,
    changed_objects: set[str],
    plane_report_path: Path,
    correction_m: float,
) -> dict[str, Any]:
    registry = load_json(source_path)
    assets: list[dict[str, Any]] = []
    for source_asset in registry.get("assets", []):
        asset = copy.deepcopy(source_asset)
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            geometry["file"] = str(output_obj)
        if str(asset.get("id")) in changed_objects:
            asset["status"] = "candidate_constrained_roof_group_fit"
            asset["evidence_level"] = "supplemental_point_cloud_group_constrained"
            asset["confidence"] = min(float(asset.get("confidence", 0.8)), 0.9)
            asset.setdefault("parameters", {}).update(
                {
                    "run09_uniform_vertical_correction_m": correction_m,
                    "run09_geometry_operation": "whole_object_vertical_translation",
                }
            )
            asset.setdefault("sources", []).append(
                {
                    "kind": "point_cloud",
                    "reference": str(plane_report_path),
                    "note": (
                        "Three segment members passed a shared vertical-translation gate; "
                        "no per-piece slope change was applied."
                    ),
                }
            )
        assets.append(asset)
    registry["assets"] = assets
    registry["release_id"] = "site_b_s2050_2200_supplement_v2_roof_group_candidate_run09"
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    write_json(output_path, registry)
    return registry


def build_constrained_canopy_group_candidate(
    *,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    plane_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_roof_group_candidate_run09"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "canopy_group_refinement_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    plane_report = load_json(Path(plane_report_path))
    gate = select_uniform_group_correction(plane_report)
    correction = float(gate["uniform_vertical_correction_m"])
    transformed = model.vertices.copy()
    changed_objects = set(RIGHT_MAIN_ROOF_OBJECTS)
    changed_indices: set[int] = set()
    for name_value in RIGHT_MAIN_ROOF_OBJECTS:
        if name_value not in model.faces_by_object:
            raise ValueError(f"Roof-group object is absent from model: {name_value}")
        indices = object_vertex_indices(model, name_value)
        transformed[indices, 2] += correction
        changed_indices.update(int(value) for value in indices)

    removed = write_refined_obj(
        source_obj,
        output_obj,
        transformed,
        output_mtl_name=output_mtl.name,
    )
    if removed:
        raise ValueError(f"Unexpected legacy objects were removed: {removed}")
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    registry = _update_registry(
        Path(source_registry),
        output_registry,
        output_obj,
        changed_objects,
        Path(plane_report_path).resolve(),
        correction,
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Constrained canopy-group candidate failed mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-group-refinement.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "plane_report": str(Path(plane_report_path).resolve()),
            "gate": gate,
            "transform": {
                "operation": "uniform_whole_object_vertical_translation",
                "changed_objects": list(RIGHT_MAIN_ROOF_OBJECTS),
                "changed_vertex_count": len(changed_indices),
                "vertical_correction_m": correction,
                "slope_change_applied": False,
            },
            "asset_count": int(registry["summary"]["asset_count"]),
            "mesh_audit": {
                "passed": True,
                "object_count": int(mesh_audit["object_count"]),
                "triangle_count": int(mesh_audit["triangle_count_after_fan_triangulation"]),
                "duplicate_face_count": int(mesh_audit["duplicate_face_count"]),
                "degenerate_triangle_count": int(mesh_audit["degenerate_triangle_count"]),
            },
            "status": "candidate_requires_point_support_seam_and_fixed_view_review",
            "withheld": [
                "SEG2100 surface-02 remains unchanged because its fitted correction and slope are unstable.",
                "SEG2150 surface-02 run-03 remains unchanged because its fitted slope change is unstable.",
            ],
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "registry": output_registry,
        "origin": output_origin,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }


def _surface_token(name: str) -> str:
    if "-SURFACE-" not in name:
        raise ValueError(f"Roof object has no surface token: {name}")
    return name.split("-SURFACE-", maxsplit=1)[1].split("-", maxsplit=1)[0]


def _boundary_indices(
    *,
    model: Any,
    object_name: str,
    vertices: np.ndarray,
    origin: np.ndarray,
    frame_origin: np.ndarray,
    along: np.ndarray,
    boundary_station_m: float,
    tolerance_m: float,
) -> np.ndarray:
    indices = object_vertex_indices(model, object_name)
    global_vertices = vertices[indices] + origin
    station = (global_vertices[:, :2] - frame_origin) @ along
    return indices[np.abs(station - boundary_station_m) <= tolerance_m]


def _warp_boundary_to_target_section(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    source_section: dict[str, Any],
    target_section: dict[str, float],
    target_station_m: float,
    origin: np.ndarray,
    frame_origin: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
) -> float:
    points = vertices[indices] + origin
    delta_xy = points[:, :2] - frame_origin
    station = delta_xy @ along
    lateral = delta_xy @ cross
    source_min = float(source_section["cross_min_m"])
    source_max = float(source_section["cross_max_m"])
    fraction = np.clip((lateral - source_min) / (source_max - source_min), 0.0, 1.0)
    source_bottom = float(source_section["bottom_z_at_min_cross_m"]) + fraction * (
        float(source_section["bottom_z_at_max_cross_m"])
        - float(source_section["bottom_z_at_min_cross_m"])
    )
    source_top = float(source_section["top_z_at_min_cross_m"]) + fraction * (
        float(source_section["top_z_at_max_cross_m"])
        - float(source_section["top_z_at_min_cross_m"])
    )
    thickness = source_top - source_bottom
    if np.any(thickness <= 1.0e-6):
        raise ValueError("Roof boundary section has zero thickness")
    vertical_fraction = np.clip((points[:, 2] - source_bottom) / thickness, 0.0, 1.0)

    target_cross = float(target_section["cross_min_m"]) + fraction * (
        float(target_section["cross_max_m"])
        - float(target_section["cross_min_m"])
    )
    target_bottom = float(target_section["bottom_z_at_min_cross_m"]) + fraction * (
        float(target_section["bottom_z_at_max_cross_m"])
        - float(target_section["bottom_z_at_min_cross_m"])
    )
    target_top = float(target_section["top_z_at_min_cross_m"]) + fraction * (
        float(target_section["top_z_at_max_cross_m"])
        - float(target_section["top_z_at_min_cross_m"])
    )
    target_z = target_bottom + vertical_fraction * (target_top - target_bottom)
    moved = points.copy()
    moved[:, :2] += (
        (target_station_m - station)[:, None] * along[None, :]
        + (target_cross - lateral)[:, None] * cross[None, :]
    )
    moved[:, 2] = target_z
    maximum_displacement = float(np.max(np.linalg.norm(moved - points, axis=1)))
    vertices[indices] = moved - origin
    return maximum_displacement


def _update_seam_registry(
    *,
    source_path: Path,
    output_path: Path,
    output_obj: Path,
    changed_objects: set[str],
    seam_evidence_path: Path,
) -> dict[str, Any]:
    registry = load_json(source_path)
    assets: list[dict[str, Any]] = []
    for source_asset in registry.get("assets", []):
        asset = copy.deepcopy(source_asset)
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            geometry["file"] = str(output_obj)
        if str(asset.get("id")) in changed_objects:
            asset["status"] = "candidate_boundary_profile_welded"
            asset.setdefault("parameters", {}).update(
                {
                    "run10_geometry_operation": "shared_boundary_profile_weld",
                    "run10_internal_partitions_preserved": True,
                }
            )
            asset.setdefault("sources", []).append(
                {
                    "kind": "geometry_seam_audit",
                    "reference": str(seam_evidence_path),
                    "note": (
                        "Adjacent 50 m roof pieces share one averaged physical boundary "
                        "profile; unsupported longitudinal roof coverage was not extended."
                    ),
                }
            )
        assets.append(asset)
    registry["assets"] = assets
    registry["release_id"] = "site_b_s2050_2200_canopy_seam_candidate_run10"
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    write_json(output_path, registry)
    return registry


def build_welded_right_canopy_seam_candidate(
    *,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    frame_report_path: str | Path,
    seam_evidence_path: str | Path,
    output_directory: str | Path,
    boundary_tolerance_m: float = 0.003,
) -> dict[str, Path]:
    """Weld only observed segment boundaries to one shared physical profile."""
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_canopy_seam_candidate_run10"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "canopy_seam_weld_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    transformed = model.vertices.copy()
    origin = np.asarray(load_json(Path(model_origin))["origin_xyz"], dtype=np.float64)
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    prefixes = ("SEG2050-CANOPY--", "SEG2100-CANOPY--", "SEG2150-CANOPY--")
    members: list[dict[str, Any]] = []
    for prefix in prefixes:
        names = sorted(
            object_name
            for object_name in model.faces_by_object
            if object_name.startswith(prefix)
            and "RIGHT-CANOPY-ROOF-CANDIDATE" in object_name
        )
        indices = np.unique(
            np.concatenate([object_vertex_indices(model, object_name) for object_name in names])
        )
        points = transformed[indices] + origin
        station = (points[:, :2] - frame_origin) @ along
        members.append(
            {
                "segment": prefix.split("-")[0],
                "names": names,
                "station_min_m": float(np.min(station)),
                "station_max_m": float(np.max(station)),
            }
        )
    members.sort(key=lambda item: float(item["station_min_m"]))

    welds: list[dict[str, Any]] = []
    changed_objects: set[str] = set()
    changed_indices: set[int] = set()
    maximum_displacement = 0.0
    for left, right in pairwise(members):
        left_station = float(left["station_max_m"])
        right_station = float(right["station_min_m"])
        target_station = (left_station + right_station) / 2.0
        left_sections = _boundary_sections(
            model=model,
            names=left["names"],
            origin=origin,
            frame_origin=frame_origin,
            along=along,
            cross=cross,
            boundary_station_m=left_station,
            boundary_tolerance_m=boundary_tolerance_m,
        )
        right_sections = _boundary_sections(
            model=model,
            names=right["names"],
            origin=origin,
            frame_origin=frame_origin,
            along=along,
            cross=cross,
            boundary_station_m=right_station,
            boundary_tolerance_m=boundary_tolerance_m,
        )
        left_by_surface = {_surface_token(item["object_name"]): item for item in left_sections}
        right_by_surface = {_surface_token(item["object_name"]): item for item in right_sections}
        if left_by_surface.keys() != right_by_surface.keys():
            raise ValueError("Adjacent roof boundaries do not expose matching surface classes")
        seam_members: list[dict[str, Any]] = []
        for token in sorted(left_by_surface):
            left_section = left_by_surface[token]
            right_section = right_by_surface[token]
            target_section = {
                key: (
                    float(left_section[key]) + float(right_section[key])
                )
                / 2.0
                for key in (
                    "cross_min_m",
                    "cross_max_m",
                    "bottom_z_at_min_cross_m",
                    "top_z_at_min_cross_m",
                    "bottom_z_at_max_cross_m",
                    "top_z_at_max_cross_m",
                )
            }
            side_records: list[dict[str, Any]] = []
            for section, station_value in (
                (left_section, left_station),
                (right_section, right_station),
            ):
                object_name = str(section["object_name"])
                indices = _boundary_indices(
                    model=model,
                    object_name=object_name,
                    vertices=transformed,
                    origin=origin,
                    frame_origin=frame_origin,
                    along=along,
                    boundary_station_m=station_value,
                    tolerance_m=boundary_tolerance_m,
                )
                if not len(indices):
                    raise ValueError(f"No boundary vertices found for {object_name}")
                displacement = _warp_boundary_to_target_section(
                    vertices=transformed,
                    indices=indices,
                    source_section=section,
                    target_section=target_section,
                    target_station_m=target_station,
                    origin=origin,
                    frame_origin=frame_origin,
                    along=along,
                    cross=cross,
                )
                maximum_displacement = max(maximum_displacement, displacement)
                changed_objects.add(object_name)
                changed_indices.update(int(index) for index in indices)
                side_records.append(
                    {
                        "object_name": object_name,
                        "changed_vertex_count": len(indices),
                        "maximum_displacement_m": displacement,
                    }
                )
            seam_members.append(
                {
                    "surface_token": token,
                    "target_section": target_section,
                    "members": side_records,
                }
            )
        welds.append(
            {
                "left_segment": left["segment"],
                "right_segment": right["segment"],
                "source_station_gap_m": right_station - left_station,
                "target_station_m": target_station,
                "surface_welds": seam_members,
            }
        )

    removed = write_refined_obj(
        source_obj,
        output_obj,
        transformed,
        output_mtl_name=output_mtl.name,
    )
    if removed:
        raise ValueError(f"Unexpected legacy objects were removed: {removed}")
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    registry = _update_seam_registry(
        source_path=Path(source_registry),
        output_path=output_registry,
        output_obj=output_obj,
        changed_objects=changed_objects,
        seam_evidence_path=Path(seam_evidence_path).resolve(),
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Welded canopy-seam candidate failed mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-seam-weld.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "operation": "shared_boundary_profile_weld",
            "welds": welds,
            "changed_objects": sorted(changed_objects),
            "changed_object_count": len(changed_objects),
            "changed_vertex_count": len(changed_indices),
            "maximum_vertex_displacement_m": maximum_displacement,
            "asset_count": int(registry["summary"]["asset_count"]),
            "mesh_audit": {
                "passed": True,
                "object_count": int(mesh_audit["object_count"]),
                "triangle_count": int(mesh_audit["triangle_count_after_fan_triangulation"]),
                "duplicate_face_count": int(mesh_audit["duplicate_face_count"]),
                "degenerate_triangle_count": int(mesh_audit["degenerate_triangle_count"]),
            },
            "geometry_limits": {
                "new_objects_added": 0,
                "unsupported_longitudinal_coverage_extended": False,
                "interior_surface_partitions_preserved": True,
            },
            "status": "candidate_requires_physical_seam_point_support_and_fixed_view_review",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "registry": output_registry,
        "origin": output_origin,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
