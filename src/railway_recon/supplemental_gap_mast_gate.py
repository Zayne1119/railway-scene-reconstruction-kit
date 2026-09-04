from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def interval_separation(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return positive interval clearance, or zero when intervals touch/overlap."""

    return max(0.0, second[0] - first[1], first[0] - second[1])


def _corridor_bounds(
    model: Any,
    object_name: str,
    model_origin: np.ndarray,
    frame: dict[str, Any],
) -> dict[str, list[float]]:
    indexes = object_vertex_indices(model, object_name)
    if not len(indexes):
        raise ValueError(f"Object is absent from candidate mesh: {object_name}")
    vertices = model.vertices[indexes] + model_origin
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    local = vertices[:, :2] - frame_origin
    station = local @ along
    lateral = local @ cross
    return {
        "station_range_m": [float(np.min(station)), float(np.max(station))],
        "cross_range_m": [float(np.min(lateral)), float(np.max(lateral))],
        "z_range_m": [float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))],
    }


def _nearest_context(
    mast: dict[str, list[float]],
    contexts: list[tuple[str, dict[str, list[float]]]],
) -> dict[str, Any]:
    station = sum(mast["station_range_m"]) / 2.0
    candidates = [
        (name, bounds)
        for name, bounds in contexts
        if bounds["station_range_m"][0] - 0.25
        <= station
        <= bounds["station_range_m"][1] + 0.25
    ]
    if not candidates:
        raise ValueError("No station-overlapping context object for mast clearance")
    name, bounds = min(
        candidates,
        key=lambda item: interval_separation(
            tuple(mast["cross_range_m"]), tuple(item[1]["cross_range_m"])
        ),
    )
    return {
        "object_name": name,
        "bounds": bounds,
        "cross_clearance_m": interval_separation(
            tuple(mast["cross_range_m"]), tuple(bounds["cross_range_m"])
        ),
    }


def gate_supplemental_gap_masts(
    *,
    candidate_report_path: str | Path,
    candidate_obj_path: str | Path,
    candidate_origin_path: str | Path,
    frame_report_path: str | Path,
    photo_evidence_path: str | Path,
    fixed_view_manifest_path: str | Path,
    mesh_audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gap-mast gate: {output}")
    report = load_json(Path(candidate_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    photo = load_json(Path(photo_evidence_path))
    fixed = load_json(Path(fixed_view_manifest_path))
    mesh_audit = load_json(Path(mesh_audit_path))
    model = parse_obj_model(candidate_obj_path)
    origin = np.asarray(load_json(Path(candidate_origin_path))["origin_xyz"], dtype=np.float64)

    track_names = [
        name
        for name in model.faces_by_object
        if name.startswith("TRACKGRAPH--TRACK-") and name.endswith("-BED")
    ]
    platform_names = [
        name
        for name in model.faces_by_object
        if "OPPOSITE-PLATFORM-" in name
        and name.endswith(("-TOP", "-VOLUME"))
    ]
    track_context = [
        (name, _corridor_bounds(model, name, origin, frame)) for name in track_names
    ]
    platform_context = [
        (name, _corridor_bounds(model, name, origin, frame)) for name in platform_names
    ]
    photo_by_id = {
        str(item["candidate_id"]): item for item in photo.get("candidates", [])
    }
    records: list[dict[str, Any]] = []
    for item in report.get("records", []):
        candidate_id = str(item["candidate_id"])
        asset_id = str(item["asset_id"])
        mast = _corridor_bounds(model, asset_id, origin, frame)
        track = _nearest_context(mast, track_context)
        platform = _nearest_context(mast, platform_context)
        photo_record = photo_by_id.get(candidate_id)
        if photo_record is None:
            raise ValueError(f"Photo evidence is absent for {candidate_id}")
        trusted_views = int(photo_record["trusted_reviewable_view_count"])
        continuous = bool(item["fit"].get("continuous_shaft_observed"))
        gates = {
            "mesh_object_exists": asset_id in model.faces_by_object,
            "point_band_surface_support_passed": bool(item["point_to_mesh_support"]["passed"]),
            "at_least_two_trusted_photo_views": trusted_views >= 2,
            "occluded_middle_explicitly_documented": not continuous,
            "track_bed_plan_clearance_at_least_0_05m": track["cross_clearance_m"] >= 0.05,
            "opposite_platform_plan_clearance_at_least_0_05m": (
                platform["cross_clearance_m"] >= 0.05
            ),
        }
        records.append(
            {
                "candidate_id": candidate_id,
                "asset_id": asset_id,
                "mast_bounds": mast,
                "nearest_track_bed": track,
                "nearest_opposite_platform": platform,
                "trusted_reviewable_photo_views": trusted_views,
                "continuous_shaft_observed": continuous,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    global_gates = {
        "two_candidate_records": len(records) == 2,
        "all_candidate_records_passed": bool(records) and all(item["passed"] for item in records),
        "mesh_audit_passed": bool(mesh_audit.get("passed")),
        "six_fixed_views_rendered": int(fixed.get("view_count", 0)) == 6,
        "fixed_view_preset_is_gap_masts": fixed.get("preset") == "gap_masts",
    }
    passed = all(global_gates.values())
    result = {
        "schema_version": "railway.supplemental-gap-mast-gate.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_report": str(Path(candidate_report_path).resolve()),
        "candidate_obj": str(Path(candidate_obj_path).resolve()),
        "records": records,
        "global_gates": global_gates,
        "passed": passed,
        "status": (
            "photo_interpreted_candidate_accepted_formal_baseline_unchanged"
            if passed
            else "candidate_rejected"
        ),
        "limitations": [
            "The mast middle is confirmed by calibrated panoramas but is not continuously sampled by LiDAR.",
            "The exact H-section catalogue profile, cantilever and insulators remain unresolved.",
        ],
    }
    write_json(output, result)
    if not passed:
        raise ValueError("Supplemental gap-mast candidate failed acceptance gate")
    return result
