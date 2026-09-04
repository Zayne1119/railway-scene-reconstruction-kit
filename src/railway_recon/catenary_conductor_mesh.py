from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms.mesh import ObjWriter, sweep_mesh
from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value


def circular_profile(radius_m: float, segment_count: int = 8) -> np.ndarray:
    """Return a closed circular sweep profile in lateral/vertical coordinates."""
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if segment_count < 6:
        raise ValueError("segment_count must be at least 6")
    angle = np.linspace(0.0, 2.0 * np.pi, segment_count, endpoint=False)
    return np.column_stack((radius_m * np.cos(angle), radius_m * np.sin(angle)))


def clip_observed_intervals(
    intervals: list[list[float]],
    core_range_m: list[float],
    *,
    minimum_length_m: float = 2.0,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Clip detector-supported spans to the modelling core without filling gaps."""
    low, high = (float(value) for value in core_range_m)
    if low >= high:
        raise ValueError("core_range_m must be increasing")
    kept: list[list[float]] = []
    rejected: list[dict[str, Any]] = []
    for source in intervals:
        source_low, source_high = (float(value) for value in source)
        clipped = [max(source_low, low), min(source_high, high)]
        length = clipped[1] - clipped[0]
        if length >= minimum_length_m:
            kept.append(clipped)
        else:
            rejected.append(
                {
                    "source_interval_m": [source_low, source_high],
                    "clipped_interval_m": clipped if length > 0.0 else None,
                    "clipped_length_m": max(length, 0.0),
                    "reason": "outside_core_or_below_minimum_supported_length",
                }
            )
    return kept, rejected


def conductor_centerline(
    frame: dict[str, Any],
    candidate: dict[str, Any],
    interval_m: list[float],
    *,
    reference_s_m: float,
    sample_spacing_m: float = 2.0,
) -> np.ndarray:
    """Build a point-fit centerline; no unsupported interpolation across intervals."""
    low, high = (float(value) for value in interval_m)
    if high <= low:
        raise ValueError("interval_m must be increasing")
    count = max(2, int(np.ceil((high - low) / sample_spacing_m)) + 1)
    s = np.linspace(low, high, count)
    cross = float(candidate["cross_position_m"]) + float(
        candidate.get("cross_slope_m_per_m", 0.0)
    ) * (s - reference_s_m)
    z = float(candidate["elevation_z_m"]) + float(
        candidate.get("elevation_slope_m_per_m", 0.0)
    ) * (s - reference_s_m)
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along_xy = np.asarray(frame["along_xy"], dtype=np.float64)
    cross_xy = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = (
        origin_xy[None, :]
        + s[:, None] * along_xy[None, :]
        + cross[:, None] * cross_xy[None, :]
    )
    return np.column_stack((xy, z))


def audit_conductor_track_alignment(
    context: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    """Audit lateral and vertical alignment for topology-approved conductors."""
    candidate_by_id = {
        str(item["candidate_id"]): item for item in context.get("candidates", [])
    }
    checks: list[dict[str, Any]] = []
    for item in gate.get("approved_conductors", []):
        candidate = candidate_by_id[str(item["candidate_id"])]
        lateral = float(candidate["nearest_track_center_distance_m"])
        height = float(candidate["height_above_reference_rail_m"])
        passed = (
            bool(candidate["nearest_track_accepted_for_model"])
            and lateral <= 0.30
            and 4.5 <= height <= 7.5
        )
        checks.append(
            {
                "candidate_id": item["candidate_id"],
                "linked_track_id": item["linked_track_id"],
                "lateral_track_center_offset_m": lateral,
                "height_above_reference_rail_m": height,
                "nearest_track_accepted_for_model": bool(
                    candidate["nearest_track_accepted_for_model"]
                ),
                "status": "pass" if passed else "fail",
            }
        )
    cross_positions = sorted(
        float(candidate_by_id[str(item["candidate_id"])]["cross_position_m"])
        for item in gate.get("approved_conductors", [])
    )
    pair_separations = [
        cross_positions[index] - cross_positions[index - 1]
        for index in range(1, len(cross_positions))
    ]
    passed = bool(checks) and all(item["status"] == "pass" for item in checks)
    return {
        "schema_version": "railway.catenary-conductor-track-alignment-audit.v1",
        "criteria": {
            "maximum_lateral_track_center_offset_m": 0.30,
            "contact_height_review_range_m": [4.5, 7.5],
        },
        "checks": checks,
        "minimum_pair_lateral_separation_m": min(pair_separations, default=None),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "limitations": [
            "The height check uses the current internal rail reference, not an external surveyed datum.",
            "This is a reconstruction consistency gate, not an electrification design compliance check.",
        ],
    }


def _write_materials(path: Path) -> None:
    content = """# Evidence-gated overhead conductor materials
newmtl ContactWireObserved
Kd 0.72 0.48 0.16
Ks 0.55 0.55 0.55
Ns 75

newmtl MessengerWireObserved
Kd 0.42 0.47 0.50
Ks 0.45 0.45 0.45
Ns 60
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_catenary_conductor_mesh(
    project: ProjectConfig,
    context_audit_path: str | Path,
    semantic_review_path: str | Path,
    topology_gate_path: str | Path,
    *,
    output_dir: str | Path,
    report_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build only topology-approved, point-supported conductor fragments."""
    context_path = project.resolve(context_audit_path)
    review_path = project.resolve(semantic_review_path)
    gate_path = project.resolve(topology_gate_path)
    context = load_json(context_path)
    review = load_json(review_path)
    gate = load_json(gate_path)
    if review.get("schema_version") != "railway.cable-candidate-semantic-review.v1":
        raise ValueError("Expected railway.cable-candidate-semantic-review.v1")
    if gate.get("schema_version") != "railway.catenary-topology-gate.v1":
        raise ValueError("Expected railway.catenary-topology-gate.v1")
    approved = list(gate.get("approved_conductors", []))
    if not approved:
        raise ValueError("No conductors passed the topology gate")

    candidate_by_id = {
        str(item["candidate_id"]): item for item in context.get("candidates", [])
    }
    frame = gate["frame"]
    core_range = [float(value) for value in gate["core_longitudinal_range_m"]]
    reference_s = float(np.mean(core_range))
    output_root = project.resolve(output_dir)
    reports = project.resolve(report_dir)
    output_obj = output_root / "catenary_conductors.obj"
    output_mtl = output_root / "catenary_conductors.mtl"
    output_origin = output_root / "model_origin.json"
    output_registry = reports / "catenary_conductor_assets.json"
    output_audit = reports / "catenary_conductor_mesh_audit.json"
    output_alignment_audit = reports / "catenary_conductor_track_alignment_audit.json"
    output_report = reports / "catenary_conductor_mesh_build.json"
    outputs = (
        output_obj,
        output_mtl,
        output_origin,
        output_registry,
        output_audit,
        output_alignment_audit,
        output_report,
    )
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite conductor outputs: {existing}")

    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    origin = np.asarray([origin_xy[0], origin_xy[1], 0.0], dtype=np.float64)
    writer = ObjWriter(origin, material_library=output_mtl.name)
    registry = new_registry(project.project_id)
    rejected_short_intervals: list[dict[str, Any]] = []
    built_fragments: list[dict[str, Any]] = []
    for approved_item in approved:
        candidate_id = str(approved_item["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Approved conductor is absent from context audit: {candidate_id}")
        conductor_class = str(approved_item["conductor_class"])
        if conductor_class not in {"contact_wire", "messenger_wire"}:
            raise ValueError(f"Unsupported approved conductor class: {conductor_class}")
        intervals, rejected = clip_observed_intervals(
            list(candidate.get("observed_longitudinal_intervals_m", [])),
            core_range,
            minimum_length_m=float(gate.get("minimum_fragment_length_m", 2.0)),
        )
        rejected_short_intervals.extend(
            {"candidate_id": candidate_id, **item} for item in rejected
        )
        if not intervals:
            raise ValueError(f"Approved conductor has no buildable core interval: {candidate_id}")
        radius = float(approved_item.get("render_radius_m", 0.018))
        profile = circular_profile(radius, int(approved_item.get("radial_segments", 8)))
        track_id = str(approved_item["linked_track_id"])
        prefix = "CONTACT-WIRE" if conductor_class == "contact_wire" else "MESSENGER-WIRE"
        for sequence, interval in enumerate(intervals, start=1):
            asset_id = f"{prefix}-{track_id}-OBS-{sequence:02d}"
            centerline = conductor_centerline(
                frame,
                candidate,
                interval,
                reference_s_m=reference_s,
                sample_spacing_m=float(gate.get("sample_spacing_m", 2.0)),
            )
            vertices, faces = sweep_mesh(centerline, profile)
            material = (
                "ContactWireObserved"
                if conductor_class == "contact_wire"
                else "MessengerWireObserved"
            )
            writer.add_mesh(asset_id, vertices, faces, material)
            sources = [
                {"kind": "point_cloud", "reference": f"{context_path}#{candidate_id}"},
                {
                    "kind": "panorama",
                    "reference": str(approved_item["evidence_sheet"]),
                },
                {"kind": "rule", "reference": str(gate_path)},
            ]
            registry["assets"].append(
                {
                    "id": asset_id,
                    "type": conductor_class,
                    "subtype": "point_supported_fragment",
                    "status": "candidate",
                    "evidence_level": "photo_interpreted",
                    "confidence": float(approved_item["confidence"]),
                    "sources": sources,
                    "parameters": {
                        "source_candidate_id": candidate_id,
                        "linked_track_id": track_id,
                        "longitudinal_interval_m": interval,
                        "cross_position_m": float(candidate["cross_position_m"]),
                        "elevation_z_m": float(candidate["elevation_z_m"]),
                        "render_radius_m": radius,
                        "unsupported_gaps_filled": False,
                    },
                    "geometry": {"file": str(output_obj), "node": asset_id},
                    "limitations": [
                        "The radius is enlarged for review visibility and is not a product diameter claim.",
                        "Only observed point-supported spans are built; gaps remain explicit.",
                    ],
                }
            )
            built_fragments.append(
                {
                    "asset_id": asset_id,
                    "candidate_id": candidate_id,
                    "linked_track_id": track_id,
                    "longitudinal_interval_m": interval,
                    "length_m": float(interval[1] - interval[0]),
                }
            )

    writer.write(output_obj)
    _write_materials(output_mtl)
    write_json(output_origin, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Conductor registry is invalid: " + "; ".join(errors))
    write_json(output_registry, registry)
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Conductor OBJ failed mesh audit")
    alignment_audit = audit_conductor_track_alignment(context, gate)
    write_json(output_alignment_audit, alignment_audit)
    if not alignment_audit["passed"]:
        raise ValueError("Conductor-to-track alignment audit failed")
    report = {
        "schema_version": "railway.catenary-conductor-mesh-build.v1",
        "project_id": project.project_id,
        "status": "point_supported_conductor_fragments_built_review_required",
        "source_context_audit": str(context_path),
        "source_semantic_review": str(review_path),
        "source_topology_gate": str(gate_path),
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_registry": str(output_registry),
        "output_mesh_audit": str(output_audit),
        "output_track_alignment_audit": str(output_alignment_audit),
        "model_sha256": sha256_file(output_obj),
        "asset_count": len(registry["assets"]),
        "built_fragments": built_fragments,
        "rejected_short_or_outside_core_intervals": rejected_short_intervals,
        "limitations": [
            "No line is added for candidates with unresolved conductor subtype.",
            "No gap is bridged without point support.",
            "No cantilever or messenger relation is inferred when the supporting mast lies outside the accepted core relation gate.",
        ],
    }
    write_json(output_report, report)
    return report
