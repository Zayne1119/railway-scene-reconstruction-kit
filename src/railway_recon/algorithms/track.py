from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ProjectConfig
from ..geometry import CorridorFrame
from ..io import load_json, write_json
from ..registry import new_registry, summarize_registry, validate_registry_value
from .mesh import (
    ObjWriter,
    interpolate_polyline,
    oriented_box,
    rail_profile,
    sweep_mesh,
    write_track_materials,
)


def _safe_prefix(value: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", value.upper())


def build_parametric_track(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_json(project.resolve(project.value["algorithms"]["track_build"]))
    candidate_report = project.workspace_path("reports") / f"{segment_id}_rail_candidates.json"
    if not candidate_report.is_file():
        raise FileNotFoundError(candidate_report)
    evidence = load_json(candidate_report)
    pairs = evidence.get("rail_pairs", [])
    if not pairs:
        raise ValueError("No reviewed rail pair is available for parametric track generation")
    frame = CorridorFrame.from_json(evidence["frame"])
    start, end = [float(value) for value in evidence["longitudinal_range_m"]]
    step = float(config["polyline_step_m"])
    longitudinal = np.arange(start, end + step * 0.5, step)
    line_by_cross = {
        round(float(item["cross_position_m"]), 6): item for item in evidence.get("rail_lines", [])
    }

    output_dir = project.workspace_path("exports") / segment_id / "track"
    obj_path = output_dir / "track.obj"
    mtl_path = output_dir / "railway_model.mtl"
    origin_path = output_dir / "model_origin.json"
    report_path = project.workspace_path("reports") / f"{segment_id}_track_build.json"
    for path in (obj_path, mtl_path, origin_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    gauge = float(config["gauge_m"])
    track_lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, pair in enumerate(pairs, start=1):
        positions = [float(value) for value in pair["cross_positions_m"]]
        center_cross = float(np.mean(positions))
        if bool(config.get("constrain_nominal_gauge", True)):
            positions = [center_cross - gauge / 2.0, center_cross + gauge / 2.0]
        z_values = []
        for original in pair["cross_positions_m"]:
            record = line_by_cross.get(round(float(original), 6), {})
            z_values.append(record.get("median_z_m"))
        valid_z = [float(value) for value in z_values if value is not None]
        rail_z = float(np.median(valid_z)) if valid_z else float(np.mean(evidence["search_z_range_m"]))
        left_xy = frame.world_xy(longitudinal, np.full_like(longitudinal, positions[0]))
        right_xy = frame.world_xy(longitudinal, np.full_like(longitudinal, positions[1]))
        left = np.column_stack((left_xy, np.full_like(longitudinal, rail_z)))
        right = np.column_stack((right_xy, np.full_like(longitudinal, rail_z)))
        track_lines.append((f"TRACK-{index:04d}", left, right))

    first_center = (track_lines[0][1][0] + track_lines[0][2][0]) / 2.0
    origin = np.asarray([first_center[0], first_center[1], 0.0], dtype=np.float64)
    writer = ObjWriter(origin)
    assets: list[dict[str, Any]] = []
    build_stats: list[dict[str, Any]] = []
    prefix = _safe_prefix(segment_id)
    profile = rail_profile()
    for track_index, (track_id, left, right) in enumerate(track_lines, start=1):
        track_asset_id = f"{prefix}-TRACK-{track_index:04d}"
        for side, line in (("LEFT", left), ("RIGHT", right)):
            vertices, faces = sweep_mesh(line, profile)
            object_name = f"{track_asset_id}-RAIL-{side}"
            writer.add_mesh(object_name, vertices, faces, "RailSteel")
            assets.append(
                {
                    "id": object_name,
                    "type": "rail",
                    "subtype": side.lower(),
                    "status": "accepted",
                    "chainage_m": 0.0,
                    "evidence_level": "observed",
                    "confidence": float(config["default_observed_confidence"]),
                    "sources": [
                        {"kind": "point_cloud", "reference": f"report:{candidate_report.name}"}
                    ],
                    "parameters": {"profile": "generic_60kg_style", "gauge_m": gauge},
                    "geometry": {"node": object_name, "file": str(obj_path)},
                    "limitations": ["Baseline fit; review switches, joints and curved transitions"],
                }
            )

        centerline = (left + right) / 2.0
        length = float(np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1).sum())
        sleeper = config["sleeper"]
        spacing = float(sleeper["spacing_m"])
        distances = np.arange(spacing / 2.0, length, spacing)
        centers = interpolate_polyline(centerline, distances)
        for sleeper_index, center in enumerate(centers, start=1):
            tangent = frame.along_xy.tolist() + [0.0]
            top_z = center[2] - float(sleeper["top_below_rail_m"])
            vertices, faces = oriented_box(
                center,
                np.asarray(tangent),
                float(sleeper["length_m"]),
                float(sleeper["width_m"]),
                top_z - float(sleeper["height_m"]),
                top_z,
            )
            writer.add_mesh(
                f"{track_asset_id}-SLEEPER-{sleeper_index:04d}",
                vertices,
                faces,
                "SleeperConcrete",
            )
        bed = config["track_bed"]
        bed_profile = np.asarray(
            [
                [-float(bed["bottom_width_m"]) / 2.0, -float(bed["top_below_rail_m"]) - float(bed["depth_m"])],
                [float(bed["bottom_width_m"]) / 2.0, -float(bed["top_below_rail_m"]) - float(bed["depth_m"])],
                [float(bed["top_width_m"]) / 2.0, -float(bed["top_below_rail_m"])],
                [-float(bed["top_width_m"]) / 2.0, -float(bed["top_below_rail_m"])],
            ],
            dtype=np.float64,
        )
        vertices, faces = sweep_mesh(centerline, bed_profile)
        writer.add_mesh(f"{track_asset_id}-BED", vertices, faces, "Ballast")
        assets.extend(
            [
                {
                    "id": track_asset_id,
                    "type": "track",
                    "subtype": "standard_gauge",
                    "status": "accepted",
                    "chainage_m": 0.0,
                    "evidence_level": "observed",
                    "confidence": float(config["default_observed_confidence"]),
                    "sources": [{"kind": "point_cloud", "reference": f"report:{candidate_report.name}"}],
                    "parameters": {"gauge_m": gauge, "length_m": length},
                    "geometry": {"file": str(obj_path)},
                    "limitations": ["Candidate-derived baseline; manual acceptance required"],
                },
                {
                    "id": f"{track_asset_id}-SLEEPERS",
                    "type": "sleeper_group",
                    "subtype": "generated_regular_spacing",
                    "status": "accepted",
                    "chainage_m": 0.0,
                    "evidence_level": "rule_inferred",
                    "confidence": float(config["default_inferred_confidence"]),
                    "sources": [{"kind": "rule", "reference": "track_build.sleeper"}],
                    "parameters": {**sleeper, "count": int(len(centers))},
                    "geometry": {"file": str(obj_path)},
                    "limitations": ["Individual sleeper positions are regularized"],
                },
                {
                    "id": f"{track_asset_id}-BED",
                    "type": "track_bed",
                    "subtype": "parametric_ballast",
                    "status": "accepted",
                    "chainage_m": 0.0,
                    "evidence_level": "rule_inferred",
                    "confidence": float(config["default_inferred_confidence"]),
                    "sources": [{"kind": "rule", "reference": "track_build.track_bed"}],
                    "parameters": bed,
                    "geometry": {"node": f"{track_asset_id}-BED", "file": str(obj_path)},
                    "limitations": ["Cross-section is a configurable engineering approximation"],
                },
            ]
        )
        build_stats.append({"track_id": track_asset_id, "length_m": length, "sleeper_count": len(centers)})

    writer.write(obj_path)
    write_track_materials(mtl_path)
    write_json(origin_path, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})

    registry_path = project.workspace_path("asset_registry")
    registry = load_json(registry_path) if registry_path.is_file() else new_registry(project.project_id)
    known = {item["id"] for item in registry["assets"]}
    duplicates = [item["id"] for item in assets if item["id"] in known]
    if duplicates and not overwrite:
        raise ValueError(f"Asset registry already contains generated ids: {duplicates}")
    registry["assets"] = [item for item in registry["assets"] if item["id"] not in {a["id"] for a in assets}]
    registry["assets"].extend(assets)
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Generated registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)

    report = {
        "schema_version": "railway.parametric-track-build.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "source_candidates": str(candidate_report),
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "origin": str(origin_path),
        "vertex_count": writer.vertex_count,
        "face_count": writer.face_count,
        "asset_count_added": len(assets),
        "tracks": build_stats,
        "status": "baseline_model_manual_mesh_and_evidence_review_required",
    }
    write_json(report_path, report)
    return report

