from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .algorithms.mesh import ObjWriter
from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model


def transition_slab_mesh(
    frame: CorridorFrame,
    *,
    start_station_m: float,
    end_station_m: float,
    start_section: tuple[float, float, float, float],
    end_section: tuple[float, float, float, float],
    start_thickness_m: float,
    end_thickness_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    if end_station_m <= start_station_m:
        raise ValueError("Transition interval must be increasing")
    start_low_c, start_high_c, start_low_z, start_high_z = start_section
    end_low_c, end_high_c, end_low_z, end_high_z = end_section
    if start_high_c <= start_low_c or end_high_c <= end_low_c:
        raise ValueError("Transition section widths must be positive")
    if start_thickness_m <= 0 or end_thickness_m <= 0:
        raise ValueError("Transition thickness values must be positive")
    station = np.asarray(
        [start_station_m, start_station_m, end_station_m, end_station_m]
    )
    cross = np.asarray([start_low_c, start_high_c, end_high_c, end_low_c])
    top_z = np.asarray([start_low_z, start_high_z, end_high_z, end_low_z])
    xy = frame.world_xy(station, cross)
    top = np.column_stack((xy, top_z))
    bottom = top.copy()
    bottom[:2, 2] -= start_thickness_m
    bottom[2:, 2] -= end_thickness_m
    vertices = np.vstack((bottom, top))
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def _frame(value: dict[str, Any]) -> CorridorFrame:
    return CorridorFrame.from_json(value.get("frame", value))


def _prefix_scz(
    obj_path: Path,
    origin_path: Path,
    frame: CorridorFrame,
    prefixes: tuple[str, ...],
) -> np.ndarray:
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(origin_path)["origin_xyz"], dtype=np.float64)
    indexes = sorted(
        {
            int(index)
            for name in model.faces_by_object
            if name.startswith(prefixes)
            for index in object_vertex_indices(model, name)
        }
    )
    if not indexes:
        raise ValueError(f"No platform objects matched prefixes: {prefixes}")
    world = model.vertices[np.asarray(indexes, dtype=np.int64)] + origin
    station, cross = frame.project(world[:, 0], world[:, 1])
    return np.column_stack((station, cross, world[:, 2]))


def _edge_section(points: np.ndarray, *, edge: str, tolerance_m: float = 0.01) -> dict[str, float]:
    if edge not in {"maximum", "minimum"}:
        raise ValueError("edge must be maximum or minimum")
    station = float(np.max(points[:, 0]) if edge == "maximum" else np.min(points[:, 0]))
    selected = points[
        points[:, 0] >= station - tolerance_m
        if edge == "maximum"
        else points[:, 0] <= station + tolerance_m
    ]
    low_cross = float(np.min(selected[:, 1]))
    high_cross = float(np.max(selected[:, 1]))
    cross_tolerance = max(0.01, 0.01 * (high_cross - low_cross))
    low_z = float(np.max(selected[selected[:, 1] <= low_cross + cross_tolerance, 2]))
    high_z = float(np.max(selected[selected[:, 1] >= high_cross - cross_tolerance, 2]))
    return {
        "station_m": station,
        "low_cross_m": low_cross,
        "high_cross_m": high_cross,
        "low_top_z_m": low_z,
        "high_top_z_m": high_z,
    }


def transition_surface_residuals(
    points_scz: np.ndarray,
    *,
    start_station_m: float,
    end_station_m: float,
    start_section: tuple[float, float, float, float],
    end_section: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_scz, dtype=np.float64)
    amount = np.clip(
        (points[:, 0] - start_station_m) / (end_station_m - start_station_m), 0.0, 1.0
    )
    start_low_c, start_high_c, start_low_z, start_high_z = start_section
    end_low_c, end_high_c, end_low_z, end_high_z = end_section
    low_c = start_low_c * (1.0 - amount) + end_low_c * amount
    high_c = start_high_c * (1.0 - amount) + end_high_c * amount
    low_z = start_low_z * (1.0 - amount) + end_low_z * amount
    high_z = start_high_z * (1.0 - amount) + end_high_z * amount
    within = (
        (points[:, 0] >= start_station_m)
        & (points[:, 0] <= end_station_m)
        & (points[:, 1] >= low_c)
        & (points[:, 1] <= high_c)
    )
    lateral_amount = np.clip(
        (points[:, 1] - low_c) / np.maximum(high_c - low_c, 1.0e-9), 0.0, 1.0
    )
    predicted = low_z * (1.0 - lateral_amount) + high_z * lateral_amount
    return points[:, 2] - predicted, within


def _write_materials(path: Path) -> None:
    content = """# Adjacent platform transition material
newmtl AdjacentPlatformTransitionObserved
Kd 0.32 0.68 0.75
Ks 0.08 0.08 0.08
Ns 10
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_adjacent_platform_transition(
    *,
    current_obj: str | Path,
    current_origin: str | Path,
    adjacent_obj: str | Path,
    adjacent_origin: str | Path,
    cloud_path: str | Path,
    frame_report: str | Path,
    output_directory: str | Path,
    current_prefixes: tuple[str, ...],
    adjacent_prefixes: tuple[str, ...],
    seam_clearance_m: float = 0.002,
    current_thickness_m: float = 0.35,
    adjacent_thickness_m: float = 0.35,
) -> dict[str, Path]:
    current_model = Path(current_obj).resolve()
    current_origin_path = Path(current_origin).resolve()
    adjacent_model = Path(adjacent_obj).resolve()
    adjacent_origin_path = Path(adjacent_origin).resolve()
    cloud = Path(cloud_path).resolve()
    frame_path = Path(frame_report).resolve()
    output = Path(output_directory).resolve()
    for path in (
        current_model,
        current_origin_path,
        adjacent_model,
        adjacent_origin_path,
        cloud,
        frame_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = _frame(load_json(frame_path))
    current_points = _prefix_scz(
        current_model, current_origin_path, frame, current_prefixes
    )
    adjacent_points = _prefix_scz(
        adjacent_model, adjacent_origin_path, frame, adjacent_prefixes
    )
    current_edge = _edge_section(current_points, edge="maximum")
    adjacent_edge = _edge_section(adjacent_points, edge="minimum")
    start_station = float(current_edge["station_m"] + seam_clearance_m)
    end_station = float(adjacent_edge["station_m"] - seam_clearance_m)
    start_section = (
        current_edge["low_cross_m"],
        current_edge["high_cross_m"],
        current_edge["low_top_z_m"],
        current_edge["high_top_z_m"],
    )
    end_section = (
        adjacent_edge["low_cross_m"],
        adjacent_edge["high_cross_m"],
        adjacent_edge["low_top_z_m"],
        adjacent_edge["high_top_z_m"],
    )
    if end_station - start_station < 0.5:
        raise ValueError("Platform transition is too short for a stable width reconciliation")

    collected: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            station, cross = frame.project(x, y)
            broad = (
                (station >= start_station)
                & (station <= end_station)
                & (cross >= min(start_section[0], end_section[0]) - 0.2)
                & (cross <= max(start_section[1], end_section[1]) + 0.2)
                & (z >= 20.8)
                & (z <= 21.5)
            )
            if np.any(broad):
                collected.append(np.column_stack((station[broad], cross[broad], z[broad])))
    points = np.vstack(collected) if collected else np.empty((0, 3), dtype=np.float64)
    residual, within = transition_surface_residuals(
        points,
        start_station_m=start_station,
        end_station_m=end_station,
        start_section=start_section,
        end_section=end_section,
    )
    near = within & (np.abs(residual) <= 0.12)
    selected = np.abs(residual[near])
    p90 = float(np.percentile(selected, 90)) if len(selected) else float("inf")
    gates = {
        "minimum_point_count": int(np.count_nonzero(near)) >= 100,
        "residual_p90_within_80mm": p90 <= 0.08,
        "start_clearance_within_3mm": seam_clearance_m <= 0.003,
        "end_clearance_within_3mm": seam_clearance_m <= 0.003,
    }
    if not all(gates.values()):
        raise ValueError(f"Platform transition failed point support gates: {gates}")

    origin = np.asarray(load_json(current_origin_path)["origin_xyz"], dtype=np.float64)
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    vertices, faces = transition_slab_mesh(
        frame,
        start_station_m=start_station,
        end_station_m=end_station,
        start_section=start_section,
        end_section=end_section,
        start_thickness_m=current_thickness_m,
        end_thickness_m=adjacent_thickness_m,
    )
    writer = ObjWriter(origin, material_library=output_mtl.name)
    writer.add_mesh(
        "ADJACENT-NEXT-RIGHT-PLATFORM-TRANSITION",
        vertices,
        faces,
        "AdjacentPlatformTransitionObserved",
    )
    writer.write(output_obj)
    _write_materials(output_mtl)
    write_json(output_origin, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    mesh = audit_obj(output_obj)
    output_audit = output / "mesh_audit.json"
    write_json(output_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Platform transition failed mesh audit")
    output_report = output / "adjacent_platform_transition_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.adjacent-platform-transition.v1",
            "source_obj": str(current_model),
            "source_cloud": str(cloud),
            "adjacent_obj": str(adjacent_model),
            "frame_report": str(frame_path),
            "owned_longitudinal_interval_m": [start_station, end_station],
            "current_edge": current_edge,
            "adjacent_edge": adjacent_edge,
            "start_section": list(start_section),
            "end_section": list(end_section),
            "seam_clearance_m": seam_clearance_m,
            "point_count": len(points),
            "inlier_point_count": int(np.count_nonzero(near)),
            "absolute_residual_p90_m": p90,
            "gates": gates,
            "evidence_level": "observed",
            "confidence": 0.88,
            "subtype": "point_supported_width_transition",
            "geometry_method": "matched_boundary_sections_with_bilinear_transition",
            "output_obj": str(output_obj),
            "output_obj_sha256": sha256_file(output_obj),
            "mesh_audit_passed": True,
            "passed": True,
            "status": "candidate_built_not_promoted",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "mesh_audit": output_audit,
        "report": output_report,
    }
