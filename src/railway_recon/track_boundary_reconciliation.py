from __future__ import annotations

import copy
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import ObjModel, object_vertex_indices, parse_obj_model
from .registry import summarize_registry, validate_registry_value

CURRENT_ROOT = re.compile(r"^(TRACKGRAPH--TRACK-\d{4})-")
ADJACENT_ROOT = re.compile(r"^(TRACK-\d{4})-")


def _roots(model: ObjModel, pattern: re.Pattern[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name in model.faces_by_object:
        match = pattern.match(name)
        if match:
            result.setdefault(match.group(1), []).append(name)
    return result


def _group_indices(model: ObjModel, names: list[str]) -> np.ndarray:
    indexes = {
        int(index)
        for name in names
        for index in object_vertex_indices(model, name)
    }
    return np.asarray(sorted(indexes), dtype=np.int64)


def _rail_indices(model: ObjModel, names: list[str]) -> np.ndarray:
    rails = [name for name in names if "-RAIL-LEFT" in name or "-RAIL-RIGHT" in name]
    if len(rails) != 2:
        raise ValueError(f"Expected two rail objects in track group; got {rails}")
    return _group_indices(model, rails)


def _local_coordinates(
    vertices: np.ndarray,
    *,
    origin_xyz: np.ndarray,
    frame: dict[str, Any],
) -> np.ndarray:
    world = vertices + origin_xyz
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = world[:, :2] - frame_origin
    return np.column_stack((xy @ along, xy @ cross, world[:, 2]))


def _endpoint_signature(
    model: ObjModel,
    names: list[str],
    *,
    origin_xyz: np.ndarray,
    frame: dict[str, Any],
    seam_station_m: float,
    side: str,
    search_radius_m: float = 1.0,
) -> dict[str, float]:
    rail_names = [
        name for name in names if "-RAIL-LEFT" in name or "-RAIL-RIGHT" in name
    ]
    rail_indexes = _rail_indices(model, names)
    local = _local_coordinates(model.vertices[rail_indexes], origin_xyz=origin_xyz, frame=frame)
    near = local[np.abs(local[:, 0] - seam_station_m) <= search_radius_m]
    if not len(near):
        raise ValueError("Track rail geometry does not reach the requested boundary")
    edge_station = float(np.max(near[:, 0]) if side == "current" else np.min(near[:, 0]))
    edge = near[np.abs(near[:, 0] - edge_station) <= 0.02]
    return {
        "edge_station_m": edge_station,
        "track_center_cross_m": float(np.mean(edge[:, 1])),
        "rail_top_mean_z_m": float(np.mean([
            _rail_endpoint_top(
                model,
                name,
                origin_xyz=origin_xyz,
                frame=frame,
                seam_station_m=seam_station_m,
                side=side,
                search_radius_m=search_radius_m,
            )
            for name in rail_names
        ])),
    }


def _rail_endpoint_top(
    model: ObjModel,
    name: str,
    *,
    origin_xyz: np.ndarray,
    frame: dict[str, Any],
    seam_station_m: float,
    side: str,
    search_radius_m: float,
) -> float:
    indexes = object_vertex_indices(model, name)
    local = _local_coordinates(model.vertices[indexes], origin_xyz=origin_xyz, frame=frame)
    near = local[np.abs(local[:, 0] - seam_station_m) <= search_radius_m]
    if not len(near):
        raise ValueError(f"Rail object does not reach the requested boundary: {name}")
    edge_station = float(np.max(near[:, 0]) if side == "current" else np.min(near[:, 0]))
    endpoint = near[np.abs(near[:, 0] - edge_station) <= 0.02]
    return float(np.max(endpoint[:, 2]))


def smoothstep_weights(
    station_m: np.ndarray, *, transition_start_m: float, seam_station_m: float
) -> np.ndarray:
    if transition_start_m >= seam_station_m:
        raise ValueError("Transition start must precede the seam")
    t = np.clip(
        (np.asarray(station_m, dtype=np.float64) - transition_start_m)
        / (seam_station_m - transition_start_m),
        0.0,
        1.0,
    )
    return t * t * (3.0 - 2.0 * t)


def reconcile_track_boundary_vertices(
    model: ObjModel,
    *,
    origin_xyz: np.ndarray,
    frame: dict[str, Any],
    seam_station_m: float,
    transition_length_m: float,
    maximum_lateral_correction_m: float = 0.10,
    maximum_vertical_correction_m: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    if transition_length_m <= 0:
        raise ValueError("Transition length must be positive")
    current = _roots(model, CURRENT_ROOT)
    adjacent = _roots(model, ADJACENT_ROOT)
    if not current or not adjacent or len(current) != len(adjacent):
        raise ValueError(
            "Boundary reconciliation requires equal non-empty current and adjacent track groups"
        )
    current_signatures = {
        root: _endpoint_signature(
            model,
            names,
            origin_xyz=origin_xyz,
            frame=frame,
            seam_station_m=seam_station_m,
            side="current",
        )
        for root, names in current.items()
    }
    adjacent_signatures = {
        root: _endpoint_signature(
            model,
            names,
            origin_xyz=origin_xyz,
            frame=frame,
            seam_station_m=seam_station_m,
            side="adjacent",
        )
        for root, names in adjacent.items()
    }
    current_roots = sorted(current)
    adjacent_roots = sorted(adjacent)
    costs = np.asarray(
        [
            [
                abs(
                    current_signatures[left]["track_center_cross_m"]
                    - adjacent_signatures[right]["track_center_cross_m"]
                )
                for right in adjacent_roots
            ]
            for left in current_roots
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    transformed = model.vertices.copy()
    cross_vector = np.asarray(frame["cross_xy"], dtype=np.float64)
    transition_start = seam_station_m - transition_length_m
    mappings: list[dict[str, Any]] = []
    changed: set[int] = set()
    for row, column in zip(rows, columns, strict=True):
        current_root = current_roots[int(row)]
        adjacent_root = adjacent_roots[int(column)]
        before = current_signatures[current_root]
        after = adjacent_signatures[adjacent_root]
        lateral = after["track_center_cross_m"] - before["track_center_cross_m"]
        vertical = after["rail_top_mean_z_m"] - before["rail_top_mean_z_m"]
        if abs(lateral) > maximum_lateral_correction_m + 1.0e-9:
            raise ValueError(f"Unsafe lateral correction for {current_root}: {lateral}")
        if abs(vertical) > maximum_vertical_correction_m + 1.0e-9:
            raise ValueError(f"Unsafe vertical correction for {current_root}: {vertical}")
        indexes = _group_indices(model, current[current_root])
        local = _local_coordinates(
            model.vertices[indexes], origin_xyz=origin_xyz, frame=frame
        )
        weights = smoothstep_weights(
            local[:, 0],
            transition_start_m=transition_start,
            seam_station_m=seam_station_m,
        )
        transformed[indexes, 0] += weights * cross_vector[0] * lateral
        transformed[indexes, 1] += weights * cross_vector[1] * lateral
        transformed[indexes, 2] += weights * vertical
        changed.update(int(value) for value in indexes[weights > 0])
        mappings.append(
            {
                "current_track_root": current_root,
                "adjacent_track_root": adjacent_root,
                "endpoint_lateral_difference_before_m": float(costs[row, column]),
                "applied_endpoint_cross_correction_m": float(lateral),
                "applied_endpoint_vertical_correction_m": float(vertical),
                "current_endpoint": before,
                "adjacent_endpoint": after,
                "changed_vertex_count": int(np.sum(weights > 0)),
            }
        )
    return transformed, {
        "seam_station_m": seam_station_m,
        "transition_start_m": transition_start,
        "transition_length_m": transition_length_m,
        "mapping_count": len(mappings),
        "changed_vertex_count": len(changed),
        "track_mappings": mappings,
        "maximum_lateral_correction_m": maximum_lateral_correction_m,
        "maximum_vertical_correction_m": maximum_vertical_correction_m,
    }


def _material_path(obj_path: Path) -> Path:
    references = [
        line.split(maxsplit=1)[1]
        for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip().startswith("mtllib ")
    ]
    if len(references) != 1:
        raise ValueError(f"Expected one material library in {obj_path}")
    path = (obj_path.parent / references[0]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _write_transformed_obj(
    source: Path, output: Path, vertices: np.ndarray, output_mtl_name: str
) -> None:
    vertex_index = 0
    with source.open("r", encoding="utf-8", errors="replace") as reader, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        writer.write("# Evidence-gated track boundary reconciliation candidate.\n")
        for raw in reader:
            line = raw.strip()
            if line.startswith("mtllib "):
                writer.write(f"mtllib {output_mtl_name}\n")
            elif line.startswith("v "):
                value = vertices[vertex_index]
                writer.write(f"v {value[0]:.9f} {value[1]:.9f} {value[2]:.9f}\n")
                vertex_index += 1
            else:
                writer.write(raw if raw.endswith("\n") else raw + "\n")
    if vertex_index != len(vertices):
        raise ValueError("OBJ vertex count changed during track reconciliation")


def _update_registry(
    source: Path,
    output: Path,
    output_obj: Path,
    report_path: Path,
    changed_roots: set[str],
) -> dict[str, Any]:
    registry = copy.deepcopy(load_json(source))
    for asset in registry.get("assets", []):
        geometry = asset.get("geometry")
        node = str(geometry.get("node", "")) if isinstance(geometry, dict) else ""
        if isinstance(geometry, dict):
            geometry["file"] = str(output_obj)
        if any(node.startswith(root) for root in changed_roots):
            asset["status"] = "candidate"
            asset.setdefault("sources", []).append(
                {
                    "kind": "rule",
                    "reference": str(report_path),
                    "note": "Smooth boundary reconciliation constrained by adjacent track identity.",
                }
            )
            asset.setdefault("parameters", {})[
                "track_boundary_reconciliation"
            ] = "smoothstep_cross_and_vertical_alignment"
    registry["release_id"] = output_obj.stem
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["model_sha256"] = sha256_file(output_obj)
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Reconciled candidate registry is invalid: " + "; ".join(errors))
    write_json(output, registry)
    return registry


def build_track_boundary_reconciliation_candidate(
    *,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    frame_report: str | Path,
    output_directory: str | Path,
    seam_station_m: float,
    transition_length_m: float = 12.0,
) -> dict[str, Path]:
    source = Path(source_obj).resolve()
    origin_path = Path(source_origin).resolve()
    registry_path = Path(source_registry).resolve()
    frame_path = Path(frame_report).resolve()
    output = Path(output_directory).resolve()
    for path in (source, origin_path, registry_path, frame_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = output.name
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_origin = output / "model_origin.json"
    output_registry = output / "asset_registry.json"
    output_audit = output / "mesh_audit.json"
    output_report = output / "track_boundary_reconciliation_report.json"

    model = parse_obj_model(source)
    origin = np.asarray(load_json(origin_path)["origin_xyz"], dtype=np.float64)
    frame = load_json(frame_path)["frame"]
    transformed, transform_report = reconcile_track_boundary_vertices(
        model,
        origin_xyz=origin,
        frame=frame,
        seam_station_m=seam_station_m,
        transition_length_m=transition_length_m,
    )
    _write_transformed_obj(source, output_obj, transformed, output_mtl.name)
    shutil.copyfile(_material_path(source), output_mtl)
    shutil.copyfile(origin_path, output_origin)
    changed_roots = {
        str(item["current_track_root"])
        for item in transform_report["track_mappings"]
    }
    registry = _update_registry(
        registry_path,
        output_registry,
        output_obj,
        output_report,
        changed_roots,
    )
    mesh = audit_obj(output_obj)
    write_json(output_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Track boundary reconciliation candidate failed mesh audit")
    report = {
        "schema_version": "railway.track-boundary-reconciliation.v1",
        "source_obj": str(source),
        "source_obj_sha256": sha256_file(source),
        "frame_report": str(frame_path),
        "transform": transform_report,
        "output_obj": str(output_obj),
        "output_obj_sha256": sha256_file(output_obj),
        "asset_count": registry["summary"]["asset_count"],
        "mesh_audit": {
            "passed": mesh["passed"],
            "object_count": mesh["object_count"],
            "triangle_count": mesh["triangle_count_after_fan_triangulation"],
            "degenerate_triangle_count": mesh["degenerate_triangle_count"],
            "duplicate_face_count": mesh["duplicate_face_count"],
        },
        "gates": {
            "four_track_identities_matched": transform_report["mapping_count"] == 4,
            "mesh_audit_passed": bool(mesh["passed"]),
        },
        "status": "candidate_written_requires_seam_and_point_support_reaudit",
    }
    if not all(report["gates"].values()):
        raise ValueError("Track boundary reconciliation candidate failed build gates")
    write_json(output_report, report)
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_audit,
        "report": output_report,
    }
