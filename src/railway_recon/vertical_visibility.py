from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def _local_clouds(
    path: str | Path,
    bounds_by_name: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    horizontal_margin_m: float = 0.8,
    vertical_margin_m: float = 0.4,
    chunk_size: int = 2_000_000,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in bounds_by_name}
    with laspy.open(Path(path)) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            for name, (minimum, maximum) in bounds_by_name.items():
                selected = (
                    (x >= minimum[0] - horizontal_margin_m)
                    & (x <= maximum[0] + horizontal_margin_m)
                    & (y >= minimum[1] - horizontal_margin_m)
                    & (y <= maximum[1] + horizontal_margin_m)
                    & (z >= minimum[2] - vertical_margin_m)
                    & (z <= maximum[2] + vertical_margin_m)
                )
                if np.any(selected):
                    collected[name].append(np.column_stack((x[selected], y[selected], z[selected])))
    return {
        name: np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float64)
        for name, chunks in collected.items()
    }


def _support_intervals(levels: np.ndarray, supported: np.ndarray) -> list[list[float]]:
    intervals: list[list[float]] = []
    start: float | None = None
    previous = 0.0
    for level, is_supported in zip(levels, supported, strict=True):
        if is_supported and start is None:
            start = float(level)
        if not is_supported and start is not None:
            intervals.append([start, previous])
            start = None
        previous = float(level)
    if start is not None:
        intervals.append([start, previous])
    return intervals


def audit_vertical_object_visibility(
    vertices: np.ndarray,
    local_cloud: np.ndarray,
    *,
    level_step_m: float = 0.20,
    support_distance_m: float = 0.12,
) -> dict[str, Any]:
    if not len(local_cloud):
        return {"decision": "insufficient_local_points", "local_point_count": 0}
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    levels = np.arange(
        minimum[2], maximum[2] + level_step_m * 0.5, level_step_m, dtype=np.float64
    )
    cross_section_xy = np.unique(np.round(vertices[:, :2], 6), axis=0)
    samples = np.vstack(
        [
            np.column_stack(
                (
                    cross_section_xy,
                    np.full(len(cross_section_xy), level, dtype=np.float64),
                )
            )
            for level in levels
        ]
    )
    distances, indices = cKDTree(local_cloud).query(samples, workers=-1)
    distances = distances.reshape(len(levels), len(cross_section_xy))
    indices = indices.reshape(len(levels), len(cross_section_xy))
    best_cross = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(levels)), best_cross]
    best_indices = indices[np.arange(len(levels)), best_cross]
    best_samples = samples.reshape(len(levels), len(cross_section_xy), 3)[
        np.arange(len(levels)), best_cross
    ]
    supported = best_distances <= support_distance_m
    lateral_deltas = local_cloud[best_indices, :2] - best_samples[:, :2]
    supported_fraction = float(np.mean(supported))
    core = (levels >= minimum[2] + 0.5) & (levels <= maximum[2] - 0.5)
    core_supported_fraction = float(np.mean(supported[core])) if np.any(core) else supported_fraction
    if core_supported_fraction >= 0.70:
        decision = "visible_supported"
    elif core_supported_fraction >= 0.20:
        decision = "partial_visibility_do_not_refit_from_distance_only"
    else:
        decision = "insufficient_shaft_visibility_do_not_auto_delete"
    unsupported_runs: list[int] = []
    current = 0
    for value in supported:
        if value:
            if current:
                unsupported_runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        unsupported_runs.append(current)
    return {
        "decision": decision,
        "local_point_count": len(local_cloud),
        "z_range_m": [float(minimum[2]), float(maximum[2])],
        "level_step_m": level_step_m,
        "level_count": len(levels),
        "support_distance_m": support_distance_m,
        "supported_level_count": int(np.count_nonzero(supported)),
        "supported_fraction": supported_fraction,
        "core_supported_fraction": core_supported_fraction,
        "longest_unsupported_run_m": (
            max(unsupported_runs, default=0) * level_step_m
        ),
        "support_intervals_z_m": _support_intervals(levels, supported),
        "nearest_distance_m": {
            "p50": float(np.percentile(best_distances, 50)),
            "p90": float(np.percentile(best_distances, 90)),
            "maximum": float(np.max(best_distances)),
        },
        "supported_lateral_delta_median_xy_m": (
            [float(value) for value in np.median(lateral_deltas[supported], axis=0)]
            if np.any(supported)
            else None
        ),
        "levels": [
            {
                "z_m": float(level),
                "nearest_distance_m": float(distance),
                "supported": bool(is_supported),
            }
            for level, distance, is_supported in zip(levels, best_distances, supported, strict=True)
        ],
    }


def audit_catenary_mast_visibility(
    obj_path: str | Path,
    origin_path: str | Path,
    registry_path: str | Path,
    cloud_path: str | Path,
    output_path: str | Path,
) -> Path:
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(origin_path))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(registry_path))
    mast_names = sorted(
        str(asset["id"])
        for asset in registry.get("assets", [])
        if asset.get("type") == "catenary_mast"
        and str(asset.get("id", "")) in model.faces_by_object
    )
    vertices_by_name: dict[str, np.ndarray] = {}
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in mast_names:
        vertices = model.vertices[object_vertex_indices(model, name)] + origin
        vertices_by_name[name] = vertices
        bounds[name] = (np.min(vertices, axis=0), np.max(vertices, axis=0))
    local_clouds = _local_clouds(cloud_path, bounds)
    records = [
        {
            "object_name": name,
            **audit_vertical_object_visibility(vertices_by_name[name], local_clouds[name]),
        }
        for name in mast_names
    ]
    report = {
        "schema_version": "railway.catenary-mast-visibility.v1",
        "inputs": {
            "obj": str(Path(obj_path).resolve()),
            "origin": str(Path(origin_path).resolve()),
            "registry": str(Path(registry_path).resolve()),
            "cloud": str(Path(cloud_path).resolve()),
        },
        "mast_count": len(records),
        "records": records,
        "policy": (
            "Sparse shaft visibility cannot authorize automatic mast deletion or translation; "
            "photo evidence or a vertical candidate fit is required."
        ),
    }
    destination = Path(output_path).resolve()
    write_json(destination, report)
    return destination
