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


def _plane_fit(points_scz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_scz, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("At least three points are required to fit a surface plane")
    design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    return np.linalg.lstsq(design, points[:, 2], rcond=None)[0]


def _predict_plane(plane: np.ndarray, station: np.ndarray, cross: np.ndarray) -> np.ndarray:
    return plane[0] * station + plane[1] * cross + plane[2]


def top_envelope_scz(points_scz: np.ndarray, *, decimals: int = 5) -> np.ndarray:
    groups: dict[tuple[float, float], np.ndarray] = {}
    for point in np.asarray(points_scz, dtype=np.float64):
        key = tuple(float(value) for value in np.round(point[:2], decimals))
        previous = groups.get(key)
        if previous is None or point[2] > previous[2]:
            groups[key] = point
    return np.asarray(list(groups.values()), dtype=np.float64)


def endpoint_cross_plane(
    envelope_scz: np.ndarray, *, edge_station_m: float, tolerance_m: float = 0.01
) -> tuple[np.ndarray, dict[str, float]]:
    envelope = np.asarray(envelope_scz, dtype=np.float64)
    edge = envelope[envelope[:, 0] >= edge_station_m - tolerance_m]
    if len(edge) < 2:
        raise ValueError("Surface endpoint contains fewer than two top-envelope points")
    cross_low = float(np.min(edge[:, 1]))
    cross_high = float(np.max(edge[:, 1]))
    if cross_high - cross_low <= 1.0e-6:
        raise ValueError("Surface endpoint has no measurable cross width")
    cross_tolerance = max(0.01, 0.01 * (cross_high - cross_low))
    z_low = float(np.max(edge[edge[:, 1] <= cross_low + cross_tolerance, 2]))
    z_high = float(np.max(edge[edge[:, 1] >= cross_high - cross_tolerance, 2]))
    slope = (z_high - z_low) / (cross_high - cross_low)
    plane = np.asarray([0.0, slope, z_low - slope * cross_low], dtype=np.float64)
    residual = np.abs(
        edge[:, 2] - _predict_plane(plane, edge[:, 0], edge[:, 1])
    )
    return plane, {
        "edge_top_point_count": len(edge),
        "cross_low_m": cross_low,
        "cross_high_m": cross_high,
        "z_low_m": z_low,
        "z_high_m": z_high,
        "endpoint_line_residual_p90_m": float(np.percentile(residual, 90)),
        "endpoint_line_residual_maximum_m": float(np.max(residual)),
    }


def robust_plane_refit(
    points_scz: np.ndarray,
    seed_plane: np.ndarray,
    *,
    seed_residual_m: float = 0.25,
    final_residual_m: float = 0.08,
) -> dict[str, Any]:
    points = np.asarray(points_scz, dtype=np.float64)
    seed = np.asarray(seed_plane, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Surface points must use an N x 3 station/cross/elevation array")
    if seed.shape != (3,):
        raise ValueError("Seed plane must contain three coefficients")
    seed_residual = np.abs(points[:, 2] - _predict_plane(seed, points[:, 0], points[:, 1]))
    candidate = points[seed_residual <= seed_residual_m]
    if len(candidate) < 100:
        return {
            "passed": False,
            "status": "insufficient_seed_consistent_points",
            "input_point_count": len(points),
            "candidate_point_count": len(candidate),
        }
    inliers = np.ones(len(candidate), dtype=bool)
    fitted = seed.copy()
    for _ in range(5):
        fitted = _plane_fit(candidate[inliers])
        residual = candidate[:, 2] - _predict_plane(
            fitted, candidate[:, 0], candidate[:, 1]
        )
        center = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - center)))
        tolerance = min(seed_residual_m, max(0.025, 3.0 * 1.4826 * mad))
        refined = np.abs(residual - center) <= tolerance
        if np.count_nonzero(refined) < 100 or np.array_equal(refined, inliers):
            break
        inliers = refined
    residual = candidate[:, 2] - _predict_plane(
        fitted, candidate[:, 0], candidate[:, 1]
    )
    final = np.abs(residual) <= final_residual_m
    selected = np.abs(residual[final])
    p90 = float(np.percentile(selected, 90)) if len(selected) else float("inf")
    gates = {
        "minimum_inlier_count": int(np.count_nonzero(final)) >= 100,
        "minimum_inlier_fraction": float(np.mean(final)) >= 0.25,
        "residual_p90_within_limit": p90 <= final_residual_m,
    }
    return {
        "passed": all(gates.values()),
        "status": "pass" if all(gates.values()) else "surface_fit_gate_failed",
        "input_point_count": len(points),
        "candidate_point_count": len(candidate),
        "inlier_point_count": int(np.count_nonzero(final)),
        "inlier_fraction": float(np.mean(final)),
        "absolute_residual_p50_m": (
            float(np.percentile(selected, 50)) if len(selected) else None
        ),
        "absolute_residual_p90_m": p90,
        "seed_plane_z_equals_a_s_plus_b_c_plus_d": seed.tolist(),
        "fitted_plane_z_equals_a_s_plus_b_c_plus_d": fitted.tolist(),
        "gates": gates,
    }


def extension_slab_mesh(
    frame: CorridorFrame,
    *,
    station_start_m: float,
    station_end_m: float,
    cross_range_m: tuple[float, float],
    start_plane: np.ndarray,
    end_plane: np.ndarray,
    thickness_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    cross_low, cross_high = cross_range_m
    if station_end_m <= station_start_m or cross_high <= cross_low or thickness_m <= 0:
        raise ValueError("Extension slab dimensions must be positive")
    station = np.asarray(
        [station_start_m, station_start_m, station_end_m, station_end_m]
    )
    cross = np.asarray([cross_low, cross_high, cross_high, cross_low])
    xy = frame.world_xy(station, cross)
    top_z = np.asarray(
        [
            _predict_plane(start_plane, station[:2], cross[:2])[0],
            _predict_plane(start_plane, station[:2], cross[:2])[1],
            _predict_plane(end_plane, station[2:], cross[2:])[0],
            _predict_plane(end_plane, station[2:], cross[2:])[1],
        ]
    )
    top = np.column_stack((xy, top_z))
    bottom = top.copy()
    bottom[:, 2] -= thickness_m
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
    frame = value.get("frame", value)
    return CorridorFrame.from_json(frame)


def _model_object_scz(
    model: Any,
    model_origin: np.ndarray,
    frame: CorridorFrame,
    object_name: str,
) -> np.ndarray:
    indexes = object_vertex_indices(model, object_name)
    world = model.vertices[indexes] + model_origin
    station, cross = frame.project(world[:, 0], world[:, 1])
    return np.column_stack((station, cross, world[:, 2]))


def _write_materials(path: Path) -> None:
    content = """# Adjacent observed surface extension materials
newmtl AdjacentRoofObserved
Kd 0.70 0.78 0.79
Ks 0.10 0.10 0.10
Ns 12

newmtl AdjacentPlatformObserved
Kd 0.32 0.68 0.75
Ks 0.08 0.08 0.08
Ns 10
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_adjacent_surface_extensions(
    *,
    source_obj: str | Path,
    source_origin: str | Path,
    cloud_path: str | Path,
    frame_report: str | Path,
    evidence_report: str | Path,
    target_settings: str | Path,
    output_directory: str | Path,
    chunk_size: int = 2_000_000,
) -> dict[str, Path]:
    source_model_path = Path(source_obj).resolve()
    source_origin_path = Path(source_origin).resolve()
    cloud = Path(cloud_path).resolve()
    frame_path = Path(frame_report).resolve()
    evidence_path = Path(evidence_report).resolve()
    settings_path = Path(target_settings).resolve()
    output = Path(output_directory).resolve()
    for path in (
        source_model_path,
        source_origin_path,
        cloud,
        frame_path,
        evidence_path,
        settings_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = parse_obj_model(source_model_path)
    origin = np.asarray(load_json(source_origin_path)["origin_xyz"], dtype=np.float64)
    frame = _frame(load_json(frame_path))
    evidence = load_json(evidence_path)
    evidence_by_name = {item["name"]: item for item in evidence["windows"]}
    settings = load_json(settings_path)
    if settings.get("schema_version") != "railway.adjacent-surface-extension-settings.v1":
        raise ValueError("Unsupported adjacent surface extension settings schema")
    targets = list(settings.get("targets", []))
    if not targets:
        raise ValueError("Adjacent surface settings contain no targets")

    prepared: list[dict[str, Any]] = []
    for target in targets:
        object_name = str(target["current_object"])
        if object_name not in model.faces_by_object:
            raise ValueError(f"Current surface object is missing: {object_name}")
        item = evidence_by_name[str(target["evidence_window"])]
        runs = list(item.get("occupied_station_runs_m", []))
        if not runs:
            prepared.append({**target, "passed": False, "status": "no_supported_run"})
            continue
        current = _model_object_scz(model, origin, frame, object_name)
        envelope = top_envelope_scz(current)
        current_edge = float(np.max(current[:, 0]))
        matching = [run for run in runs if float(run[1]) > current_edge]
        if not matching:
            prepared.append(
                {**target, "passed": False, "status": "supported_run_does_not_extend_boundary"}
            )
            continue
        supported_run = max(matching, key=lambda run: float(run[1]) - float(run[0]))
        station_end = float(supported_run[1])
        station_start = current_edge + float(target.get("seam_clearance_m", 0.002))
        seed_plane = _plane_fit(envelope)
        seam_plane, seam = endpoint_cross_plane(
            envelope, edge_station_m=current_edge
        )
        prepared.append(
            {
                **target,
                "passed": None,
                "current_edge_station_m": current_edge,
                "station_start_m": station_start,
                "station_end_m": station_end,
                "cross_range_m": [
                    float(seam["cross_low_m"]),
                    float(seam["cross_high_m"]),
                ],
                "z_range_m": [float(value) for value in item["z_range_m"]],
                "seed_plane": seed_plane,
                "seam_plane": seam_plane,
                "seam_endpoint": seam,
                "points": [],
                "supported_run_m": [float(value) for value in supported_run],
            }
        )

    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            station, cross = frame.project(x, y)
            for target in prepared:
                if target.get("passed") is False:
                    continue
                cross_low, cross_high = target["cross_range_m"]
                z_low, z_high = target["z_range_m"]
                selected = (
                    (station >= float(target["current_edge_station_m"]) - 0.5)
                    & (station <= float(target["station_end_m"]))
                    & (cross >= cross_low)
                    & (cross <= cross_high)
                    & (z >= z_low)
                    & (z <= z_high)
                )
                if np.any(selected):
                    target["points"].append(
                        np.column_stack((station[selected], cross[selected], z[selected]))
                    )

    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    writer = ObjWriter(origin, material_library=output_mtl.name)
    records: list[dict[str, Any]] = []
    for target in prepared:
        if target.get("passed") is False:
            records.append({key: value for key, value in target.items() if key != "points"})
            continue
        points = np.vstack(target["points"]) if target["points"] else np.empty((0, 3))
        fit = robust_plane_refit(points, target["seed_plane"])
        passed = bool(fit["passed"])
        record = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in target.items()
            if key != "points"
        }
        record["fit"] = fit
        record["passed"] = passed
        record["status"] = "extension_built" if passed else fit["status"]
        if passed:
            vertices, faces = extension_slab_mesh(
                frame,
                station_start_m=float(target["station_start_m"]),
                station_end_m=float(target["station_end_m"]),
                cross_range_m=tuple(float(value) for value in target["cross_range_m"]),
                start_plane=np.asarray(target["seam_plane"], dtype=np.float64),
                end_plane=np.asarray(
                    fit["fitted_plane_z_equals_a_s_plus_b_c_plus_d"], dtype=np.float64
                ),
                thickness_m=float(target["thickness_m"]),
            )
            writer.add_mesh(
                str(target["id"]),
                vertices,
                faces,
                "AdjacentPlatformObserved"
                if str(target["kind"]) == "platform"
                else "AdjacentRoofObserved",
            )
        records.append(record)
    built = [record for record in records if record.get("passed")]
    if not built:
        raise ValueError("No adjacent observed surface extension passed its point gate")
    writer.write(output_obj)
    _write_materials(output_mtl)
    write_json(output_origin, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    mesh = audit_obj(output_obj)
    output_audit = output / "mesh_audit.json"
    write_json(output_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Adjacent observed surface extensions failed mesh audit")
    output_report = output / "adjacent_surface_extensions_report.json"
    interval = [
        min(float(record["station_start_m"]) for record in built),
        max(float(record["station_end_m"]) for record in built),
    ]
    write_json(
        output_report,
        {
            "schema_version": "railway.adjacent-surface-extensions.v1",
            "source_obj": str(source_model_path),
            "source_cloud": str(cloud),
            "frame_report": str(frame_path),
            "evidence_report": str(evidence_path),
            "target_settings": str(settings_path),
            "owned_longitudinal_interval_m": interval,
            "target_count": len(records),
            "built_count": len(built),
            "records": records,
            "output_obj": str(output_obj),
            "output_obj_sha256": sha256_file(output_obj),
            "evidence_level": "observed",
            "confidence": 0.90,
            "subtype": "point_cloud_supported_boundary_surface_extension",
            "geometry_method": "current_surface_seed_plus_robust_point_plane_refit",
            "mesh_audit_passed": True,
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
