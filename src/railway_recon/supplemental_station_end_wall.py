from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .algorithms.mesh import ObjWriter
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value
from .supplemental_station_entry import _local_xyz_to_world
from .targeted_canopy_integration import merge_candidate_objs


def fit_station_end_wall(points_station_cross_z: np.ndarray) -> dict[str, float | int]:
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    if len(points) < 1_000:
        raise ValueError("Insufficient points for station end-wall fitting")
    station_edges = np.arange(95.35, 96.21, 0.02)
    counts, _ = np.histogram(points[:, 0], bins=station_edges)
    peak = int(np.argmax(counts))
    station_peak = float((station_edges[peak] + station_edges[peak + 1]) * 0.5)
    plane_points = points[np.abs(points[:, 0] - station_peak) <= 0.16]
    plane_points = plane_points[
        (plane_points[:, 1] >= 21.8)
        & (plane_points[:, 1] <= 29.0)
        & (plane_points[:, 2] >= 18.8)
        & (plane_points[:, 2] <= 24.9)
    ]
    if len(plane_points) < 500:
        raise ValueError("The detected end-wall plane is not sufficiently sampled")
    station_p05, station_p95 = np.percentile(plane_points[:, 0], (5.0, 95.0))
    cross_p02, cross_p98 = np.percentile(plane_points[:, 1], (2.0, 98.0))
    z_p02, z_p98 = np.percentile(plane_points[:, 2], (2.0, 98.0))
    return {
        "source_point_count": len(points),
        "plane_point_count": len(plane_points),
        "station_peak_m": station_peak,
        "station_min_m": float(station_p05),
        "station_max_m": float(max(station_p95, station_p05 + 0.10)),
        "cross_min_m": float(cross_p02),
        "cross_max_m": float(cross_p98),
        "bottom_z_m": float(z_p02),
        "top_z_m": float(z_p98),
    }


def _load_local_points(
    cloud: Path,
    frame: dict[str, Any],
    *,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    local_parts: list[np.ndarray] = []
    world_parts: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                (
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
            )
            local_xy = xyz[:, :2] - frame_origin
            station = local_xy @ along
            lateral = local_xy @ cross
            selected = (
                (station >= 92.5)
                & (station <= 99.0)
                & (lateral >= 18.0)
                & (lateral <= 32.5)
                & (xyz[:, 2] >= 17.5)
                & (xyz[:, 2] <= 25.2)
            )
            if np.any(selected):
                local_parts.append(np.column_stack((station[selected], lateral[selected], xyz[selected, 2])))
                world_parts.append(xyz[selected])
    if not local_parts:
        raise ValueError("No station end-wall points were loaded")
    return np.vstack(local_parts), np.vstack(world_parts)


def _wall_prism_local(fit: dict[str, float | int]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    s0 = float(fit["station_min_m"])
    s1 = float(fit["station_max_m"])
    c0 = float(fit["cross_min_m"])
    c1 = float(fit["cross_max_m"])
    z0 = float(fit["bottom_z_m"])
    z1 = float(fit["top_z_m"])
    vertices = np.asarray(
        (
            (s0, c0, z0),
            (s0, c1, z0),
            (s0, c1, z1),
            (s0, c0, z1),
            (s1, c0, z0),
            (s1, c1, z0),
            (s1, c1, z1),
            (s1, c0, z1),
        ),
        dtype=np.float64,
    )
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def _wall_surface_support(
    fit: dict[str, float | int],
    frame: dict[str, Any],
    world_points: np.ndarray,
    *,
    spacing_m: float = 0.15,
) -> dict[str, float | int | bool]:
    cross_values = np.arange(
        float(fit["cross_min_m"]),
        float(fit["cross_max_m"]) + spacing_m * 0.5,
        spacing_m,
    )
    z_values = np.arange(
        float(fit["bottom_z_m"]),
        float(fit["top_z_m"]) + spacing_m * 0.5,
        spacing_m,
    )
    cross_grid, z_grid = np.meshgrid(cross_values, z_values)
    local = np.column_stack(
        (
            np.full(cross_grid.size, float(fit["station_peak_m"])),
            cross_grid.ravel(),
            z_grid.ravel(),
        )
    )
    samples = _local_xyz_to_world(local, frame)
    distances, _ = cKDTree(world_points).query(samples, workers=-1)
    p90 = float(np.percentile(distances, 90))
    coverage = float(np.mean(distances <= 0.30))
    return {
        "sampling_spacing_m": spacing_m,
        "sample_count": len(samples),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": p90,
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "coverage_at_0_30m": coverage,
        "passed_review_gate": p90 <= 0.35 and coverage >= 0.75,
    }


def build_supplemental_station_end_wall(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    group_evidence_path: str | Path,
    photo_evidence_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cloud = Path(cloud_path).resolve()
    frame = load_json(Path(frame_report_path))["frame"]
    local_points, world_points = _load_local_points(cloud, frame)
    fit = fit_station_end_wall(local_points)
    source_model = Path(source_obj).resolve()
    source_origin_path = Path(source_origin).resolve()
    source_registry_path = Path(source_registry).resolve()
    origin_value = load_json(source_origin_path)
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)

    addon_obj = output / "supplemental_station_end_wall.obj"
    addon_mtl = output / "supplemental_station_end_wall.mtl"
    addon_origin = output / "supplemental_station_end_wall_origin.json"
    node = "SUPPLEMENTAL-STATION-END-WALL-01"
    local_vertices, faces = _wall_prism_local(fit)
    writer = ObjWriter(model_origin, material_library=addon_mtl.name)
    writer.add_mesh(
        node,
        _local_xyz_to_world(local_vertices, frame),
        faces,
        "SupplementalStationEndWallTile",
    )
    writer.write(addon_obj)
    addon_mtl.write_text(
        """# Observed white tile station end wall
newmtl SupplementalStationEndWallTile
Ka 0.24 0.24 0.23
Kd 0.78 0.79 0.76
Ks 0.12 0.12 0.11
Ns 24.0
d 1.0
illum 2
""",
        encoding="utf-8",
        newline="\n",
    )
    write_json(addon_origin, origin_value)

    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [(source_model, source_origin_path), (addon_obj, addon_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )
    support = _wall_surface_support(fit, frame, world_points)
    mesh = audit_obj(output_obj)
    if not mesh["passed"]:
        raise ValueError("Station end-wall candidate failed mesh audit")

    registry = copy.deepcopy(load_json(source_registry_path))
    for asset in registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    asset = {
        "id": node,
        "type": "station_end_wall",
        "subtype": "visible_tiled_end_wall",
        "status": "candidate",
        "chainage_m": float(fit["station_peak_m"]),
        "evidence_level": "observed",
        "confidence": 0.88,
        "sources": [
            {"kind": "point_cloud", "reference": str(Path(group_evidence_path).resolve())},
            {"kind": "panorama", "reference": str(Path(photo_evidence_path).resolve())},
        ],
        "parameters": {
            **fit,
            "point_support": support,
            "geometry_method": "dense_cloud_end_plane_fit_with_photo_wall_semantics",
        },
        "geometry": {"file": str(output_obj), "node": node},
        "limitations": [
            "Only the continuous lower tiled wall is represented.",
            "The upper roof overhang and open-air interval are deliberately not vertically filled.",
            "No interior or unobserved rear volume is inferred.",
        ],
    }
    registry["assets"].append(asset)
    registry["release_id"] = output_name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    additions = new_registry(str(registry.get("project_id", "site-b")))
    additions["assets"] = [copy.deepcopy(asset)]
    additions["summary"] = summarize_registry(additions)
    errors = validate_registry_value(additions)
    if errors:
        raise ValueError("Station end-wall asset is invalid: " + "; ".join(errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)
    mesh_path = output / "mesh_audit.json"
    write_json(mesh_path, mesh)
    report = output / "supplemental_station_end_wall_report.json"
    write_json(
        report,
        {
            "schema_version": "railway.supplemental-station-end-wall.v1",
            "source_obj": str(source_model),
            "output_obj": str(output_obj),
            "fit": fit,
            "point_support": support,
            "mesh_audit_passed": True,
            "merge": merge,
            "asset": asset,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_built_fixed_view_and_gap_regression_review_required",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": mesh_path,
        "report": report,
    }
