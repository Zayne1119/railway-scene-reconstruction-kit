from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .algorithms.mesh import ObjWriter
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import parse_obj_model, sample_object_surfaces
from .registry import new_registry, summarize_registry, validate_registry_value
from .supplemental_station_entry import _local_xyz_to_world
from .targeted_canopy_integration import merge_candidate_objs


def _rgb8(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.column_stack((red, green, blue)).astype(np.float64)
    scale = 257.0 if np.any(rgb) and float(np.percentile(rgb, 99)) > 255.0 else 1.0
    return np.clip(rgb / scale, 0.0, 255.0)


def _load_clock_crop(
    cloud: Path,
    frame: dict[str, Any],
    *,
    station_m: float,
    cross_m: float,
    z_m: float,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    local_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local_xy = np.column_stack((x, y)) - frame_origin
            station = local_xy @ along
            lateral = local_xy @ cross
            selected = (
                (np.abs(station - station_m) <= 0.75)
                & (np.abs(lateral - cross_m) <= 1.25)
                & (z >= z_m - 1.35)
                & (z <= z_m + 1.35)
            )
            if not np.any(selected):
                continue
            local_parts.append(np.column_stack((station[selected], lateral[selected], z[selected])))
            rgb_parts.append(
                _rgb8(
                    np.asarray(chunk.red)[selected],
                    np.asarray(chunk.green)[selected],
                    np.asarray(chunk.blue)[selected],
                )
            )
    if not local_parts:
        raise ValueError("No points found around the platform-clock candidate")
    return np.vstack(local_parts), np.vstack(rgb_parts)


def fit_platform_clock(
    local_points: np.ndarray,
    rgb: np.ndarray,
    *,
    station_hint_m: float,
) -> dict[str, float | int]:
    points = np.asarray(local_points, dtype=np.float64)
    colours = np.asarray(rgb, dtype=np.float64)
    plane = np.abs(points[:, 0] - station_hint_m) <= 0.20
    blue_grey = (
        (colours[:, 2] >= colours[:, 0] + 7.0)
        & (colours[:, 2] >= colours[:, 1] + 4.0)
        & (np.mean(colours, axis=1) >= 45.0)
        & (np.mean(colours, axis=1) <= 220.0)
    )
    selected = plane & blue_grey
    target = points[selected]
    if len(target) < 80:
        raise ValueError("Insufficient blue-grey points on the candidate clock plane")
    projected = target[:, 1:3]
    cell_size = 0.08
    minimum = np.min(projected, axis=0)
    cell = np.floor((projected - minimum) / cell_size).astype(np.int64)
    shape = np.max(cell, axis=0) + 1
    occupancy = np.zeros((int(shape[1]), int(shape[0])), dtype=bool)
    occupancy[cell[:, 1], cell[:, 0]] = True
    labels, count = ndimage.label(occupancy, structure=np.ones((3, 3), dtype=np.uint8))
    point_labels = labels[cell[:, 1], cell[:, 0]]
    components: list[tuple[int, np.ndarray]] = []
    for label in range(1, count + 1):
        member = point_labels == label
        values = projected[member]
        if len(values) < 60:
            continue
        extent = np.ptp(values, axis=0)
        if 0.35 <= extent[0] <= 1.40 and 0.35 <= extent[1] <= 1.40:
            components.append((len(values), member))
    if not components:
        raise ValueError("No compact round clock component was isolated")
    _, member = max(components, key=lambda item: item[0])
    clock_points = target[member]
    cross_z = clock_points[:, 1:3]
    design = np.column_stack((2.0 * cross_z[:, 0], 2.0 * cross_z[:, 1], np.ones(len(cross_z))))
    rhs = np.sum(cross_z**2, axis=1)
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center_cross, center_z, constant = solution
    radius = float(np.sqrt(max(constant + center_cross**2 + center_z**2, 0.0)))
    radial = np.linalg.norm(cross_z - np.asarray((center_cross, center_z)), axis=1)
    inlier = np.abs(radial - radius) <= max(0.10, float(np.percentile(np.abs(radial - radius), 75)))
    if np.sum(inlier) >= 50:
        cross_z = cross_z[inlier]
        design = np.column_stack(
            (2.0 * cross_z[:, 0], 2.0 * cross_z[:, 1], np.ones(len(cross_z)))
        )
        rhs = np.sum(cross_z**2, axis=1)
        solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        center_cross, center_z, constant = solution
        radius = float(np.sqrt(max(constant + center_cross**2 + center_z**2, 0.0)))
    if not 0.20 <= radius <= 0.75:
        raise ValueError(f"Fitted clock radius is implausible: {radius:.3f} m")
    angles = np.arctan2(cross_z[:, 1] - center_z, cross_z[:, 0] - center_cross)
    occupied_angle_bins = len(np.unique(np.floor((angles + np.pi) / (2.0 * np.pi) * 16)))
    if occupied_angle_bins < 8:
        raise ValueError("Clock points do not cover enough of a round outline")
    station_values = clock_points[:, 0]
    return {
        "crop_point_count": len(points),
        "colour_plane_point_count": len(target),
        "circle_point_count": len(clock_points),
        "station_center_m": float(np.median(station_values)),
        "station_thickness_m": float(
            np.clip(np.percentile(station_values, 95) - np.percentile(station_values, 5), 0.08, 0.16)
        ),
        "cross_center_m": float(center_cross),
        "z_center_m": float(center_z),
        "radius_m": radius,
        "occupied_angle_bins_of_16": occupied_angle_bins,
        "hanger_top_z_m": float(center_z + radius + 0.45),
    }


def _clock_disc_local(
    fit: dict[str, float | int],
    *,
    sides: int = 32,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    station = float(fit["station_center_m"])
    half_thickness = float(fit["station_thickness_m"]) * 0.5
    cross = float(fit["cross_center_m"])
    z = float(fit["z_center_m"])
    radius = float(fit["radius_m"])
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rings = []
    for station_value in (station - half_thickness, station + half_thickness):
        rings.append(
            np.column_stack(
                (
                    np.full(sides, station_value),
                    cross + radius * np.cos(angle),
                    z + radius * np.sin(angle),
                )
            )
        )
    vertices = np.vstack(rings)
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sides))), tuple(range(sides, 2 * sides))]
    for index in range(sides):
        following = (index + 1) % sides
        faces.append((index, following, sides + following, sides + index))
    return vertices, faces


def _hanger_local(fit: dict[str, float | int]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    station = float(fit["station_center_m"])
    cross = float(fit["cross_center_m"])
    bottom = float(fit["z_center_m"]) + float(fit["radius_m"])
    top = max(bottom + 0.08, float(fit["hanger_top_z_m"]))
    half = 0.035
    vertices = np.asarray(
        [
            (station - half, cross - half, bottom),
            (station + half, cross - half, bottom),
            (station + half, cross + half, bottom),
            (station - half, cross + half, bottom),
            (station - half, cross - half, top),
            (station + half, cross - half, top),
            (station + half, cross + half, top),
            (station - half, cross + half, top),
        ],
        dtype=np.float64,
    )
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return vertices, faces


def build_supplemental_platform_clock(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    gap_report_path: str | Path,
    candidate_id: str,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    point_evidence_path: str | Path,
    photo_evidence_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gap = load_json(Path(gap_report_path))
    candidate = next(
        (item for item in gap["vertical_candidates"] if item["candidate_id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Unknown gap candidate: {candidate_id}")
    frame = load_json(Path(frame_report_path))["frame"]
    cloud = Path(cloud_path).resolve()
    local_points, rgb = _load_clock_crop(
        cloud,
        frame,
        station_m=float(candidate["station_m"]),
        cross_m=float(candidate["cross_m"]),
        z_m=float(candidate["centroid_xyz_m"][2]),
    )
    fit = fit_platform_clock(
        local_points,
        rgb,
        station_hint_m=float(candidate["station_m"]),
    )
    origin_value = load_json(Path(source_origin))
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    addon_obj = output / "supplemental_platform_clock.obj"
    addon_mtl = output / "supplemental_platform_clock.mtl"
    addon_origin = output / "supplemental_platform_clock_origin.json"
    writer = ObjWriter(model_origin, material_library=addon_mtl.name)
    disc_vertices, disc_faces = _clock_disc_local(fit)
    hanger_vertices, hanger_faces = _hanger_local(fit)
    node = "SUPPLEMENTAL-OPPOSITE-PLATFORM-CLOCK-001"
    hanger_node = f"{node}-HANGER"
    writer.add_mesh(
        node,
        _local_xyz_to_world(disc_vertices, frame),
        disc_faces,
        "PlatformClockDarkFace",
    )
    writer.add_mesh(
        hanger_node,
        _local_xyz_to_world(hanger_vertices, frame),
        hanger_faces,
        "PlatformClockHangerSteel",
    )
    writer.write(addon_obj)
    addon_mtl.write_text(
        """# Observed opposite-platform hanging clock
newmtl PlatformClockDarkFace
Ka 0.03 0.05 0.08
Kd 0.12 0.18 0.28
Ks 0.34 0.38 0.42
Ns 64.0
d 1.0
illum 2

newmtl PlatformClockHangerSteel
Ka 0.16 0.17 0.18
Kd 0.48 0.51 0.53
Ks 0.35 0.35 0.35
Ns 48.0
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
        [(Path(source_obj).resolve(), Path(source_origin).resolve()), (addon_obj, addon_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )
    addon_model = parse_obj_model(addon_obj)
    samples = sample_object_surfaces(
        addon_model,
        origin_xyz=model_origin,
        maximum_samples_per_object=10_000,
    )[node]
    crop_world = _local_xyz_to_world(local_points, frame)
    distances, _ = cKDTree(crop_world).query(samples, workers=-1)
    support = {
        "sample_count": len(samples),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "coverage_at_0_30m": float(np.mean(distances <= 0.30)),
    }
    support["passed_review_gate"] = bool(
        support["p90_m"] <= 0.35 and support["coverage_at_0_30m"] >= 0.60
    )
    mesh = audit_obj(output_obj)
    if not mesh["passed"]:
        raise ValueError("Platform-clock candidate failed mesh audit")
    registry = copy.deepcopy(load_json(Path(source_registry)))
    for item in registry.get("assets", []):
        if isinstance(item.get("geometry"), dict):
            item["geometry"]["file"] = str(output_obj)
    asset = {
        "id": node,
        "type": "station_information_sign",
        "subtype": "round_hanging_platform_clock",
        "status": "candidate",
        "chainage_m": float(fit["station_center_m"]),
        "evidence_level": "observed",
        "confidence": 0.84,
        "sources": [
            {"kind": "point_cloud", "reference": str(Path(point_evidence_path).resolve())},
            {"kind": "panorama", "reference": str(Path(photo_evidence_path).resolve())},
        ],
        "parameters": {**fit, "point_support": support, "source_candidate_id": candidate_id},
        "geometry": {"file": str(output_obj), "node": node},
        "limitations": [
            "Clock face graphics and hands are not resolved by the current evidence.",
            "The hanger top is conservatively extended to the inferred canopy underside.",
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
        raise ValueError("Platform-clock registry record is invalid: " + "; ".join(errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)
    mesh_path = output / "mesh_audit.json"
    write_json(mesh_path, mesh)
    report_path = output / "supplemental_platform_clock_report.json"
    write_json(
        report_path,
        {
            "schema_version": "railway.supplemental-platform-clock.v1",
            "candidate_id": candidate_id,
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
        "report": report_path,
    }
