from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms.mesh import ObjWriter
from .config import ProjectConfig
from .io import load_json, write_json
from .mesh_audit import audit_obj


def build_segment_ownership_plan(
    project: ProjectConfig,
    segment_frame_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one continuous Voronoi-style ownership frame for adjacent segments."""
    if len(segment_frame_paths) < 2:
        raise ValueError("At least two segment frames are required")
    output = project.resolve(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)

    records: list[dict[str, Any]] = []
    reference_along: np.ndarray | None = None
    for segment_id, raw_path in segment_frame_paths.items():
        path = project.resolve(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        value = load_json(path)
        frame = value.get("frame", value)
        origin = np.asarray(frame["origin_xy"], dtype=np.float64)
        along = np.asarray(frame["along_xy"], dtype=np.float64)
        along /= max(float(np.linalg.norm(along)), 1e-12)
        if reference_along is None:
            reference_along = along
        elif float(np.dot(along, reference_along)) < 0:
            along = -along
        records.append(
            {
                "segment_id": segment_id,
                "frame_report": str(path),
                "origin": origin,
                "along": along,
            }
        )

    assert reference_along is not None
    canonical_along = np.sum([record["along"] for record in records], axis=0)
    canonical_along /= max(float(np.linalg.norm(canonical_along)), 1e-12)
    global_origin = records[0]["origin"]
    lateral = np.asarray([-canonical_along[1], canonical_along[0]])
    for record in records:
        offset = record["origin"] - global_origin
        record["center_s_m"] = float(np.dot(offset, canonical_along))
        record["center_lateral_offset_m"] = float(np.dot(offset, lateral))
        cosine = float(np.clip(np.dot(record["along"], canonical_along), -1.0, 1.0))
        record["axis_deviation_deg"] = float(np.degrees(np.arccos(cosine)))
    records.sort(key=lambda record: record["center_s_m"])
    center_stations = np.asarray([record["center_s_m"] for record in records])
    spacings = np.diff(center_stations)
    if np.any(spacings <= 1e-6):
        raise ValueError("Segment centers are not uniquely ordered along the corridor")
    if max(record["axis_deviation_deg"] for record in records) > 5.0:
        raise ValueError("Segment axes diverge by more than five degrees")

    seam_stations = ((center_stations[:-1] + center_stations[1:]) * 0.5).tolist()
    outer_minimum = float(center_stations[0] - spacings[0] * 0.5)
    outer_maximum = float(center_stations[-1] + spacings[-1] * 0.5)
    boundaries = [outer_minimum, *seam_stations, outer_maximum]
    segments = []
    for index, record in enumerate(records):
        segments.append(
            {
                "segment_id": record["segment_id"],
                "frame_report": record["frame_report"],
                "center_station_m": record["center_s_m"],
                "center_lateral_offset_m": record["center_lateral_offset_m"],
                "axis_deviation_deg": record["axis_deviation_deg"],
                "owned_interval_m": [boundaries[index], boundaries[index + 1]],
            }
        )
    result = {
        "schema_version": "railway.segment-ownership-plan.v1",
        "project_id": project.project_id,
        "status": "pass",
        "policy": "shared_corridor_frame_with_midpoint_voronoi_boundaries",
        "frame": {
            "origin_xy": global_origin.tolist(),
            "along_xy": canonical_along.tolist(),
            "lateral_xy": lateral.tolist(),
        },
        "segment_count": len(segments),
        "segment_center_spacing_m": spacings.tolist(),
        "seam_stations_m": seam_stations,
        "coverage_interval_m": [outer_minimum, outer_maximum],
        "coverage_length_m": outer_maximum - outer_minimum,
        "segments": segments,
        "passed": True,
        "limitations": [
            "Midpoint ownership prevents overlap and gaps but does not force adjacent surface elevations to match.",
            "Run a cross-segment geometry seam audit after clipping.",
        ],
    }
    write_json(output, result)
    return result


def _clip_polygon_halfspace(
    polygon: np.ndarray,
    origin_xy: np.ndarray,
    along_xy: np.ndarray,
    boundary_m: float,
    *,
    keep_greater: bool,
    tolerance: float = 1e-9,
) -> np.ndarray:
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=np.float64)

    def signed(point: np.ndarray) -> float:
        return float(np.dot(point[:2] - origin_xy, along_xy) - boundary_m)

    def inside(value: float) -> bool:
        return value >= -tolerance if keep_greater else value <= tolerance

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = signed(previous)
    previous_inside = inside(previous_value)
    for current in polygon:
        current_value = signed(current)
        current_inside = inside(current_value)
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                amount = previous_value / denominator
                output.append(previous + amount * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return (
        np.asarray(output, dtype=np.float64)
        if len(output) >= 3
        else np.empty((0, 3), dtype=np.float64)
    )


def clip_polygon_to_longitudinal_interval(
    polygon: np.ndarray,
    origin_xy: np.ndarray,
    along_xy: np.ndarray,
    interval_m: tuple[float, float],
) -> np.ndarray:
    minimum, maximum = interval_m
    if minimum >= maximum:
        raise ValueError("Ownership interval must have positive length")
    clipped = _clip_polygon_halfspace(
        np.asarray(polygon, dtype=np.float64),
        origin_xy,
        along_xy,
        minimum,
        keep_greater=True,
    )
    if not len(clipped):
        return clipped
    return _clip_polygon_halfspace(
        clipped,
        origin_xy,
        along_xy,
        maximum,
        keep_greater=False,
    )


def _parse_obj(path: Path) -> tuple[np.ndarray, list[dict[str, Any]], Path]:
    vertices: list[list[float]] = []
    faces: list[dict[str, Any]] = []
    object_name = "default"
    material_name = "default"
    material_reference: str | None = None
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "mtllib" and len(parts) >= 2:
            if material_reference is not None:
                raise ValueError("Ownership clipping expects one material library")
            material_reference = " ".join(parts[1:])
        elif parts[0] in {"o", "g"} and len(parts) >= 2:
            object_name = " ".join(parts[1:])
        elif parts[0] == "usemtl" and len(parts) >= 2:
            material_name = " ".join(parts[1:])
        elif parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f" and len(parts) >= 4:
            indexes = []
            for token in parts[1:]:
                value = int(token.split("/")[0])
                indexes.append(value - 1 if value > 0 else len(vertices) + value)
            faces.append(
                {
                    "object_name": object_name,
                    "material_name": material_name,
                    "indexes": indexes,
                }
            )
    if not vertices or not faces or material_reference is None:
        raise ValueError(f"OBJ is incomplete for ownership clipping: {path}")
    material_path = (path.parent / material_reference).resolve()
    if not material_path.is_file():
        raise FileNotFoundError(material_path)
    return np.asarray(vertices, dtype=np.float64), faces, material_path


def _append_polygon(
    vertices: list[list[float]],
    faces: list[tuple[int, ...]],
    vertex_lookup: dict[tuple[float, float, float], int],
    polygon: np.ndarray,
    serialization_origin: np.ndarray | None = None,
) -> None:
    points = [np.asarray(point, dtype=np.float64) for point in polygon]
    changed = True
    while changed and len(points) >= 3:
        changed = False
        cleaned: list[np.ndarray] = []
        count = len(points)
        for index, current in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % count]
            first = current - previous
            second = following - current
            scale = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0)
            if float(np.linalg.norm(np.cross(first, second))) <= 1e-10 * scale:
                changed = True
                continue
            cleaned.append(current)
        if cleaned:
            points = cleaned
        else:
            points = []
    if len(points) < 3:
        return
    indexes: list[int] = []
    for point in points:
        # ObjWriter serializes *local* coordinates to six decimals. Deduplicate
        # using that exact representation; rounding world coordinates is not
        # equivalent when the model origin itself has fractional metres.
        local_point = point if serialization_origin is None else point - serialization_origin
        key = tuple(float(f"{value:.6f}") for value in local_point)
        index = vertex_lookup.get(key)
        if index is None:
            index = len(vertices)
            vertex_lookup[key] = index
            vertices.append(point.tolist())
        indexes.append(index)
    compact = [index for position, index in enumerate(indexes) if position == 0 or index != indexes[position - 1]]
    if len(compact) >= 3 and compact[0] == compact[-1]:
        compact.pop()
    if len(set(compact)) >= 3:
        faces.append(tuple(compact))


def clip_segment_mesh_ownership(
    project: ProjectConfig,
    source_obj_path: str | Path,
    source_origin_path: str | Path,
    frame_report_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_longitudinal_m: float,
    maximum_longitudinal_m: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_obj = project.resolve(source_obj_path)
    source_origin_file = project.resolve(source_origin_path)
    frame_path = project.resolve(frame_report_path)
    for path in (source_obj, source_origin_file, frame_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = project.resolve(output_dir)
    output_obj = output_root / f"{source_obj.stem}_owned.obj"
    output_mtl = output_root / f"{source_obj.stem}_owned.mtl"
    output_origin = output_root / "model_origin.json"
    output_audit = output_root / "mesh_audit.json"
    output_report = output_root / "ownership_clip_report.json"
    outputs = (output_obj, output_mtl, output_origin, output_audit, output_report)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite ownership outputs: {existing}")

    local_vertices, source_faces, source_mtl = _parse_obj(source_obj)
    source_origin = np.asarray(
        load_json(source_origin_file)["origin_xyz"], dtype=np.float64
    )
    world_vertices = local_vertices + source_origin
    frame_value = load_json(frame_path)
    frame = frame_value.get("frame", frame_value)
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    along /= max(float(np.linalg.norm(along)), 1e-12)
    interval = (float(minimum_longitudinal_m), float(maximum_longitudinal_m))
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"vertices": [], "faces": [], "lookup": {}, "source_lookup": {}}
    )
    kept_faces = clipped_faces = removed_faces = 0
    source_triangle_count = 0
    for face in source_faces:
        indexes = face["indexes"]
        source_triangle_count += max(len(indexes) - 2, 0)
        source_polygon = world_vertices[indexes]
        source_station = (source_polygon[:, :2] - frame_origin) @ along
        entirely_inside = bool(
            np.all(source_station >= interval[0] - 1e-9)
            and np.all(source_station <= interval[1] + 1e-9)
        )
        if entirely_inside:
            kept_faces += 1
            group = groups[(face["object_name"], face["material_name"])]
            owned_indexes = []
            for source_index in indexes:
                owned_index = group["source_lookup"].get(source_index)
                if owned_index is None:
                    owned_index = len(group["vertices"])
                    group["source_lookup"][source_index] = owned_index
                    group["vertices"].append(world_vertices[source_index].tolist())
                owned_indexes.append(owned_index)
            group["faces"].append(tuple(owned_indexes))
            continue
        for offset in range(1, len(indexes) - 1):
            polygon = world_vertices[[indexes[0], indexes[offset], indexes[offset + 1]]]
            clipped = clip_polygon_to_longitudinal_interval(
                polygon, frame_origin, along, interval
            )
            if not len(clipped):
                removed_faces += 1
                continue
            clipped_faces += 1
            group = groups[(face["object_name"], face["material_name"])]
            _append_polygon(
                group["vertices"],
                group["faces"],
                group["lookup"],
                clipped,
                source_origin,
            )

    writer = ObjWriter(source_origin, material_library=output_mtl.name)
    emitted_objects = 0
    used_names: set[str] = set()
    for (name, material), group in groups.items():
        if not group["faces"]:
            continue
        emitted_name = name
        if emitted_name in used_names:
            emitted_name = f"{name}-MATERIAL-{emitted_objects + 1:02d}"
        used_names.add(emitted_name)
        writer.add_mesh(
            emitted_name,
            np.asarray(group["vertices"], dtype=np.float64),
            group["faces"],
            material,
        )
        emitted_objects += 1
    if emitted_objects == 0:
        raise ValueError("Ownership interval removed all source geometry")
    writer.write(output_obj)
    output_mtl.parent.mkdir(parents=True, exist_ok=True)
    temporary_mtl = output_mtl.with_suffix(output_mtl.suffix + ".tmp")
    shutil.copyfile(source_mtl, temporary_mtl)
    os.replace(temporary_mtl, output_mtl)
    write_json(
        output_origin,
        {"origin_xyz": source_origin.tolist(), "units": "metre", "axis": "Z-up"},
    )
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Owned segment OBJ failed mesh audit")
    report = {
        "schema_version": "railway.segment-mesh-ownership-clip.v1",
        "project_id": project.project_id,
        "source_obj": str(source_obj),
        "source_origin": str(source_origin_file),
        "frame_report": str(frame_path),
        "owned_longitudinal_interval_m": list(interval),
        "source_face_count": len(source_faces),
        "source_triangle_count_after_fan_triangulation": source_triangle_count,
        "fully_owned_face_count": kept_faces,
        "boundary_clipped_face_count": clipped_faces,
        "outside_face_count": removed_faces,
        "emitted_object_count": emitted_objects,
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_mesh_audit": str(output_audit),
        "boundary_cap_policy": "open_at_internal_segment_seams_pending_cross_segment_reconciliation",
        "passed": True,
        "status": "owned_interval_mesh_written_cross_segment_seam_review_required",
        "limitations": [
            "Cut faces are not capped because they are internal multi-segment ownership seams.",
            "Adjacent fitted surfaces still require a shared seam continuity audit before composition.",
        ],
    }
    write_json(output_report, report)
    return report


def _sample_boundary_edges(
    obj_path: Path,
    frame_origin: np.ndarray,
    along: np.ndarray,
    lateral: np.ndarray,
    seam_station_m: float,
    *,
    tolerance_m: float,
    sample_spacing_m: float,
) -> tuple[np.ndarray, int]:
    vertices, faces, _ = _parse_obj(obj_path)
    origin_path = obj_path.parent / "model_origin.json"
    if not origin_path.is_file():
        raise FileNotFoundError(origin_path)
    model_origin = np.asarray(load_json(origin_path)["origin_xyz"], dtype=np.float64)
    world = vertices + model_origin
    stations = (world[:, :2] - frame_origin) @ along
    cross_height = np.column_stack(
        ((world[:, :2] - frame_origin) @ lateral, world[:, 2])
    )
    boundary_edges: dict[tuple[tuple[float, float], tuple[float, float]], tuple[np.ndarray, np.ndarray]] = {}
    for face in faces:
        indexes = face["indexes"]
        for first, second in zip(indexes, indexes[1:] + indexes[:1]):
            if first == second:
                continue
            if (
                abs(float(stations[first]) - seam_station_m) > tolerance_m
                or abs(float(stations[second]) - seam_station_m) > tolerance_m
            ):
                continue
            point_a = cross_height[first]
            point_b = cross_height[second]
            if float(np.linalg.norm(point_b - point_a)) <= 1e-9:
                continue
            rounded_a = tuple(float(value) for value in np.round(point_a, 6))
            rounded_b = tuple(float(value) for value in np.round(point_b, 6))
            key = tuple(sorted((rounded_a, rounded_b)))
            boundary_edges[key] = (point_a, point_b)
    samples: list[np.ndarray] = []
    for point_a, point_b in boundary_edges.values():
        sample_count = max(
            2,
            int(np.ceil(float(np.linalg.norm(point_b - point_a)) / sample_spacing_m))
            + 1,
        )
        amount = np.linspace(0.0, 1.0, sample_count)[:, None]
        samples.append(point_a * (1.0 - amount) + point_b * amount)
    if not samples:
        return np.empty((0, 2), dtype=np.float64), 0
    return np.vstack(samples), len(boundary_edges)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, len(source), 500):
        delta = source[start : start + 500, None, :] - target[None, :, :]
        chunks.append(np.linalg.norm(delta, axis=2).min(axis=1))
    return np.concatenate(chunks)


def audit_owned_mesh_seams(
    project: ProjectConfig,
    ownership_plan_path: str | Path,
    segment_mesh_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    p90_max_m: float,
    boundary_tolerance_m: float = 0.002,
    sample_spacing_m: float = 0.02,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure shared cut-edge continuity without depending on triangulation layout."""
    if p90_max_m <= 0 or boundary_tolerance_m <= 0 or sample_spacing_m <= 0:
        raise ValueError("Seam audit tolerances must be positive")
    plan_path = project.resolve(ownership_plan_path)
    output = project.resolve(output_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    plan = load_json(plan_path)
    frame = plan["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    lateral = np.asarray(frame["lateral_xy"], dtype=np.float64)
    segment_ids = [segment["segment_id"] for segment in plan["segments"]]
    if set(segment_mesh_paths) != set(segment_ids):
        raise ValueError("Segment mesh IDs must exactly match the ownership plan")
    meshes = {key: project.resolve(value) for key, value in segment_mesh_paths.items()}
    for path in meshes.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    seam_results = []
    for index, seam_station in enumerate(plan["seam_stations_m"]):
        left_id = segment_ids[index]
        right_id = segment_ids[index + 1]
        left, left_edge_count = _sample_boundary_edges(
            meshes[left_id],
            frame_origin,
            along,
            lateral,
            float(seam_station),
            tolerance_m=boundary_tolerance_m,
            sample_spacing_m=sample_spacing_m,
        )
        right, right_edge_count = _sample_boundary_edges(
            meshes[right_id],
            frame_origin,
            along,
            lateral,
            float(seam_station),
            tolerance_m=boundary_tolerance_m,
            sample_spacing_m=sample_spacing_m,
        )
        if not len(left) or not len(right):
            distances = np.asarray([np.inf])
        else:
            distances = np.concatenate(
                (_nearest_distances(left, right), _nearest_distances(right, left))
            )
        metrics = {
            "p50_m": float(np.percentile(distances, 50)),
            "p90_m": float(np.percentile(distances, 90)),
            "p95_m": float(np.percentile(distances, 95)),
            "maximum_m": float(np.max(distances)),
        }
        passed = bool(np.isfinite(metrics["p90_m"]) and metrics["p90_m"] <= p90_max_m)
        seam_results.append(
            {
                "seam_index": index,
                "seam_station_m": float(seam_station),
                "left_segment_id": left_id,
                "right_segment_id": right_id,
                "left_boundary_edge_count": left_edge_count,
                "right_boundary_edge_count": right_edge_count,
                "left_sample_count": len(left),
                "right_sample_count": len(right),
                "symmetric_nearest_edge_distance": metrics,
                "p90_max_m": p90_max_m,
                "passed": passed,
            }
        )
    passed = all(seam["passed"] for seam in seam_results)
    result = {
        "schema_version": "railway.owned-mesh-seam-audit.v1",
        "project_id": project.project_id,
        "ownership_plan": str(plan_path),
        "segment_meshes": {key: str(path) for key, path in meshes.items()},
        "boundary_tolerance_m": boundary_tolerance_m,
        "sample_spacing_m": sample_spacing_m,
        "p90_max_m": p90_max_m,
        "seams": seam_results,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "metric_note": (
            "Symmetric nearest distance between sampled boundary edges in corridor lateral-Z space; "
            "this is insensitive to different internal triangulations."
        ),
    }
    write_json(output, result)
    return result
