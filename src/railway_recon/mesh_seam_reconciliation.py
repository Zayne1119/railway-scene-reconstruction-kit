from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import summarize_registry, validate_registry_value


def _corridor_frame(value: dict[str, Any]) -> CorridorFrame:
    frame = value.get("frame", value)
    cross = frame.get("cross_xy", frame.get("lateral_xy"))
    if cross is None:
        raise ValueError("Frame requires cross_xy or lateral_xy")
    return CorridorFrame(
        np.asarray(frame["origin_xy"], dtype=np.float64),
        np.asarray(frame["along_xy"], dtype=np.float64),
        np.asarray(cross, dtype=np.float64),
    )


def _parse_obj(
    path: Path,
) -> tuple[list[str], np.ndarray, dict[str, set[int]], list[dict[str, Any]], list[str]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    vertices: list[list[float]] = []
    objects: dict[str, set[int]] = {}
    faces: list[dict[str, Any]] = []
    material_libraries: list[str] = []
    current_object: str | None = None
    for line_index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("mtllib "):
            material_libraries.extend(line.split()[1:])
        elif line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("o "):
            current_object = line.split(maxsplit=1)[1]
            objects.setdefault(current_object, set())
        elif line.startswith("f "):
            if current_object is None:
                current_object = "default"
                objects.setdefault(current_object, set())
            indexes: list[int] = []
            for token in line.split()[1:]:
                value = int(token.split("/")[0])
                index = value - 1 if value > 0 else len(vertices) + value
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"Invalid OBJ face index on line {line_index + 1}")
                indexes.append(index)
                objects[current_object].add(index)
            if len(indexes) < 3:
                raise ValueError(f"OBJ face has fewer than three vertices on line {line_index + 1}")
            faces.append(
                {
                    "line_index": line_index,
                    "object": current_object,
                    "indexes": indexes,
                }
            )
    if not vertices or not faces:
        raise ValueError(f"OBJ has no mesh geometry: {path}")
    return lines, np.asarray(vertices, dtype=np.float64), objects, faces, material_libraries


def _endpoint_clusters(
    indices: set[int],
    vertices: np.ndarray,
    origin: np.ndarray,
    frame: CorridorFrame,
    seam_station_m: float,
    *,
    search_tolerance_m: float,
    cluster_tolerance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = np.asarray(sorted(indices), dtype=np.int64)
    world = vertices[ordered] + origin
    station, cross = frame.project(world[:, 0], world[:, 1])
    residual = np.abs(station - seam_station_m)
    selected = residual <= search_tolerance_m
    if not np.any(selected):
        raise ValueError(
            f"No endpoint vertex is within {search_tolerance_m} m of seam {seam_station_m}"
        )
    selected_indices = ordered[selected]
    selected_cross = cross[selected]
    selected_z = world[selected, 2]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, c_value, z_value in zip(selected_indices, selected_cross, selected_z, strict=True):
        key = (
            round(float(c_value) / cluster_tolerance_m),
            round(float(z_value) / cluster_tolerance_m),
        )
        groups.setdefault(key, []).append(int(index))
    clusters = []
    for indexes in groups.values():
        cluster_world = vertices[np.asarray(indexes, dtype=np.int64)] + origin
        _, cluster_cross = frame.project(cluster_world[:, 0], cluster_world[:, 1])
        clusters.append(
            {
                "indexes": indexes,
                "cross_m": float(np.mean(cluster_cross)),
                "z_m": float(np.mean(cluster_world[:, 2])),
            }
        )
    clusters.sort(key=lambda item: (item["cross_m"], item["z_m"]))
    return clusters, {
        "selected_vertex_count": int(selected_indices.size),
        "unique_endpoint_count": len(clusters),
        "nearest_station_residual_m": float(np.min(residual)),
        "maximum_selected_station_residual_m": float(np.max(residual[selected])),
    }


def weld_object_pair_endpoints(
    left_object: str,
    right_object: str,
    seam_station_m: float,
    vertices: np.ndarray,
    objects: dict[str, set[int]],
    origin: np.ndarray,
    frame: CorridorFrame,
    *,
    search_tolerance_m: float,
    cluster_tolerance_m: float,
    maximum_cross_delta_m: float,
    maximum_z_delta_m: float,
) -> dict[str, Any]:
    """Move paired endpoint clusters onto one shared corridor seam."""
    for name in (left_object, right_object):
        if name not in objects:
            raise ValueError(f"Seam object is missing from OBJ: {name}")
    left, left_summary = _endpoint_clusters(
        objects[left_object],
        vertices,
        origin,
        frame,
        seam_station_m,
        search_tolerance_m=search_tolerance_m,
        cluster_tolerance_m=cluster_tolerance_m,
    )
    right, right_summary = _endpoint_clusters(
        objects[right_object],
        vertices,
        origin,
        frame,
        seam_station_m,
        search_tolerance_m=search_tolerance_m,
        cluster_tolerance_m=cluster_tolerance_m,
    )
    if len(left) != len(right):
        raise ValueError(
            f"Endpoint cluster mismatch for {left_object} and {right_object}: "
            f"{len(left)} != {len(right)}"
        )
    if len(left) < 2:
        raise ValueError("A seam pair requires at least two unique endpoint vertices")
    maximum_cross = max(
        abs(float(a["cross_m"]) - float(b["cross_m"])) for a, b in zip(left, right, strict=True)
    )
    maximum_z = max(
        abs(float(a["z_m"]) - float(b["z_m"])) for a, b in zip(left, right, strict=True)
    )
    if maximum_cross > maximum_cross_delta_m:
        raise ValueError("Seam cross-position delta exceeds the configured repair gate")
    if maximum_z > maximum_z_delta_m:
        raise ValueError("Seam elevation delta exceeds the configured repair gate")

    targets = []
    for cluster_index, (left_cluster, right_cluster) in enumerate(zip(left, right, strict=True)):
        target_cross = 0.5 * (float(left_cluster["cross_m"]) + float(right_cluster["cross_m"]))
        target_z = 0.5 * (float(left_cluster["z_m"]) + float(right_cluster["z_m"]))
        target_xy = frame.world_xy(np.asarray([seam_station_m]), np.asarray([target_cross]))[0]
        target_local = np.asarray([target_xy[0], target_xy[1], target_z], dtype=np.float64) - origin
        for index in [*left_cluster["indexes"], *right_cluster["indexes"]]:
            vertices[index] = target_local
        targets.append(
            {
                "cluster_index": cluster_index,
                "target_cross_m": target_cross,
                "target_z_m": target_z,
                "left_vertex_count": len(left_cluster["indexes"]),
                "right_vertex_count": len(right_cluster["indexes"]),
            }
        )
    return {
        "left_object": left_object,
        "right_object": right_object,
        "left_endpoint": left_summary,
        "right_endpoint": right_summary,
        "unique_endpoint_count": len(left),
        "maximum_cross_delta_before_m": maximum_cross,
        "maximum_z_delta_before_m": maximum_z,
        "maximum_cross_delta_after_m": 0.0,
        "maximum_z_delta_after_m": 0.0,
        "targets": targets,
        "status": "pass_exact_shared_endpoint_clusters",
    }


def _face_triangle_keys(
    face: dict[str, Any], vertices: np.ndarray
) -> set[tuple[tuple[float, float, float], ...]]:
    indexes = face["indexes"]
    result = set()
    for offset in range(1, len(indexes) - 1):
        triangle = [indexes[0], indexes[offset], indexes[offset + 1]]
        points = [
            tuple(float(value) for value in np.round(vertices[index], 6)) for index in triangle
        ]
        result.add(tuple(sorted(points)))
    return result


def remove_duplicate_internal_cap(
    left_object: str,
    right_object: str,
    seam_station_m: float,
    faces: list[dict[str, Any]],
    vertices: np.ndarray,
    origin: np.ndarray,
    frame: CorridorFrame,
    *,
    cap_tolerance_m: float,
    drop_side: str,
) -> tuple[set[int], dict[str, Any]]:
    """Remove only cap polygons proven identical across a welded object pair."""
    if drop_side not in {"left", "right"}:
        raise ValueError("drop_side must be 'left' or 'right'")

    def cap_faces(name: str) -> list[dict[str, Any]]:
        selected = []
        for face in faces:
            if face["object"] != name:
                continue
            world = vertices[np.asarray(face["indexes"], dtype=np.int64)] + origin
            station, _ = frame.project(world[:, 0], world[:, 1])
            if np.all(np.abs(station - seam_station_m) <= cap_tolerance_m):
                selected.append(face)
        return selected

    left_faces = cap_faces(left_object)
    right_faces = cap_faces(right_object)
    left_keys = (
        set().union(*(_face_triangle_keys(face, vertices) for face in left_faces))
        if left_faces
        else set()
    )
    right_keys = (
        set().union(*(_face_triangle_keys(face, vertices) for face in right_faces))
        if right_faces
        else set()
    )
    duplicate_keys = left_keys & right_keys
    candidates = left_faces if drop_side == "left" else right_faces
    removed_lines: set[int] = set()
    removed_triangles = 0
    for face in candidates:
        keys = _face_triangle_keys(face, vertices)
        if keys and keys <= duplicate_keys:
            removed_lines.add(int(face["line_index"]))
            removed_triangles += len(keys)
    if duplicate_keys and not removed_lines:
        raise ValueError("Duplicate seam caps were found but no complete polygon was removable")
    remaining_left = [face for face in left_faces if face["line_index"] not in removed_lines]
    remaining_right = [face for face in right_faces if face["line_index"] not in removed_lines]
    remaining_left_keys = (
        set().union(*(_face_triangle_keys(face, vertices) for face in remaining_left))
        if remaining_left
        else set()
    )
    remaining_right_keys = (
        set().union(*(_face_triangle_keys(face, vertices) for face in remaining_right))
        if remaining_right
        else set()
    )
    remaining_duplicates = remaining_left_keys & remaining_right_keys
    if remaining_duplicates:
        raise ValueError("Duplicate internal seam-cap triangles remain after cleanup")
    return removed_lines, {
        "policy": f"drop_{drop_side}_only_when_triangle_coordinates_match",
        "left_cap_polygon_count": len(left_faces),
        "right_cap_polygon_count": len(right_faces),
        "duplicate_triangle_count_before": len(duplicate_keys),
        "removed_polygon_count": len(removed_lines),
        "removed_triangle_equivalent_count": removed_triangles,
        "duplicate_triangle_count_after": len(remaining_duplicates),
        "status": "pass" if not remaining_duplicates else "fail",
    }


def _write_obj(
    lines: list[str],
    vertices: np.ndarray,
    removed_face_lines: set[int],
    output_obj: Path,
    material_libraries: list[str],
) -> None:
    output_lines: list[str] = []
    vertex_index = 0
    material_written = False
    for line_index, raw in enumerate(lines):
        stripped = raw.strip()
        if line_index in removed_face_lines:
            continue
        if stripped.startswith("mtllib "):
            if not material_written:
                for library in material_libraries:
                    output_lines.append(f"mtllib {library}")
                material_written = True
            continue
        if stripped.startswith("v "):
            point = vertices[vertex_index]
            output_lines.append(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}")
            vertex_index += 1
        else:
            output_lines.append(raw)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_obj.with_suffix(output_obj.suffix + ".tmp")
    temporary.write_text("\n".join(output_lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, output_obj)


def reconcile_mesh_seams(
    project: ProjectConfig,
    source_obj_path: str | Path,
    source_origin_path: str | Path,
    frame_report_path: str | Path,
    settings_path: str | Path,
    output_dir: str | Path,
    *,
    registry_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Weld configured object pairs and remove proven duplicate seam caps."""
    source_obj = project.resolve(source_obj_path)
    source_origin_file = project.resolve(source_origin_path)
    frame_path = project.resolve(frame_report_path)
    settings_file = project.resolve(settings_path)
    for path in (source_obj, source_origin_file, frame_path, settings_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = load_json(settings_file)
    if settings.get("schema_version") != "railway.mesh-seam-reconciliation-settings.v1":
        raise ValueError("Unsupported mesh-seam reconciliation settings")
    seams = list(settings.get("seams", []))
    if not seams:
        raise ValueError("Mesh-seam settings contain no seams")

    output = project.resolve(output_dir)
    output_obj = output / "reconciled.obj"
    output_origin = output / "model_origin.json"
    output_audit = output / "mesh_audit.json"
    output_report = output / "seam_reconciliation.json"
    output_registry = output / "asset_registry.json"
    outputs = [output_obj, output_origin, output_audit, output_report]
    if registry_path:
        outputs.append(output_registry)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    lines, vertices, objects, faces, source_libraries = _parse_obj(source_obj)
    if len(source_libraries) != 1:
        raise ValueError("Mesh-seam reconciliation currently requires one OBJ material library")
    source_mtl = source_obj.parent / source_libraries[0]
    if not source_mtl.is_file():
        raise FileNotFoundError(source_mtl)
    output_mtl = output / "reconciled.mtl"
    material_names = [output_mtl.name]
    origin = np.asarray(load_json(source_origin_file)["origin_xyz"], dtype=np.float64)
    frame = _corridor_frame(load_json(frame_path))
    search_tolerance = float(settings.get("endpoint_search_tolerance_m", 0.10))
    cluster_tolerance = float(settings.get("endpoint_cluster_tolerance_m", 1e-5))
    cap_tolerance = float(settings.get("cap_station_tolerance_m", 1e-4))
    maximum_cross = float(settings.get("maximum_cross_delta_m", 0.50))
    maximum_z = float(settings.get("maximum_z_delta_m", 0.50))
    if min(search_tolerance, cluster_tolerance, cap_tolerance) <= 0:
        raise ValueError("Mesh-seam tolerances must be positive")

    seam_records = []
    removed_face_lines: set[int] = set()
    for seam in seams:
        seam_station = float(seam["station_m"])
        pair_records = []
        for pair in seam.get("pairs", []):
            left_object = str(pair["left_object"])
            right_object = str(pair["right_object"])
            weld = weld_object_pair_endpoints(
                left_object,
                right_object,
                seam_station,
                vertices,
                objects,
                origin,
                frame,
                search_tolerance_m=search_tolerance,
                cluster_tolerance_m=cluster_tolerance,
                maximum_cross_delta_m=float(pair.get("maximum_cross_delta_m", maximum_cross)),
                maximum_z_delta_m=float(pair.get("maximum_z_delta_m", maximum_z)),
            )
            cap_policy = str(pair.get("internal_cap_policy", "keep"))
            if cap_policy == "keep":
                cap = {
                    "policy": "keep",
                    "removed_polygon_count": 0,
                    "removed_triangle_equivalent_count": 0,
                    "duplicate_triangle_count_after": None,
                    "status": "not_requested",
                }
            elif cap_policy in {"drop_left_duplicate", "drop_right_duplicate"}:
                drop_side = "left" if "left" in cap_policy else "right"
                removed, cap = remove_duplicate_internal_cap(
                    left_object,
                    right_object,
                    seam_station,
                    faces,
                    vertices,
                    origin,
                    frame,
                    cap_tolerance_m=cap_tolerance,
                    drop_side=drop_side,
                )
                expected = pair.get("expected_removed_cap_polygon_count")
                if expected is not None and len(removed) != int(expected):
                    raise ValueError(
                        f"Unexpected removed cap count for {left_object}/{right_object}: "
                        f"{len(removed)} != {expected}"
                    )
                removed_face_lines.update(removed)
            else:
                raise ValueError(f"Unsupported internal_cap_policy: {cap_policy}")
            pair_records.append({"weld": weld, "internal_cap_cleanup": cap})
        if not pair_records:
            raise ValueError(f"Seam {seam.get('id')} contains no object pairs")
        seam_records.append(
            {
                "id": str(seam["id"]),
                "station_m": seam_station,
                "pairs": pair_records,
                "status": "pass",
            }
        )

    _write_obj(lines, vertices, removed_face_lines, output_obj, material_names)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(source_origin_file, output_origin)
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Reconciled OBJ failed mesh audit")

    resolved_registry: Path | None = None
    if registry_path:
        resolved_registry = project.resolve(registry_path)
        if not resolved_registry.is_file():
            raise FileNotFoundError(resolved_registry)
        registry = load_json(resolved_registry)
        for asset in registry.get("assets", []):
            geometry = asset.get("geometry")
            if isinstance(geometry, dict):
                geometry["file"] = str(output_obj)
        registry["summary"] = summarize_registry(registry)
        errors = validate_registry_value(registry)
        if errors:
            raise ValueError("Reconciled registry is invalid: " + "; ".join(errors))
        write_json(output_registry, registry)

    report = {
        "schema_version": "railway.mesh-seam-reconciliation.v1",
        "project_id": project.project_id,
        "source_policy": "read_only",
        "source_obj": str(source_obj),
        "source_origin": str(source_origin_file),
        "source_registry": str(resolved_registry) if resolved_registry else None,
        "frame_report": str(frame_path),
        "settings": str(settings_file),
        "source_sha256": sha256_file(source_obj),
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_registry": str(output_registry) if registry_path else None,
        "output_mesh_audit": str(output_audit),
        "output_sha256": sha256_file(output_obj),
        "seam_count": len(seam_records),
        "pair_count": sum(len(item["pairs"]) for item in seam_records),
        "removed_internal_cap_polygon_count": len(removed_face_lines),
        "seams": seam_records,
        "mesh_audit_passed": True,
        "passed": True,
        "status": "mesh_seams_welded_and_requested_internal_caps_cleaned",
        "limitations": [
            "Only explicitly configured object pairs are modified.",
            "A cap is removed only when its triangulated coordinates exactly match the opposite cap.",
            "Interior fitted vertices away from the configured seam remain unchanged.",
        ],
    }
    write_json(output_report, report)
    return report
