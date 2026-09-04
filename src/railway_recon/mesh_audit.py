from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def audit_obj(
    path: Path,
    area_tolerance: float = 1e-12,
    *,
    fail_on_unreferenced_vertices: bool = False,
) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[tuple[tuple[int, ...], str]] = []
    object_names: list[str] = []
    current_object = "default"
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed vertex at line {line_number}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indexes = []
                for token in line.split()[1:]:
                    value = int(token.split("/")[0])
                    indexes.append(value - 1 if value > 0 else len(vertices) + value)
                faces.append((tuple(indexes), current_object))
            elif line.startswith("o "):
                current_object = line[2:].strip()
                object_names.append(current_object)

    xyz = np.asarray(vertices, dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(xyz))) if len(xyz) else 0
    invalid_index_faces = 0
    degenerate = 0
    duplicate_keys: set[tuple[int, ...]] = set()
    duplicate_faces = 0
    coordinate_triangle_owners: dict[
        tuple[tuple[float, float, float], ...], str
    ] = {}
    coordinate_duplicate_triangles = 0
    cross_object_coordinate_duplicate_triangles = 0
    coordinate_duplicate_examples: list[dict[str, str]] = []
    referenced_vertex_indexes: set[int] = set()
    triangle_count = 0
    for face, object_name in faces:
        if len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face):
            invalid_index_faces += 1
            continue
        referenced_vertex_indexes.update(face)
        key = tuple(sorted(face))
        if key in duplicate_keys:
            duplicate_faces += 1
        duplicate_keys.add(key)
        for offset in range(1, len(face) - 1):
            triangle_count += 1
            a, b, c = xyz[face[0]], xyz[face[offset]], xyz[face[offset + 1]]
            area = float(np.linalg.norm(np.cross(b - a, c - a)) * 0.5)
            if not math.isfinite(area) or area <= area_tolerance:
                degenerate += 1
            coordinate_key = tuple(
                sorted(
                    tuple(float(value) for value in xyz[index])
                    for index in (face[0], face[offset], face[offset + 1])
                )
            )
            first_owner = coordinate_triangle_owners.get(coordinate_key)
            if first_owner is None:
                coordinate_triangle_owners[coordinate_key] = object_name
            else:
                coordinate_duplicate_triangles += 1
                if first_owner != object_name:
                    cross_object_coordinate_duplicate_triangles += 1
                if len(coordinate_duplicate_examples) < 25:
                    coordinate_duplicate_examples.append(
                        {
                            "first_object": first_owner,
                            "duplicate_object": object_name,
                        }
                    )

    rounded = [tuple(np.round(vertex, 9)) for vertex in xyz]
    duplicate_vertices = len(rounded) - len(set(rounded))
    unreferenced_vertices = len(vertices) - len(referenced_vertex_indexes)
    duplicate_object_names = len(object_names) - len(set(object_names))
    passed = (
        bool(vertices)
        and bool(faces)
        and nonfinite == 0
        and invalid_index_faces == 0
        and degenerate == 0
        and duplicate_faces == 0
        and coordinate_duplicate_triangles == 0
        and duplicate_object_names == 0
        and (not fail_on_unreferenced_vertices or unreferenced_vertices == 0)
    )
    return {
        "schema_version": "railway.obj-mesh-audit.v3",
        "path": str(path.resolve()),
        "passed": passed,
        "object_count": len(set(object_names)),
        "object_declaration_count": len(object_names),
        "object_names": sorted(set(object_names)),
        "duplicate_object_name_count": duplicate_object_names,
        "vertex_count": len(vertices),
        "referenced_vertex_count": len(referenced_vertex_indexes),
        "unreferenced_vertex_count": unreferenced_vertices,
        "unreferenced_vertex_ratio": (
            unreferenced_vertices / len(vertices) if vertices else 0.0
        ),
        "fail_on_unreferenced_vertices": fail_on_unreferenced_vertices,
        "face_count": len(faces),
        "triangle_count_after_fan_triangulation": triangle_count,
        "nonfinite_coordinate_count": nonfinite,
        "invalid_index_face_count": invalid_index_faces,
        "degenerate_triangle_count": degenerate,
        "duplicate_face_count": duplicate_faces,
        "coordinate_duplicate_triangle_count": coordinate_duplicate_triangles,
        "cross_object_coordinate_duplicate_triangle_count": (
            cross_object_coordinate_duplicate_triangles
        ),
        "coordinate_duplicate_examples": coordinate_duplicate_examples,
        "duplicate_vertex_count": duplicate_vertices,
        "bounds": {
            "minimum": xyz.min(axis=0).tolist() if len(xyz) else None,
            "maximum": xyz.max(axis=0).tolist() if len(xyz) else None,
        },
        "limitations": [
            "Duplicate vertices across independent asset objects may be intentional.",
            "Unreferenced vertices are reported and may be made fatal by policy.",
            "Exact cross-object duplicate triangles are rejected by coordinate identity.",
            "Near-coplanar overlap and runtime material flicker still require a tolerance-based engine audit.",
            "Run Blender/UE visual acceptance with back-face culling and six fixed views.",
        ],
    }
