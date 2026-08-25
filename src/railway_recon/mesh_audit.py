from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def audit_obj(path: Path, area_tolerance: float = 1e-12) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    object_names: set[str] = set()
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
                faces.append(tuple(indexes))
            elif line.startswith("o "):
                object_names.add(line[2:].strip())

    xyz = np.asarray(vertices, dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(xyz))) if len(xyz) else 0
    invalid_index_faces = 0
    degenerate = 0
    duplicate_keys: set[tuple[int, ...]] = set()
    duplicate_faces = 0
    triangle_count = 0
    for face in faces:
        if len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face):
            invalid_index_faces += 1
            continue
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

    rounded = [tuple(np.round(vertex, 9)) for vertex in xyz]
    duplicate_vertices = len(rounded) - len(set(rounded))
    passed = (
        bool(vertices)
        and bool(faces)
        and nonfinite == 0
        and invalid_index_faces == 0
        and degenerate == 0
        and duplicate_faces == 0
    )
    return {
        "schema_version": "railway.obj-mesh-audit.v1",
        "path": str(path.resolve()),
        "passed": passed,
        "object_count": len(object_names),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "triangle_count_after_fan_triangulation": triangle_count,
        "nonfinite_coordinate_count": nonfinite,
        "invalid_index_face_count": invalid_index_faces,
        "degenerate_triangle_count": degenerate,
        "duplicate_face_count": duplicate_faces,
        "duplicate_vertex_count": duplicate_vertices,
        "bounds": {
            "minimum": xyz.min(axis=0).tolist() if len(xyz) else None,
            "maximum": xyz.max(axis=0).tolist() if len(xyz) else None,
        },
        "limitations": [
            "Duplicate vertices across independent asset objects may be intentional.",
            "This OBJ audit does not detect cross-object coplanar overlap or runtime material flicker.",
            "Run Blender/UE visual acceptance with back-face culling and six fixed views.",
        ],
    }

