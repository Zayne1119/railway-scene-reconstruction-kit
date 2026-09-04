from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ObjFace:
    vertices: tuple[int, ...]
    material: str | None


@dataclass
class ObjObject:
    name: str
    faces: list[ObjFace] = field(default_factory=list)


@dataclass
class Triangle:
    vertices: tuple[int, int, int]
    material: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_obj(path: Path) -> tuple[np.ndarray, list[ObjObject], list[str]]:
    vertices: list[list[float]] = []
    objects: list[ObjObject] = []
    material_libraries: list[str] = []
    current: ObjObject | None = None
    material: str | None = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                material_libraries.extend(line.split()[1:])
            elif line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed vertex at line {line_number}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("o "):
                name = line[2:].strip()
                if not name:
                    raise ValueError(f"Empty object name at line {line_number}")
                current = ObjObject(name=name)
                objects.append(current)
            elif line.startswith("usemtl "):
                material = line[7:].strip() or None
            elif line.startswith("f "):
                if current is None:
                    current = ObjObject(name="default")
                    objects.append(current)
                indexes: list[int] = []
                for token in line.split()[1:]:
                    value = int(token.split("/")[0])
                    index = value - 1 if value > 0 else len(vertices) + value
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"Invalid face index at line {line_number}: {value}")
                    indexes.append(index)
                if len(indexes) < 3:
                    raise ValueError(f"Face has fewer than three vertices at line {line_number}")
                current.faces.append(ObjFace(tuple(indexes), material))

    if not vertices or not any(obj.faces for obj in objects):
        raise ValueError(f"OBJ has no renderable geometry: {path}")
    xyz = np.asarray(vertices, dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"OBJ contains non-finite coordinates: {path}")
    return xyz, objects, material_libraries


def _triangulate(obj: ObjObject) -> list[Triangle]:
    triangles: list[Triangle] = []
    for face in obj.faces:
        for offset in range(1, len(face.vertices) - 1):
            triangles.append(
                Triangle(
                    (face.vertices[0], face.vertices[offset], face.vertices[offset + 1]),
                    face.material,
                )
            )
    return triangles


def _edge_direction(triangle: Triangle, edge: tuple[int, int]) -> bool:
    vertices = triangle.vertices
    for offset in range(3):
        a = vertices[offset]
        b = vertices[(offset + 1) % 3]
        if {a, b} == {edge[0], edge[1]}:
            return (a, b) == edge
    raise ValueError("Triangle does not contain requested edge")


def _repair_winding(
    xyz: np.ndarray, triangles: list[Triangle]
) -> tuple[list[Triangle], dict[str, int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        a, b, c = triangle.vertices
        for edge in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted(edge))].append(face_index)

    adjacency: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    nonmanifold_edges = 0
    for edge, face_indexes in edge_faces.items():
        if len(face_indexes) > 2:
            nonmanifold_edges += 1
            continue
        if len(face_indexes) != 2:
            continue
        first, second = face_indexes
        same_direction = _edge_direction(triangles[first], edge) == _edge_direction(
            triangles[second], edge
        )
        adjacency[first].append((second, same_direction))
        adjacency[second].append((first, same_direction))

    flip: dict[int, bool] = {}
    components: list[list[int]] = []
    winding_conflicts = 0
    for seed in range(len(triangles)):
        if seed in flip:
            continue
        flip[seed] = False
        component: list[int] = []
        queue: deque[int] = deque([seed])
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour, same_direction in adjacency.get(current, []):
                expected = flip[current] ^ same_direction
                if neighbour in flip:
                    if flip[neighbour] != expected:
                        winding_conflicts += 1
                    continue
                flip[neighbour] = expected
                queue.append(neighbour)
        components.append(component)

    repaired: list[Triangle] = []
    adjacency_flipped = 0
    for index, triangle in enumerate(triangles):
        if flip[index]:
            a, b, c = triangle.vertices
            repaired.append(Triangle((a, c, b), triangle.material))
            adjacency_flipped += 1
        else:
            repaired.append(triangle)

    closed_components_inverted = 0
    for component in components:
        component_set = set(component)
        component_edges: dict[tuple[int, int], int] = defaultdict(int)
        for face_index in component:
            a, b, c = repaired[face_index].vertices
            for edge in ((a, b), (b, c), (c, a)):
                component_edges[tuple(sorted(edge))] += 1
        closed = bool(component_edges) and all(count == 2 for count in component_edges.values())
        if not closed:
            continue
        volume6 = 0.0
        for face_index in component_set:
            a, b, c = xyz[list(repaired[face_index].vertices)]
            volume6 += float(np.dot(a, np.cross(b, c)))
        if volume6 < 0:
            closed_components_inverted += 1
            for face_index in component:
                triangle = repaired[face_index]
                a, b, c = triangle.vertices
                repaired[face_index] = Triangle((a, c, b), triangle.material)

    boundary_edges = sum(1 for indexes in edge_faces.values() if len(indexes) == 1)
    return repaired, {
        "connected_component_count": len(components),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "winding_constraint_conflict_count": winding_conflicts,
        "adjacency_repaired_triangle_count": adjacency_flipped,
        "closed_component_outward_flip_count": closed_components_inverted,
    }


def _normalised(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-15 or not math.isfinite(length):
        raise ValueError("Cannot normalise zero or non-finite vector")
    return vector / length


def _corner_normals(
    xyz: np.ndarray,
    triangles: list[Triangle],
    crease_angle_degrees: float,
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    area_vectors: list[np.ndarray] = []
    unit_normals: list[np.ndarray] = []
    incident: dict[int, list[int]] = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        a, b, c = xyz[list(triangle.vertices)]
        area_vector = np.cross(b - a, c - a)
        area_vectors.append(area_vector)
        unit_normals.append(_normalised(area_vector))
        for vertex in triangle.vertices:
            incident[vertex].append(face_index)

    cosine_limit = math.cos(math.radians(crease_angle_degrees))
    normals: list[np.ndarray] = []
    normal_lookup: dict[tuple[float, float, float], int] = {}
    corner_indexes: list[tuple[int, int, int]] = []
    for face_index, triangle in enumerate(triangles):
        corner: list[int] = []
        reference = unit_normals[face_index]
        for vertex in triangle.vertices:
            selected = [
                area_vectors[other]
                for other in incident[vertex]
                if float(np.dot(reference, unit_normals[other])) >= cosine_limit
            ]
            value = _normalised(np.sum(selected, axis=0))
            key = tuple(float(item) for item in np.round(value, 9))
            normal_index = normal_lookup.get(key)
            if normal_index is None:
                normal_index = len(normals)
                normal_lookup[key] = normal_index
                normals.append(value)
            corner.append(normal_index)
        corner_indexes.append(tuple(corner))
    return normals, corner_indexes


def _coordinate_duplicate_triangles(
    xyz: np.ndarray, objects: list[tuple[ObjObject, list[Triangle]]]
) -> list[dict[str, Any]]:
    seen: dict[tuple[tuple[float, float, float], ...], tuple[str, int]] = {}
    duplicates: list[dict[str, Any]] = []
    for obj, triangles in objects:
        for triangle_index, triangle in enumerate(triangles):
            points = [tuple(float(item) for item in np.round(xyz[index], 6)) for index in triangle.vertices]
            key = tuple(sorted(points))
            if key in seen:
                source_object, source_triangle = seen[key]
                duplicates.append(
                    {
                        "first_object": source_object,
                        "first_triangle_index": source_triangle,
                        "duplicate_object": obj.name,
                        "duplicate_triangle_index": triangle_index,
                        "rounded_vertices_m": [list(point) for point in key],
                    }
                )
            else:
                seen[key] = (obj.name, triangle_index)
    return duplicates


def prepare_obj_interchange(
    source: Path,
    output: Path,
    *,
    crease_angle_degrees: float = 35.0,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Input and output OBJ paths must differ")
    if not 0.0 < crease_angle_degrees < 180.0:
        raise ValueError("crease_angle_degrees must be between 0 and 180")

    xyz, parsed_objects, material_libraries = _parse_obj(source)
    repaired_objects: list[tuple[ObjObject, list[Triangle], dict[str, int]]] = []
    for obj in parsed_objects:
        triangles = _triangulate(obj)
        repaired, source_topology = _repair_winding(xyz, triangles)
        repaired_objects.append((obj, repaired, source_topology))

    source_coordinate_duplicates = _coordinate_duplicate_triangles(
        xyz, [(obj, triangles) for obj, triangles, _ in repaired_objects]
    )
    duplicate_indexes = {
        (item["duplicate_object"], item["duplicate_triangle_index"])
        for item in source_coordinate_duplicates
        if item["first_object"] == item["duplicate_object"]
    }

    processed: list[tuple[ObjObject, list[Triangle], list[np.ndarray], list[tuple[int, int, int]]]] = []
    object_reports: list[dict[str, Any]] = []
    for obj, triangles, source_topology in repaired_objects:
        filtered = [
            triangle
            for triangle_index, triangle in enumerate(triangles)
            if (obj.name, triangle_index) not in duplicate_indexes
        ]
        repaired, topology = _repair_winding(xyz, filtered)
        topology["winding_constraint_conflict_count"] += source_topology[
            "winding_constraint_conflict_count"
        ]
        topology["adjacency_repaired_triangle_count"] += source_topology[
            "adjacency_repaired_triangle_count"
        ]
        topology["closed_component_outward_flip_count"] += source_topology[
            "closed_component_outward_flip_count"
        ]
        normals, corners = _corner_normals(xyz, repaired, crease_angle_degrees)
        processed.append((obj, repaired, normals, corners))
        object_reports.append(
            {
                "name": obj.name,
                "source_face_count": len(obj.faces),
                "triangle_count": len(repaired),
                "normal_count": len(normals),
                "removed_duplicate_triangle_count": len(triangles) - len(filtered),
                **topology,
            }
        )

    flattened = [(obj, triangles) for obj, triangles, _, _ in processed]
    output_coordinate_duplicates = _coordinate_duplicate_triangles(xyz, flattened)
    normal_offsets: list[int] = []
    normal_count = 0
    for _, _, normals, _ in processed:
        normal_offsets.append(normal_count)
        normal_count += len(normals)

    output.parent.mkdir(parents=True, exist_ok=True)
    copied_materials: list[str] = []
    for library_index, library in enumerate(material_libraries):
        material_source = source.parent / library
        if material_source.is_file():
            material_target = (
                output.with_suffix(".mtl")
                if len(material_libraries) == 1 and library_index == 0
                else output.parent / Path(library).name
            )
            if material_source.resolve() != material_target.resolve():
                shutil.copy2(material_source, material_target)
            copied_materials.append(material_target.name)

    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Deterministic real-time interchange OBJ\n")
        stream.write(f"# Source SHA256: {_sha256(source)}\n")
        stream.write(f"# Crease angle: {crease_angle_degrees:.3f} degrees\n")
        for library in copied_materials:
            stream.write(f"mtllib {library}\n")
        for vertex in xyz:
            stream.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for _, _, normals, _ in processed:
            for normal in normals:
                stream.write(f"vn {normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f}\n")
        for object_index, (obj, triangles, _, corner_indexes) in enumerate(processed):
            stream.write(f"\no {obj.name}\n")
            stream.write("s off\n")
            current_material: str | None = None
            normal_offset = normal_offsets[object_index]
            for triangle, corner in zip(triangles, corner_indexes, strict=True):
                if triangle.material != current_material:
                    if triangle.material:
                        stream.write(f"usemtl {triangle.material}\n")
                    current_material = triangle.material
                tokens = [
                    f"{vertex + 1}//{normal_offset + normal + 1}"
                    for vertex, normal in zip(triangle.vertices, corner, strict=True)
                ]
                stream.write("f " + " ".join(tokens) + "\n")

    total_triangles = sum(len(triangles) for _, triangles, _, _ in processed)
    report = {
        "schema_version": "railway.obj-interchange-preparation.v1",
        "source": str(source),
        "output": str(output),
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "object_count": len(processed),
        "vertex_count": len(xyz),
        "triangle_count": total_triangles,
        "explicit_normal_count": normal_count,
        "crease_angle_degrees": crease_angle_degrees,
        "source_coordinate_duplicate_triangle_count": len(source_coordinate_duplicates),
        "removed_coordinate_duplicate_triangle_count": len(duplicate_indexes),
        "removed_coordinate_duplicate_triangles": [
            item
            for item in source_coordinate_duplicates
            if item["first_object"] == item["duplicate_object"]
        ],
        "coordinate_duplicate_triangle_count": len(output_coordinate_duplicates),
        "coordinate_duplicate_triangles": output_coordinate_duplicates,
        "nonmanifold_edge_count": sum(item["nonmanifold_edge_count"] for item in object_reports),
        "winding_constraint_conflict_count": sum(
            item["winding_constraint_conflict_count"] for item in object_reports
        ),
        "adjacency_repaired_triangle_count": sum(
            item["adjacency_repaired_triangle_count"] for item in object_reports
        ),
        "closed_component_outward_flip_count": sum(
            item["closed_component_outward_flip_count"] for item in object_reports
        ),
        "material_libraries": copied_materials,
        "objects": object_reports,
        "passed": not output_coordinate_duplicates
        and all(item["winding_constraint_conflict_count"] == 0 for item in object_reports),
        "limitations": [
            "Boundary edges are reported but may be intentional for observed surface assets.",
            "Near-coplanar cross-object overlap still requires fixed-view and target-renderer review.",
            "This tool does not create UV coordinates or textures.",
        ],
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
