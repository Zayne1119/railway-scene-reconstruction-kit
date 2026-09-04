from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json

GLTF_ARRAY_BUFFER = 34962
GLTF_FLOAT = 5126


@dataclass(frozen=True)
class _Corner:
    vertex: int
    normal: int | None


@dataclass
class _Material:
    colour: tuple[float, float, float] = (0.62, 0.68, 0.70)
    alpha: float = 1.0


def _resolve_index(raw: str, count: int) -> int:
    value = int(raw)
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ValueError(f"OBJ index {value} is outside a collection of {count} values")
    return index


def _material_library(obj_path: Path) -> Path | None:
    references = [
        line.strip().split(maxsplit=1)[1]
        for line in obj_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip().startswith("mtllib ")
    ]
    if not references:
        return None
    if len(references) != 1:
        raise ValueError(f"Expected at most one OBJ material library; got {references}")
    material_path = (obj_path.parent / references[0]).resolve()
    if not material_path.is_file():
        raise FileNotFoundError(material_path)
    return material_path


def _parse_materials(path: Path | None) -> dict[str, _Material]:
    if path is None:
        return {}
    materials: dict[str, _Material] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("newmtl "):
            current = line.split(maxsplit=1)[1]
            materials[current] = _Material()
        elif current is not None and line.startswith("Kd "):
            values = tuple(float(value) for value in line.split()[1:4])
            materials[current].colour = tuple(
                min(1.0, max(0.0, value)) for value in values
            )
        elif current is not None and line.startswith("d "):
            materials[current].alpha = min(1.0, max(0.0, float(line.split()[1])))
        elif current is not None and line.startswith("Tr "):
            materials[current].alpha = 1.0 - min(
                1.0, max(0.0, float(line.split()[1]))
            )
    return materials


def _parse_obj(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[str, str], list[tuple[_Corner, _Corner, _Corner]]],
]:
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    groups: dict[
        tuple[str, str], list[tuple[_Corner, _Corner, _Corner]]
    ] = defaultdict(list)
    object_name = "default"
    material_name = "default"
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append(tuple(float(value) for value in parts[1:4]))
        elif parts[0] == "vn" and len(parts) >= 4:
            normals.append(tuple(float(value) for value in parts[1:4]))
        elif parts[0] in {"o", "g"} and len(parts) >= 2:
            object_name = " ".join(parts[1:])
        elif parts[0] == "usemtl" and len(parts) >= 2:
            material_name = " ".join(parts[1:])
        elif parts[0] == "f" and len(parts) >= 4:
            corners = []
            for token in parts[1:]:
                values = token.split("/")
                vertex = _resolve_index(values[0], len(vertices))
                normal = (
                    _resolve_index(values[2], len(normals))
                    if len(values) >= 3 and values[2]
                    else None
                )
                corners.append(_Corner(vertex=vertex, normal=normal))
            for index in range(1, len(corners) - 1):
                groups[(object_name, material_name)].append(
                    (corners[0], corners[index], corners[index + 1])
                )
    if not vertices or not groups:
        raise ValueError(f"OBJ has no renderable geometry: {path}")
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(normals, dtype=np.float64).reshape((-1, 3)),
        dict(groups),
    )


def _expanded_primitive(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: list[tuple[_Corner, _Corner, _Corner]],
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(triangles) * 3, 3), dtype=np.float32)
    expanded_normals = np.empty_like(positions)
    for face_index, triangle in enumerate(triangles):
        face_positions = vertices[[corner.vertex for corner in triangle]]
        normal_raw = np.cross(
            face_positions[1] - face_positions[0],
            face_positions[2] - face_positions[0],
        )
        length = float(np.linalg.norm(normal_raw))
        if length <= 1.0e-12:
            raise ValueError("Degenerate OBJ face reached GLB export")
        face_normal = normal_raw / length
        for corner_index, corner in enumerate(triangle):
            output_index = face_index * 3 + corner_index
            positions[output_index] = face_positions[corner_index]
            if corner.normal is None:
                expanded_normals[output_index] = face_normal
            else:
                candidate = normals[corner.normal]
                candidate_length = float(np.linalg.norm(candidate))
                expanded_normals[output_index] = (
                    candidate / candidate_length
                    if candidate_length > 1.0e-12
                    else face_normal
                )
    return positions, expanded_normals


def _append_vec3_accessor(
    document: dict[str, Any],
    binary: bytearray,
    values: np.ndarray,
    *,
    include_bounds: bool,
) -> int:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    data = contiguous.tobytes()
    binary.extend(data)
    view_index = len(document["bufferViews"])
    document["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
            "target": GLTF_ARRAY_BUFFER,
        }
    )
    accessor: dict[str, Any] = {
        "bufferView": view_index,
        "componentType": GLTF_FLOAT,
        "count": len(contiguous),
        "type": "VEC3",
    }
    if include_bounds:
        accessor["min"] = contiguous.min(axis=0).astype(float).tolist()
        accessor["max"] = contiguous.max(axis=0).astype(float).tolist()
    accessor_index = len(document["accessors"])
    document["accessors"].append(accessor)
    return accessor_index


def _glb_material(name: str, material: _Material) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [*material.colour, material.alpha],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.78,
        },
        "doubleSided": False,
    }
    if material.alpha < 0.999:
        value["alphaMode"] = "BLEND"
    return value


def inspect_web_glb(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        magic, version, total_length = struct.unpack("<4sII", stream.read(12))
        json_length, json_type = struct.unpack("<II", stream.read(8))
        json_bytes = stream.read(json_length)
        binary_length, binary_type = struct.unpack("<II", stream.read(8))
        stream.seek(binary_length, 1)
        actual_length = stream.tell()
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Invalid GLB header: {source}")
    if json_type != 0x4E4F534A or binary_type != 0x004E4942:
        raise ValueError(f"Invalid GLB chunk types: {source}")
    if total_length != actual_length or total_length != source.stat().st_size:
        raise ValueError(f"GLB byte length mismatch: {source}")
    document = json.loads(json_bytes.decode("utf-8").rstrip(" \x00"))
    declared_binary_length = int(document["buffers"][0]["byteLength"])
    if declared_binary_length > binary_length:
        raise ValueError("GLB buffer declaration exceeds the binary chunk")
    triangle_count = sum(
        int(document["accessors"][primitive["attributes"]["POSITION"]]["count"]) // 3
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    )
    return {
        "version": version,
        "byte_length": total_length,
        "node_count": len(document.get("nodes", [])),
        "mesh_count": len(document.get("meshes", [])),
        "primitive_count": sum(
            len(mesh.get("primitives", [])) for mesh in document.get("meshes", [])
        ),
        "material_count": len(document.get("materials", [])),
        "triangle_count": triangle_count,
        "node_names": [str(node.get("name", "")) for node in document.get("nodes", [])],
    }


def export_obj_to_web_glb(
    source_obj: str | Path,
    output_glb: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_obj).resolve()
    output = Path(output_glb).resolve()
    report_output = (
        Path(report_path).resolve()
        if report_path is not None
        else output.with_suffix(".conversion.json")
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".obj" or output.suffix.lower() != ".glb":
        raise ValueError("Expected an OBJ input and GLB output")
    if output.exists() or report_output.exists():
        raise FileExistsError(output if output.exists() else report_output)

    vertices, normals, groups = _parse_obj(source)
    material_values = _parse_materials(_material_library(source))
    used_material_names = list(dict.fromkeys(material for _, material in groups))
    material_indexes = {name: index for index, name in enumerate(used_material_names)}
    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "railway_recon.web_glb_export",
            "extras": {"coordinateSystem": "Z-up"},
        },
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [
            _glb_material(name, material_values.get(name, _Material()))
            for name in used_material_names
        ],
        "buffers": [{"byteLength": 0}],
        "bufferViews": [],
        "accessors": [],
    }
    binary = bytearray()
    object_primitives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    triangle_count = 0
    expanded_vertex_count = 0
    for (object_name, material_name), triangles in groups.items():
        positions, expanded_normals = _expanded_primitive(vertices, normals, triangles)
        position_accessor = _append_vec3_accessor(
            document, binary, positions, include_bounds=True
        )
        normal_accessor = _append_vec3_accessor(
            document, binary, expanded_normals, include_bounds=False
        )
        object_primitives[object_name].append(
            {
                "attributes": {
                    "POSITION": position_accessor,
                    "NORMAL": normal_accessor,
                },
                "material": material_indexes[material_name],
                "mode": 4,
            }
        )
        triangle_count += len(triangles)
        expanded_vertex_count += len(positions)
    for object_name, primitives in object_primitives.items():
        mesh_index = len(document["meshes"])
        document["meshes"].append({"name": object_name, "primitives": primitives})
        node_index = len(document["nodes"])
        document["nodes"].append({"name": object_name, "mesh": mesh_index})
        document["scenes"][0]["nodes"].append(node_index)

    while len(binary) % 4:
        binary.append(0)
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    binary_header = struct.pack("<II", len(binary), 0x004E4942)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(json_header)
        stream.write(json_bytes)
        stream.write(binary_header)
        stream.write(binary)
    os.replace(temporary, output)

    inspection = inspect_web_glb(output)
    gates = {
        "object_names_preserved": inspection["node_count"] == len(object_primitives)
        and set(inspection["node_names"]) == set(object_primitives),
        "triangle_count_preserved": inspection["triangle_count"] == triangle_count,
        "glb_structure_valid": inspection["mesh_count"] == len(object_primitives),
    }
    report = {
        "schema_version": "railway.web-glb-conversion.v2",
        "source_obj": str(source),
        "output_glb": str(output),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "source_size_bytes": source.stat().st_size,
        "output_size_bytes": output.stat().st_size,
        "coordinate_system": "Z-up",
        "geometry_operations": [
            "triangulate_obj_polygons_by_fan",
            "expand_face_corners_for_stable_normals",
        ],
        "source_vertex_count": len(vertices),
        "expanded_vertex_count": expanded_vertex_count,
        "triangle_count": triangle_count,
        "inspection": inspection,
        "gates": gates,
        "passed": all(gates.values()),
        "status": "pass" if all(gates.values()) else "failed",
        "limitations": [
            "This is a web interchange conversion, not a geometry repair operation.",
            "Texture coordinates are not emitted because the candidate currently uses flat MTL colours.",
        ],
    }
    write_json(report_output, report)
    if not report["passed"]:
        raise ValueError("Exported GLB failed structural validation")
    return report
