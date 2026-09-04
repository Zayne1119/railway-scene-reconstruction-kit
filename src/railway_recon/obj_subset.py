from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _GeometryLine:
    raw: str
    vertex_count: int
    texture_count: int
    normal_count: int


@dataclass
class _ObjectBlock:
    name: str
    lines: list[_GeometryLine]


def _resolve_index(value: str, count: int) -> int:
    index = int(value)
    resolved = index - 1 if index > 0 else count + index
    if resolved < 0 or resolved >= count:
        raise ValueError(f"OBJ index {index} resolves outside 1..{count}")
    return resolved


def _referenced_indexes(
    line: _GeometryLine,
    vertices: set[int],
    textures: set[int],
    normals: set[int],
) -> None:
    parts = line.raw.split()
    if not parts:
        return
    if parts[0] == "f":
        for token in parts[1:]:
            values = token.split("/")
            if values[0]:
                vertices.add(_resolve_index(values[0], line.vertex_count))
            if len(values) > 1 and values[1]:
                textures.add(_resolve_index(values[1], line.texture_count))
            if len(values) > 2 and values[2]:
                normals.add(_resolve_index(values[2], line.normal_count))
    elif parts[0] in {"l", "p"}:
        for token in parts[1:]:
            values = token.split("/")
            vertices.add(_resolve_index(values[0], line.vertex_count))
            if len(values) > 1 and values[1]:
                textures.add(_resolve_index(values[1], line.texture_count))


def _rewrite_geometry_line(
    line: _GeometryLine,
    vertex_map: dict[int, int],
    texture_map: dict[int, int],
    normal_map: dict[int, int],
) -> str:
    parts = line.raw.split()
    if not parts or parts[0] not in {"f", "l", "p"}:
        return line.raw
    rewritten = [parts[0]]
    for token in parts[1:]:
        values = token.split("/")
        vertex = vertex_map[_resolve_index(values[0], line.vertex_count)]
        output = [str(vertex)]
        if len(values) > 1:
            output.append(
                str(texture_map[_resolve_index(values[1], line.texture_count)])
                if values[1]
                else ""
            )
        if len(values) > 2:
            output.append(
                str(normal_map[_resolve_index(values[2], line.normal_count)])
                if values[2]
                else ""
            )
        rewritten.append("/".join(output))
    return " ".join(rewritten)


def subset_obj_objects(
    source_obj: str | Path,
    output_obj: str | Path,
    object_names: Iterable[str],
    *,
    output_mtl: str | Path | None = None,
) -> dict[str, Any]:
    """Write a compact OBJ containing only the requested renderable objects.

    The function remaps vertex, texture and normal indexes instead of leaving the
    unused geometry from the source file in the subset. Object names and material
    assignments are preserved so asset registries can continue to address nodes.
    """

    source = Path(source_obj).resolve()
    output = Path(output_obj).resolve()
    requested = list(dict.fromkeys(str(value) for value in object_names))
    if not requested:
        raise ValueError("At least one OBJ object name is required")
    if not source.is_file():
        raise FileNotFoundError(source)

    vertices: list[str] = []
    textures: list[str] = []
    normals: list[str] = []
    material_libraries: list[str] = []
    blocks: list[_ObjectBlock] = []
    current: _ObjectBlock | None = None
    for raw in source.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("v "):
            vertices.append(line)
            continue
        if line.startswith("vt "):
            textures.append(line)
            continue
        if line.startswith("vn "):
            normals.append(line)
            continue
        if line.startswith("mtllib "):
            material_libraries.append(line.split(maxsplit=1)[1])
            continue
        if line.startswith("o "):
            current = _ObjectBlock(line[2:].strip(), [])
            blocks.append(current)
            continue
        if current is not None and line:
            current.lines.append(
                _GeometryLine(line, len(vertices), len(textures), len(normals))
            )

    by_name = {block.name: block for block in blocks}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"Requested OBJ objects are missing: {missing}")
    selected = [by_name[name] for name in requested]

    used_vertices: set[int] = set()
    used_textures: set[int] = set()
    used_normals: set[int] = set()
    face_count = 0
    for block in selected:
        for line in block.lines:
            _referenced_indexes(line, used_vertices, used_textures, used_normals)
            if line.raw.startswith("f "):
                face_count += 1
    if not used_vertices or not face_count:
        raise ValueError("Selected OBJ objects contain no faces")

    vertex_order = sorted(used_vertices)
    texture_order = sorted(used_textures)
    normal_order = sorted(used_normals)
    vertex_map = {old: new for new, old in enumerate(vertex_order, start=1)}
    texture_map = {old: new for new, old in enumerate(texture_order, start=1)}
    normal_map = {old: new for new, old in enumerate(normal_order, start=1)}

    if output_mtl is None:
        output_material = output.with_suffix(".mtl")
    else:
        output_material = Path(output_mtl).resolve()
    if len(material_libraries) != 1:
        raise ValueError(
            f"Expected exactly one material library in {source}; got {material_libraries}"
        )
    source_material = (source.parent / material_libraries[0]).resolve()
    if not source_material.is_file():
        raise FileNotFoundError(source_material)

    lines = [
        "# Compact asset subset generated by railway_recon.obj_subset",
        f"# Source: {source}",
        f"mtllib {output_material.name}",
        *(vertices[index] for index in vertex_order),
        *(textures[index] for index in texture_order),
        *(normals[index] for index in normal_order),
    ]
    for block in selected:
        lines.extend(("", f"o {block.name}"))
        lines.extend(
            _rewrite_geometry_line(line, vertex_map, texture_map, normal_map)
            for line in block.lines
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_material.parent.mkdir(parents=True, exist_ok=True)
    temporary_obj = output.with_suffix(output.suffix + ".tmp")
    temporary_mtl = output_material.with_suffix(output_material.suffix + ".tmp")
    temporary_obj.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(source_material, temporary_mtl)
    os.replace(temporary_obj, output)
    os.replace(temporary_mtl, output_material)
    return {
        "source_obj": str(source),
        "output_obj": str(output),
        "output_mtl": str(output_material),
        "object_count": len(selected),
        "vertex_count": len(vertex_order),
        "texture_coordinate_count": len(texture_order),
        "normal_count": len(normal_order),
        "face_count": face_count,
        "object_names": requested,
    }
