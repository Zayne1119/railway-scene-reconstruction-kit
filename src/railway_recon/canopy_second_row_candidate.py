from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .canopy_column_alignment_refinement import derive_column_row_translations
from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import summarize_registry


def _object_blocks(lines: list[str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("o "):
            current = stripped[2:].strip()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return blocks


def _remap_face(line: str, mapping: dict[int, int]) -> str:
    tokens = line.strip().split()
    remapped = ["f"]
    for token in tokens[1:]:
        components = token.split("/")
        old_index = int(components[0])
        if old_index <= 0:
            raise ValueError("Negative/relative OBJ face indices are not supported")
        components[0] = str(mapping[old_index])
        remapped.append("/".join(components))
    return " ".join(remapped) + "\n"


def clone_obj_objects(
    source_path: str | Path,
    output_path: str | Path,
    clone_specs: list[dict[str, Any]],
    *,
    output_mtl_name: str,
    omit_objects: set[str] | None = None,
) -> None:
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    blocks = _object_blocks(lines)
    model = parse_obj_model(source)
    omitted = omit_objects or set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as writer:
        current_omitted = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("o "):
                current_omitted = stripped[2:].strip() in omitted
            if current_omitted:
                # OBJ indexes are global. Some generators interleave vertex and
                # normal declarations inside object blocks, so removing those
                # declarations would shift every subsequent cloned face index.
                # Keep index-bearing declarations while withholding the object's
                # name, materials and faces.
                if stripped.startswith(("v ", "vn ", "vt ")):
                    writer.write(line if line.endswith("\n") else line + "\n")
                continue
            if stripped.startswith("mtllib "):
                writer.write(f"mtllib {output_mtl_name}\n")
            else:
                writer.write(line if line.endswith("\n") else line + "\n")
        writer.write("\n# Supplemental V2 second canopy-column row candidates.\n")
        next_vertex_index = len(model.vertices) + 1
        for spec in clone_specs:
            source_name = str(spec["source_object"])
            clone_name = str(spec["clone_object"])
            if source_name not in blocks:
                raise ValueError(f"Source OBJ object block missing: {source_name}")
            indices = object_vertex_indices(model, source_name)
            translation = np.asarray(spec["translation_local_xyz_m"], dtype=np.float64)
            mapping: dict[int, int] = {}
            for zero_based in indices:
                original_one_based = int(zero_based) + 1
                mapping[original_one_based] = next_vertex_index
                value = model.vertices[int(zero_based)] + translation
                writer.write(f"v {value[0]:.9f} {value[1]:.9f} {value[2]:.9f}\n")
                next_vertex_index += 1
            writer.write(f"o {clone_name}\n")
            for line in blocks[source_name]:
                stripped = line.strip()
                if stripped.startswith("f "):
                    writer.write(_remap_face(line, mapping))
                elif not stripped.startswith(("o ", "v ", "vn ", "vt ")):
                    writer.write(line if line.endswith("\n") else line + "\n")


def _clone_specs(
    model: Any,
    rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row_name, row in rows.items():
        delta_xy = row["translation_world_xy_m"]
        for column_id in row["column_ids"]:
            clone_column_id = f"{column_id}-SUPPLEMENTAL-ROW-02"
            associated = sorted(
                name
                for name in model.faces_by_object
                if name == column_id or name.startswith(f"{column_id}-")
            )
            for source_name in associated:
                suffix = source_name[len(column_id) :]
                specs.append(
                    {
                        "row": row_name,
                        "source_column": column_id,
                        "source_object": source_name,
                        "clone_object": f"{clone_column_id}{suffix}",
                        "translation_local_xyz_m": [delta_xy[0], delta_xy[1], 0.0],
                    }
                )
    return specs


def _update_registry(
    source_path: Path,
    output_path: Path,
    output_obj: Path,
    clone_specs: list[dict[str, Any]],
    comparison_path: Path,
    photo_review_path: Path,
) -> dict[str, Any]:
    registry = load_json(source_path)
    assets = registry.get("assets", [])
    source_by_id = {str(asset["id"]): asset for asset in assets}
    for asset in assets:
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    for spec in clone_specs:
        source_id = str(spec["source_object"])
        if source_id not in source_by_id:
            raise ValueError(f"Registry source asset missing: {source_id}")
        cloned = copy.deepcopy(source_by_id[source_id])
        cloned["id"] = str(spec["clone_object"])
        cloned["status"] = "candidate"
        cloned["evidence_level"] = "supplemental_point_cloud_and_multi_view_photo"
        cloned["confidence"] = 0.88
        cloned["geometry"]["file"] = str(output_obj)
        cloned["geometry"]["node"] = str(spec["clone_object"])
        cloned.setdefault("parameters", {}).update(
            {
                "source_template_asset_id": source_id,
                "row": spec["row"],
                "construction": "parallel_periodic_row_from_observed_phase",
            }
        )
        cloned.setdefault("sources", []).extend(
            [
                {
                    "kind": "point_cloud",
                    "reference": str(comparison_path),
                    "note": "Supplemental V2 periodic vertical row support.",
                },
                {
                    "kind": "panorama",
                    "reference": str(photo_review_path),
                    "note": "Multi-view column semantics review package.",
                },
            ]
        )
        cloned["limitations"] = [
            "Candidate second-row component; retain per-object point-support QA.",
            "Unobserved positions inherit the existing 9 m column-grid phase.",
        ]
        assets.append(cloned)
    registry["assets"] = assets
    registry["release_id"] = "site_b_s2050_2200_supplement_v2_second_row_candidate"
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    write_json(output_path, registry)
    return registry


def build_canopy_second_row_candidate(
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    comparison_path: str | Path,
    frame_report_path: str | Path,
    photo_review_path: str | Path,
    output_directory: str | Path,
    semantic_decisions_path: str | Path | None = None,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_second_row_candidate"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "second_row_candidate_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    origin = np.asarray(load_json(Path(model_origin))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    comparison = load_json(Path(comparison_path))
    frame = load_json(Path(frame_report_path))["frame"]
    rows = derive_column_row_translations(model, origin, registry, comparison, frame)
    specs = _clone_specs(model, rows)
    confirmed_candidate_ids: set[str] | None = None
    if semantic_decisions_path is not None:
        decisions = load_json(Path(semantic_decisions_path))
        confirmed_candidate_ids = {
            str(item["candidate_id"])
            for item in decisions.get("decisions", [])
            if item.get("decision") == "confirmed_canopy_column"
        }
        confirmed_columns = {
            str(match["column_id"])
            for row in rows.values()
            for match in row["matches"]
            if str(match["candidate_id"]) in confirmed_candidate_ids
        }
        specs = [spec for spec in specs if spec["source_column"] in confirmed_columns]
        if not specs:
            raise ValueError("No clone objects remain after semantic decision filtering")
    clone_obj_objects(source_obj, output_obj, specs, output_mtl_name=output_mtl.name)
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    updated_registry = _update_registry(
        Path(source_registry),
        output_registry,
        output_obj,
        specs,
        Path(comparison_path).resolve(),
        Path(photo_review_path).resolve(),
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Second-row canopy candidate failed OBJ mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-second-row-candidate.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "rows": rows,
            "clone_count": len(specs),
            "clones": specs,
            "semantic_decisions": (
                str(Path(semantic_decisions_path).resolve())
                if semantic_decisions_path is not None
                else None
            ),
            "confirmed_candidate_ids": (
                sorted(confirmed_candidate_ids)
                if confirmed_candidate_ids is not None
                else None
            ),
            "asset_count": updated_registry["summary"]["asset_count"],
            "mesh_audit": {
                "passed": mesh_audit["passed"],
                "object_count": mesh_audit["object_count"],
                "triangle_count": mesh_audit["triangle_count_after_fan_triangulation"],
                "duplicate_face_count": mesh_audit["duplicate_face_count"],
                "degenerate_triangle_count": mesh_audit["degenerate_triangle_count"],
            },
            "status": (
                "photo_confirmed_partial_second_row_candidate_requires_fixed_view_review"
                if semantic_decisions_path is not None
                else "candidate_requires_per_object_point_support_gate"
            ),
            "failed_hypothesis_predecessor": "run04 translated original rows; rejected",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "registry": output_registry,
        "origin": output_origin,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
