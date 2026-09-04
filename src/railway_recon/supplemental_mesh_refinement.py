from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import ObjModel, object_vertex_indices, parse_obj_model
from .registry import summarize_registry

PLATFORM_PATTERN = re.compile(r"^(S2050|S2100|S2150)-OPPOSITE-PLATFORM-(TOP|VOLUME)$")
COLUMN_PATTERN = re.compile(r"^(S2050|S2100|S2150)-OPPOSITE-CANOPY-COLUMN-\d+$")
UNSUPPORTED_LEGACY_PREFIXES = (
    "SEG2050-CANOPY--RIGHT-CANOPY-GRID-004",
    "SEG2050-CANOPY--RIGHT-CANOPY-GRID-009",
    "SEG2100-CANOPY--RIGHT-CANOPY-GRID-010",
)


def platform_vertical_knots(
    model: ObjModel, support_report: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    support = {str(item["object_name"]): item for item in support_report["objects"]}
    records: list[dict[str, Any]] = []
    for prefix in ("S2050", "S2100", "S2150"):
        name = f"{prefix}-OPPOSITE-PLATFORM-TOP"
        indices = object_vertex_indices(model, name)
        if not len(indices) or name not in support:
            raise ValueError(f"Missing platform support evidence: {name}")
        delta_z = float(support[name]["nearest_delta_median_xyz_m"][2])
        if not -0.25 <= delta_z <= 0.05:
            raise ValueError(f"Unsafe platform vertical correction for {name}: {delta_z}")
        records.append(
            {
                "object_name": name,
                "center_y_local_m": float(np.mean(model.vertices[indices, 1])),
                "vertical_correction_m": delta_z,
                "support_p90_m": float(support[name]["p90_m"]),
            }
        )
    records.sort(key=lambda item: float(item["center_y_local_m"]))
    return (
        np.asarray([item["center_y_local_m"] for item in records], dtype=np.float64),
        np.asarray([item["vertical_correction_m"] for item in records], dtype=np.float64),
        records,
    )


def apply_refinement_vertices(
    model: ObjModel,
    knot_y: np.ndarray,
    knot_delta_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = model.vertices.copy()
    changed_indices: set[int] = set()
    platform_objects: list[str] = []
    extended_columns: list[dict[str, Any]] = []
    for name in sorted(model.faces_by_object):
        indices = object_vertex_indices(model, name)
        if not len(indices):
            continue
        if PLATFORM_PATTERN.fullmatch(name):
            corrections = np.interp(vertices[indices, 1], knot_y, knot_delta_z)
            vertices[indices, 2] += corrections
            changed_indices.update(int(value) for value in indices)
            platform_objects.append(name)
        elif COLUMN_PATTERN.fullmatch(name):
            minimum_z = float(np.min(vertices[indices, 2]))
            bottom = indices[np.abs(vertices[indices, 2] - minimum_z) <= 1e-6]
            center_y = float(np.mean(vertices[indices, 1]))
            correction = float(np.interp(center_y, knot_y, knot_delta_z))
            vertices[bottom, 2] += correction
            changed_indices.update(int(value) for value in bottom)
            extended_columns.append(
                {
                    "object_name": name,
                    "bottom_vertex_count": len(bottom),
                    "bottom_extension_m": -correction,
                }
            )
    return vertices, {
        "changed_vertex_count": len(changed_indices),
        "platform_objects": platform_objects,
        "extended_columns": extended_columns,
    }


def _is_removed(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in UNSUPPORTED_LEGACY_PREFIXES)


def write_refined_obj(
    source_path: str | Path,
    output_path: str | Path,
    transformed_vertices: np.ndarray,
    *,
    output_mtl_name: str,
) -> list[str]:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vertex_index = 0
    current_removed = False
    removed_objects: list[str] = []
    with source.open("r", encoding="utf-8", errors="replace") as reader, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        writer.write("# Supplemental V2 evidence refinement; source remains unchanged.\n")
        for raw_line in reader:
            line = raw_line.strip()
            if line.startswith("mtllib "):
                writer.write(f"mtllib {output_mtl_name}\n")
            elif line.startswith("v "):
                value = transformed_vertices[vertex_index]
                writer.write(f"v {value[0]:.9f} {value[1]:.9f} {value[2]:.9f}\n")
                vertex_index += 1
            elif line.startswith("o "):
                name = line[2:].strip()
                current_removed = _is_removed(name)
                if current_removed:
                    removed_objects.append(name)
                else:
                    writer.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            elif not current_removed:
                writer.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
    if vertex_index != len(transformed_vertices):
        raise ValueError("OBJ vertex count changed during rewrite")
    return removed_objects


def update_candidate_registry(
    source_path: str | Path,
    output_path: str | Path,
    output_obj: str | Path,
    removed_objects: list[str],
    changed_objects: set[str],
    support_report_path: str | Path,
) -> dict[str, Any]:
    registry = load_json(Path(source_path))
    removed = set(removed_objects)
    retained: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        asset_id = str(asset.get("id", ""))
        if asset_id in removed:
            continue
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            geometry["file"] = str(Path(output_obj).resolve())
        if asset_id in changed_objects:
            asset.setdefault("sources", []).append(
                {
                    "kind": "point_cloud",
                    "reference": str(Path(support_report_path).resolve()),
                    "note": "Supplemental V2 evidence-driven vertical refinement.",
                }
            )
        retained.append(asset)
    registry["assets"] = retained
    registry["release_id"] = "site_b_s2050_2200_supplement_v2_candidate"
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    write_json(Path(output_path), registry)
    return registry


def build_supplemental_refinement_candidate(
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    support_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_candidate"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "supplemental_refinement_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    support_report = load_json(Path(support_report_path))
    knot_y, knot_delta_z, knot_records = platform_vertical_knots(model, support_report)
    transformed, transform_report = apply_refinement_vertices(model, knot_y, knot_delta_z)
    removed_objects = write_refined_obj(
        source_obj,
        output_obj,
        transformed,
        output_mtl_name=output_mtl.name,
    )
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    changed_objects = set(transform_report["platform_objects"])
    changed_objects.update(item["object_name"] for item in transform_report["extended_columns"])
    registry = update_candidate_registry(
        source_registry,
        output_registry,
        output_obj,
        removed_objects,
        changed_objects,
        support_report_path,
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Supplemental refinement candidate failed OBJ mesh audit")
    report = {
        "schema_version": "railway.supplemental-mesh-refinement.v1",
        "candidate_id": name,
        "source_obj": str(Path(source_obj).resolve()),
        "output_obj": str(output_obj),
        "support_report": str(Path(support_report_path).resolve()),
        "platform_vertical_knots": knot_records,
        "transform": transform_report,
        "removed_unsupported_legacy_objects": removed_objects,
        "asset_count_after_refinement": registry["summary"]["asset_count"],
        "mesh_audit": {
            "passed": mesh_audit["passed"],
            "object_count": mesh_audit["object_count"],
            "triangle_count": mesh_audit["triangle_count_after_fan_triangulation"],
            "duplicate_face_count": mesh_audit["duplicate_face_count"],
            "degenerate_triangle_count": mesh_audit["degenerate_triangle_count"],
        },
        "preserved_without_geometry_change": [
            "all 8 rail objects",
            "all supported contact-wire fragments",
            "all canopy roof surfaces pending segmented plane refit",
            "catenary mast 0002 pending visibility-aware review",
        ],
        "limitations": [
            "This is an evidence-refined candidate, not a frozen final release.",
            "Platform correction is interpolated continuously across segment ownership seams.",
            "Only column bottoms are extended; canopy roof connections remain fixed.",
        ],
    }
    write_json(output_report, report)
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "registry": output_registry,
        "origin": output_origin,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
