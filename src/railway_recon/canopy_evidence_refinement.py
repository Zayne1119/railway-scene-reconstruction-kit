from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .supplemental_mesh_refinement import update_candidate_registry, write_refined_obj

OPPOSITE_ROOF = re.compile(r"^S(2050|2100|2150)-OPPOSITE-CANOPY-ROOF-RUN-01$")
OPPOSITE_COLUMN = re.compile(r"^S(2050|2100|2150)-OPPOSITE-CANOPY-COLUMN-\d+$")
OPPOSITE_ATTACHMENT = re.compile(
    r"^S(2050|2100|2150)-OPPOSITE-CANOPY-COLUMN-\d+-(CAPITAL|LOCAL-UNDERROOF)$"
)


def opposite_roof_knots(
    model_vertices: np.ndarray,
    faces_by_object: dict[str, list[tuple[int, ...]]],
    plane_report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    records_by_name = {
        str(item["object_name"]): item
        for item in plane_report["records"]
        if OPPOSITE_ROOF.fullmatch(str(item["object_name"]))
    }
    records: list[dict[str, Any]] = []
    temporary_model = type("TemporaryObjModel", (), {})()
    temporary_model.vertices = model_vertices
    temporary_model.faces_by_object = faces_by_object
    for prefix in ("S2050", "S2100", "S2150"):
        name = f"{prefix}-OPPOSITE-CANOPY-ROOF-RUN-01"
        fit = records_by_name.get(name)
        if fit is None or not fit.get("passed"):
            raise ValueError(f"Opposite roof did not pass the plane gate: {name}")
        indices = object_vertex_indices(temporary_model, name)
        records.append(
            {
                "object_name": name,
                "center_y_local_m": float(np.mean(model_vertices[indices, 1])),
                "vertical_correction_m": float(fit["vertical_correction_at_center_m"]),
                "fit_residual_p90_m": float(fit["residual_p90_m"]),
                "fit_angle_change_deg": float(fit["plane_angle_change_deg"]),
            }
        )
    records.sort(key=lambda item: float(item["center_y_local_m"]))
    return (
        np.asarray([item["center_y_local_m"] for item in records], dtype=np.float64),
        np.asarray([item["vertical_correction_m"] for item in records], dtype=np.float64),
        records,
    )


def apply_opposite_canopy_correction(
    vertices: np.ndarray,
    faces_by_object: dict[str, list[tuple[int, ...]]],
    knot_y: np.ndarray,
    knot_delta_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    transformed = vertices.copy()
    temporary_model = type("TemporaryObjModel", (), {})()
    temporary_model.vertices = vertices
    temporary_model.faces_by_object = faces_by_object
    changed: list[dict[str, Any]] = []
    changed_vertices: set[int] = set()
    for name in sorted(faces_by_object):
        indices = object_vertex_indices(temporary_model, name)
        if not len(indices):
            continue
        if OPPOSITE_ROOF.fullmatch(name):
            correction = np.interp(transformed[indices, 1], knot_y, knot_delta_z)
            transformed[indices, 2] += correction
            changed_vertices.update(int(value) for value in indices)
            changed.append(
                {
                    "object_name": name,
                    "operation": "continuous_roof_vertical_correction",
                    "minimum_correction_m": float(np.min(correction)),
                    "maximum_correction_m": float(np.max(correction)),
                }
            )
        elif OPPOSITE_ATTACHMENT.fullmatch(name):
            center_y = float(np.mean(transformed[indices, 1]))
            correction = float(np.interp(center_y, knot_y, knot_delta_z))
            transformed[indices, 2] += correction
            changed_vertices.update(int(value) for value in indices)
            changed.append(
                {
                    "object_name": name,
                    "operation": "attachment_translation",
                    "vertical_correction_m": correction,
                }
            )
        elif OPPOSITE_COLUMN.fullmatch(name):
            maximum_z = float(np.max(transformed[indices, 2]))
            top = indices[np.abs(transformed[indices, 2] - maximum_z) <= 1e-6]
            center_y = float(np.mean(transformed[indices, 1]))
            correction = float(np.interp(center_y, knot_y, knot_delta_z))
            transformed[top, 2] += correction
            changed_vertices.update(int(value) for value in top)
            changed.append(
                {
                    "object_name": name,
                    "operation": "column_top_reconnect",
                    "top_vertex_count": len(top),
                    "vertical_correction_m": correction,
                }
            )
    return transformed, {
        "changed_vertex_count": len(changed_vertices),
        "changed_object_count": len(changed),
        "objects": changed,
    }


def build_canopy_evidence_candidate(
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    plane_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_canopy_candidate"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "canopy_evidence_refinement_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    plane_report = load_json(Path(plane_report_path))
    knot_y, knot_delta_z, knots = opposite_roof_knots(
        model.vertices, model.faces_by_object, plane_report
    )
    transformed, transform_report = apply_opposite_canopy_correction(
        model.vertices, model.faces_by_object, knot_y, knot_delta_z
    )
    removed = write_refined_obj(
        source_obj,
        output_obj,
        transformed,
        output_mtl_name=output_mtl.name,
    )
    if removed:
        raise ValueError("Unexpected unsupported legacy objects remained in the run02 source")
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    changed_objects = {item["object_name"] for item in transform_report["objects"]}
    registry = update_candidate_registry(
        source_registry,
        output_registry,
        output_obj,
        [],
        changed_objects,
        plane_report_path,
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Canopy evidence candidate failed OBJ mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-evidence-refinement.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "plane_report": str(Path(plane_report_path).resolve()),
            "opposite_roof_knots": knots,
            "transform": transform_report,
            "asset_count": registry["summary"]["asset_count"],
            "mesh_audit": {
                "passed": mesh_audit["passed"],
                "object_count": mesh_audit["object_count"],
                "triangle_count": mesh_audit["triangle_count_after_fan_triangulation"],
                "duplicate_face_count": mesh_audit["duplicate_face_count"],
                "degenerate_triangle_count": mesh_audit["degenerate_triangle_count"],
            },
            "withheld": [
                "Right canopy surface-01 group: retain until joint/seam topology is constrained.",
                "Two right multi-slope roof records that failed plane-angle gates.",
            ],
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
