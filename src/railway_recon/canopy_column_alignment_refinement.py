from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import ObjModel, object_vertex_indices, parse_obj_model
from .supplemental_mesh_refinement import update_candidate_registry, write_refined_obj


def _project_xy(point: np.ndarray, frame: dict[str, Any]) -> tuple[float, float]:
    delta = point - np.asarray(frame["origin_xy"], dtype=np.float64)
    return (
        float(delta @ np.asarray(frame["along_xy"], dtype=np.float64)),
        float(delta @ np.asarray(frame["cross_xy"], dtype=np.float64)),
    )


def estimate_row_alignment(
    column_centers_s_c: list[tuple[str, float, float]],
    candidates_s_c: list[tuple[str, float, float, float]],
    *,
    cross_delta_range_m: tuple[float, float],
    maximum_along_delta_m: float = 0.75,
    minimum_matches: int = 3,
    maximum_cross_mad_m: float = 0.40,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for candidate_id, candidate_s, candidate_c, height in candidates_s_c:
        if not 3.5 <= height <= 6.5:
            continue
        name, column_s, column_c = min(
            column_centers_s_c,
            key=lambda item: abs(candidate_s - item[1]),
        )
        delta_s = candidate_s - column_s
        delta_c = candidate_c - column_c
        if (
            abs(delta_s) <= maximum_along_delta_m
            and cross_delta_range_m[0] <= delta_c <= cross_delta_range_m[1]
        ):
            matches.append(
                {
                    "candidate_id": candidate_id,
                    "column_id": name,
                    "delta_along_m": delta_s,
                    "delta_cross_m": delta_c,
                }
            )
    if len(matches) < minimum_matches:
        raise ValueError(f"Insufficient row alignment matches: {len(matches)}")
    delta_s = np.asarray([item["delta_along_m"] for item in matches])
    delta_c = np.asarray([item["delta_cross_m"] for item in matches])
    median_s = float(np.median(delta_s))
    median_c = float(np.median(delta_c))
    mad_s = float(np.median(np.abs(delta_s - median_s)))
    mad_c = float(np.median(np.abs(delta_c - median_c)))
    if mad_c > maximum_cross_mad_m:
        raise ValueError(f"Incoherent row cross-offset MAD: {mad_c}")
    return {
        "match_count": len(matches),
        "median_delta_along_m": median_s,
        "median_delta_cross_m": median_c,
        "mad_delta_along_m": mad_s,
        "mad_delta_cross_m": mad_c,
        "matches": matches,
    }


def _column_centers(
    model: ObjModel,
    origin: np.ndarray,
    assets: list[dict[str, Any]],
    frame: dict[str, Any],
    marker: str,
) -> list[tuple[str, float, float]]:
    result = []
    for asset in assets:
        name = str(asset.get("id", ""))
        if asset.get("type") != "canopy_column" or marker not in name:
            continue
        vertices = model.vertices[object_vertex_indices(model, name)] + origin
        center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0
        along, cross = _project_xy(center[:2], frame)
        result.append((name, along, cross))
    return result


def derive_column_row_translations(
    model: ObjModel,
    origin: np.ndarray,
    registry: dict[str, Any],
    comparison: dict[str, Any],
    frame: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidates = []
    for item in comparison["records"]:
        if item["decision"] != "unmatched_requires_photo_or_point_review":
            continue
        along, cross = _project_xy(
            np.asarray([item["center_x"], item["center_y"]], dtype=np.float64), frame
        )
        candidates.append(
            (
                str(item["id"]),
                along,
                cross,
                float(item["maximum_z"] - item["minimum_z"]),
            )
        )
    assets = registry.get("assets", [])
    rows = {
        "right": {
            "columns": _column_centers(model, origin, assets, frame, "RIGHT"),
            "cross_delta_range_m": (1.0, 4.0),
        },
        "opposite": {
            "columns": _column_centers(model, origin, assets, frame, "OPPOSITE"),
            "cross_delta_range_m": (-4.0, -1.0),
        },
    }
    along_vector = np.asarray(frame["along_xy"], dtype=np.float64)
    cross_vector = np.asarray(frame["cross_xy"], dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    for row_name, row in rows.items():
        estimate = estimate_row_alignment(
            row["columns"],
            candidates,
            cross_delta_range_m=row["cross_delta_range_m"],
        )
        delta_xy = (
            along_vector * float(estimate["median_delta_along_m"])
            + cross_vector * float(estimate["median_delta_cross_m"])
        )
        result[row_name] = {
            **estimate,
            "column_ids": [item[0] for item in row["columns"]],
            "translation_world_xy_m": [float(value) for value in delta_xy],
        }
    return result


def apply_column_row_translations(
    model: ObjModel,
    rows: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, set[str]]:
    transformed = model.vertices.copy()
    changed_objects: set[str] = set()
    changed_vertices: set[int] = set()
    for row in rows.values():
        translation = np.asarray(row["translation_world_xy_m"], dtype=np.float64)
        for column_id in row["column_ids"]:
            associated = [
                name
                for name in model.faces_by_object
                if name == column_id or name.startswith(f"{column_id}-")
            ]
            for name in associated:
                indices = object_vertex_indices(model, name)
                pending = np.asarray(
                    [index for index in indices if int(index) not in changed_vertices],
                    dtype=np.int64,
                )
                if len(pending):
                    transformed[pending, :2] += translation
                    changed_vertices.update(int(index) for index in pending)
                changed_objects.add(name)
    return transformed, changed_objects


def build_canopy_column_alignment_candidate(
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    comparison_path: str | Path,
    frame_report_path: str | Path,
    photo_review_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_column_aligned_candidate"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "column_alignment_refinement_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    origin = np.asarray(load_json(Path(model_origin))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    comparison = load_json(Path(comparison_path))
    frame = load_json(Path(frame_report_path))["frame"]
    rows = derive_column_row_translations(model, origin, registry, comparison, frame)
    transformed, changed_objects = apply_column_row_translations(model, rows)
    removed = write_refined_obj(
        source_obj, output_obj, transformed, output_mtl_name=output_mtl.name
    )
    if removed:
        raise ValueError("Unexpected object removal during column alignment")
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    updated_registry = update_candidate_registry(
        source_registry,
        output_registry,
        output_obj,
        [],
        changed_objects,
        comparison_path,
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Column-aligned candidate failed OBJ mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-column-alignment-refinement.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "supplemental_candidate_comparison": str(Path(comparison_path).resolve()),
            "photo_review": str(Path(photo_review_path).resolve()),
            "rows": rows,
            "changed_object_count": len(changed_objects),
            "changed_objects": sorted(changed_objects),
            "asset_count": updated_registry["summary"]["asset_count"],
            "mesh_audit": {
                "passed": mesh_audit["passed"],
                "object_count": mesh_audit["object_count"],
                "triangle_count": mesh_audit["triangle_count_after_fan_triangulation"],
                "duplicate_face_count": mesh_audit["duplicate_face_count"],
                "degenerate_triangle_count": mesh_audit["degenerate_triangle_count"],
            },
            "status": "candidate_requires_fixed_view_and_point_support_review",
            "limitations": [
                "Row translations are propagated from periodic point matches and multi-view photos.",
                "Roof geometry is preserved; only columns and their named attachments move.",
                "This candidate does not infer additional unobserved columns.",
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
