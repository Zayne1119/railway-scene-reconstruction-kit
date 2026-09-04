from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json
from .registry import validate_registry_value


def evaluate_adjacent_candidate_closure(
    *,
    mesh: dict[str, Any],
    seams: dict[str, Any],
    columns: dict[str, Any],
    conductors: dict[str, Any],
    surfaces: dict[str, Any],
    transition: dict[str, Any],
    registry: dict[str, Any],
    fixed_views: list[dict[str, Any]],
) -> dict[str, Any]:
    object_names = [str(value) for value in mesh.get("object_names", [])]
    column_fits = list(columns.get("fit_records", []))
    surface_records = list(surfaces.get("records", []))
    asset_ids = [str(asset.get("id")) for asset in registry.get("assets", [])]
    registry_errors = validate_registry_value(registry)
    gates = {
        "mesh_audit_passed": bool(mesh.get("passed")),
        "no_degenerate_or_duplicate_faces": int(mesh.get("degenerate_triangle_count", -1)) == 0
        and int(mesh.get("duplicate_face_count", -1)) == 0,
        "all_boundary_subsystem_seams_passed": bool(seams.get("passed")),
        "three_endpoint_columns_point_supported": len(column_fits) == 3
        and all(bool(item.get("point_to_mesh_support", {}).get("passed")) for item in column_fits),
        "adjacent_conductor_gate_passed": bool(conductors.get("passed")),
        "all_observed_surface_extensions_passed": bool(surface_records)
        and int(surfaces.get("built_count", -1)) == int(surfaces.get("target_count", -2))
        and all(bool(item.get("passed")) for item in surface_records),
        "platform_transition_gate_passed": bool(transition.get("passed"))
        and all(bool(value) for value in transition.get("gates", {}).values()),
        "normalized_registry_schema_valid": not registry_errors,
        "asset_ids_unique": len(asset_ids) == len(set(asset_ids)),
        "two_six_view_reviews_written": len(fixed_views) == 2
        and all(int(item.get("view_count", 0)) == 6 for item in fixed_views),
        "known_wrong_automatic_canopy_excluded": not any(
            name.startswith("STATION--CANOPY--") for name in object_names
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "passed": passed,
        "registry_validation_errors": registry_errors,
        "status": (
            "candidate_passed_all_geometry_and_evidence_gates_not_promoted"
            if passed
            else "candidate_blocked_by_closure_gate"
        ),
    }


def build_adjacent_candidate_closure(
    *,
    final_obj: str | Path,
    mesh_audit: str | Path,
    seam_audit: str | Path,
    column_report: str | Path,
    conductor_report: str | Path,
    surface_report: str | Path,
    transition_report: str | Path,
    normalized_registry: str | Path,
    fixed_view_manifests: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    final_model = Path(final_obj).resolve()
    paths = {
        "mesh_audit": Path(mesh_audit).resolve(),
        "seam_audit": Path(seam_audit).resolve(),
        "column_report": Path(column_report).resolve(),
        "conductor_report": Path(conductor_report).resolve(),
        "surface_report": Path(surface_report).resolve(),
        "transition_report": Path(transition_report).resolve(),
        "normalized_registry": Path(normalized_registry).resolve(),
    }
    view_paths = [Path(path).resolve() for path in fixed_view_manifests]
    output = Path(output_path).resolve()
    for path in (final_model, *paths.values(), *view_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    values = {key: load_json(path) for key, path in paths.items()}
    views = [load_json(path) for path in view_paths]
    result = evaluate_adjacent_candidate_closure(
        mesh=values["mesh_audit"],
        seams=values["seam_audit"],
        columns=values["column_report"],
        conductors=values["conductor_report"],
        surfaces=values["surface_report"],
        transition=values["transition_report"],
        registry=values["normalized_registry"],
        fixed_views=views,
    )
    report = {
        "schema_version": "railway.adjacent-candidate-closure.v1",
        "final_candidate_obj": str(final_model),
        "final_candidate_obj_sha256": sha256_file(final_model),
        "inputs": {key: str(path) for key, path in paths.items()},
        "fixed_view_manifests": [str(path) for path in view_paths],
        "metrics": {
            "object_count": values["mesh_audit"].get("object_count"),
            "triangle_count": values["mesh_audit"].get(
                "triangle_count_after_fan_triangulation"
            ),
            "asset_count": len(values["normalized_registry"].get("assets", [])),
            "seam_subsystem_count": len(values["seam_audit"].get("subsystems", {})),
            "surface_extension_count": values["surface_report"].get("built_count"),
            "endpoint_column_fit_p90_m": [
                item["point_to_mesh_support"]["p90_m"]
                for item in values["column_report"].get("fit_records", [])
            ],
            "surface_fit_p90_m": {
                item["id"]: item["fit"]["absolute_residual_p90_m"]
                for item in values["surface_report"].get("records", [])
            },
            "platform_transition_fit_p90_m": values["transition_report"].get(
                "absolute_residual_p90_m"
            ),
        },
        **result,
        "promotion_policy": (
            "This report closes the candidate QA phase only. It does not replace the formal "
            "baseline without explicit owner authorization."
        ),
    }
    write_json(output, report)
    return report
