from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, write_json


def gate_supplemental_conductor_candidate(
    *,
    refinement_report_path: str | Path,
    mesh_audit_path: str | Path,
    point_support_report_path: str | Path,
    fixed_view_manifest_path: str | Path,
    output_path: str | Path,
    maximum_fit_p90_m: float = 0.02,
    maximum_surface_support_p90_m: float = 0.08,
    minimum_coverage_at_0_10m: float = 0.99,
) -> dict[str, Any]:
    refinement = load_json(Path(refinement_report_path))
    mesh_audit = load_json(Path(mesh_audit_path))
    point_support = load_json(Path(point_support_report_path))
    fixed_views = load_json(Path(fixed_view_manifest_path))
    asset_ids = [str(item["asset_id"]) for item in refinement["fit_records"]]
    support_by_name = {
        str(item["object_name"]): item for item in point_support.get("objects", [])
    }
    asset_results: list[dict[str, Any]] = []
    for fit_record in refinement["fit_records"]:
        asset_id = str(fit_record["asset_id"])
        support = support_by_name.get(asset_id)
        span_fits = fit_record.get("span_fits", [])
        gates = {
            "all_span_fits_passed": bool(span_fits)
            and all(bool(item.get("passed")) for item in span_fits),
            "fit_p90_within_limit": float(
                fit_record["maximum_span_residual_p90_m"]
            )
            <= maximum_fit_p90_m,
            "point_support_record_present": support is not None,
            "surface_support_p90_within_limit": support is not None
            and float(support["p90_m"]) <= maximum_surface_support_p90_m,
            "surface_coverage_at_0_10m": support is not None
            and float(support["coverage_at_0_10m"])
            >= minimum_coverage_at_0_10m,
            "point_support_disposition_keep": support is not None
            and support.get("disposition") == "supported_keep",
        }
        asset_results.append(
            {
                "asset_id": asset_id,
                "maximum_span_fit_p90_m": float(
                    fit_record["maximum_span_residual_p90_m"]
                ),
                "surface_support_p90_m": (
                    float(support["p90_m"]) if support is not None else None
                ),
                "coverage_at_0_10m": (
                    float(support["coverage_at_0_10m"])
                    if support is not None
                    else None
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    global_gates = {
        "expected_two_assets": len(asset_ids) == 2,
        "new_assets_schema_valid": bool(refinement.get("new_assets_schema_valid")),
        "mesh_audit_passed": bool(mesh_audit.get("passed")),
        "six_fixed_views_rendered": int(fixed_views.get("view_count", 0)) == 6,
        "all_assets_passed": bool(asset_results)
        and all(item["passed"] for item in asset_results),
    }
    passed = all(global_gates.values())
    result = {
        "schema_version": "railway.supplemental-conductor-gate.v1",
        "asset_ids": asset_ids,
        "thresholds": {
            "maximum_fit_p90_m": maximum_fit_p90_m,
            "maximum_surface_support_p90_m": maximum_surface_support_p90_m,
            "minimum_coverage_at_0_10m": minimum_coverage_at_0_10m,
        },
        "asset_results": asset_results,
        "global_gates": global_gates,
        "passed": passed,
        "status": (
            "geometry_accepted_semantic_subtype_pending"
            if passed
            else "candidate_rejected_or_requires_refit"
        ),
        "limitations": [
            "Geometry acceptance does not resolve the professional conductor subtype.",
            "The two assets remain candidates until a railway electrification review names the conductor type.",
            "Inherited registry vocabulary findings are non-blocking for this isolated geometry gate.",
        ],
    }
    write_json(Path(output_path), result)
    return result
