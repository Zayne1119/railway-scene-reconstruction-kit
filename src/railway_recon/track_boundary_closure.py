from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

from .io import load_json, sha256_file, write_json


def _rail_metrics(report: dict[str, Any]) -> dict[str, Any]:
    rails = [
        item
        for item in report.get("objects", [])
        if str(item.get("object_name", "")).endswith(("RAIL-LEFT", "RAIL-RIGHT"))
    ]
    if not rails:
        raise ValueError("Point-support report contains no rail objects")
    return {
        "object_names": sorted(str(item["object_name"]) for item in rails),
        "rail_count": len(rails),
        "mean_p50_m": fmean(float(item["p50_m"]) for item in rails),
        "mean_p90_m": fmean(float(item["p90_m"]) for item in rails),
        "maximum_object_p90_m": max(float(item["p90_m"]) for item in rails),
        "mean_coverage_at_0_10m": fmean(
            float(item["coverage_at_0_10m"]) for item in rails
        ),
    }


def evaluate_track_boundary_closure(
    *,
    candidate_build: dict[str, Any],
    mesh: dict[str, Any],
    baseline_seams: dict[str, Any],
    candidate_seams: dict[str, Any],
    baseline_support: dict[str, Any],
    candidate_support: dict[str, Any],
    fixed_views: dict[str, Any],
    seam_p90_limit_m: float = 0.05,
    seam_p95_limit_m: float = 0.03,
    support_p90_regression_limit_m: float = 0.001,
    coverage_regression_limit: float = 0.005,
) -> dict[str, Any]:
    before_track = baseline_seams["subsystems"]["track"]
    after_track = candidate_seams["subsystems"]["track"]
    before_distance = before_track["symmetric_cross_z_distance"]
    after_distance = after_track["symmetric_cross_z_distance"]
    before_support = _rail_metrics(baseline_support)
    after_support = _rail_metrics(candidate_support)
    seam_p90_before = float(before_distance["p90_m"])
    seam_p90_after = float(after_distance["p90_m"])
    seam_p95_after = float(after_distance["p95_m"])
    support_p90_delta = after_support["mean_p90_m"] - before_support["mean_p90_m"]
    coverage_delta = (
        after_support["mean_coverage_at_0_10m"]
        - before_support["mean_coverage_at_0_10m"]
    )
    build_gates = candidate_build.get("gates", {})
    gates = {
        "four_track_identities_reconciled": int(
            candidate_build.get("transform", {}).get("mapping_count", 0)
        )
        == 4
        and all(bool(value) for value in build_gates.values()),
        "mesh_audit_passed": bool(mesh.get("passed")),
        "no_degenerate_or_duplicate_faces": int(
            mesh.get("degenerate_triangle_count", -1)
        )
        == 0
        and int(mesh.get("duplicate_face_count", -1)) == 0,
        "all_boundary_subsystems_passed": bool(candidate_seams.get("passed")),
        "track_seam_p90_within_50mm": seam_p90_after <= seam_p90_limit_m,
        "track_seam_p95_within_30mm": seam_p95_after <= seam_p95_limit_m,
        "track_seam_improved": seam_p90_after < seam_p90_before,
        "rail_object_identity_set_preserved": (
            before_support["object_names"] == after_support["object_names"]
        ),
        "rail_point_support_p90_not_regressed": (
            support_p90_delta <= support_p90_regression_limit_m
        ),
        "rail_10cm_coverage_not_regressed": (
            coverage_delta >= -coverage_regression_limit
        ),
        "six_fixed_views_written": int(fixed_views.get("view_count", 0)) == 6
        and len(fixed_views.get("views", [])) == 6,
    }
    passed = all(gates.values())
    return {
        "metrics": {
            "track_seam_p90_before_m": seam_p90_before,
            "track_seam_p90_after_m": seam_p90_after,
            "track_seam_p90_improvement_m": seam_p90_before - seam_p90_after,
            "track_seam_p95_after_m": seam_p95_after,
            "track_seam_maximum_after_m": float(after_distance["maximum_m"]),
            "signed_station_gap_after_m": float(after_track["signed_station_gap_m"]),
            "rail_support_before": before_support,
            "rail_support_after": after_support,
            "rail_mean_p90_delta_m": support_p90_delta,
            "rail_mean_10cm_coverage_delta": coverage_delta,
        },
        "limits": {
            "seam_p90_limit_m": seam_p90_limit_m,
            "seam_p95_limit_m": seam_p95_limit_m,
            "support_p90_regression_limit_m": support_p90_regression_limit_m,
            "coverage_regression_limit": coverage_regression_limit,
        },
        "gates": gates,
        "passed": passed,
        "status": (
            "candidate_closed_all_track_boundary_gates_not_promoted"
            if passed
            else "candidate_blocked_by_track_boundary_gate"
        ),
    }


def build_track_boundary_closure(
    *,
    candidate_obj: str | Path,
    candidate_build_report: str | Path,
    mesh_audit: str | Path,
    baseline_seam_audit: str | Path,
    candidate_seam_audit: str | Path,
    baseline_support_report: str | Path,
    candidate_support_report: str | Path,
    fixed_view_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    model = Path(candidate_obj).resolve()
    inputs = {
        "candidate_build_report": Path(candidate_build_report).resolve(),
        "mesh_audit": Path(mesh_audit).resolve(),
        "baseline_seam_audit": Path(baseline_seam_audit).resolve(),
        "candidate_seam_audit": Path(candidate_seam_audit).resolve(),
        "baseline_support_report": Path(baseline_support_report).resolve(),
        "candidate_support_report": Path(candidate_support_report).resolve(),
        "fixed_view_manifest": Path(fixed_view_manifest).resolve(),
    }
    output = Path(output_path).resolve()
    for path in (model, *inputs.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    values = {key: load_json(path) for key, path in inputs.items()}
    result = evaluate_track_boundary_closure(
        candidate_build=values["candidate_build_report"],
        mesh=values["mesh_audit"],
        baseline_seams=values["baseline_seam_audit"],
        candidate_seams=values["candidate_seam_audit"],
        baseline_support=values["baseline_support_report"],
        candidate_support=values["candidate_support_report"],
        fixed_views=values["fixed_view_manifest"],
    )
    report = {
        "schema_version": "railway.track-boundary-closure.v1",
        "candidate_obj": str(model),
        "candidate_obj_sha256": sha256_file(model),
        "inputs": {key: str(path) for key, path in inputs.items()},
        **result,
        "promotion_policy": (
            "This report closes the track-boundary candidate QA only. It does not replace "
            "the formal baseline without explicit owner authorization."
        ),
        "limitations": [
            (
                "The maximum cross-section distance is retained as a review statistic but is "
                "not a gate because different ballast edge profiles create isolated outliers."
            ),
            "Final Blender or UE material and render acceptance remains separate.",
        ],
    }
    write_json(output, report)
    return report
