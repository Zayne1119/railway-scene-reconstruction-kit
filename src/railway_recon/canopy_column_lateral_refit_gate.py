from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def _p0_current_segment_refinements(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report.get("vertical_candidates", [])
        if item.get("priority") == "P0"
        and item.get("corridor_scope_ownership") == "within_current_segment"
        and item.get("asset_relation") == "existing_asset_geometry_refinement"
    ]


def gate_canopy_column_lateral_refit(
    *,
    refit_report_path: str | Path,
    candidate_obj_path: str | Path,
    model_origin_path: str | Path,
    frame_report_path: str | Path,
    baseline_gap_report_path: str | Path,
    candidate_gap_report_path: str | Path,
    mesh_audit_path: str | Path,
    fixed_view_manifest_path: str | Path,
    output_path: str | Path,
    expected_column_count: int = 13,
    maximum_station_delta_m: float = 0.02,
    maximum_cross_delta_m: float = 0.02,
) -> dict[str, Any]:
    refit = load_json(Path(refit_report_path))
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    baseline_gap = load_json(Path(baseline_gap_report_path))
    candidate_gap = load_json(Path(candidate_gap_report_path))
    mesh_audit = load_json(Path(mesh_audit_path))
    fixed_views = load_json(Path(fixed_view_manifest_path))
    model = parse_obj_model(Path(candidate_obj_path))

    placement_results: list[dict[str, Any]] = []
    for plan in refit.get("plans", []):
        asset_id = str(plan["asset_id"])
        indexes = object_vertex_indices(model, asset_id)
        centre_world = np.median(model.vertices[indexes], axis=0) + origin
        observed_station = float(np.dot(centre_world[:2] - frame_origin, along))
        observed_cross = float(np.dot(centre_world[:2] - frame_origin, cross))
        station_delta = abs(observed_station - float(plan["current_station_m"]))
        cross_delta = abs(observed_cross - float(plan["target_cross_center_m"]))
        placement_results.append(
            {
                "asset_id": asset_id,
                "station_m": observed_station,
                "station_delta_m": station_delta,
                "cross_m": observed_cross,
                "cross_delta_m": cross_delta,
                "passed": station_delta <= maximum_station_delta_m
                and cross_delta <= maximum_cross_delta_m,
            }
        )

    baseline_refinements = _p0_current_segment_refinements(baseline_gap)
    candidate_refinements = _p0_current_segment_refinements(candidate_gap)
    baseline_overall = baseline_gap["overall"]
    candidate_overall = candidate_gap["overall"]
    gates = {
        "expected_column_count": len(placement_results) == expected_column_count,
        "all_placements_within_tolerance": bool(placement_results)
        and all(item["passed"] for item in placement_results),
        "p0_column_refinement_fragments_closed": len(candidate_refinements) == 0,
        "p0_column_refinement_fragments_reduced": len(candidate_refinements)
        < len(baseline_refinements),
        "coverage_at_0_10m_not_degraded": float(
            candidate_overall["coverage_at_0_10m"]
        )
        >= float(baseline_overall["coverage_at_0_10m"]),
        "unexplained_over_0_25m_not_degraded": float(
            candidate_overall["unexplained_over_0_25m"]
        )
        <= float(baseline_overall["unexplained_over_0_25m"]),
        "mesh_audit_passed": bool(mesh_audit.get("passed")),
        "six_fixed_views_rendered": int(fixed_views.get("view_count", 0)) == 6,
    }
    passed = all(gates.values())
    result = {
        "schema_version": "railway.canopy-column-lateral-refit-gate.v1",
        "thresholds": {
            "expected_column_count": expected_column_count,
            "maximum_station_delta_m": maximum_station_delta_m,
            "maximum_cross_delta_m": maximum_cross_delta_m,
        },
        "placement_results": placement_results,
        "metrics": {
            "baseline_p0_column_refinement_fragments": len(baseline_refinements),
            "candidate_p0_column_refinement_fragments": len(candidate_refinements),
            "baseline_coverage_at_0_10m": float(
                baseline_overall["coverage_at_0_10m"]
            ),
            "candidate_coverage_at_0_10m": float(
                candidate_overall["coverage_at_0_10m"]
            ),
            "baseline_unexplained_over_0_25m": float(
                baseline_overall["unexplained_over_0_25m"]
            ),
            "candidate_unexplained_over_0_25m": float(
                candidate_overall["unexplained_over_0_25m"]
            ),
        },
        "gates": gates,
        "passed": passed,
        "status": "geometry_candidate_accepted" if passed else "candidate_rejected",
        "limitations": [
            "This gate accepts geometry against the supplied point cloud, not structural design accuracy.",
            "The candidate remains separate from the formal baseline until explicit promotion.",
        ],
    }
    write_json(Path(output_path), result)
    return result
