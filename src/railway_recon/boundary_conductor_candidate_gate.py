from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, write_json


def _in_scope_high_priority(report: dict[str, Any]) -> list[dict[str, Any]]:
    scope = report["corridor_scope"]
    minimum = float(scope["station_minimum_m"])
    maximum = float(scope["station_maximum_m"])
    records: list[dict[str, Any]] = []
    for candidate in report["overhead_linear_candidates"]:
        if candidate["priority"] not in {"P0", "P1"}:
            continue
        start, end = (float(value) for value in candidate["station_range_m"])
        if min(end, maximum) - max(start, minimum) > 0.0:
            records.append(candidate)
    return records


def gate_boundary_conductor_candidate(
    *,
    candidate_report_path: str | Path,
    baseline_gap_report_path: str | Path,
    candidate_gap_report_path: str | Path,
    fixed_view_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    candidate_report = load_json(Path(candidate_report_path))
    baseline = load_json(Path(baseline_gap_report_path))
    candidate = load_json(Path(candidate_gap_report_path))
    fixed_views = load_json(Path(fixed_view_manifest_path))
    mesh_audit = load_json(Path(candidate_report_path).parent / "mesh_audit.json")
    fits = candidate_report["fit_records"]
    baseline_high = _in_scope_high_priority(baseline)
    candidate_high = _in_scope_high_priority(candidate)
    baseline_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    gates = {
        "mesh_audit_passed": bool(mesh_audit["passed"]),
        "three_boundary_fragments_built": int(candidate_report["added_asset_count"]) == 3,
        "all_point_fits_passed": bool(fits) and all(item["fit"]["passed"] for item in fits),
        "all_fit_residual_p90_below_8cm": bool(fits)
        and all(float(item["fit"]["residual_p90_m"]) <= 0.08 for item in fits),
        "six_fixed_views_rendered": int(fixed_views["view_count"]) == 6,
        "three_in_scope_high_priority_targets_in_baseline": len(baseline_high) == 3,
        "no_in_scope_high_priority_linear_candidate_remains": len(candidate_high) == 0,
        "p90_reduced": float(candidate_overall["p90_m"])
        < float(baseline_overall["p90_m"]),
        "coverage_20cm_improved": float(candidate_overall["coverage_at_0_20m"])
        > float(baseline_overall["coverage_at_0_20m"]),
        "unexplained_ratio_reduced": float(
            candidate_overall["unexplained_over_0_25m"]
        )
        < float(baseline_overall["unexplained_over_0_25m"]),
        "no_p0_linear_candidate_remains": int(
            candidate["overhead_linear_priority_counts"]["P0"]
        )
        == 0,
    }
    passed = all(gates.values())
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.boundary-conductor-candidate-gate.v1",
            "candidate_report": str(Path(candidate_report_path).resolve()),
            "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
            "candidate_gap_report": str(Path(candidate_gap_report_path).resolve()),
            "fixed_view_manifest": str(Path(fixed_view_manifest_path).resolve()),
            "target": {
                "baseline_candidate_ids": [
                    str(item["candidate_id"]) for item in baseline_high
                ],
                "candidate_in_scope_high_priority_ids": [
                    str(item["candidate_id"]) for item in candidate_high
                ],
                "remaining_adjacent_segment_high_priority_ids": [
                    str(item["candidate_id"])
                    for item in candidate["overhead_linear_candidates"]
                    if item["priority"] in {"P0", "P1"}
                    and item not in candidate_high
                ],
            },
            "regression": {
                "p90_m": {
                    "baseline": float(baseline_overall["p90_m"]),
                    "candidate": float(candidate_overall["p90_m"]),
                },
                "coverage_at_0_20m": {
                    "baseline": float(baseline_overall["coverage_at_0_20m"]),
                    "candidate": float(candidate_overall["coverage_at_0_20m"]),
                },
                "unexplained_over_0_25m": {
                    "baseline": float(baseline_overall["unexplained_over_0_25m"]),
                    "candidate": float(candidate_overall["unexplained_over_0_25m"]),
                },
            },
            "gates": gates,
            "passed": passed,
            "status": (
                "candidate_gate_passed_formal_baseline_unchanged"
                if passed
                else "candidate_gate_failed_not_eligible_for_promotion"
            ),
            "limitations": [
                "Professional conductor subtype remains pending.",
                "Fragments stop at the current segment boundary; continuation belongs to the next segment.",
                "Formal baseline promotion requires explicit authorization.",
            ],
        },
    )
    return destination
