from __future__ import annotations

from pathlib import Path

from .io import load_json, write_json


def _targets(report: dict[str, object], fit: dict[str, object]) -> list[dict[str, object]]:
    s0 = float(fit["station_min_m"]) - 0.75
    s1 = float(fit["station_max_m"]) + 0.75
    cross = float(fit["cross_peak_m"])
    return [
        item
        for item in report["vertical_candidates"]  # type: ignore[index]
        if s0 <= float(item["station_m"]) <= s1
        and abs(float(item["cross_m"]) - cross) <= 0.75
    ]


def gate_remote_context_facade_candidate(
    *,
    candidate_report_path: str | Path,
    baseline_gap_report_path: str | Path,
    candidate_gap_report_path: str | Path,
    fixed_view_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    report = load_json(Path(candidate_report_path))
    baseline = load_json(Path(baseline_gap_report_path))
    candidate = load_json(Path(candidate_gap_report_path))
    views = load_json(Path(fixed_view_manifest_path))
    fit = report["fit"]
    support = report["point_support"]
    base_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    baseline_targets = _targets(baseline, fit)
    candidate_targets = _targets(candidate, fit)
    vertical_reduction = int(baseline["vertical_candidate_count"]) - int(
        candidate["vertical_candidate_count"]
    )
    p1_reduction = int(baseline["priority_counts"].get("P1", 0)) - int(
        candidate["priority_counts"].get("P1", 0)
    )
    gates = {
        "mesh_audit_passed": bool(report["mesh_audit_passed"]),
        "dense_surface_support_passed": bool(support["passed_review_gate"]),
        "dense_surface_sample_count_at_least_1000": int(support["sample_count"]) >= 1_000,
        "dense_surface_p90_below_30cm": float(support["p90_m"]) <= 0.30,
        "six_fixed_views_rendered": int(views["view_count"]) == 6,
        "two_target_fragments_present_in_baseline": len(baseline_targets) == 2,
        "target_fragments_resolved": len(candidate_targets) == 0,
        "two_vertical_fragments_resolved": vertical_reduction == 2,
        "two_p1_fragments_resolved": p1_reduction == 2,
        "p90_reduced": float(candidate_overall["p90_m"]) < float(base_overall["p90_m"]),
        "unexplained_ratio_reduced": float(candidate_overall["unexplained_over_0_25m"])
        < float(base_overall["unexplained_over_0_25m"]),
        "no_new_p0_candidate": int(candidate["priority_counts"].get("P0", 0))
        <= int(baseline["priority_counts"].get("P0", 0)),
    }
    passed = all(gates.values())
    output = Path(output_path).resolve()
    write_json(
        output,
        {
            "schema_version": "railway.remote-context-facade-candidate-gate.v1",
            "candidate_report": str(Path(candidate_report_path).resolve()),
            "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
            "candidate_gap_report": str(Path(candidate_gap_report_path).resolve()),
            "fixed_view_manifest": str(Path(fixed_view_manifest_path).resolve()),
            "target": {
                "baseline_candidate_ids": [item["candidate_id"] for item in baseline_targets],
                "candidate_candidate_ids": [item["candidate_id"] for item in candidate_targets],
            },
            "regression": {
                "p90_m": {
                    "baseline": base_overall["p90_m"],
                    "candidate": candidate_overall["p90_m"],
                },
                "coverage_at_0_20m": {
                    "baseline": base_overall["coverage_at_0_20m"],
                    "candidate": candidate_overall["coverage_at_0_20m"],
                },
                "unexplained_over_0_25m": {
                    "baseline": base_overall["unexplained_over_0_25m"],
                    "candidate": candidate_overall["unexplained_over_0_25m"],
                },
                "vertical_candidate_reduction": vertical_reduction,
                "p1_candidate_reduction": p1_reduction,
            },
            "gates": gates,
            "passed": passed,
            "status": (
                "candidate_gate_passed_formal_baseline_unchanged"
                if passed
                else "candidate_gate_failed_formal_baseline_unchanged"
            ),
        },
    )
    return output
