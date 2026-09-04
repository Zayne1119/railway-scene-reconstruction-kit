from __future__ import annotations

from pathlib import Path

from .io import load_json, write_json


def gate_station_end_wall_candidate(
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
    base_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    vertical_reduction = int(baseline["vertical_candidate_count"]) - int(
        candidate["vertical_candidate_count"]
    )
    p1_reduction = int(baseline["priority_counts"].get("P1", 0)) - int(
        candidate["priority_counts"].get("P1", 0)
    )
    gates = {
        "mesh_audit_passed": bool(report["mesh_audit_passed"]),
        "dense_surface_support_passed": bool(report["point_support"]["passed_review_gate"]),
        "dense_surface_sample_count_at_least_1000": int(report["point_support"]["sample_count"])
        >= 1_000,
        "six_fixed_views_rendered": int(views["view_count"]) == 6,
        "unexplained_ratio_reduced": float(candidate_overall["unexplained_over_0_25m"])
        < float(base_overall["unexplained_over_0_25m"]),
        "p90_reduced": float(candidate_overall["p90_m"]) < float(base_overall["p90_m"]),
        "six_false_vertical_fragments_resolved": vertical_reduction == 6,
        "six_p1_fragments_resolved": p1_reduction == 6,
        "no_new_p0_candidate": int(candidate["priority_counts"].get("P0", 0))
        <= int(baseline["priority_counts"].get("P0", 0)),
    }
    output = Path(output_path).resolve()
    write_json(
        output,
        {
            "schema_version": "railway.station-end-wall-candidate-gate.v1",
            "candidate_report": str(Path(candidate_report_path).resolve()),
            "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
            "candidate_gap_report": str(Path(candidate_gap_report_path).resolve()),
            "fixed_view_manifest": str(Path(fixed_view_manifest_path).resolve()),
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
            "passed": all(gates.values()),
            "status": (
                "candidate_gate_passed_formal_baseline_unchanged"
                if all(gates.values())
                else "candidate_gate_failed_formal_baseline_unchanged"
            ),
        },
    )
    return output
