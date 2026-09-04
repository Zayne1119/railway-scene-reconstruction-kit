from __future__ import annotations

from pathlib import Path

from .io import load_json, write_json


def _target_candidates(
    report: dict[str, object],
    *,
    station_m: float,
    cross_m: float,
    tolerance_m: float = 0.75,
) -> list[dict[str, object]]:
    return [
        item
        for item in report["vertical_candidates"]  # type: ignore[index]
        if abs(float(item["station_m"]) - station_m) <= tolerance_m
        and abs(float(item["cross_m"]) - cross_m) <= tolerance_m
    ]


def gate_platform_clock_candidate(
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
    point_support = report["point_support"]
    base_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    station_m = float(fit["station_center_m"])
    cross_m = float(fit["cross_center_m"])
    baseline_targets = _target_candidates(
        baseline,
        station_m=station_m,
        cross_m=cross_m,
    )
    candidate_targets = _target_candidates(
        candidate,
        station_m=station_m,
        cross_m=cross_m,
    )
    priority_reduction = int(baseline["priority_counts"].get("P1", 0)) - int(
        candidate["priority_counts"].get("P1", 0)
    )
    gates = {
        "mesh_audit_passed": bool(report["mesh_audit_passed"]),
        "dense_point_support_passed": bool(point_support["passed_review_gate"]),
        "dense_point_support_p90_below_10cm": float(point_support["p90_m"]) <= 0.10,
        "full_round_profile_observed": int(fit["occupied_angle_bins_of_16"]) >= 12,
        "six_fixed_views_rendered": int(views["view_count"]) == 6,
        "target_was_present_in_baseline": len(baseline_targets) >= 1,
        "target_candidate_resolved": len(candidate_targets) == 0,
        "one_p1_candidate_resolved": priority_reduction == 1,
        "p90_not_regressed": float(candidate_overall["p90_m"])
        <= float(base_overall["p90_m"]) + 0.002,
        "unexplained_ratio_not_regressed": float(
            candidate_overall["unexplained_over_0_25m"]
        )
        <= float(base_overall["unexplained_over_0_25m"]) + 0.00001,
        "no_new_p0_candidate": int(candidate["priority_counts"].get("P0", 0))
        <= int(baseline["priority_counts"].get("P0", 0)),
    }
    passed = all(gates.values())
    output = Path(output_path).resolve()
    write_json(
        output,
        {
            "schema_version": "railway.platform-clock-candidate-gate.v1",
            "candidate_report": str(Path(candidate_report_path).resolve()),
            "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
            "candidate_gap_report": str(Path(candidate_gap_report_path).resolve()),
            "fixed_view_manifest": str(Path(fixed_view_manifest_path).resolve()),
            "target": {
                "station_m": station_m,
                "cross_m": cross_m,
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
                "p1_candidate_reduction": priority_reduction,
            },
            "gates": gates,
            "passed": passed,
            "status": (
                "candidate_gate_passed_formal_baseline_unchanged"
                if passed
                else "candidate_gate_failed_formal_baseline_unchanged"
            ),
            "limitations": [
                "The clock face graphics and hands remain unresolved.",
                "The hanger-top connection remains a low-confidence inference because no continuous observed canopy roof exists at this chainage in the current model.",
            ],
        },
    )
    return output
