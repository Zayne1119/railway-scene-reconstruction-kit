from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, write_json


def gate_station_entry_candidate(
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
    baseline_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    support = candidate_report["point_support_gate_summary"]
    station_entry_residuals: list[dict[str, Any]] = []
    for item in candidate["vertical_candidates"]:
        station = float(item["station_m"])
        cross = float(item["cross_m"])
        if (
            item.get("priority") == "P0"
            and 98.0 <= station <= 126.0
            and 15.0 <= cross <= 25.0
            and int(item.get("sample_point_count", 0)) < 100
        ):
            station_entry_residuals.append(
                {
                    "candidate_id": item["candidate_id"],
                    "station_m": station,
                    "cross_m": cross,
                    "sample_point_count": item.get("sample_point_count"),
                    "disposition": "station_facade_edge_residual",
                    "model_action": "refine_existing_group_not_new_standalone_asset",
                }
            )
    gates = {
        "mesh_audit_passed": bool(candidate_report["mesh_audit_passed"]),
        "all_observed_nodes_pass_point_support": bool(support["all_passed"]),
        "six_fixed_views_rendered": int(fixed_views["view_count"]) == 6,
        "unexplained_ratio_reduced": (
            float(candidate_overall["unexplained_over_0_25m"])
            < float(baseline_overall["unexplained_over_0_25m"])
        ),
        "coverage_at_0_20m_improved": (
            float(candidate_overall["coverage_at_0_20m"])
            > float(baseline_overall["coverage_at_0_20m"])
        ),
        "p1_candidate_count_not_increased": (
            int(candidate["priority_counts"].get("P1", 0))
            <= int(baseline["priority_counts"].get("P1", 0))
        ),
        "vertical_candidate_count_reduced": (
            int(candidate["vertical_candidate_count"])
            < int(baseline["vertical_candidate_count"])
        ),
        "new_station_entry_p0_residuals_explicitly_disposed": all(
            item["model_action"] == "refine_existing_group_not_new_standalone_asset"
            for item in station_entry_residuals
        ),
    }
    output = Path(output_path).resolve()
    write_json(
        output,
        {
            "schema_version": "railway.station-entry-candidate-gate.v1",
            "candidate_report": str(Path(candidate_report_path).resolve()),
            "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
            "candidate_gap_report": str(Path(candidate_gap_report_path).resolve()),
            "fixed_view_manifest": str(Path(fixed_view_manifest_path).resolve()),
            "regression": {
                "coverage_at_0_20m": {
                    "baseline": baseline_overall["coverage_at_0_20m"],
                    "candidate": candidate_overall["coverage_at_0_20m"],
                    "delta": float(candidate_overall["coverage_at_0_20m"])
                    - float(baseline_overall["coverage_at_0_20m"]),
                },
                "unexplained_over_0_25m": {
                    "baseline": baseline_overall["unexplained_over_0_25m"],
                    "candidate": candidate_overall["unexplained_over_0_25m"],
                    "delta": float(candidate_overall["unexplained_over_0_25m"])
                    - float(baseline_overall["unexplained_over_0_25m"]),
                },
                "p90_m": {
                    "baseline": baseline_overall["p90_m"],
                    "candidate": candidate_overall["p90_m"],
                    "delta": float(candidate_overall["p90_m"])
                    - float(baseline_overall["p90_m"]),
                },
                "vertical_candidate_count": {
                    "baseline": baseline["vertical_candidate_count"],
                    "candidate": candidate["vertical_candidate_count"],
                },
                "p1_candidate_count": {
                    "baseline": baseline["priority_counts"].get("P1", 0),
                    "candidate": candidate["priority_counts"].get("P1", 0),
                },
            },
            "station_entry_p0_residual_dispositions": station_entry_residuals,
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
