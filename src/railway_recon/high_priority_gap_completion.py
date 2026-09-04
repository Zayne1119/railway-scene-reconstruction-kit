from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


def _in_scope_high_priority_linear(report: dict[str, Any]) -> list[dict[str, Any]]:
    scope = report["corridor_scope"]
    low = float(scope["station_minimum_m"])
    high = float(scope["station_maximum_m"])
    return [
        item
        for item in report["overhead_linear_candidates"]
        if item["priority"] in {"P0", "P1"}
        and min(float(item["station_range_m"][1]), high)
        - max(float(item["station_range_m"][0]), low)
        > 0.0
    ]


def build_high_priority_gap_completion_report(
    *,
    model_path: str | Path,
    registry_path: str | Path,
    current_gap_report_path: str | Path,
    vertical_disposition_ledger_path: str | Path,
    candidate_gate_paths: tuple[str | Path, ...],
    output_path: str | Path,
) -> Path:
    model = Path(model_path).resolve()
    registry_path = Path(registry_path).resolve()
    registry = load_json(registry_path)
    current = load_json(Path(current_gap_report_path))
    vertical_ledger = load_json(Path(vertical_disposition_ledger_path))
    gates = [(Path(path).resolve(), load_json(Path(path))) for path in candidate_gate_paths]
    current_p0_vertical = [
        item
        for item in current["vertical_candidates"]
        if item["priority"] == "P0"
        and item["corridor_scope_ownership"] == "within_current_segment"
    ]
    current_high_linear = _in_scope_high_priority_linear(current)
    checks = {
        "all_baseline_vertical_p1_disposed": bool(
            vertical_ledger["summary"]["all_baseline_p1_disposed"]
        ),
        "no_in_scope_p0_vertical_candidate": len(current_p0_vertical) == 0,
        "no_in_scope_p0_or_p1_linear_candidate": len(current_high_linear) == 0,
        "all_candidate_gates_passed": bool(gates)
        and all(bool(value.get("passed")) for _, value in gates),
    }
    passed = all(checks.values())
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.high-priority-gap-completion.v1",
            "candidate_release": {
                "model": str(model),
                "model_sha256": sha256_file(model),
                "registry": str(registry_path),
                "registry_sha256": sha256_file(registry_path),
                "asset_count": int(registry["summary"]["asset_count"]),
            },
            "evidence": {
                "current_gap_report": str(Path(current_gap_report_path).resolve()),
                "vertical_disposition_ledger": str(
                    Path(vertical_disposition_ledger_path).resolve()
                ),
                "candidate_gates": [str(path) for path, _ in gates],
            },
            "raw_detector_counts": {
                "vertical": current["priority_counts"],
                "overhead_linear": current["overhead_linear_priority_counts"],
            },
            "effective_current_segment_counts": {
                "unresolved_baseline_vertical_p1": int(
                    vertical_ledger["summary"]["unresolved_p1_count"]
                ),
                "in_scope_p0_vertical": len(current_p0_vertical),
                "in_scope_p0_or_p1_linear": len(current_high_linear),
                "adjacent_segment_p0_vertical": sum(
                    item["priority"] == "P0"
                    and item["corridor_scope_ownership"] != "within_current_segment"
                    for item in current["vertical_candidates"]
                ),
                "adjacent_or_outside_p0_or_p1_linear": sum(
                    item["priority"] in {"P0", "P1"}
                    and item not in current_high_linear
                    for item in current["overhead_linear_candidates"]
                ),
            },
            "quality": current["overall"],
            "checks": checks,
            "passed": passed,
            "formal_baseline_changed": False,
            "status": (
                "high_priority_current_segment_candidate_chain_closed_not_promoted"
                if passed
                else "high_priority_current_segment_gaps_remain"
            ),
            "limitations": [
                "P2/P3 residuals remain available for lower-priority refinement and context-layer classification.",
                "Adjacent-segment candidates are not current-segment defects and remain for segment stitching.",
                "The candidate release is not the formal baseline until explicitly promoted.",
            ],
        },
    )
    return destination
