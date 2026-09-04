from __future__ import annotations

import json
from pathlib import Path

from railway_recon.platform_clock_candidate_gate import gate_platform_clock_candidate


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_platform_clock_gate_passes_resolved_supported_candidate(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    views = tmp_path / "views.json"
    _write(
        report,
        {
            "mesh_audit_passed": True,
            "fit": {
                "station_center_m": 84.15,
                "cross_center_m": -27.86,
                "occupied_angle_bins_of_16": 16,
            },
            "point_support": {
                "passed_review_gate": True,
                "p90_m": 0.08,
            },
        },
    )
    overall = {
        "p90_m": 1.42,
        "coverage_at_0_20m": 0.62,
        "unexplained_over_0_25m": 0.3524,
    }
    _write(
        baseline,
        {
            "overall": overall,
            "priority_counts": {"P0": 4, "P1": 24},
            "vertical_candidates": [
                {
                    "candidate_id": "GAP-VERTICAL-0011",
                    "station_m": 84.18,
                    "cross_m": -27.87,
                }
            ],
        },
    )
    _write(
        candidate,
        {
            "overall": {
                "p90_m": 1.419,
                "coverage_at_0_20m": 0.621,
                "unexplained_over_0_25m": 0.3523,
            },
            "priority_counts": {"P0": 4, "P1": 23},
            "vertical_candidates": [],
        },
    )
    _write(views, {"view_count": 6})
    output = gate_platform_clock_candidate(
        candidate_report_path=report,
        baseline_gap_report_path=baseline,
        candidate_gap_report_path=candidate,
        fixed_view_manifest_path=views,
        output_path=tmp_path / "gate.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["target"]["baseline_candidate_ids"] == ["GAP-VERTICAL-0011"]
