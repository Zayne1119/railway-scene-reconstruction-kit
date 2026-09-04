from __future__ import annotations

import json
from pathlib import Path

from railway_recon.station_entry_candidate_gate import gate_station_entry_candidate


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_station_entry_candidate_gate_passes_improving_candidate(tmp_path: Path) -> None:
    candidate_report = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "gap.json"
    views = tmp_path / "views.json"
    _write(
        candidate_report,
        {"mesh_audit_passed": True, "point_support_gate_summary": {"all_passed": True}},
    )
    _write(
        baseline,
        {
            "overall": {"coverage_at_0_20m": 0.60, "unexplained_over_0_25m": 0.36, "p90_m": 1.5},
            "priority_counts": {"P1": 37},
            "vertical_candidate_count": 224,
        },
    )
    _write(
        candidate,
        {
            "overall": {"coverage_at_0_20m": 0.62, "unexplained_over_0_25m": 0.35, "p90_m": 1.4},
            "priority_counts": {"P1": 34},
            "vertical_candidate_count": 222,
            "vertical_candidates": [
                {
                    "candidate_id": "GAP-VERTICAL-0004",
                    "priority": "P0",
                    "station_m": 103.0,
                    "cross_m": 20.4,
                    "sample_point_count": 44,
                }
            ],
        },
    )
    _write(views, {"view_count": 6})
    output = gate_station_entry_candidate(
        candidate_report_path=candidate_report,
        baseline_gap_report_path=baseline,
        candidate_gap_report_path=candidate,
        fixed_view_manifest_path=views,
        output_path=tmp_path / "gate.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["station_entry_p0_residual_dispositions"][0]["model_action"].startswith("refine")
