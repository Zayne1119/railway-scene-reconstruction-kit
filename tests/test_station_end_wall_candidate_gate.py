from __future__ import annotations

import json
from pathlib import Path

from railway_recon.station_end_wall_candidate_gate import gate_station_end_wall_candidate


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_station_end_wall_gate_passes_six_fragment_reduction(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    views = tmp_path / "views.json"
    _write(
        report,
        {
            "mesh_audit_passed": True,
            "point_support": {"passed_review_gate": True, "sample_count": 1_739},
        },
    )
    _write(
        baseline,
        {
            "overall": {"p90_m": 1.42, "coverage_at_0_20m": 0.62, "unexplained_over_0_25m": 0.3524},
            "vertical_candidate_count": 222,
            "priority_counts": {"P0": 4, "P1": 34},
        },
    )
    _write(
        candidate,
        {
            "overall": {"p90_m": 1.41, "coverage_at_0_20m": 0.621, "unexplained_over_0_25m": 0.3523},
            "vertical_candidate_count": 216,
            "priority_counts": {"P0": 4, "P1": 28},
        },
    )
    _write(views, {"view_count": 6})
    output = gate_station_end_wall_candidate(
        candidate_report_path=report,
        baseline_gap_report_path=baseline,
        candidate_gap_report_path=candidate,
        fixed_view_manifest_path=views,
        output_path=tmp_path / "gate.json",
    )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
