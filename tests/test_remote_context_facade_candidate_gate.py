from __future__ import annotations

import json
from pathlib import Path

from railway_recon.remote_context_facade_candidate_gate import (
    gate_remote_context_facade_candidate,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_remote_context_facade_gate_passes_two_resolved_fragments(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    views = tmp_path / "views.json"
    _write(
        report,
        {
            "mesh_audit_passed": True,
            "fit": {
                "station_min_m": 95.8,
                "station_max_m": 100.0,
                "cross_peak_m": 42.59,
            },
            "point_support": {
                "passed_review_gate": True,
                "sample_count": 1_160,
                "p90_m": 0.21,
            },
        },
    )
    _write(
        baseline,
        {
            "overall": {"p90_m": 1.42, "coverage_at_0_20m": 0.62, "unexplained_over_0_25m": 0.3524},
            "vertical_candidate_count": 215,
            "priority_counts": {"P0": 4, "P1": 19},
            "vertical_candidates": [
                {"candidate_id": "A", "station_m": 96.5, "cross_m": 42.6},
                {"candidate_id": "B", "station_m": 99.0, "cross_m": 42.58},
            ],
        },
    )
    _write(
        candidate,
        {
            "overall": {"p90_m": 1.41, "coverage_at_0_20m": 0.621, "unexplained_over_0_25m": 0.3521},
            "vertical_candidate_count": 213,
            "priority_counts": {"P0": 4, "P1": 17},
            "vertical_candidates": [],
        },
    )
    _write(views, {"view_count": 6})
    output = gate_remote_context_facade_candidate(
        candidate_report_path=report,
        baseline_gap_report_path=baseline,
        candidate_gap_report_path=candidate,
        fixed_view_manifest_path=views,
        output_path=tmp_path / "gate.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["regression"]["p1_candidate_reduction"] == 2
