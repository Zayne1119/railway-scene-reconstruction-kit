from __future__ import annotations

import json
from pathlib import Path

from railway_recon.boundary_conductor_candidate_gate import (
    gate_boundary_conductor_candidate,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_gate_boundary_conductor_candidate(tmp_path: Path) -> None:
    report = _write(
        tmp_path / "candidate.json",
        {
            "added_asset_count": 3,
            "fit_records": [
                {"fit": {"passed": True, "residual_p90_m": 0.02}}
                for _ in range(3)
            ],
        },
    )
    _write(tmp_path / "mesh_audit.json", {"passed": True})
    baseline = _write(
        tmp_path / "baseline.json",
        {
            "corridor_scope": {"station_minimum_m": 0.0, "station_maximum_m": 100.0},
            "overall": {"p90_m": 1.0, "coverage_at_0_20m": 0.5, "unexplained_over_0_25m": 0.4},
            "overhead_linear_candidates": [
                {"candidate_id": name, "priority": "P1", "station_range_m": [95.0, 110.0]}
                for name in ("A", "B", "C")
            ],
        },
    )
    candidate = _write(
        tmp_path / "after.json",
        {
            "corridor_scope": {"station_minimum_m": 0.0, "station_maximum_m": 100.0},
            "overall": {"p90_m": 0.9, "coverage_at_0_20m": 0.6, "unexplained_over_0_25m": 0.3},
            "overhead_linear_priority_counts": {"P0": 0, "P1": 1, "P2": 0},
            "overhead_linear_candidates": [
                {"candidate_id": "NEXT", "priority": "P1", "station_range_m": [100.2, 112.0]}
            ],
        },
    )
    views = _write(tmp_path / "views.json", {"view_count": 6})
    output = gate_boundary_conductor_candidate(
        candidate_report_path=report,
        baseline_gap_report_path=baseline,
        candidate_gap_report_path=candidate,
        fixed_view_manifest_path=views,
        output_path=tmp_path / "gate.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["target"]["candidate_in_scope_high_priority_ids"] == []
