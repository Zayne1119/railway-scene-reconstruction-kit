from __future__ import annotations

import json
from pathlib import Path

from railway_recon.vertical_gap_disposition_ledger import (
    build_vertical_gap_disposition_ledger,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _candidate(candidate_id: str, station: float, cross: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "priority": "P1",
        "station_m": station,
        "cross_m": cross,
        "extent_xyz_m": [0.2, 0.2, 2.5],
    }


def test_build_vertical_gap_disposition_ledger(tmp_path: Path) -> None:
    baseline = _write(
        tmp_path / "baseline.json",
        {
            "vertical_candidates": [
                _candidate("A", 10.0, 1.0),
                _candidate("B", 20.0, 2.0),
                _candidate("C", 30.0, 3.0),
            ]
        },
    )
    current = _write(
        tmp_path / "current.json",
        {
            "vertical_candidates": [
                {
                    **_candidate("RENAMED", 30.0, 3.0),
                    "priority": "P2",
                    "semantic_review_hint": "existing_station_entry_surface_edge_or_residual",
                    "automatic_geometry_action": "refine_existing_group_not_new_asset",
                }
            ]
        },
    )
    gate = _write(
        tmp_path / "gate.json",
        {"passed": True, "target": {"baseline_candidate_ids": ["A"]}},
    )
    closure = _write(
        tmp_path / "closure.json",
        {
            "records": [
                {
                    "candidate_id": "OLD-ID",
                    "station_m": 20.05,
                    "cross_m": 2.0,
                    "semantic_class": "reviewed_context_residual",
                }
            ]
        },
    )
    output = build_vertical_gap_disposition_ledger(
        baseline_gap_report_path=baseline,
        current_gap_report_path=current,
        candidate_gate_paths=(gate,),
        closure_report_paths=(closure,),
        output_path=tmp_path / "ledger.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["all_baseline_p1_disposed"] is True
    assert result["summary"]["disposition_counts"] == {
        "built_candidate_gate_passed": 1,
        "closed_no_geometry": 1,
        "existing_group_refinement_not_new_asset": 1,
    }
