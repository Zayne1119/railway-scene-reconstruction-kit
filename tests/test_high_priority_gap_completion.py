from __future__ import annotations

import json
from pathlib import Path

from railway_recon.high_priority_gap_completion import (
    build_high_priority_gap_completion_report,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_build_high_priority_gap_completion_report(tmp_path: Path) -> None:
    model = tmp_path / "model.obj"
    model.write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"summary": {"asset_count": 12}})
    gap = _write(
        tmp_path / "gap.json",
        {
            "corridor_scope": {"station_minimum_m": 0.0, "station_maximum_m": 100.0},
            "priority_counts": {"P0": 1, "P1": 0, "P2": 1, "P3": 0},
            "overhead_linear_priority_counts": {"P0": 0, "P1": 1, "P2": 1},
            "vertical_candidates": [
                {"priority": "P0", "corridor_scope_ownership": "adjacent_next_segment"}
            ],
            "overhead_linear_candidates": [
                {"priority": "P1", "station_range_m": [101.0, 110.0]}
            ],
            "overall": {"p90_m": 1.0},
        },
    )
    ledger = _write(
        tmp_path / "ledger.json",
        {"summary": {"all_baseline_p1_disposed": True, "unresolved_p1_count": 0}},
    )
    gate = _write(tmp_path / "gate.json", {"passed": True})
    output = build_high_priority_gap_completion_report(
        model_path=model,
        registry_path=registry,
        current_gap_report_path=gap,
        vertical_disposition_ledger_path=ledger,
        candidate_gate_paths=(gate,),
        output_path=tmp_path / "completion.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["effective_current_segment_counts"]["in_scope_p0_vertical"] == 0
    assert result["formal_baseline_changed"] is False
