from __future__ import annotations

import json
from pathlib import Path

from railway_recon.canopy_joint_residual_closure import close_canopy_joint_residuals


def test_closes_ten_known_joint_residuals(tmp_path: Path) -> None:
    candidates = []
    index = 0
    for station, crosses in ((7.7, (4.3, 4.7, -24.6, -25.2)), (61.74, (4.4, -25.4, -24.7)), (115.69, (4.4, -25.0, -25.8))):
        for cross in crosses:
            index += 1
            candidates.append(
                {
                    "candidate_id": f"GAP-VERTICAL-{index:04d}",
                    "priority": "P1",
                    "station_m": station,
                    "cross_m": cross,
                    "extent_xyz_m": [0.2, 0.2, 2.1],
                }
            )
    gap = tmp_path / "gap.json"
    gate = tmp_path / "gate.json"
    evidence = tmp_path / "evidence.json"
    gap.write_text(json.dumps({"vertical_candidates": candidates}), encoding="utf-8")
    gate.write_text(json.dumps({"accepted": False}), encoding="utf-8")
    evidence.write_text(json.dumps({"seams": []}), encoding="utf-8")
    output = close_canopy_joint_residuals(
        gap_report_path=gap,
        rejected_weld_gate_path=gate,
        local_seam_evidence_path=evidence,
        output_path=tmp_path / "closure.json",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["closed_candidate_count"] == 10
    assert report["geometry_changed"] is False
