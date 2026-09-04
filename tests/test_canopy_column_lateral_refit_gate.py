from __future__ import annotations

from pathlib import Path

from railway_recon.canopy_column_lateral_refit_gate import (
    gate_canopy_column_lateral_refit,
)
from railway_recon.io import write_json


def test_lateral_refit_gate_accepts_index_safe_supported_candidate(
    tmp_path: Path,
) -> None:
    obj = tmp_path / "candidate.obj"
    obj.write_text(
        "v 10 2 0\nv 10 2 1\nv 10 2 2\no COLUMN-01\nf 1 2 3\n",
        encoding="utf-8",
    )
    write_json(tmp_path / "origin.json", {"origin_xyz": [0.0, 0.0, 0.0]})
    write_json(
        tmp_path / "frame.json",
        {"frame": {"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0], "cross_xy": [0.0, 1.0]}},
    )
    write_json(
        tmp_path / "refit.json",
        {
            "plans": [
                {
                    "asset_id": "COLUMN-01",
                    "current_station_m": 10.0,
                    "target_cross_center_m": 2.0,
                }
            ]
        },
    )
    p0 = {
        "priority": "P0",
        "corridor_scope_ownership": "within_current_segment",
        "asset_relation": "existing_asset_geometry_refinement",
    }
    write_json(
        tmp_path / "baseline.json",
        {
            "vertical_candidates": [p0],
            "overall": {"coverage_at_0_10m": 0.4, "unexplained_over_0_25m": 0.3},
        },
    )
    write_json(
        tmp_path / "candidate.json",
        {
            "vertical_candidates": [],
            "overall": {"coverage_at_0_10m": 0.5, "unexplained_over_0_25m": 0.2},
        },
    )
    write_json(tmp_path / "mesh.json", {"passed": True})
    write_json(tmp_path / "views.json", {"view_count": 6})
    result = gate_canopy_column_lateral_refit(
        refit_report_path=tmp_path / "refit.json",
        candidate_obj_path=obj,
        model_origin_path=tmp_path / "origin.json",
        frame_report_path=tmp_path / "frame.json",
        baseline_gap_report_path=tmp_path / "baseline.json",
        candidate_gap_report_path=tmp_path / "candidate.json",
        mesh_audit_path=tmp_path / "mesh.json",
        fixed_view_manifest_path=tmp_path / "views.json",
        output_path=tmp_path / "gate.json",
        expected_column_count=1,
    )
    assert result["passed"] is True
    assert result["status"] == "geometry_candidate_accepted"
