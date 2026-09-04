from __future__ import annotations

import json
from pathlib import Path

from railway_recon.catenary_companion_residual_closure import (
    close_catenary_companion_residuals,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_close_catenary_companion_residuals(tmp_path: Path) -> None:
    model = tmp_path / "model.obj"
    model.write_text("o mast\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"assets": []})
    gap = _write(
        tmp_path / "gap.json",
        {
            "vertical_candidates": [
                {
                    "candidate_id": "GAP-VERTICAL-0001",
                    "station_m": 10.0,
                    "cross_m": -2.0,
                    "extent_xyz_m": [0.2, 0.2, 5.0],
                    "vertical_occupancy": 0.5,
                    "semantic_review_hint": (
                        "station_aligned_catenary_companion_trace_or_attachment"
                    ),
                    "aligned_existing_catenary_asset": {
                        "asset_id": "MAST-1",
                        "station_delta_m": 0.02,
                        "cross_delta_m": 1.8,
                    },
                }
            ]
        },
    )
    detail = _write(
        tmp_path / "detail.json",
        {
            "candidates": [
                {
                    "candidate_id": "GAP-VERTICAL-0001",
                    "rendered_candidate_point_count": 100,
                    "evidence_figure": "detail.png",
                }
            ]
        },
    )
    photo = _write(
        tmp_path / "photo.json",
        {
            "candidates": [
                {
                    "candidate_id": "GAP-VERTICAL-0001",
                    "trusted_reviewable_view_count": 2,
                    "evidence_sheet": "photo.jpg",
                }
            ]
        },
    )
    output = tmp_path / "closure.json"
    close_catenary_companion_residuals(
        gap_report_path=gap,
        detail_evidence_path=detail,
        photo_evidence_path=photo,
        model_path=model,
        registry_path=registry,
        candidate_ids=("GAP-VERTICAL-0001",),
        output_path=output,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["closed_candidate_count"] == 1
    assert result["summary"]["geometry_write"] is False
    assert result["records"][0]["geometry_action"] == (
        "retain_existing_mast_no_new_vertical_asset"
    )
