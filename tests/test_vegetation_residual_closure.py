from __future__ import annotations

import json
from pathlib import Path

from railway_recon.vegetation_residual_closure import (
    close_rgb_vegetation_residuals,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_close_rgb_vegetation_residual(tmp_path: Path) -> None:
    model = tmp_path / "model.obj"
    model.write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"assets": []})
    gap = _write(
        tmp_path / "gap.json",
        {
            "vertical_candidates": [
                {
                    "candidate_id": "A",
                    "station_m": 10.0,
                    "cross_m": 40.0,
                    "extent_xyz_m": [0.2, 0.2, 3.0],
                }
            ]
        },
    )
    detail = _write(
        tmp_path / "detail.json",
        {"candidates": [{"candidate_id": "A", "rendered_candidate_point_count": 80, "evidence_figure": "a.png"}]},
    )
    photo = _write(
        tmp_path / "photo.json",
        {"candidates": [{"candidate_id": "A", "trusted_reviewable_view_count": 3, "evidence_sheet": "a.jpg"}]},
    )
    attributes = _write(
        tmp_path / "attributes.json",
        {
            "records": [
                {
                    "candidate_id": "A",
                    "interpretation": "rgb_vegetation_like_review_no_build",
                    "green_dominant_fraction": 0.9,
                    "median_normalized_excess_green": 0.11,
                    "median_rgb_8bit": [90.0, 120.0, 95.0],
                }
            ]
        },
    )
    output = close_rgb_vegetation_residuals(
        gap_report_path=gap,
        detail_evidence_path=detail,
        photo_evidence_path=photo,
        attribute_evidence_path=attributes,
        model_path=model,
        registry_path=registry,
        candidate_ids=("A",),
        output_path=tmp_path / "closure.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["closed_candidate_count"] == 1
    assert result["records"][0]["geometry_action"] == (
        "exclude_from_structured_asset_model"
    )
