from __future__ import annotations

import json
from pathlib import Path

from railway_recon.external_context_post_closure import (
    close_external_context_post_row,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_close_external_context_post_row(tmp_path: Path) -> None:
    ids = ["A", "B", "C"]
    model = tmp_path / "model.obj"
    model.write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"assets": []})
    gap = _write(
        tmp_path / "gap.json",
        {
            "vertical_candidates": [
                {
                    "candidate_id": candidate_id,
                    "station_m": station,
                    "cross_m": 29.7 + offset,
                    "extent_xyz_m": [0.1, 0.1, 2.8],
                    "nearest_existing_vertical_asset": {"footprint_distance_m": 24.0},
                }
                for candidate_id, station, offset in zip(
                    ids, [32.0, 49.0, 64.0], [-0.02, 0.01, 0.02], strict=True
                )
            ]
        },
    )
    detail = _write(
        tmp_path / "detail.json",
        {
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "rendered_candidate_point_count": 100,
                    "evidence_figure": f"{candidate_id}.png",
                }
                for candidate_id in ids
            ]
        },
    )
    photo = _write(
        tmp_path / "photo.json",
        {
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "trusted_reviewable_view_count": 3,
                    "evidence_sheet": f"{candidate_id}.jpg",
                }
                for candidate_id in ids
            ]
        },
    )
    attributes = _write(
        tmp_path / "attributes.json",
        {
            "records": [
                {
                    "candidate_id": candidate_id,
                    "vegetation_class_fraction": 0.0,
                    "median_rgb_8bit": [110.0, 120.0, 122.0],
                    "median_rgb_saturation": 0.08,
                }
                for candidate_id in ids
            ]
        },
    )
    output = close_external_context_post_row(
        gap_report_path=gap,
        detail_evidence_path=detail,
        photo_evidence_path=photo,
        attribute_evidence_path=attributes,
        model_path=model,
        registry_path=registry,
        candidate_ids=tuple(ids),
        output_path=tmp_path / "closure.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["closed_candidate_count"] == 3
    assert result["summary"]["geometry_write"] is False
    assert result["semantic_decision"]["class"] == (
        "external_non_railway_context_post_row"
    )
