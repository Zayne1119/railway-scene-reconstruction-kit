from __future__ import annotations

import json
from pathlib import Path

from railway_recon.station_entry_residual_closure import (
    BLUE_EDGE,
    SPARSE_CONTEXT,
    close_station_entry_surface_residuals,
)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_close_station_entry_surface_residuals(tmp_path: Path) -> None:
    model = tmp_path / "model.obj"
    model.write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"assets": []})
    candidates = [
        {
            "candidate_id": "BLUE",
            "station_m": 108.0,
            "cross_m": 20.2,
            "extent_xyz_m": [0.1, 0.2, 2.1],
        },
        {
            "candidate_id": "SPARSE",
            "station_m": 120.0,
            "cross_m": 24.8,
            "extent_xyz_m": [0.2, 0.2, 2.9],
        },
    ]
    gap = _write(tmp_path / "gap.json", {"vertical_candidates": candidates})
    detail = _write(
        tmp_path / "detail.json",
        {
            "candidates": [
                {"candidate_id": "BLUE", "rendered_candidate_point_count": 98, "evidence_figure": "blue.png"},
                {"candidate_id": "SPARSE", "rendered_candidate_point_count": 87, "evidence_figure": "sparse.png"},
            ]
        },
    )
    photo = _write(
        tmp_path / "photo.json",
        {
            "candidates": [
                {"candidate_id": "BLUE", "trusted_reviewable_view_count": 2, "evidence_sheet": "blue.jpg"},
                {"candidate_id": "SPARSE", "trusted_reviewable_view_count": 2, "evidence_sheet": "sparse.jpg"},
            ]
        },
    )
    attributes = _write(
        tmp_path / "attributes.json",
        {
            "records": [
                {"candidate_id": "BLUE", "full_resolution_point_count": 98, "median_rgb_8bit": [17, 31, 58], "median_rgb_saturation": 0.72},
                {"candidate_id": "SPARSE", "full_resolution_point_count": 87, "median_rgb_8bit": [79, 80, 85], "median_rgb_saturation": 0.07},
            ]
        },
    )
    surface = _write(
        tmp_path / "surface.json",
        {
            "segmented_cross_histogram_peaks": [
                {
                    "station_range_m": [103.5, 112.5],
                    "cross_peaks": [{"cross_range_m": [15.6, 15.625], "point_count": 1000}],
                },
                {
                    "station_range_m": [115.0, 124.0],
                    "cross_peaks": [{"cross_range_m": [20.075, 20.1], "point_count": 1000}],
                },
            ]
        },
    )
    output = close_station_entry_surface_residuals(
        gap_report_path=gap,
        detail_evidence_path=detail,
        photo_evidence_path=photo,
        attribute_evidence_path=attributes,
        surface_analysis_path=surface,
        model_path=model,
        registry_path=registry,
        decisions={"BLUE": BLUE_EDGE, "SPARSE": SPARSE_CONTEXT},
        output_path=tmp_path / "closure.json",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"] == {
        "reviewed_candidate_count": 2,
        "closed_candidate_count": 2,
        "new_asset_count": 0,
        "geometry_write": False,
    }
    assert result["records"][0]["semantic_class"] == (
        "station_entry_blue_facade_or_door_edge_residual"
    )
    assert result["records"][1]["geometry_action"] == (
        "withhold_geometry_until_continuous_surface_evidence_exists"
    )
