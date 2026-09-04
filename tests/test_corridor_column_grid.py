from __future__ import annotations

import pytest

from railway_recon.corridor_column_grid import recover_corridor_column_grid_data


def _candidate(identifier: str, x: float, y: float, predicted: str = "canopy_column") -> dict:
    return {
        "id": identifier,
        "center_x": x,
        "center_y": y,
        "minimum_z": 1.0,
        "maximum_z": 6.0,
        "height_m": 5.0,
        "footprint_m": 0.8,
        "predicted_class": predicted,
        "features": {"vertical_occupied_ratio": 0.95},
    }


def _settings() -> dict:
    return {
        "schema_version": "railway.corridor-column-grid-settings.v1",
        "minimum_seed_count": 2,
        "minimum_column_height_m": 3.5,
        "maximum_column_height_m": 7.0,
        "minimum_vertical_occupied_ratio": 0.85,
        "minimum_absolute_cross_position_m": 2.0,
        "maximum_absolute_cross_position_m": 40.0,
        "nominal_column_spacing_m": 9.0,
        "minimum_column_spacing_m": 7.0,
        "maximum_column_spacing_m": 11.0,
        "seed_grid_match_tolerance_m": 0.75,
        "maximum_seed_grid_residual_m": 0.75,
        "require_reviewed_seed_ids": False,
        "reviewed_seed_ids": [],
    }


def test_grid_uses_one_phase_across_segment_boundary_and_fills_missing_positions() -> None:
    plan = {
        "frame": {
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "lateral_xy": [0.0, 1.0],
        },
        "coverage_interval_m": [0.0, 100.0],
        "segments": [
            {"segment_id": "A", "owned_interval_m": [0.0, 50.0]},
            {"segment_id": "B", "owned_interval_m": [50.0, 100.0]},
        ],
    }
    reports = {
        "A": {
            "candidates": [
                _candidate("A1", 5.0, -5.0),
                _candidate("A2", 23.0, -5.0),
                _candidate("IGNORE", 41.0, -5.0, "catenary_support"),
            ]
        },
        "B": {
            "candidates": [
                _candidate("B1", 59.0, -5.0),
                _candidate("B2", 77.0, -5.0),
                _candidate("B3", 95.0, -5.0),
            ]
        },
    }

    result = recover_corridor_column_grid_data(plan, reports, side="left", settings=_settings())

    assert result["passed"] is True
    assert result["seed_count"] == 5
    assert result["inferred_position_count"] > 0
    assert result["fit"]["refined_spacing_m"] == pytest.approx(9.0)
    assert {item["owner_segment_id"] for item in result["grid"]} == {"A", "B"}
    stations = [item["predicted_longitudinal_position_m"] for item in result["grid"]]
    assert any(abs(value - 50.0) <= 1.0 for value in stations)


def test_reviewed_seed_mode_rejects_unlisted_predictions() -> None:
    plan = {
        "frame": {
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "lateral_xy": [0.0, 1.0],
        },
        "coverage_interval_m": [0.0, 50.0],
        "segments": [{"segment_id": "A", "owned_interval_m": [0.0, 50.0]}],
    }
    reports = {
        "A": {
            "candidates": [
                _candidate("A1", 5.0, -5.0),
                _candidate("A2", 23.0, -5.0),
            ]
        }
    }
    settings = _settings()
    settings["require_reviewed_seed_ids"] = True
    settings["reviewed_seed_ids"] = ["A::A1"]

    try:
        recover_corridor_column_grid_data(plan, reports, side="left", settings=settings)
    except ValueError as error:
        assert "Too few" in str(error)
    else:
        raise AssertionError("Reviewed mode must fail with only one accepted seed")
