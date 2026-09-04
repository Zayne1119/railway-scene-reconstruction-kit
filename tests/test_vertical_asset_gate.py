from __future__ import annotations

from railway_recon.vertical_asset_gate import evaluate_vertical_asset_gate


def _candidate(occupancy: float, rail_distance: float) -> dict[str, object]:
    return {
        "id": "VERTICAL-HYPOTHESIS-TEST",
        "features": {"vertical_occupied_ratio": occupancy},
        "rail_context": {"nearest_rail_distance_m": rail_distance},
    }


def test_catenary_gate_rejects_sparse_wire_aligned_ghost() -> None:
    result = evaluate_vertical_asset_gate(
        _candidate(0.053, 0.607),
        {
            "trusted_reviewable_view_count": 4,
            "visible_full_height_shaft": False,
            "base_support_confirmed": False,
        },
        asset_class="catenary_support",
    )
    assert result["passed"] is False
    assert "visible_full_height_shaft" in result["failed_checks"]
    assert "minimum_rail_clearance_1_0m" in result["failed_checks"]


def test_catenary_gate_accepts_continuous_photo_confirmed_mast() -> None:
    result = evaluate_vertical_asset_gate(
        _candidate(0.98, 2.3),
        {
            "trusted_reviewable_view_count": 4,
            "visible_full_height_shaft": True,
            "base_support_confirmed": True,
        },
        asset_class="catenary_support",
    )
    assert result["passed"] is True


def test_canopy_column_does_not_use_catenary_rail_clearance_rule() -> None:
    result = evaluate_vertical_asset_gate(
        _candidate(0.9, 0.4),
        {
            "trusted_reviewable_view_count": 3,
            "visible_full_height_shaft": True,
            "base_support_confirmed": True,
        },
        asset_class="canopy_column",
    )
    assert result["passed"] is True
    assert "minimum_rail_clearance_1_0m" not in result["checks"]
