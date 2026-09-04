from __future__ import annotations

from railway_recon.model_support_comparison import compare_support_values


def test_compare_support_values() -> None:
    before = {
        "objects": [
            {
                "object_name": "A",
                "asset_type": "platform_surface",
                "p90_m": 0.2,
                "coverage_at_0_10m": 0.1,
            },
            {
                "object_name": "REMOVED",
                "asset_type": "column",
                "p90_m": 1.0,
                "coverage_at_0_10m": 0.0,
            },
        ]
    }
    after = {
        "objects": [
            {
                "object_name": "A",
                "asset_type": "platform_surface",
                "p90_m": 0.1,
                "coverage_at_0_10m": 0.9,
            }
        ]
    }
    result = compare_support_values(before, after)
    assert result["removed_objects"] == ["REMOVED"]
    assert result["opposite_platform_surface"][0]["p90_change_m"] == -0.1
    assert result["opposite_platform_surface"][0]["coverage_at_0_10m_change"] == 0.8
