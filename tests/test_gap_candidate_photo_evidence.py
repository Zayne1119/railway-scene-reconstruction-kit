from railway_recon.gap_candidate_photo_evidence import (
    gap_candidate_to_vertical_hypothesis,
)


def test_gap_candidate_adapter_preserves_corridor_coordinates() -> None:
    result = gap_candidate_to_vertical_hypothesis(
        {
            "candidate_id": "GAP-VERTICAL-0010",
            "station_m": 61.68,
            "cross_m": -21.23,
            "minimum_xyz_m": [1.0, 2.0, 20.0],
            "maximum_xyz_m": [1.4, 2.8, 28.2],
            "extent_xyz_m": [0.4, 0.8, 8.2],
            "classification": "large_vertical_surface_or_mixed_cluster",
            "priority": "P1",
        }
    )
    assert result["id"] == "GAP-VERTICAL-0010"
    assert result["longitudinal_position_m"] == 61.68
    assert result["cross_position_m"] == -21.23
    assert result["minimum_z"] == 20.0
    assert result["maximum_z"] == 28.2
    assert result["footprint_m"] == 0.8
    assert result["confidence"] == 0.65
