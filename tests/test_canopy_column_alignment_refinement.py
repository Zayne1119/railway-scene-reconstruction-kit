from __future__ import annotations

from railway_recon.canopy_column_alignment_refinement import estimate_row_alignment


def test_estimate_periodic_row_alignment() -> None:
    columns = [("C1", 0.0, 5.0), ("C2", 9.0, 5.0), ("C3", 18.0, 5.0)]
    candidates = [
        ("P1", 0.1, 7.2, 5.0),
        ("P2", 9.1, 7.3, 5.1),
        ("P3", 18.0, 7.2, 4.9),
        ("EDGE", 9.0, 15.0, 4.0),
    ]
    result = estimate_row_alignment(
        columns, candidates, cross_delta_range_m=(1.0, 4.0)
    )
    assert result["match_count"] == 3
    assert abs(result["median_delta_cross_m"] - 2.2) < 1e-6


def test_reject_incoherent_row_alignment() -> None:
    columns = [("C1", 0.0, 5.0), ("C2", 9.0, 5.0), ("C3", 18.0, 5.0)]
    candidates = [
        ("P1", 0.0, 6.1, 5.0),
        ("P2", 9.0, 7.5, 5.0),
        ("P3", 18.0, 8.8, 5.0),
    ]
    try:
        estimate_row_alignment(columns, candidates, cross_delta_range_m=(1.0, 4.0))
    except ValueError as error:
        assert "Incoherent" in str(error)
    else:
        raise AssertionError("Incoherent row offsets must fail closed")
