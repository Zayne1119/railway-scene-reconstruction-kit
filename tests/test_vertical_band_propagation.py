from __future__ import annotations

import pytest

from railway_recon.vertical_band_propagation import evaluate_vertical_band_propagation


def candidate(
    candidate_id: str,
    *,
    occupancy: float,
    points: int,
    footprint: float,
    height: float = 5.0,
) -> dict:
    return {
        "id": candidate_id,
        "height_m": height,
        "footprint_m": footprint,
        "features": {
            "vertical_occupied_ratio": occupancy,
            "local_point_count": points,
        },
    }


def test_similar_band_is_covered_by_reviewed_representative() -> None:
    reviewed = candidate("R1", occupancy=0.82, points=1200, footprint=0.58)
    target = candidate("T1", occupancy=0.75, points=900, footprint=0.65)
    report = evaluate_vertical_band_propagation([reviewed], [reviewed, target])
    assert report["passed"] is True
    assert report["uncovered_candidate_ids"] == []


def test_feature_heterogeneous_band_requires_individual_review() -> None:
    reviewed = candidate("R1", occupancy=1.0, points=4500, footprint=1.30)
    weak = candidate("T1", occupancy=0.2, points=180, footprint=0.40)
    report = evaluate_vertical_band_propagation([reviewed], [reviewed, weak])
    assert report["passed"] is False
    assert report["uncovered_candidate_ids"] == ["T1"]
    assert report["status"] == "individual_review_required_for_uncovered_feature_clusters"


def test_empty_inputs_are_rejected() -> None:
    item = candidate("R1", occupancy=1.0, points=1000, footprint=0.5)
    with pytest.raises(ValueError):
        evaluate_vertical_band_propagation([], [item])
    with pytest.raises(ValueError):
        evaluate_vertical_band_propagation([item], [])
