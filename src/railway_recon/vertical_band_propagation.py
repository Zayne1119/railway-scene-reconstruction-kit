from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def _features(candidate: dict[str, Any]) -> dict[str, float]:
    features = candidate.get("features", {})
    return {
        "height_m": float(candidate.get("height_m", 0.0)),
        "footprint_m": float(candidate.get("footprint_m", 0.0)),
        "vertical_occupied_ratio": float(features.get("vertical_occupied_ratio", 0.0)),
        "log10_local_point_count": math.log10(
            max(float(features.get("local_point_count", 0.0)), 1.0)
        ),
    }


def evaluate_vertical_band_propagation(
    representatives: Iterable[dict[str, Any]],
    band_candidates: Iterable[dict[str, Any]],
    *,
    maximum_height_delta_m: float = 0.75,
    maximum_footprint_delta_m: float = 0.45,
    maximum_occupancy_delta: float = 0.20,
    maximum_log10_point_count_delta: float = 0.75,
) -> dict[str, Any]:
    """Check whether reviewed representatives cover every geometry phenotype in a band.

    Passing this gate only permits semantic decisions to be considered for reuse. It never
    confirms an asset class and never replaces per-candidate photo checks when projections
    disagree.
    """
    representative_list = list(representatives)
    candidate_list = list(band_candidates)
    if not representative_list:
        raise ValueError("At least one reviewed representative is required")
    if not candidate_list:
        raise ValueError("At least one band candidate is required")
    thresholds = {
        "height_m": float(maximum_height_delta_m),
        "footprint_m": float(maximum_footprint_delta_m),
        "vertical_occupied_ratio": float(maximum_occupancy_delta),
        "log10_local_point_count": float(maximum_log10_point_count_delta),
    }
    representative_features = [
        (str(item.get("id")), _features(item)) for item in representative_list
    ]
    coverage = []
    for candidate in candidate_list:
        candidate_id = str(candidate.get("id"))
        values = _features(candidate)
        comparisons = []
        for representative_id, reviewed in representative_features:
            deltas = {name: abs(values[name] - reviewed[name]) for name in thresholds}
            comparisons.append(
                {
                    "representative_id": representative_id,
                    "deltas": deltas,
                    "within_feature_envelope": all(
                        deltas[name] <= thresholds[name] for name in thresholds
                    ),
                }
            )
        covered = any(item["within_feature_envelope"] for item in comparisons)
        coverage.append(
            {
                "candidate_id": candidate_id,
                "features": values,
                "covered_by_reviewed_phenotype": covered,
                "comparisons": comparisons,
            }
        )
    uncovered = [
        item["candidate_id"]
        for item in coverage
        if not item["covered_by_reviewed_phenotype"]
    ]
    passed = not uncovered
    return {
        "schema_version": "railway.vertical-band-propagation-gate.v1",
        "representative_count": len(representative_list),
        "candidate_count": len(candidate_list),
        "thresholds": thresholds,
        "coverage": coverage,
        "uncovered_candidate_ids": uncovered,
        "passed": passed,
        "status": (
            "feature_envelope_covered_photo_consistency_still_required"
            if passed
            else "individual_review_required_for_uncovered_feature_clusters"
        ),
        "limitations": [
            "Feature similarity is not semantic confirmation.",
            "A band-level decision must not be propagated when any candidate phenotype is uncovered.",
            "Contradictory photo evidence always forces individual review even when this gate passes.",
        ],
    }
