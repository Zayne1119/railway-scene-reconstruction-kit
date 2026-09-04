from __future__ import annotations

import pytest

from railway_recon.gap_candidate_disposition import compile_gap_candidate_dispositions


def _gap_report() -> dict:
    return {
        "vertical_candidates": [
            {
                "candidate_id": "GAP-VERTICAL-0001",
                "station_m": 12.0,
                "cross_m": -4.0,
                "minimum_xyz_m": [1.0, 2.0, 3.0],
                "maximum_xyz_m": [1.2, 2.2, 9.0],
                "sample_point_count": 120,
                "corridor_scope_ownership": "within_current_segment",
            }
        ]
    }


def test_compile_disposition_keeps_semantics_explicit() -> None:
    result = compile_gap_candidate_dispositions(
        _gap_report(),
        {
            "decisions": [
                {
                    "candidate_id": "GAP-VERTICAL-0001",
                    "reviewed_class": "catenary_mast",
                    "model_action": "build_candidate",
                    "confidence": 0.9,
                    "evidence": ["photo.jpg", "gap.json#GAP-VERTICAL-0001"],
                }
            ]
        },
    )
    assert result["build_candidate_ids"] == ["GAP-VERTICAL-0001"]
    assert result["decisions"][0]["candidate_geometry"]["station_m"] == 12.0


def test_compile_disposition_rejects_implicit_or_duplicate_decisions() -> None:
    decision = {
        "candidate_id": "GAP-VERTICAL-0001",
        "reviewed_class": "catenary_mast",
        "model_action": "build_candidate",
        "confidence": 0.9,
        "evidence": ["photo.jpg"],
    }
    with pytest.raises(ValueError, match="duplicate"):
        compile_gap_candidate_dispositions(_gap_report(), {"decisions": [decision, decision]})
