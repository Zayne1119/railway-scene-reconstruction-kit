from __future__ import annotations

import numpy as np

from railway_recon.supplemental_vertical_comparison import (
    compare_vertical_candidates_to_assets,
    deduplicate_vertical_candidates,
)


def test_deduplicate_overlapping_segment_candidates() -> None:
    candidates = [
        {
            "id": "A",
            "segment_id": "s1",
            "center_x": 0.0,
            "center_y": 0.0,
            "minimum_z": 1.0,
            "maximum_z": 6.0,
            "point_count": 100,
        },
        {
            "id": "B",
            "segment_id": "s2",
            "center_x": 0.1,
            "center_y": 0.1,
            "minimum_z": 1.1,
            "maximum_z": 6.1,
            "point_count": 50,
        },
    ]
    result = deduplicate_vertical_candidates(candidates)
    assert len(result) == 1
    assert result[0]["source_occurrence_count"] == 2


def test_match_candidate_to_asset_box() -> None:
    candidates = [
        {
            "id": "C",
            "center_x": 0.1,
            "center_y": 0.1,
            "minimum_z": 1.0,
            "maximum_z": 6.0,
        }
    ]
    assets = [
        {
            "id": "COLUMN-1",
            "type": "canopy_column",
            "minimum_xyz": np.asarray([0.0, 0.0, 0.8]),
            "maximum_xyz": np.asarray([0.3, 0.3, 6.2]),
        }
    ]
    result = compare_vertical_candidates_to_assets(candidates, assets)
    assert result[0]["decision"] == "explained_by_existing_vertical_asset"
