from __future__ import annotations

import numpy as np

from railway_recon.small_asset_rescan import classify_small_asset_component


def test_small_asset_shape_classes() -> None:
    assert (
        classify_small_asset_component(
            np.asarray([4.0, 0.4, 1.2]),
            bottom_above_platform_m=0.1,
            top_below_canopy_m=4.0,
        )
        == "railing_or_fence_gap_candidate"
    )
    assert (
        classify_small_asset_component(
            np.asarray([1.6, 0.3, 1.8]),
            bottom_above_platform_m=0.2,
            top_below_canopy_m=3.0,
        )
        == "freestanding_or_hanging_sign_candidate"
    )
    assert (
        classify_small_asset_component(
            np.asarray([0.8, 1.5, 1.0]),
            bottom_above_platform_m=0.1,
            top_below_canopy_m=4.0,
        )
        == "equipment_box_or_cabinet_candidate"
    )
