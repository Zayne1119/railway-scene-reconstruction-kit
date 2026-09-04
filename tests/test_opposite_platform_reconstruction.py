from __future__ import annotations

import copy

import numpy as np
import pytest

from railway_recon.geometry import CorridorFrame
from railway_recon.opposite_platform_reconstruction import (
    local_owned_interval_from_plan,
    prepare_opposite_platform_component,
)


def _candidate(
    identifier: str,
    *,
    side: str = "left",
    outer: float = -5.0,
    rail: float = -2.0,
    elevation: float = 1.0,
) -> dict:
    cross_range = [outer, rail] if side == "left" else [rail, outer]
    component = {
        "id": f"{identifier}-COMPONENT",
        "side": side,
        "occupied_cell_count": 500,
        "longitudinal_range_m": [-10.0, 10.0],
        "cross_range_m": cross_range,
        "z_range_m": [elevation - 0.03, elevation + 0.03],
        "median_height_above_rail_m": elevation,
        "fit_segments": [
            {
                "id": f"{identifier}-FIT-1",
                "longitudinal_range_m": [-10.0, 0.0],
                "observed_cross_range_m": list(cross_range),
                "rail_side_edge_cross_m": rail,
                "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, elevation],
            },
            {
                "id": f"{identifier}-FIT-2",
                "longitudinal_range_m": [0.0, 10.0],
                "observed_cross_range_m": list(cross_range),
                "rail_side_edge_cross_m": rail,
                "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, elevation],
            },
        ],
    }
    return {
        "configuration_id": identifier,
        "settings": {"surface_grid_m": 0.5},
        "report": {"rail_reference_z_m": 0.0, "platform_components": [component]},
    }


def test_left_platform_uses_common_conservative_outer_extent() -> None:
    candidates = [
        _candidate("A", outer=-7.0, rail=-2.0),
        _candidate("B", outer=-4.0, rail=-2.1, elevation=1.02),
        _candidate("C", outer=-5.5, rail=-1.9, elevation=0.98),
    ]
    component, gate = prepare_opposite_platform_component(
        candidates,
        side="left",
        owned_interval_m=(-10.0, 10.0),
        selected_configuration_id="B",
        component_id="LEFT-PLATFORM",
    )

    assert gate["passed"] is True
    assert gate["generic_consensus_passed"] is False
    assert gate["conservative_outer_edge_cross_m"] == -4.0
    assert component["cross_range_m"] == [-4.0, -2.0]
    assert all(fit["observed_cross_range_m"][0] >= -4.0 for fit in component["fit_segments"])


def test_right_platform_mirrors_conservative_intersection_policy() -> None:
    candidates = [
        _candidate("A", side="right", outer=7.0, rail=2.0),
        _candidate("B", side="right", outer=4.0, rail=2.1),
        _candidate("C", side="right", outer=5.5, rail=1.9),
    ]
    component, gate = prepare_opposite_platform_component(
        candidates,
        side="right",
        owned_interval_m=(-10.0, 10.0),
        selected_configuration_id="B",
        component_id="RIGHT-PLATFORM",
    )

    assert gate["conservative_outer_edge_cross_m"] == 4.0
    assert component["cross_range_m"] == [2.0, 4.0]
    assert all(fit["observed_cross_range_m"][1] <= 4.0 for fit in component["fit_segments"])


def test_platform_fails_closed_on_unsupported_longitudinal_gap() -> None:
    candidates = [_candidate(identifier) for identifier in ("A", "B", "C")]
    for candidate in candidates:
        component = candidate["report"]["platform_components"][0]
        component["fit_segments"][1]["longitudinal_range_m"] = [2.0, 10.0]
    with pytest.raises(ValueError, match="unsupported longitudinal gap"):
        prepare_opposite_platform_component(
            candidates,
            side="left",
            owned_interval_m=(-10.0, 10.0),
            selected_configuration_id="B",
            component_id="LEFT-PLATFORM",
            maximum_fit_gap_m=0.25,
        )


def test_selected_configuration_must_be_accepted() -> None:
    candidates = [_candidate(identifier) for identifier in ("A", "B", "C")]
    rejected = copy.deepcopy(_candidate("D"))
    rejected["report"]["platform_components"][0]["z_range_m"] = [0.0, 1.0]
    candidates.append(rejected)
    with pytest.raises(ValueError, match="did not pass"):
        prepare_opposite_platform_component(
            candidates,
            side="left",
            owned_interval_m=(-10.0, 10.0),
            selected_configuration_id="D",
            component_id="LEFT-PLATFORM",
        )


def test_shared_ownership_interval_is_converted_to_segment_local_station() -> None:
    plan = {
        "frame": {
            "origin_xy": [100.0, 200.0],
            "along_xy": [1.0, 0.0],
            "lateral_xy": [0.0, 1.0],
        },
        "segments": [{"segment_id": "B", "owned_interval_m": [50.0, 100.0]}],
    }
    local_frame = CorridorFrame(
        np.asarray([175.0, 200.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
    )

    interval = local_owned_interval_from_plan(plan, "B", local_frame)

    assert interval == pytest.approx((-25.0, 25.0))
