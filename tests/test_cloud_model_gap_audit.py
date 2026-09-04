from __future__ import annotations

import numpy as np

from railway_recon.cloud_model_gap_audit import (
    assign_corridor_scope_ownership,
    detect_overhead_linear_gap_candidates,
    detect_vertical_gap_candidates,
    match_linear_candidates_to_assets,
    match_vertical_candidates_to_assets,
    reverse_model_distances,
    sample_model_surfaces,
)
from railway_recon.model_point_support import ObjModel


def test_surface_sampling_fills_large_triangle() -> None:
    model = ObjModel(
        vertices=np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
        faces_by_object={"PLANE": [(0, 1, 2)]},
    )
    points, report = sample_model_surfaces(
        model,
        origin_xyz=np.zeros(3),
        spacing_m=0.2,
        random_seed=1,
    )
    assert len(points) >= 100
    assert report["surface_area_m2"] == 2.0


def test_reverse_distance_finds_unexplained_vertical_asset() -> None:
    model = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    z = np.linspace(0.0, 5.0, 80)
    pole = np.column_stack((np.full_like(z, 3.0), np.full_like(z, 2.0), z))
    distances = reverse_model_distances(pole, model)
    candidates = detect_vertical_gap_candidates(
        pole,
        distances,
        unexplained_distance_m=0.25,
        cell_size_m=0.25,
        minimum_sample_points_per_cell=10,
    )
    assert len(candidates) == 1
    assert candidates[0]["classification"] == "high_confidence_pole_or_mast"
    assert candidates[0]["priority"] == "P0"


def test_scope_ownership_blocks_adjacent_segment_verticals() -> None:
    model = ObjModel(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 1.0, 0.0))
        ),
        faces_by_object={"TRACK-001": [(0, 1, 2)]},
    )
    verticals = [{"station_m": -2.0}, {"station_m": 5.0}, {"station_m": 12.0}]
    linears = [{"station_range_m": [-1.0, 8.0]}]
    report = assign_corridor_scope_ownership(
        verticals,
        linears,
        model=model,
        origin_xyz=np.zeros(3),
        frame={"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0]},
    )
    assert report["length_m"] == 10.0
    assert [item["corridor_scope_ownership"] for item in verticals] == [
        "adjacent_previous_segment",
        "within_current_segment",
        "adjacent_next_segment",
    ]
    assert linears[0]["corridor_scope_ownership"] == "crosses_segment_boundary"


def test_scope_ownership_accepts_namespaced_track_objects() -> None:
    model = ObjModel(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 1.0, 0.0))
        ),
        faces_by_object={"SCENE--BASELINE--TRACK-001-BED": [(0, 1, 2)]},
    )
    verticals = [{"station_m": 5.0}]
    report = assign_corridor_scope_ownership(
        verticals,
        [],
        model=model,
        origin_xyz=np.zeros(3),
        frame={"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0]},
    )
    assert report["length_m"] == 10.0
    assert verticals[0]["corridor_scope_ownership"] == "within_current_segment"


def test_candidate_matching_separates_existing_asset_from_new_pole() -> None:
    model = ObjModel(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.0, 0.2, 4.0),
            )
        ),
        faces_by_object={"MAST-1": [(0, 1, 2)]},
    )
    candidates = [
            {
                "minimum_xyz_m": [-0.1, -0.1, 0.0],
                "maximum_xyz_m": [0.3, 0.3, 4.0],
                "centroid_xyz_m": [0.1, 0.1, 2.0],
            },
            {
                "minimum_xyz_m": [5.0, 5.0, 0.0],
                "maximum_xyz_m": [5.2, 5.2, 4.0],
                "centroid_xyz_m": [5.1, 5.1, 2.0],
            },
    ]
    counts = match_vertical_candidates_to_assets(
        candidates,
        model=model,
        origin_xyz=np.zeros(3),
        registry={
            "assets": [
                {
                    "id": "MAST-1",
                    "type": "catenary_mast",
                    "geometry": {"node": "MAST-1"},
                }
            ]
        },
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert candidates[0]["asset_relation"] == "existing_asset_geometry_refinement"
    assert candidates[1]["asset_relation"] == "new_standalone_vertical_candidate"
    assert counts == {
        "existing_asset_geometry_refinement": 1,
        "new_standalone_vertical_candidate": 1,
    }


def test_station_aligned_discontinuous_catenary_trace_is_demoted() -> None:
    model = ObjModel(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.0, 0.2, 8.0),
            )
        ),
        faces_by_object={"MAST-1": [(0, 1, 2)]},
    )
    candidate = {
        "priority": "P1",
        "minimum_xyz_m": [-0.1, 1.75, 24.0],
        "maximum_xyz_m": [0.15, 2.0, 29.0],
        "centroid_xyz_m": [0.03, 1.88, 26.5],
        "extent_xyz_m": [0.25, 0.25, 5.0],
        "vertical_occupancy": 0.5,
    }
    match_vertical_candidates_to_assets(
        [candidate],
        model=model,
        origin_xyz=np.zeros(3),
        registry={
            "assets": [
                {
                    "id": "MAST-1",
                    "type": "catenary_mast",
                    "geometry": {"node": "MAST-1"},
                }
            ]
        },
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert candidate["asset_relation"] == "existing_asset_position_or_detail_refinement"
    assert candidate["priority_before_relation_review"] == "P1"
    assert candidate["priority"] == "P2"
    assert candidate["automatic_geometry_action"] == "withhold_pending_photo_review"


def test_station_entry_surface_edge_is_not_promoted_as_new_pole() -> None:
    model = ObjModel(
        vertices=np.asarray(
            (
                (10.0, 5.0, 0.0),
                (12.0, 5.0, 0.0),
                (10.0, 5.0, 4.0),
            )
        ),
        faces_by_object={"ENTRY-WALL": [(0, 1, 2)]},
    )
    candidate = {
        "priority": "P1",
        "minimum_xyz_m": [12.8, 5.3, 0.0],
        "maximum_xyz_m": [13.0, 5.5, 3.0],
        "centroid_xyz_m": [12.9, 5.4, 1.5],
        "extent_xyz_m": [0.2, 0.2, 3.0],
        "vertical_occupancy": 0.8,
    }
    match_vertical_candidates_to_assets(
        [candidate],
        model=model,
        origin_xyz=np.zeros(3),
        registry={
            "assets": [
                {
                    "id": "ENTRY-WALL",
                    "type": "station_entry_outer_wall",
                    "parameters": {
                        "station_range_m": [10.0, 12.0],
                        "cross_range_m": [5.0, 5.2],
                    },
                    "geometry": {"node": "ENTRY-WALL"},
                }
            ]
        },
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert candidate["asset_relation"] == (
        "possible_attachment_or_position_refinement"
    )
    assert candidate["priority"] == "P2"
    assert candidate["semantic_review_hint"] == (
        "existing_station_entry_surface_edge_or_residual"
    )


def test_p0_trace_directly_on_station_entry_wall_is_demoted() -> None:
    model = ObjModel(
        vertices=np.asarray(
            ((10.0, 5.0, 0.0), (12.0, 5.0, 0.0), (10.0, 5.0, 4.0))
        ),
        faces_by_object={"ENTRY-WALL": [(0, 1, 2)]},
    )
    candidate = {
        "priority": "P0",
        "minimum_xyz_m": [11.8, 5.1, 0.0],
        "maximum_xyz_m": [12.0, 5.3, 4.0],
        "centroid_xyz_m": [11.9, 5.2, 2.0],
        "extent_xyz_m": [0.2, 0.2, 4.0],
        "vertical_occupancy": 0.9,
    }
    match_vertical_candidates_to_assets(
        [candidate],
        model=model,
        origin_xyz=np.zeros(3),
        registry={
            "assets": [
                {
                    "id": "ENTRY-WALL",
                    "type": "station_entry_outer_wall",
                    "parameters": {
                        "station_range_m": [10.0, 12.0],
                        "cross_range_m": [5.0, 5.2],
                    },
                    "geometry": {"node": "ENTRY-WALL"},
                }
            ]
        },
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert candidate["priority_before_relation_review"] == "P0"
    assert candidate["priority"] == "P2"
    assert candidate["automatic_geometry_action"] == (
        "refine_existing_station_entry_group_not_new_standalone_asset"
    )


def test_overhead_linear_detector_recovers_long_thin_residual() -> None:
    station = np.linspace(0.0, 30.0, 400)
    points = np.column_stack(
        (
            station,
            np.full_like(station, 2.0),
            27.0 - 0.002 * (station - 15.0) ** 2,
        )
    )
    candidates = detect_overhead_linear_gap_candidates(
        points,
        np.full(len(points), 0.8),
        {
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert len(candidates) == 1
    assert candidates[0]["classification"] == "overhead_wire_or_linear_asset"
    assert candidates[0]["priority"] == "P0"


def test_canopy_roof_edge_trace_is_not_promoted_as_missing_wire() -> None:
    model = ObjModel(
        vertices=np.asarray(
            (
                (0.0, -4.0, 26.0),
                (20.0, -4.0, 26.0),
                (20.0, -1.0, 27.0),
                (0.0, -1.0, 27.0),
            )
        ),
        faces_by_object={"ROOF": [(0, 1, 2, 3)]},
    )
    candidate = {
        "classification": "overhead_wire_or_linear_asset",
        "priority": "P0",
        "station_range_m": [2.0, 18.0],
        "cross_range_m": [-1.18, -1.02],
        "z_range_m": [26.8, 27.4],
    }
    counts = match_linear_candidates_to_assets(
        [candidate],
        model=model,
        origin_xyz=np.zeros(3),
        registry={
            "assets": [
                {
                    "id": "ROOF",
                    "type": "canopy_roof_surface",
                    "geometry": {"node": "ROOF"},
                }
            ]
        },
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
    )
    assert candidate["priority"] == "P2"
    assert candidate["classification"] == "canopy_roof_edge_or_fascia_residual"
    assert candidate["automatic_geometry_action"] == (
        "refine_existing_canopy_group_not_new_wire"
    )
    assert counts == {"existing_canopy_roof_edge_or_fascia_refinement": 1}
