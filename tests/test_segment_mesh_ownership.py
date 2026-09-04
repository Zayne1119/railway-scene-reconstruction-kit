import numpy as np

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.segment_mesh_ownership import (
    _append_polygon,
    build_segment_ownership_plan,
    clip_polygon_to_longitudinal_interval,
)


def test_polygon_is_clipped_to_both_ownership_boundaries() -> None:
    polygon = np.asarray(
        [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
            [-2.0, 1.0, 0.0],
        ]
    )
    clipped = clip_polygon_to_longitudinal_interval(
        polygon,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        (-1.0, 1.0),
    )
    assert len(clipped) == 4
    assert np.isclose(clipped[:, 0].min(), -1.0)
    assert np.isclose(clipped[:, 0].max(), 1.0)
    assert np.isclose(clipped[:, 1].min(), -1.0)
    assert np.isclose(clipped[:, 1].max(), 1.0)


def test_polygon_outside_ownership_interval_is_removed() -> None:
    polygon = np.asarray(
        [
            [3.0, -1.0, 0.0],
            [4.0, -1.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    clipped = clip_polygon_to_longitudinal_interval(
        polygon,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        (-1.0, 1.0),
    )
    assert clipped.shape == (0, 3)


def test_append_polygon_removes_collinear_boundary_vertices() -> None:
    vertices: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    _append_polygon(
        vertices,
        faces,
        {},
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
    assert len(vertices) == 4
    assert faces == [(0, 1, 2, 3)]


def test_segment_ownership_plan_uses_shared_midpoint_boundaries(tmp_path) -> None:
    project = load_project(initialize_project(tmp_path / "project", "sample", "Sample"))
    frames = {}
    for index, center in enumerate((0.0, 52.0, 102.0)):
        segment_id = f"s{index}"
        path = project.root / f"{segment_id}.json"
        write_json(
            path,
            {"frame": {"origin_xy": [center, 1.0], "along_xy": [1.0, 0.0]}},
        )
        frames[segment_id] = path

    result = build_segment_ownership_plan(project, frames, "ownership.json")

    assert result["seam_stations_m"] == [26.0, 77.0]
    assert result["segments"][0]["owned_interval_m"] == [-26.0, 26.0]
    assert result["segments"][1]["owned_interval_m"] == [26.0, 77.0]
    assert result["segments"][2]["owned_interval_m"] == [77.0, 127.0]
    assert load_json(project.root / "ownership.json")["passed"] is True
