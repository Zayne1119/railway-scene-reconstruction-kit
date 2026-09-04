from __future__ import annotations

import numpy as np

from railway_recon.geometry import CorridorFrame
from railway_recon.observed_roof_surface import (
    fit_observed_roof_surface,
    roof_profile_slab_mesh,
    roof_slab_mesh,
)


def _synthetic_roof(*, gap: tuple[float, float] | None = None) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(17)
    s_values = np.arange(0.0, 10.0, 0.1)
    c_values = np.arange(-4.0, -0.99, 0.2)
    s, c = np.meshgrid(s_values, c_values, indexing="xy")
    s = s.ravel()
    c = c.ravel()
    keep = np.ones(s.size, dtype=bool)
    if gap is not None:
        keep &= ~((s >= gap[0]) & (s < gap[1]))
    s = s[keep]
    c = c[keep]
    z = 0.01 * s + 0.2 * c + 5.0 + rng.normal(0.0, 0.006, s.size)
    return s, c, z


def test_fit_observed_roof_surface_accepts_supported_profiles() -> None:
    s, c, z = _synthetic_roof()
    report = fit_observed_roof_surface(
        s,
        c,
        z,
        longitudinal_range_m=[0.0, 10.0],
        cross_range_m=[-4.0, -1.0],
        initial_cross_line=[0.2, 5.05],
        minimum_profile_points=80,
    )
    assert report["passed"] is True
    assert report["supported_runs_m"] == [[0.0, 10.0]]
    assert report["passed_profile_ratio"] == 1.0
    np.testing.assert_allclose(
        report["plane_z_equals_a_s_plus_b_c_plus_d"], [0.01, 0.2, 5.0], atol=0.003
    )


def test_fit_observed_roof_surface_preserves_real_gap() -> None:
    s, c, z = _synthetic_roof(gap=(4.0, 6.0))
    report = fit_observed_roof_surface(
        s,
        c,
        z,
        longitudinal_range_m=[0.0, 10.0],
        cross_range_m=[-4.0, -1.0],
        initial_cross_line=[0.2, 5.05],
        minimum_profile_points=80,
        minimum_passed_profile_ratio=0.75,
    )
    assert report["passed"] is True
    assert report["supported_runs_m"] == [[0.0, 4.0], [6.0, 10.0]]
    assert report["unsupported_gaps_filled"] is False


def test_roof_slab_mesh_is_closed_and_uses_world_frame() -> None:
    frame = CorridorFrame(
        origin_xy=np.asarray([100.0, 200.0]),
        along_xy=np.asarray([1.0, 0.0]),
        cross_xy=np.asarray([0.0, 1.0]),
    )
    vertices, faces = roof_slab_mesh(
        frame,
        [0.01, 0.2, 5.0],
        [0.0, 10.0],
        [-4.0, -1.0],
        thickness_m=0.12,
    )
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    np.testing.assert_allclose(vertices[4, :2], [100.0, 196.0])
    np.testing.assert_allclose(vertices[4, 2], 4.2)
    np.testing.assert_allclose(vertices[4:, 2] - vertices[:4, 2], 0.12)


def test_roof_profile_slab_mesh_preserves_variable_exact_boundaries() -> None:
    frame = CorridorFrame(
        origin_xy=np.asarray([100.0, 200.0]),
        along_xy=np.asarray([1.0, 0.0]),
        cross_xy=np.asarray([0.0, 1.0]),
    )
    profiles = np.asarray(
        [
            [0.0, -4.0, 4.20, -1.0, 4.80],
            [2.0, -3.8, 4.25, -0.9, 4.82],
            [4.0, -3.7, 4.30, -0.8, 4.85],
        ]
    )
    vertices, faces = roof_profile_slab_mesh(frame, profiles, thickness_m=0.12)
    assert vertices.shape == (12, 3)
    assert len(faces) == 10
    np.testing.assert_allclose(vertices[0], [100.0, 196.0, 4.20])
    np.testing.assert_allclose(vertices[5], [104.0, 199.2, 4.85])
    np.testing.assert_allclose(vertices[6:, 2], vertices[:6, 2] - 0.12)


def test_roof_profile_slab_mesh_can_leave_owned_assembly_seam_uncapped() -> None:
    frame = CorridorFrame(
        origin_xy=np.asarray([0.0, 0.0]),
        along_xy=np.asarray([1.0, 0.0]),
        cross_xy=np.asarray([0.0, 1.0]),
    )
    profiles = [
        [0.0, -2.0, 4.0, -1.0, 4.1],
        [1.0, -2.0, 4.0, -1.0, 4.1],
    ]
    _, no_start = roof_profile_slab_mesh(
        frame, profiles, thickness_m=0.12, cap_start=False
    )
    _, no_end = roof_profile_slab_mesh(
        frame, profiles, thickness_m=0.12, cap_end=False
    )
    _, no_caps = roof_profile_slab_mesh(
        frame, profiles, thickness_m=0.12, cap_start=False, cap_end=False
    )
    assert len(no_start) == 5
    assert len(no_end) == 5
    assert len(no_caps) == 4
    assert (0, 1, 5, 4) not in no_start
    assert (2, 6, 7, 3) not in no_end


def test_roof_profile_slab_mesh_rejects_reversed_or_collapsed_profiles() -> None:
    frame = CorridorFrame(
        origin_xy=np.asarray([0.0, 0.0]),
        along_xy=np.asarray([1.0, 0.0]),
        cross_xy=np.asarray([0.0, 1.0]),
    )
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        roof_profile_slab_mesh(
            frame,
            [[2.0, -2.0, 4.0, -1.0, 4.1], [1.0, -2.0, 4.0, -1.0, 4.1]],
            thickness_m=0.12,
        )
    with np.testing.assert_raises_regex(ValueError, "increasing cross"):
        roof_profile_slab_mesh(
            frame,
            [[0.0, -1.0, 4.0, -1.0, 4.1], [1.0, -1.0, 4.0, -1.0, 4.1]],
            thickness_m=0.12,
        )
