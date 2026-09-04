from __future__ import annotations

import numpy as np

from railway_recon.catenary_candidate_mesh import (
    audit_catenary_track_clearance,
    h_section_prism,
)


def test_h_section_prism_is_closed_and_uses_measured_axes() -> None:
    vertices, faces = h_section_prism(
        np.asarray([100.0, 200.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        0.25,
        0.28,
        0.028,
        0.020,
        19.8,
        30.4,
    )

    assert vertices.shape == (24, 3)
    assert len(faces) == 32
    assert np.isclose(vertices[:, 0].max() - vertices[:, 0].min(), 0.25)
    assert np.isclose(vertices[:, 1].max() - vertices[:, 1].min(), 0.28)
    assert np.isclose(vertices[:, 2].max() - vertices[:, 2].min(), 10.6)
    assert all(len(set(face)) == len(face) for face in faces)


def test_h_section_prism_supports_rotated_track_frame() -> None:
    along = np.asarray([0.6, 0.8])
    cross = np.asarray([-0.8, 0.6])
    vertices, _ = h_section_prism(
        np.asarray([0.0, 0.0]),
        along,
        cross,
        0.30,
        0.40,
        0.04,
        0.025,
        0.0,
        8.0,
    )
    local_along = vertices[:12, :2] @ along
    local_cross = vertices[:12, :2] @ cross
    assert np.isclose(local_along.max() - local_along.min(), 0.30)
    assert np.isclose(local_cross.max() - local_cross.min(), 0.40)


def test_catenary_foundation_clearance_rejects_track_bed_intersection() -> None:
    graph = {
        "observations": [
            {"global_track_id": "TRACK-0001", "lateral_offset_m": -4.0},
            {"global_track_id": "TRACK-0002", "lateral_offset_m": 0.0},
        ]
    }
    section = {
        "candidate_id": "VERTICAL-HYPOTHESIS-0001",
        "estimated_shaft": {"cross_position_m": -2.2},
        "height_bins": [{"cross_width_p90_m": 1.0}],
    }
    settings = {"track_bed": {"bottom_width_m": 3.0}}

    result = audit_catenary_track_clearance(graph, section, settings)

    assert result["passed"] is False
    assert "intersection" in {item["status"] for item in result["checks"]}
