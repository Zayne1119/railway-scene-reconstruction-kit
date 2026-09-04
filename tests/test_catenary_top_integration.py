from __future__ import annotations

import numpy as np

from railway_recon.catenary_top_integration import (
    combine_meshes,
    ribbed_insulator,
    tube_between,
)


def test_tube_between_is_closed_and_non_degenerate() -> None:
    vertices, faces = tube_between(
        np.asarray([0.0, 0.0, 0.0]), np.asarray([2.0, 1.0, 0.5]), 0.05
    )
    assert vertices.shape == (16, 3)
    assert len(faces) == 10
    assert all(len(set(face)) >= 3 for face in faces)


def test_ribbed_insulator_and_mesh_combination() -> None:
    first = ribbed_insulator(
        np.asarray([0.0, 0.0, 0.0]), np.asarray([0.4, 0.0, 0.0])
    )
    second = tube_between(
        np.asarray([1.0, 0.0, 0.0]), np.asarray([2.0, 0.0, 0.0]), 0.04
    )
    vertices, faces = combine_meshes([first, second])
    assert len(vertices) == len(first[0]) + len(second[0])
    assert len(faces) == len(first[1]) + len(second[1])
    assert max(max(face) for face in faces) < len(vertices)
