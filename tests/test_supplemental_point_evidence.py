from __future__ import annotations

import numpy as np

from railway_recon.supplemental_point_evidence import (
    GridBounds,
    accumulate_density,
    render_density_comparison,
)


def test_accumulate_density_clips_and_counts() -> None:
    bounds = GridBounds(0.0, 0.0, 2.0, 2.0)
    grid = np.zeros((2, 2), dtype=np.uint64)
    retained = accumulate_density(
        grid,
        np.asarray([0.1, 0.9, 1.1, 2.0, -0.1]),
        np.asarray([0.1, 0.9, 1.1, 2.0, 0.5]),
        bounds,
        1.0,
    )
    assert retained == 4
    assert grid.tolist() == [[2, 0], [0, 2]]


def test_render_density_comparison(tmp_path) -> None:
    old = np.asarray([[0, 1], [2, 3]], dtype=np.uint64)
    new = np.asarray([[1, 1], [5, 3]], dtype=np.uint64)
    output = render_density_comparison(old, new, tmp_path / "comparison.png")
    assert output.is_file()
    assert output.stat().st_size > 0
