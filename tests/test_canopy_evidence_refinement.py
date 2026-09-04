from __future__ import annotations

import numpy as np

from railway_recon.canopy_evidence_refinement import apply_opposite_canopy_correction


def test_apply_opposite_canopy_correction_reconnects_top() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
            [0.0, 0.5, 1.0],
            [0.2, 0.5, 1.0],
            [0.0, 0.5, 2.8],
            [0.0, 0.5, 2.9],
            [0.2, 0.5, 2.9],
        ]
    )
    faces = {
        "S2050-OPPOSITE-CANOPY-ROOF-RUN-01": [(0, 1, 0)],
        "S2050-OPPOSITE-CANOPY-COLUMN-01": [(2, 3, 4)],
        "S2050-OPPOSITE-CANOPY-COLUMN-01-CAPITAL": [(5, 6, 5)],
    }
    refined, report = apply_opposite_canopy_correction(
        vertices,
        faces,
        np.asarray([0.0, 1.0]),
        np.asarray([-0.1, -0.2]),
    )
    assert np.allclose(refined[:2, 2], [2.9, 2.8])
    assert refined[2, 2] == 1.0
    assert refined[4, 2] == 2.65
    assert np.allclose(refined[5:, 2], [2.75, 2.75])
    assert report["changed_object_count"] == 3
