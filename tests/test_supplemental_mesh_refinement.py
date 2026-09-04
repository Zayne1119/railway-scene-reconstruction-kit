from __future__ import annotations

import numpy as np

from railway_recon.model_point_support import ObjModel
from railway_recon.supplemental_mesh_refinement import apply_refinement_vertices


def test_platform_moves_and_column_bottom_extends() -> None:
    model = ObjModel(
        vertices=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [2.0, 0.5, 1.0],
                [2.2, 0.5, 1.0],
                [2.0, 0.5, 3.0],
            ]
        ),
        faces_by_object={
            "S2050-OPPOSITE-PLATFORM-TOP": [(0, 1, 2)],
            "S2050-OPPOSITE-CANOPY-COLUMN-01": [(3, 4, 5)],
        },
    )
    refined, report = apply_refinement_vertices(
        model,
        np.asarray([0.0, 1.0]),
        np.asarray([-0.1, -0.2]),
    )
    assert np.allclose(refined[:3, 2], [0.9, 0.9, 0.8])
    assert np.allclose(refined[3:5, 2], [0.85, 0.85])
    assert refined[5, 2] == 3.0
    assert report["changed_vertex_count"] == 5
