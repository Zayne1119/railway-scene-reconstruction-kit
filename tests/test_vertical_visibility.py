from __future__ import annotations

import numpy as np

from railway_recon.vertical_visibility import audit_vertical_object_visibility


def test_vertical_visibility_detects_supported_shaft() -> None:
    vertices = np.asarray(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, 0.1, 2.0],
            [0.1, 0.1, 2.0],
        ]
    )
    z = np.arange(0.0, 2.01, 0.05)
    cloud = np.column_stack((np.full(len(z), -0.1), np.full(len(z), -0.1), z))
    result = audit_vertical_object_visibility(vertices, cloud, level_step_m=0.2)
    assert result["decision"] == "visible_supported"
    assert result["supported_fraction"] == 1.0


def test_vertical_visibility_does_not_delete_partial_shaft() -> None:
    vertices = np.asarray(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, 0.1, 2.0],
            [0.1, 0.1, 2.0],
        ]
    )
    z = np.arange(1.5, 2.01, 0.05)
    cloud = np.column_stack((np.full(len(z), -0.1), np.full(len(z), -0.1), z))
    result = audit_vertical_object_visibility(vertices, cloud, level_step_m=0.2)
    assert "do_not" in result["decision"]
