from __future__ import annotations

import numpy as np

from railway_recon.final_scene_qa import bbox_gap


def test_bbox_gap_treats_long_face_contact_as_connected() -> None:
    mast = (np.asarray([0.0, 0.0, 0.0]), np.asarray([0.3, 0.3, 8.0]))
    arm = (np.asarray([0.2, 0.1, 5.9]), np.asarray([3.0, 0.2, 6.1]))
    assert bbox_gap(mast, arm) == 0.0


def test_bbox_gap_reports_detached_component() -> None:
    first = (np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 1.0, 1.0]))
    second = (np.asarray([1.0, 1.3, 1.4]), np.asarray([2.0, 2.0, 2.0]))
    assert np.isclose(bbox_gap(first, second), 0.5)
