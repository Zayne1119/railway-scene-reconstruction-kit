from __future__ import annotations

import pytest

from railway_recon.canopy_column_lateral_refit import infer_two_face_column_center


def test_two_face_column_center_uses_separated_face_clusters() -> None:
    result = infer_two_face_column_center([-25.72, -25.68, -25.01])
    assert result["target_cross_center_m"] == pytest.approx(-25.355)
    assert result["observed_cross_extent_m"] == pytest.approx(0.69)


def test_two_face_column_center_rejects_single_surface() -> None:
    with pytest.raises(ValueError, match="do not resolve two column faces"):
        infer_two_face_column_center([5.02, 5.08, 5.11])
