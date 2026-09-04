from pathlib import Path

import numpy as np
import pytest

from railway_recon.algorithms.mesh import oriented_box
from railway_recon.mesh_audit import audit_obj
from railway_recon.track_detail_refinement import (
    SleeperDetailPolicy,
    beveled_sleeper_box,
    refine_sleeper_objects_in_obj,
)


def test_beveled_sleeper_stays_inside_box_and_keeps_envelope() -> None:
    source, _ = oriented_box(
        np.asarray([4.0, 7.0, 0.0]),
        np.asarray([0.6, 0.8, 0.0]),
        2.60,
        0.26,
        1.0,
        1.22,
    )
    vertices, faces = beveled_sleeper_box(source)
    assert vertices.shape == (24, 3)
    assert len(faces) == 18
    for axis in (source[1, :2] - source[0, :2], source[3, :2] - source[0, :2]):
        axis /= np.linalg.norm(axis)
        assert np.isclose((vertices[:, :2] @ axis).min(), (source[:, :2] @ axis).min())
        assert np.isclose((vertices[:, :2] @ axis).max(), (source[:, :2] @ axis).max())
    assert np.isclose(vertices[:, 2].min(), source[:, 2].min())
    assert np.isclose(vertices[:, 2].max(), source[:, 2].max())


def test_refine_named_obj_is_closed_and_preserves_other_object(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text(
        """mtllib source.mtl
o TRACK-01-SLEEPERS
usemtl Concrete
v -1.3 -0.13 0
v 1.3 -0.13 0
v 1.3 0.13 0
v -1.3 0.13 0
v -1.3 -0.13 0.22
v 1.3 -0.13 0.22
v 1.3 0.13 0.22
v -1.3 0.13 0.22
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
o CONTEXT
usemtl Context
v 4 4 0
v 5 4 0
v 4 5 0
f 9 10 11
""",
        encoding="utf-8",
    )
    output = tmp_path / "candidate.obj"
    report = refine_sleeper_objects_in_obj(
        source,
        output,
        ["TRACK-01-SLEEPERS"],
        material_library="candidate.mtl",
    )
    assert report["passed"] is True
    assert report["refined_sleeper_count"] == 1
    audit = audit_obj(output)
    assert audit["passed"] is True
    assert audit["object_count"] == 2
    assert audit["degenerate_triangle_count"] == 0
    assert "o CONTEXT" in output.read_text(encoding="utf-8")


def test_refinement_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    output = tmp_path / "existing.obj"
    source.write_text("o X\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refine_sleeper_objects_in_obj(source, output, ["X"])


def test_policy_rejects_oversized_bevel() -> None:
    source, _ = oriented_box(
        np.zeros(3), np.asarray([1.0, 0.0, 0.0]), 2.6, 0.2, 0.0, 0.02
    )
    with pytest.raises(ValueError, match="top-edge bevel"):
        beveled_sleeper_box(
            source,
            SleeperDetailPolicy(corner_chamfer_m=0.04, top_edge_bevel_m=0.015),
        )
