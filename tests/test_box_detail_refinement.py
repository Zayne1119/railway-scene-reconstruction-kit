from pathlib import Path

import numpy as np
import pytest

from railway_recon.algorithms.mesh import oriented_box
from railway_recon.box_detail_refinement import (
    BoxDetailPolicy,
    beveled_box,
    refine_box_objects_in_obj,
)
from railway_recon.mesh_audit import audit_obj


def source_box() -> np.ndarray:
    vertices, _ = oriented_box(
        np.asarray([4.0, 7.0, 0.0]),
        np.asarray([0.6, 0.8, 0.0]),
        0.8,
        0.6,
        1.0,
        4.0,
    )
    return vertices


def test_beveled_box_preserves_oriented_envelope_and_elevations() -> None:
    source = source_box()
    vertices, faces = beveled_box(source, BoxDetailPolicy(0.02, 0.01))
    assert vertices.shape == (24, 3)
    assert len(faces) == 18
    for axis in (source[1, :2] - source[0, :2], source[3, :2] - source[0, :2]):
        axis /= np.linalg.norm(axis)
        assert np.isclose((vertices[:, :2] @ axis).min(), (source[:, :2] @ axis).min())
        assert np.isclose((vertices[:, :2] @ axis).max(), (source[:, :2] @ axis).max())
    assert np.isclose(vertices[:, 2].min(), source[:, 2].min())
    assert np.isclose(vertices[:, 2].max(), source[:, 2].max())


def test_refine_box_object_preserves_other_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    box = source_box()
    lines = ["mtllib source.mtl", "o COLUMN", "usemtl Concrete"]
    lines.extend(f"v {x} {y} {z}" for x, y, z in box)
    lines.extend(
        [
            "f 1 4 3 2",
            "f 5 6 7 8",
            "f 1 2 6 5",
            "f 2 3 7 6",
            "f 3 4 8 7",
            "f 4 1 5 8",
            "o KEEP",
            "usemtl Other",
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 9 10 11",
        ]
    )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "candidate.obj"
    report = refine_box_objects_in_obj(
        source,
        output,
        {"COLUMN": BoxDetailPolicy(0.02, 0.01)},
        material_library="candidate.mtl",
    )
    assert report["passed"] is True
    assert report["target_count"] == 1
    assert audit_obj(output)["passed"] is True
    assert "o KEEP" in output.read_text(encoding="utf-8")


def test_refinement_rejects_non_box_target(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text("o X\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a single eight-vertex box"):
        refine_box_objects_in_obj(
            source,
            tmp_path / "candidate.obj",
            {"X": BoxDetailPolicy(0.02, 0.01)},
        )


def test_refinement_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    output = tmp_path / "existing.obj"
    source.write_text("o X\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refine_box_objects_in_obj(
            source,
            output,
            {"X": BoxDetailPolicy(0.02, 0.01)},
        )
