from pathlib import Path

from railway_recon.mesh_audit import audit_obj
from railway_recon.obj_subset import subset_obj_objects


def test_subset_obj_compacts_geometry_and_preserves_object_name(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    material = tmp_path / "source.mtl"
    source.write_text(
        """mtllib source.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vn 0 0 1
o KEEP
usemtl white
f 1//1 2//1 3//1
o DROP
usemtl white
f 1//1 3//1 4//1
""",
        encoding="utf-8",
    )
    material.write_text("newmtl white\nKd 1 1 1\n", encoding="utf-8")

    output = tmp_path / "subset.obj"
    report = subset_obj_objects(source, output, ["KEEP"])

    text = output.read_text(encoding="utf-8")
    assert "o KEEP" in text
    assert "o DROP" not in text
    assert report["object_count"] == 1
    assert report["vertex_count"] == 3
    assert report["normal_count"] == 1
    assert audit_obj(output)["passed"] is True


def test_subset_obj_supports_negative_face_indexes(tmp_path: Path) -> None:
    source = tmp_path / "negative.obj"
    (tmp_path / "negative.mtl").write_text("newmtl white\n", encoding="utf-8")
    source.write_text(
        """mtllib negative.mtl
v 0 0 0
v 1 0 0
v 0 1 0
o KEEP
f -3 -2 -1
""",
        encoding="utf-8",
    )

    report = subset_obj_objects(source, tmp_path / "subset.obj", ["KEEP"])

    assert report["vertex_count"] == 3
    assert report["face_count"] == 1
