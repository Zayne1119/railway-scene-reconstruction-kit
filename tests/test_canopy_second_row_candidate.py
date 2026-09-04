from __future__ import annotations

from pathlib import Path

from railway_recon.canopy_second_row_candidate import clone_obj_objects
from railway_recon.model_point_support import object_vertex_indices, parse_obj_model


def test_clone_obj_object_with_translation(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text(
        "mtllib source.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "o COLUMN\n"
        "usemtl concrete\n"
        "f 1//1 2//1 3//1\n"
        "o OLD\n"
        "usemtl concrete\n"
        "f 1//1 3//1 2//1\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.obj"
    clone_obj_objects(
        source,
        output,
        [
            {
                "source_object": "COLUMN",
                "clone_object": "COLUMN-ROW-02",
                "translation_local_xyz_m": [2.0, 0.0, 0.0],
            }
        ],
        output_mtl_name="output.mtl",
        omit_objects={"OLD"},
    )
    model = parse_obj_model(output)
    assert "COLUMN-ROW-02" in model.faces_by_object
    assert "COLUMN" in model.faces_by_object
    assert "OLD" not in model.faces_by_object
    assert len(model.vertices) == 6
    assert model.vertices[3, 0] == 2.0


def test_clone_replacement_preserves_interleaved_global_vertex_indexes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "interleaved.obj"
    source.write_text(
        "mtllib source.mtl\n"
        "o KEEP\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
        "o REPLACE\n"
        "v 10 0 0\n"
        "v 11 0 0\n"
        "v 10 1 0\n"
        "f 4 5 6\n"
        "o NEXT\n"
        "v 20 0 0\n"
        "v 21 0 0\n"
        "v 20 1 0\n"
        "f 7 8 9\n",
        encoding="utf-8",
    )
    output = tmp_path / "replaced.obj"
    clone_obj_objects(
        source,
        output,
        [
            {
                "source_object": "REPLACE",
                "clone_object": "REPLACE",
                "translation_local_xyz_m": [0.0, 2.0, 0.0],
            }
        ],
        output_mtl_name="output.mtl",
        omit_objects={"REPLACE"},
    )
    model = parse_obj_model(output)
    indexes = object_vertex_indices(model, "REPLACE")
    assert len(model.vertices) == 12
    assert indexes.tolist() == [9, 10, 11]
    assert model.vertices[indexes, 0].tolist() == [10.0, 11.0, 10.0]
    assert model.vertices[indexes, 1].tolist() == [2.0, 2.0, 3.0]
