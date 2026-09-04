from __future__ import annotations

from pathlib import Path

from railway_recon.web_glb_export import export_obj_to_web_glb, inspect_web_glb


def test_named_obj_exports_to_structurally_valid_glb(tmp_path: Path) -> None:
    material = tmp_path / "sample.mtl"
    material.write_text(
        "newmtl rail\nKd 0.2 0.3 0.4\n"
        "newmtl glass\nKd 0.1 0.5 0.7\nd 0.5\n",
        encoding="utf-8",
    )
    source = tmp_path / "sample.obj"
    source.write_text(
        "mtllib sample.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        "o TRACK-01\n"
        "usemtl rail\n"
        "f 1//1 2//1 3//1 4//1\n"
        "o STATION-01\n"
        "usemtl glass\n"
        "f -4//1 -2//1 -1//1\n",
        encoding="utf-8",
    )
    output = tmp_path / "sample.glb"
    report = export_obj_to_web_glb(source, output)
    inspection = inspect_web_glb(output)
    assert report["passed"] is True
    assert inspection["node_count"] == 2
    assert inspection["material_count"] == 2
    assert inspection["triangle_count"] == 3
    assert inspection["node_names"] == ["TRACK-01", "STATION-01"]
