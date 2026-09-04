from pathlib import Path

from railway_recon.config import ProjectConfig
from railway_recon.io import write_json
from railway_recon.mesh_assembly import assemble_candidate_meshes


def _mesh(root: Path, name: str, offset: float) -> Path:
    directory = root / name
    directory.mkdir()
    obj = directory / f"{name}.obj"
    (directory / f"{name}.mtl").write_text(
        "newmtl Material\nKd 0.5 0.5 0.5\n", encoding="utf-8"
    )
    obj.write_text(
        f"mtllib {name}.mtl\no TRIANGLE\nusemtl Material\n"
        f"v {offset} 0 0\nv {offset + 1} 0 0\nv {offset} 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    write_json(directory / "model_origin.json", {"origin_xyz": [0.0, 0.0, 0.0]})
    return obj


def test_assemble_candidate_meshes_namespaces_objects(tmp_path: Path) -> None:
    project = ProjectConfig(tmp_path / "project.json", {"project": {"id": "test"}})
    first = _mesh(tmp_path, "first", 0.0)
    second = _mesh(tmp_path, "second", 2.0)

    result = assemble_candidate_meshes(
        project,
        tmp_path / "output",
        "assembly",
        [("FIRST", first), ("SECOND", second)],
    )

    assert result["passed"] is True
    content = Path(result["output_obj"]).read_text(encoding="utf-8")
    assert "o FIRST--TRIANGLE" in content
    assert "o SECOND--TRIANGLE" in content
