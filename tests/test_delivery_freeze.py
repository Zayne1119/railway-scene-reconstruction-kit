from __future__ import annotations

from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.delivery_freeze import _copy_obj_with_mtl, freeze_delivery
from railway_recon.io import load_json, sha256_file, write_json
from railway_recon.registry import new_registry


def test_copy_obj_rewrites_material_library(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    destination = tmp_path / "final.obj"
    source.write_text("mtllib old.mtl\no A\nv 0 0 0\n", encoding="utf-8")
    _copy_obj_with_mtl(source, destination, "final.mtl")
    assert destination.read_text(encoding="utf-8").splitlines()[0] == "mtllib final.mtl"


def test_freeze_is_explicitly_candidate_only(tmp_path: Path) -> None:
    config_path = initialize_project(tmp_path / "project", "site-b", "Site B")
    project = load_project(config_path)
    source_obj = project.root / "candidate.obj"
    source_mtl = project.root / "candidate.mtl"
    source_glb = project.root / "candidate.glb"
    origin = project.root / "origin.json"
    registry_path = project.root / "registry.json"
    qa_path = project.root / "qa.json"
    conversion_path = project.root / "conversion.json"
    source_obj.write_text("mtllib candidate.mtl\no A\nv 0 0 0\n", encoding="utf-8")
    source_mtl.write_text("newmtl A\n", encoding="utf-8")
    source_glb.write_bytes(b"glTF-candidate")
    write_json(origin, {"origin": [0, 0, 0]})
    registry = new_registry(project.project_id)
    registry["assets"] = [
        {
            "id": "TRACK-001",
            "type": "track",
            "status": "accepted",
            "evidence_level": "observed",
            "confidence": 0.95,
            "sources": [{"kind": "point_cloud", "reference": "source.laz"}],
            "geometry": {"file": str(source_obj), "node": "A"},
        }
    ]
    write_json(registry_path, registry)
    write_json(
        qa_path,
        {
            "candidate_obj_sha256": sha256_file(source_obj),
            "summary": {
                "passed": True,
                "check_count": 1,
                "failed_check_count": 0,
                "supported_catenary_assembly_count": 0,
                "added_catenary_component_count": 0,
                "withheld_low_confidence_mast_count": 0,
                "new_unresolved_small_asset_count": 0,
            },
        },
    )
    write_json(
        conversion_path,
        {"passed": True, "source_sha256": sha256_file(source_obj)},
    )

    result = freeze_delivery(
        source_obj=source_obj,
        source_mtl=source_mtl,
        source_glb=source_glb,
        source_registry=registry_path,
        source_origin=origin,
        final_qa_report=qa_path,
        glb_conversion_report=conversion_path,
        output_directory=project.root / "candidate-snapshot",
    )

    assert result["obj"].name == "site_b_200m_candidate.obj"
    assert result["glb"].name == "site_b_200m_candidate.glb"
    assert "final" not in result["registry"].name
    manifest = load_json(result["manifest"])
    assert manifest["formal_release"] is False
    assert manifest["status"] == "frozen_candidate_not_formal_release"
    frozen_registry = load_json(result["registry"])
    assert "release_id" not in frozen_registry
    assert frozen_registry["candidate_id"] == "candidate-snapshot"
