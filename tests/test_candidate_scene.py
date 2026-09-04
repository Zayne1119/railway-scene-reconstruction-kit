from __future__ import annotations

import json
from pathlib import Path

from railway_recon.candidate_scene import (
    audit_track_platform_candidate,
    compose_candidate_scene,
)
from railway_recon.config import initialize_project, load_project
from railway_recon.io import write_json
from railway_recon.registry import new_registry, summarize_registry


def _component(root: Path, name: str, object_name: str, material: str, x: float) -> dict[str, str]:
    directory = root / name
    directory.mkdir(parents=True)
    obj = directory / f"{name}.obj"
    mtl = directory / f"{name}.mtl"
    origin = directory / "model_origin.json"
    registry_path = directory / "asset_registry.json"
    obj.write_text(
        f"mtllib {mtl.name}\no {object_name}\nusemtl {material}\n"
        f"v {x} 0 0\nv {x + 1} 0 0\nv {x} 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    mtl.write_text(f"newmtl {material}\nKd 0.5 0.5 0.5\n", encoding="utf-8")
    write_json(origin, {"origin_xyz": [100.0 + x, 200.0, 0.0]})
    registry = new_registry("test-project")
    registry["assets"] = [
        {
            "id": object_name,
            "type": "test_asset",
            "status": "candidate",
            "evidence_level": "observed",
            "confidence": 0.8,
            "sources": [{"kind": "point_cloud", "reference": name}],
            "geometry": {"file": str(obj), "node": object_name},
        }
    ]
    registry["summary"] = summarize_registry(registry)
    write_json(registry_path, registry)
    return {"obj": str(obj), "origin": str(origin), "registry": str(registry_path)}


def test_compose_candidate_scene_merges_objects_and_registry(tmp_path: Path) -> None:
    project_path = initialize_project(tmp_path / "project", "test-project", "Test")
    first = _component(tmp_path, "first", "FIRST", "FirstMaterial", 0.0)
    second = _component(tmp_path, "second", "SECOND", "SecondMaterial", 5.0)

    report = compose_candidate_scene(
        load_project(project_path),
        "candidate-v1",
        [first, second],
        rejected_asset_ids={"REJECTED"},
        extra_relations=[
            {
                "id": "REL-FIRST-ALIGNED-WITH-SECOND",
                "type": "aligned_with",
                "from": "FIRST",
                "to": "SECOND",
            }
        ],
        update_canonical_registry=False,
    )

    assert report["formal_release"] is False
    assert report["asset_count"] == 2
    audit = json.loads(Path(report["output_mapping_audit"]).read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["object_count"] == 2
    registry = json.loads(Path(report["output_registry"]).read_text(encoding="utf-8"))
    assert {item["id"] for item in registry["assets"]} == {"FIRST", "SECOND"}
    assert registry["relations"] == [
        {
            "id": "REL-FIRST-ALIGNED-WITH-SECOND",
            "type": "aligned_with",
            "from": "FIRST",
            "to": "SECOND",
        }
    ]
    assert report["extra_relation_count"] == 1
    assert all(item["geometry"]["file"] == report["output_obj"] for item in registry["assets"])


def test_track_platform_audit_reports_gap_without_inventing_rejected_track() -> None:
    def observation(track_id: str, cross: float) -> dict[str, object]:
        return {
            "global_track_id": track_id,
            "local_track_id": track_id,
            "lateral_offset_m": cross,
            "rail_cross_positions_local_m": [cross - 0.754, cross + 0.754],
            "joint_support_ratio": 0.99,
            "asymmetric_support_ratio": 0.0,
            "maximum_internal_joint_gap_m": 0.0,
            "rail_top_crosslevel_m": 0.004,
            "source_reported_rail_top_crosslevel_m": 0.003,
            "pair_continuity_status": "pass",
            "rail_top_start_z_m": 10.0,
            "rail_top_end_z_m": 10.0,
        }

    result = audit_track_platform_candidate(
        {"observations": [observation("TRACK-0002", -12.5), observation("TRACK-0003", -7.5)]},
        {
            "platform_components": [
                {
                    "id": "PLATFORM",
                    "fit_segments": [
                        {
                            "longitudinal_range_m": [-5.0, 0.0],
                            "rail_side_edge_cross_m": 2.0,
                            "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 11.2],
                        }
                    ],
                }
            ]
        },
        {"longitudinal_range_m": [-10.0, 10.0]},
        {"track_bed": {"bottom_width_m": 3.5}},
        rejected_track_ids={"TRACK-0001", "TRACK-0004"},
    )
    assert result["geometry_checks_passed"] is True
    assert result["completeness_status"] == "review_required"
    assert result["track_spacing"][0]["parametric_bed_edge_clearance_m"] == 1.5
    assert result["platform_interface_samples"][0]["rail_to_platform_horizontal_gap_m"] > 0
