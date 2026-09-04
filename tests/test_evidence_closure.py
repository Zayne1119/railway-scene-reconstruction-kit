from __future__ import annotations

import json
from pathlib import Path

from railway_recon.evidence_closure import build_no_geometry_evidence_closure


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_build_no_geometry_evidence_closure(tmp_path: Path) -> None:
    model = tmp_path / "model.obj"
    model.write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = _write(tmp_path / "registry.json", {"assets": []})
    mesh = _write(
        tmp_path / "mesh.json",
        {
            "passed": True,
            "object_count": 1,
            "triangle_count_after_fan_triangulation": 1,
            "duplicate_face_count": 0,
            "degenerate_triangle_count": 0,
        },
    )
    vertical = _write(
        tmp_path / "vertical.json",
        {
            "summary": {
                "reviewed_candidate_count": 2,
                "geometry_add_count": 0,
                "adjacent_segment_hold_count": 1,
                "facade_or_mullion_edge_count": 1,
                "outside_canopy_footprint_count": 0,
                "unresolved_candidate_count": 0,
            }
        },
    )
    canopy = _write(
        tmp_path / "canopy.json",
        {
            "roof_object_count": 2,
            "passed_count": 1,
            "records": [
                {
                    "object_name": "roof-a",
                    "passed": False,
                    "vertical_correction_at_center_m": 0.2,
                    "plane_angle_change_deg": 3.0,
                    "gates": {"slope": False, "points": True},
                }
            ],
        },
    )
    support = _write(
        tmp_path / "support.json",
        {
            "objects": [
                {
                    "object_name": "platform",
                    "asset_type": "platform_surface",
                    "p90_m": 0.05,
                    "coverage_at_0_10m": 0.9,
                    "nearest_delta_median_xyz_m": [0.0, 0.0, 0.001],
                    "disposition": "supported_keep",
                }
            ]
        },
    )
    seams = _write(
        tmp_path / "seams.json",
        {
            "passed": True,
            "seams": [
                {
                    "seam_station_m": 50.0,
                    "symmetric_nearest_edge_distance": {
                        "p90_m": 0.005,
                        "maximum_m": 0.01,
                    },
                    "passed": True,
                }
            ],
        },
    )

    output = tmp_path / "closure.json"
    build_no_geometry_evidence_closure(
        model_path=model,
        registry_path=registry,
        mesh_audit_path=mesh,
        vertical_decisions_path=vertical,
        canopy_audit_path=canopy,
        platform_support_path=support,
        platform_seam_audit_path=seams,
        output_path=output,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "evidence_closed_model_hash_unchanged"
    assert result["canopy_review"]["withheld_count"] == 1
    assert result["geometry_write"] is False
