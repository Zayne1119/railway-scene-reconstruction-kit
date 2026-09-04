from __future__ import annotations

from pathlib import Path

from railway_recon.io import write_json
from railway_recon.supplemental_candidate_gate import (
    gate_supplemental_conductor_candidate,
)


def test_supplemental_conductor_gate_accepts_supported_geometry(tmp_path: Path) -> None:
    assets = ("SUPPLEMENTAL-AUX-CONDUCTOR-01", "SUPPLEMENTAL-AUX-CONDUCTOR-02")
    refinement = {
        "new_assets_schema_valid": True,
        "fit_records": [
            {
                "asset_id": asset_id,
                "maximum_span_residual_p90_m": 0.015,
                "span_fits": [{"passed": True}, {"passed": True}],
            }
            for asset_id in assets
        ],
    }
    support = {
        "objects": [
            {
                "object_name": asset_id,
                "p90_m": 0.06,
                "coverage_at_0_10m": 0.995,
                "disposition": "supported_keep",
            }
            for asset_id in assets
        ]
    }
    inputs = {
        "refinement.json": refinement,
        "mesh.json": {"passed": True},
        "support.json": support,
        "views.json": {"view_count": 6},
    }
    for name, value in inputs.items():
        write_json(tmp_path / name, value)
    result = gate_supplemental_conductor_candidate(
        refinement_report_path=tmp_path / "refinement.json",
        mesh_audit_path=tmp_path / "mesh.json",
        point_support_report_path=tmp_path / "support.json",
        fixed_view_manifest_path=tmp_path / "views.json",
        output_path=tmp_path / "gate.json",
    )
    assert result["passed"] is True
    assert result["status"] == "geometry_accepted_semantic_subtype_pending"
