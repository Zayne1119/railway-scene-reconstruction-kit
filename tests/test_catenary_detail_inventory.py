from __future__ import annotations

from railway_recon.catenary_detail_inventory import evaluate_catenary_detail_inventory


def _asset(asset_id: str, evidence: str = "observed") -> dict:
    return {
        "id": asset_id,
        "type": "catenary_mast",
        "chainage_m": 10.0,
        "evidence_level": evidence,
        "confidence": 0.9,
        "geometry": {"node": asset_id},
        "limitations": [],
    }


def test_catenary_inventory_separates_complete_and_bare_masts() -> None:
    complete = "CATENARY-MAST-001"
    bare = "CATENARY-MAST-002"
    registry = {
        "assets": [
            _asset(complete),
            _asset(bare),
            {"id": "WIRE-1", "type": "contact_wire"},
        ]
    }
    objects = [
        complete,
        f"{complete}-FOUNDATION",
        f"{complete}-CANTILEVER",
        f"{complete}-INSULATOR-UPPER",
        f"{complete}-INSULATOR-LOWER",
        f"{complete}-POSITIONER",
        bare,
        f"{bare}-FOUNDATION",
    ]
    result = evaluate_catenary_detail_inventory(registry, objects)
    assert result["passed"] is True
    assert result["summary"]["mast_count"] == 2
    assert result["summary"]["top_equipment_refinement_target_count"] == 1
    assert result["summary"]["contact_wire_asset_count"] == 1
    assert result["records"][0]["detail_status"] == "reference_assembly_complete"
    assert result["records"][1]["detail_status"] == "shaft_and_foundation_only"
