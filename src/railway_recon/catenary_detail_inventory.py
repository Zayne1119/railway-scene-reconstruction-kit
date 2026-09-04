from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json
from .model_point_support import parse_obj_model


def evaluate_catenary_detail_inventory(
    registry: dict[str, Any], object_names: list[str]
) -> dict[str, Any]:
    available = set(object_names)
    mast_assets = [
        asset for asset in registry.get("assets", []) if asset.get("type") == "catenary_mast"
    ]
    records: list[dict[str, Any]] = []
    for asset in mast_assets:
        node = str(asset.get("geometry", {}).get("node", ""))
        related = sorted(name for name in available if name == node or name.startswith(f"{node}-"))
        has_shaft = node in available
        has_foundation = any("FOUNDATION" in name for name in related)
        has_cantilever = any("CANTILEVER" in name for name in related)
        insulator_count = sum("INSULATOR" in name for name in related)
        has_positioner = any("POSITIONER" in name for name in related)
        complete = (
            has_shaft
            and has_foundation
            and has_cantilever
            and insulator_count >= 2
            and has_positioner
        )
        if complete:
            detail_status = "reference_assembly_complete"
            priority = "P2_reference_only"
            next_action = "retain_as_reference_and_validate_against_local_top_equipment_points"
        elif has_shaft and has_foundation:
            detail_status = "shaft_and_foundation_only"
            priority = "P0_extract_top_equipment_evidence"
            next_action = "extract_local_cantilever_insulator_positioner_geometry_without_template_cloning"
        elif has_shaft:
            detail_status = "shaft_only"
            priority = "P1_confirm_foundation_and_top_equipment"
            next_action = "confirm_foundation_then_extract_local_top_equipment_evidence"
        else:
            detail_status = "registered_mesh_missing"
            priority = "BLOCKED"
            next_action = "repair_registry_to_mesh_binding_before_refinement"
        records.append(
            {
                "asset_id": asset.get("id"),
                "node": node,
                "chainage_m": asset.get("chainage_m"),
                "evidence_level": asset.get("evidence_level"),
                "confidence": asset.get("confidence"),
                "components": {
                    "shaft": has_shaft,
                    "foundation": has_foundation,
                    "cantilever": has_cantilever,
                    "insulator_count": insulator_count,
                    "positioner": has_positioner,
                },
                "component_names": related,
                "detail_status": detail_status,
                "priority": priority,
                "next_action": next_action,
                "limitations": list(asset.get("limitations", [])),
            }
        )
    records.sort(
        key=lambda item: (
            item["chainage_m"] is None,
            float(item["chainage_m"] or 0.0),
            str(item["asset_id"]),
        )
    )
    status_counts = Counter(str(item["detail_status"]) for item in records)
    priorities = Counter(str(item["priority"]) for item in records)
    conductor_assets = [
        asset
        for asset in registry.get("assets", [])
        if asset.get("type") in {"contact_wire", "catenary_auxiliary_conductor"}
    ]
    gates = {
        "mast_assets_present": bool(records),
        "all_registered_masts_have_mesh": all(
            item["components"]["shaft"] for item in records
        ),
        "asset_ids_unique": len(records)
        == len({str(item["asset_id"]) for item in records}),
    }
    passed = all(gates.values())
    return {
        "summary": {
            "mast_count": len(records),
            "detail_status_counts": dict(status_counts),
            "priority_counts": dict(priorities),
            "top_equipment_refinement_target_count": sum(
                item["priority"] == "P0_extract_top_equipment_evidence"
                for item in records
            ),
            "foundation_confirmation_target_count": sum(
                item["priority"] == "P1_confirm_foundation_and_top_equipment"
                for item in records
            ),
            "contact_wire_asset_count": sum(
                asset.get("type") == "contact_wire" for asset in conductor_assets
            ),
            "auxiliary_conductor_asset_count": sum(
                asset.get("type") == "catenary_auxiliary_conductor"
                for asset in conductor_assets
            ),
        },
        "records": records,
        "gates": gates,
        "passed": passed,
        "status": (
            "inventory_complete_geometry_unchanged"
            if passed
            else "inventory_blocked_by_registry_mesh_mismatch"
        ),
        "policy": [
            "A complete reference mast is not cloned onto another location without local evidence.",
            "Observed shafts can enter top-equipment extraction; photo-interpreted shafts remain lower confidence.",
            "Inventory output authorizes review tasks only and does not write geometry.",
        ],
    }


def build_catenary_detail_inventory(
    *, registry_path: str | Path, obj_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    registry_file = Path(registry_path).resolve()
    model_file = Path(obj_path).resolve()
    output_file = Path(output_path).resolve()
    for path in (registry_file, model_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_file.exists():
        raise FileExistsError(output_file)
    registry = load_json(registry_file)
    model = parse_obj_model(model_file)
    result = evaluate_catenary_detail_inventory(
        registry, list(model.faces_by_object)
    )
    report = {
        "schema_version": "railway.catenary-detail-inventory.v1",
        "inputs": {
            "registry": str(registry_file),
            "obj": str(model_file),
            "obj_sha256": sha256_file(model_file),
        },
        **result,
    }
    write_json(output_file, report)
    return report
