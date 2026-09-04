from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import sha256_file, write_json
from .registry import summarize_registry, validate_registry_value


def _safe_asset_id(node: str) -> str:
    value = node.upper().replace(".", "-").replace(" ", "-")
    if len(value) <= 80:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12].upper()
    return f"{value[:67]}-{digest}"


def _classify_node(node: str) -> tuple[str, str, float, str]:
    upper = node.upper()
    if "CONTEXT-LOW-CONFIDENCE" in upper:
        return "context_building_mass", "rule_inferred", 0.30, "point_cloud"
    if "-RAIL-LEFT" in upper or "-RAIL-RIGHT" in upper:
        return "rail", "observed", 0.85, "point_cloud"
    if "-SLEEPERS" in upper:
        return "sleeper_group", "rule_inferred", 0.55, "rule"
    if "-BED" in upper and "TRACK" in upper:
        return "track_bed", "rule_inferred", 0.65, "rule"
    if "PLATFORM" in upper and (upper.endswith("-TOP") or "-SURFACE" in upper):
        return "platform_surface", "observed", 0.8, "point_cloud"
    if "PLATFORM" in upper and (upper.endswith("-VOLUME") or "-TRANSITION" in upper):
        return "platform_volume", "rule_inferred", 0.6, "rule"
    if "CANOPY" in upper and upper.endswith("-CAPITAL"):
        return "canopy_capital", "photo_interpreted", 0.72, "panorama"
    if "CANOPY" in upper and "UNDERROOF" in upper:
        return "canopy_underroof_connector", "photo_interpreted", 0.68, "panorama"
    if "CANOPY" in upper and "ROOF" in upper:
        return "canopy_roof_surface", "observed", 0.75, "point_cloud"
    if "CANOPY" in upper and ("GRID" in upper or "COLUMN" in upper):
        return "canopy_column", "photo_interpreted", 0.78, "panorama"
    if "CATENARY" in upper and "CANTILEVER" in upper:
        return "catenary_cantilever", "observed", 0.90, "point_cloud"
    if "CATENARY" in upper and "POSITIONER" in upper:
        return "catenary_positioner", "observed", 0.90, "point_cloud"
    if "CATENARY" in upper and "TOPOLOGY-CONSTRAINED-STEEL" in upper:
        return "catenary_cantilever", "observed", 0.90, "point_cloud"
    if "CATENARY" in upper and "INSULATOR" in upper:
        return "catenary_insulator", "observed", 0.90, "point_cloud"
    if "CATENARY" in upper and "FOUNDATION" in upper:
        return "catenary_foundation", "photo_interpreted", 0.92, "panorama"
    if "CATENARY" in upper and "MAST" in upper:
        return "catenary_mast", "photo_interpreted", 0.97, "panorama"
    if (
        "CONTACT-WIRE" in upper
        or "CATENARY-WIRE" in upper
        or "CONDUCTOR" in upper
    ):
        return "contact_wire", "photo_interpreted", 0.75, "panorama"
    if "SIGN" in upper:
        return "station_information_sign", "photo_interpreted", 0.88, "panorama"
    if "FENCE" in upper:
        return "platform_fence", "photo_interpreted", 0.82, "panorama"
    if "STATION" in upper and "GLASS" in upper:
        return "station_glazing", "photo_interpreted", 0.75, "panorama"
    if "STATION" in upper and "DOOR" in upper:
        return "station_door", "photo_interpreted", 0.75, "panorama"
    if "STATION" in upper and any(
        token in upper for token in ("CLADDING", "WALL", "FACADE")
    ):
        return "station_facade", "photo_interpreted", 0.72, "panorama"
    if "CLOCK" in upper:
        return "station_clock", "photo_interpreted", 0.72, "panorama"
    return "unclassified_renderable_asset", "unsupported", 0.0, "manual_review"


def _object_names(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("o "):
            name = line[2:].strip()
            if name and name not in names:
                names.append(name)
    if not names:
        raise ValueError(f"OBJ has no object declarations: {path}")
    return names


def build_mesh_object_registry(
    project: ProjectConfig,
    obj_path: str | Path,
    output_path: str | Path,
    *,
    evidence_reference: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register every renderable OBJ object after namespaced corridor composition."""
    obj = project.resolve(obj_path)
    output = project.resolve(output_path)
    if not obj.is_file():
        raise FileNotFoundError(obj)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    assets = []
    unclassified = []
    for node in _object_names(obj):
        asset_type, evidence_level, confidence, source_kind = _classify_node(node)
        asset_id = _safe_asset_id(node)
        if asset_type == "unclassified_renderable_asset":
            unclassified.append(asset_id)
        assets.append(
            {
                "id": asset_id,
                "type": asset_type,
                "subtype": "namespaced_composed_mesh_object",
                "status": "candidate",
                "chainage_m": None,
                "evidence_level": evidence_level,
                "confidence": confidence,
                "sources": [
                    {
                        "kind": source_kind,
                        "reference": evidence_reference,
                        "note": "Evidence class inherited from the reviewed component type.",
                    }
                ],
                "parameters": {
                    "composed_node": node,
                    "classification_policy": "mesh_object_name_candidate_mapping_v1",
                },
                "geometry": {"file": str(obj), "node": node},
                "limitations": [
                    "Candidate registry entry; retain source evidence reports for acceptance."
                ],
            }
        )
    registry: dict[str, Any] = {
        "schema_version": "railway.asset-registry.v1",
        "project_id": project.project_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "assets": assets,
        "relations": [],
        "model_sha256": sha256_file(obj),
        "mapping_policy": "one_registry_asset_per_renderable_namespaced_obj_object",
        "unclassified_asset_ids": unclassified,
    }
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Generated mesh object registry is invalid: " + "; ".join(errors))
    write_json(output, registry)
    return {
        "schema_version": "railway.mesh-object-registry-build.v1",
        "project_id": project.project_id,
        "obj": str(obj),
        "output_registry": str(output),
        "asset_count": len(assets),
        "unclassified_asset_count": len(unclassified),
        "unclassified_asset_ids": unclassified,
        "passed": not unclassified,
        "status": "pass" if not unclassified else "manual_classification_required",
    }
