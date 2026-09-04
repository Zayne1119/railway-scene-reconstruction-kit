from __future__ import annotations

import copy
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value
from .targeted_canopy_integration import merge_candidate_objs


def _context_asset_type(object_name: str) -> str:
    upper = object_name.upper()
    if upper.startswith(("TRACK--", "TRACK-")):
        if "RAIL-" in upper:
            return "rail"
        if "SLEEPER" in upper:
            return "sleeper_group"
        if "BED" in upper:
            return "track_bed"
        return "track_context"
    if upper.startswith("STATION--PLATFORM"):
        return "platform_context"
    if upper.startswith("STATION--CANOPY"):
        if "ROOF" in upper:
            return "canopy_roof_context"
        if "GRID" in upper or "COLUMN" in upper:
            return "canopy_column_context"
        return "canopy_context"
    if upper.startswith("CATENARY--"):
        return "catenary_context"
    if upper.startswith("CONDUCTOR--"):
        return "catenary_conductor_context"
    if upper.startswith("ADJACENT-") and "CANOPY-ROOF" in upper:
        return "canopy_roof_context"
    if upper.startswith("ADJACENT-") and "PLATFORM-TRANSITION" in upper:
        return "platform_transition_context"
    if upper.startswith("ADJACENT-") and "PLATFORM" in upper:
        return "platform_context"
    if "-PLATFORM-" in upper:
        return "platform_context"
    return "adjacent_segment_context"


def _next_context_sequence(registry: dict[str, Any]) -> int:
    values = []
    for asset in registry.get("assets", []):
        match = re.fullmatch(r"ADJCTX-(\d+)-[A-Z0-9_-]+", str(asset.get("id", "")))
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0) + 1


def _asset_type_code(asset_type: str) -> str:
    codes = {
        "rail": "RAIL",
        "sleeper_group": "SLEEPER",
        "track_bed": "BED",
        "track_context": "TRACK",
        "platform_context": "PLATFORM",
        "platform_transition_context": "PLATFORM-TRANS",
        "canopy_roof_context": "ROOF",
        "canopy_column_context": "COLUMN",
        "canopy_context": "CANOPY",
        "catenary_context": "CATENARY",
        "catenary_conductor_context": "CONDUCTOR",
        "adjacent_segment_context": "CONTEXT",
    }
    return codes[asset_type]


def build_adjacent_context_candidate(
    *,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    context_obj: str | Path,
    context_origin: str | Path,
    context_clip_report: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_model = Path(source_obj).resolve()
    source_origin_path = Path(source_origin).resolve()
    source_registry_path = Path(source_registry).resolve()
    context_model = Path(context_obj).resolve()
    context_origin_path = Path(context_origin).resolve()
    clip_path = Path(context_clip_report).resolve()
    for path in (
        source_model,
        source_origin_path,
        source_registry_path,
        context_model,
        context_origin_path,
        clip_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    context = parse_obj_model(context_model)
    context_names = sorted(context.faces_by_object)
    if not context_names:
        raise ValueError("Adjacent context clip contains no mesh objects")
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [(source_model, source_origin_path), (context_model, context_origin_path)],
        output_obj,
        output_mtl,
        output_origin,
    )

    registry = copy.deepcopy(load_json(source_registry_path))
    for asset in registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    clip = load_json(clip_path)
    owned_interval = [float(value) for value in clip["owned_longitudinal_interval_m"]]
    evidence_level = str(clip.get("evidence_level", "rule_inferred"))
    confidence = float(clip.get("confidence", 0.65))
    source_reference = str(clip.get("source_cloud", clip.get("source_obj", context_model)))
    subtype = str(
        clip.get("subtype", "automatic_full_corridor_candidate_owned_clip")
    )
    geometry_method = str(
        clip.get(
            "geometry_method",
            "automatic_full_corridor_candidate_then_ownership_clip",
        )
    )
    additions: list[dict[str, Any]] = []
    first_sequence = _next_context_sequence(registry)
    for sequence, object_name in enumerate(context_names, start=first_sequence):
        asset_type = _context_asset_type(object_name)
        asset = {
            "id": f"ADJCTX-{sequence:03d}-{_asset_type_code(asset_type)}",
            "type": asset_type,
            "subtype": subtype,
            "status": "candidate",
            "chainage_m": sum(owned_interval) / 2.0,
            "evidence_level": evidence_level,
            "confidence": confidence,
            "sources": [
                {"kind": "rule", "reference": str(clip_path)},
                {"kind": "point_cloud", "reference": source_reference},
            ],
            "parameters": {
                "segment_ownership": "adjacent_next_segment",
                "owned_longitudinal_interval_m": owned_interval,
                "source_object": object_name,
                "geometry_method": geometry_method,
            },
            "geometry": {"file": str(output_obj), "node": object_name},
            "limitations": [
                "This is an automatic adjacent-segment context candidate, not accepted geometry.",
                "Its seam with the refined scene requires subsystem-specific continuity review.",
            ],
        }
        additions.append(asset)
        registry["assets"].append(asset)
    registry["release_id"] = output_name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    additions_registry = new_registry(str(registry.get("project_id", "site-b")))
    additions_registry["assets"] = copy.deepcopy(additions)
    additions_registry["summary"] = summarize_registry(additions_registry)
    errors = validate_registry_value(additions_registry)
    if errors:
        raise ValueError("Adjacent context assets are invalid: " + "; ".join(errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Adjacent context candidate failed mesh audit")
    report = output / "adjacent_context_candidate_report.json"
    write_json(
        report,
        {
            "schema_version": "railway.adjacent-context-candidate.v1",
            "source_obj": str(source_model),
            "context_obj": str(context_model),
            "context_clip_report": str(clip_path),
            "owned_longitudinal_interval_m": owned_interval,
            "context_object_count": len(context_names),
            "context_type_counts": {
                kind: sum(_context_asset_type(name) == kind for name in context_names)
                for kind in sorted({_context_asset_type(name) for name in context_names})
            },
            "merge": merge,
            "output_obj": str(output_obj),
            "output_obj_sha256": sha256_file(output_obj),
            "mesh_audit_passed": True,
            "status": "adjacent_context_candidate_assembled_seam_review_required_not_promoted",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": report,
    }
