from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # Keep the field CLI usable in an offline pre-provisioned Python.
    Draft202012Validator = None  # type: ignore[assignment]

from .config import ProjectConfig
from .io import load_json, sha256_json, write_json


def new_registry(project_id: str) -> dict[str, Any]:
    return {
        "schema_version": "railway.asset-registry.v1",
        "project_id": project_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "assets": [],
        "relations": [],
        "summary": {"asset_count": 0, "by_type": {}, "by_evidence_level": {}},
    }


def initialize_registry(project: ProjectConfig, overwrite: bool = False) -> Path:
    path = project.workspace_path("asset_registry")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite registry: {path}")
    write_json(path, new_registry(project.project_id))
    return path


def summarize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_evidence: dict[str, int] = {}
    for asset in registry.get("assets", []):
        asset_type = str(asset.get("type", "unknown"))
        level = str(asset.get("evidence_level", "unsupported"))
        by_type[asset_type] = by_type.get(asset_type, 0) + 1
        by_evidence[level] = by_evidence.get(level, 0) + 1
    return {
        "asset_count": len(registry.get("assets", [])),
        "by_type": dict(sorted(by_type.items())),
        "by_evidence_level": dict(sorted(by_evidence.items())),
    }


def validate_registry_value(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if Draft202012Validator is not None:
        resource = resources.files("railway_recon.resources").joinpath(
            "asset-registry.schema.json"
        )
        with resource.open("r", encoding="utf-8") as stream:
            import json

            schema = json.load(stream)
        validator = Draft202012Validator(schema)
        errors.extend(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(validator.iter_errors(registry), key=lambda item: list(item.path))
        )
    else:
        if registry.get("schema_version") != "railway.asset-registry.v1":
            errors.append("schema_version must be 'railway.asset-registry.v1'")
        for key in ("project_id", "assets", "relations", "summary"):
            if key not in registry:
                errors.append(f"missing required registry property: {key}")
        levels = {"observed", "photo_interpreted", "rule_inferred", "unsupported"}
        statuses = {"candidate", "reviewed", "accepted", "rejected", "superseded"}
        for index, asset in enumerate(registry.get("assets", [])):
            for key in ("id", "type", "status", "evidence_level", "confidence", "sources"):
                if key not in asset:
                    errors.append(f"assets.{index}: missing required property '{key}'")
            if asset.get("evidence_level") not in levels:
                errors.append(f"assets.{index}.evidence_level is invalid")
            if asset.get("status") not in statuses:
                errors.append(f"assets.{index}.status is invalid")
            confidence = asset.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append(f"assets.{index}.confidence must be within 0..1")
    ids = [str(item.get("id")) for item in registry.get("assets", [])]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        errors.append(f"duplicate asset ids: {duplicates}")
    known = set(ids)
    for relation in registry.get("relations", []):
        for endpoint in ("from", "to"):
            value = relation.get(endpoint)
            if value not in known:
                errors.append(f"relation {relation.get('id')} has unknown {endpoint}: {value}")
    return errors


def validate_registry_file(path: Path) -> tuple[dict[str, Any], list[str]]:
    registry = load_json(path)
    return registry, validate_registry_value(registry)


def freeze_release_registry(
    project: ProjectConfig,
    release_id: str,
    output: str | Path,
) -> dict[str, Any]:
    source_path = project.workspace_path("asset_registry")
    registry = load_json(source_path)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Cannot freeze an invalid registry:\n- " + "\n- ".join(errors))
    assets = list(registry.get("assets", []))
    if not assets:
        raise ValueError("Cannot freeze an empty release registry")
    blocking = sorted(
        str(asset.get("id")) for asset in assets if asset.get("status") != "accepted"
    )
    if blocking:
        raise ValueError(
            "Release registry contains non-accepted assets: " + ", ".join(blocking[:20])
        )
    output_path = project.resolve(output)
    root = project.root.resolve()
    if output_path != root and root not in output_path.parents:
        raise ValueError(f"Release registry must stay inside the project directory: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen release registry: {output_path}")
    asset_ids = sorted(str(asset["id"]) for asset in assets)
    registry["release_id"] = release_id
    registry["asset_set_sha256"] = hashlib.sha256(
        "\n".join(asset_ids).encode("utf-8")
    ).hexdigest()
    registry["relation_set_sha256"] = sha256_json(
        sorted(registry.get("relations", []), key=lambda item: str(item.get("id", "")))
    )
    registry["frozen_at"] = datetime.now(UTC).isoformat()
    write_json(output_path, registry)
    return {
        "registry_path": str(output_path),
        "release_id": release_id,
        "asset_count": len(asset_ids),
        "asset_set_sha256": registry["asset_set_sha256"],
        "relation_set_sha256": registry["relation_set_sha256"],
    }


def _asset_from_csv(row: dict[str, str]) -> dict[str, Any]:
    limitations = [
        item.strip() for item in (row.get("limitations") or "").split("|") if item.strip()
    ]
    chainage_text = (row.get("chainage_m") or "").strip()
    return {
        "id": row["id"].strip(),
        "type": row["type"].strip(),
        "subtype": (row.get("subtype") or "").strip() or None,
        "status": (row.get("status") or "reviewed").strip(),
        "chainage_m": float(chainage_text) if chainage_text else None,
        "evidence_level": row["evidence_level"].strip(),
        "confidence": float(row["confidence"]),
        "sources": [
            {
                "kind": row["source_kind"].strip(),
                "reference": row["source_reference"].strip(),
                "note": (row.get("source_note") or "").strip(),
            }
        ],
        "parameters": {},
        "geometry": {"node": (row.get("geometry_node") or row["id"]).strip()},
        "limitations": limitations,
    }


def import_registry_records(
    project: ProjectConfig,
    source: Path,
    replace_existing_ids: bool = False,
) -> dict[str, Any]:
    registry_path = project.workspace_path("asset_registry")
    registry = load_json(registry_path) if registry_path.is_file() else new_registry(project.project_id)
    suffix = source.suffix.lower()
    relations: list[dict[str, Any]] = []
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "id", "type", "evidence_level", "confidence", "source_kind", "source_reference"
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Review CSV is missing columns: {sorted(missing)}")
            assets = [_asset_from_csv(row) for row in reader]
    elif suffix == ".json":
        value = load_json(source)
        if "assets" not in value:
            raise ValueError("JSON import must contain an 'assets' array")
        assets = list(value["assets"])
        relations = list(value.get("relations", []))
    else:
        raise ValueError("Registry import supports CSV or JSON")
    if not assets:
        raise ValueError("Registry import contains no assets")

    incoming_ids = [str(item.get("id")) for item in assets]
    duplicate_incoming = sorted({value for value in incoming_ids if incoming_ids.count(value) > 1})
    if duplicate_incoming:
        raise ValueError(f"Import contains duplicate asset ids: {duplicate_incoming}")
    existing_ids = {str(item["id"]) for item in registry.get("assets", [])}
    conflicts = sorted(existing_ids & set(incoming_ids))
    if conflicts and not replace_existing_ids:
        raise ValueError(f"Asset ids already exist; use explicit replacement: {conflicts}")
    if conflicts:
        conflict_set = set(conflicts)
        registry["assets"] = [
            item for item in registry["assets"] if item["id"] not in conflict_set
        ]
    registry["assets"].extend(assets)

    existing_relation_ids = {str(item["id"]) for item in registry.get("relations", [])}
    relation_conflicts = existing_relation_ids & {str(item.get("id")) for item in relations}
    if relation_conflicts and not replace_existing_ids:
        raise ValueError(f"Relation ids already exist: {sorted(relation_conflicts)}")
    if relation_conflicts:
        registry["relations"] = [
            item for item in registry["relations"] if item["id"] not in relation_conflicts
        ]
    registry["relations"].extend(relations)
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Imported registry records are invalid:\n- " + "\n- ".join(errors))
    write_json(registry_path, registry)
    return {
        "registry_path": str(registry_path),
        "imported_asset_count": len(assets),
        "imported_relation_count": len(relations),
        "replaced_asset_ids": conflicts,
        "summary": registry["summary"],
    }
