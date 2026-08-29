from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .registry import validate_registry_value


def _safe_path(project: ProjectConfig, value: str | Path) -> Path:
    path = project.resolve(value)
    root = project.root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Release input/output must stay inside the project directory: {path}")
    return path


def build_web_acceptance_config(
    project: ProjectConfig,
    release_id: str,
    title: str,
    model: str | Path,
    registry: str | Path,
    output: str | Path,
    *,
    model_url: str,
    registry_url: str,
) -> dict[str, Any]:
    model_path = _safe_path(project, model)
    registry_path = _safe_path(project, registry)
    output_path = _safe_path(project, output)
    for source in (model_path, registry_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite release config: {output_path}")
    registry_value = load_json(registry_path)
    errors = validate_registry_value(registry_value)
    if errors:
        raise ValueError("Release registry is invalid:\n- " + "\n- ".join(errors))
    if registry_value.get("release_id") != release_id:
        raise ValueError("Release registry release_id does not match")
    assets = list(registry_value.get("assets", []))
    blocking = [asset.get("id") for asset in assets if asset.get("status") != "accepted"]
    if not assets or blocking:
        raise ValueError("Release registry must contain accepted assets only")
    asset_ids = sorted(str(asset["id"]) for asset in assets)
    asset_set_sha256 = hashlib.sha256("\n".join(asset_ids).encode("utf-8")).hexdigest()
    if registry_value.get("asset_set_sha256") != asset_set_sha256:
        raise ValueError("Release registry asset_set_sha256 does not match its assets")
    value = {
        "schema_version": "railway.web-release-config.v2",
        "title": title,
        "acceptance_mode": True,
        "release_id": release_id,
        "model_url": model_url,
        "model_sha256": sha256_file(model_path),
        "registry_url": registry_url,
        "registry_sha256": sha256_file(registry_path),
        "asset_set_sha256": asset_set_sha256,
    }
    write_json(output_path, value)
    return {
        "config_path": str(output_path),
        **value,
    }
