from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, sha256_json, write_json


def _git_commit(start: Path) -> str | None:
    environment_value = os.environ.get("GITHUB_SHA")
    if environment_value:
        return environment_value
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_manifest(
    project: ProjectConfig,
    command: str,
    status: str,
    outputs: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + f"-{command.replace('_', '-')}"
    root = project.workspace_path("manifests") / run_id
    audit_path = project.workspace_path("reports") / "input_audit.json"
    input_hashes = load_json(audit_path).get("input_hashes", {}) if audit_path.is_file() else {}
    output_values = outputs or []
    output_artifacts: list[dict[str, Any]] = []
    for output in output_values:
        candidate = Path(output)
        resolved = candidate.resolve() if candidate.is_absolute() else project.resolve(candidate)
        record: dict[str, Any] = {
            "path": str(resolved),
            "exists": resolved.exists(),
            "kind": "file" if resolved.is_file() else "directory" if resolved.is_dir() else "missing",
        }
        if resolved.is_file():
            record["bytes"] = resolved.stat().st_size
            record["sha256"] = sha256_file(resolved)
        output_artifacts.append(record)
    value = {
        "schema_version": "railway.run-manifest.v1",
        "run_id": run_id,
        "project_id": project.project_id,
        "command": command,
        "status": status,
        "generated_at": timestamp.isoformat(),
        "config_path": str(project.path),
        "config_sha256": sha256_json(project.value),
        "git_commit": _git_commit(project.root),
        "input_hashes": input_hashes,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "outputs": output_values,
        "output_artifacts": output_artifacts,
        "metrics": metrics or {},
    }
    path = root / "manifest.json"
    write_json(root / "config.snapshot.json", project.value)
    write_json(path, value)
    return path
