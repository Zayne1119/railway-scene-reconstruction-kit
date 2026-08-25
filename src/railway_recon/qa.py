from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, write_json
from .registry import summarize_registry, validate_registry_value


def quality_report(project: ProjectConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name in ("point_cloud", "camera_csv"):
        path = project.input_path(name)
        passed = bool(path and path.is_file())
        checks.append({"id": f"input.{name}", "passed": passed, "path": str(path)})

    audit_path = project.workspace_path("reports") / "input_audit.json"
    checks.append({"id": "audit.exists", "passed": audit_path.is_file(), "path": str(audit_path)})
    segment_manifest = project.workspace_path("segment_manifest")
    checks.append(
        {
            "id": "segments.manifest_exists",
            "passed": segment_manifest.is_file(),
            "path": str(segment_manifest),
        }
    )

    registry_path = project.workspace_path("asset_registry")
    registry_summary: dict[str, Any] | None = None
    registry_errors: list[str] = []
    if registry_path.is_file():
        registry = load_json(registry_path)
        registry_errors = validate_registry_value(registry)
        registry_summary = summarize_registry(registry)
    checks.append(
        {
            "id": "registry.valid",
            "passed": registry_path.is_file() and not registry_errors,
            "path": str(registry_path),
            "errors": registry_errors,
        }
    )

    crs = project.value["project"].get("crs", {})
    absolute_precision_ready = bool(crs.get("epsg") and crs.get("vertical_datum"))
    result = {
        "schema_version": "railway.quality-report.v1",
        "project_id": project.project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "asset_summary": registry_summary,
        "absolute_precision_ready": absolute_precision_ready,
        "limitations": [] if absolute_precision_ready else [
            "CRS and/or vertical datum is missing; report internal fit only."
        ],
    }
    write_json(project.workspace_path("reports") / "quality_report.json", result)
    return result

