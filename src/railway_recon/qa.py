from __future__ import annotations

from datetime import UTC, datetime
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

    track_graph_configured = bool(project.value.get("algorithms", {}).get("track_graph"))
    track_graph_audit_path = project.workspace_path("reports") / "track_graph_audit.json"
    track_graph_status = None
    if track_graph_audit_path.is_file():
        track_graph_status = load_json(track_graph_audit_path).get("status")
    track_graph_ready = not track_graph_configured or track_graph_status == "pass"
    track_mesh_audit_path = project.workspace_path("reports") / "track_graph_mesh_audit.json"
    track_mesh_status = None
    track_mesh_mapping_complete = False
    if track_mesh_audit_path.is_file():
        track_mesh_audit = load_json(track_mesh_audit_path)
        track_mesh_status = track_mesh_audit.get("status")
        track_mesh_mapping_complete = bool(
            track_mesh_audit.get("registry_mapping_complete")
            and track_mesh_audit.get("visible_node_set_matches_expected")
        )
    track_mesh_ready = (
        not track_graph_configured
        or (track_mesh_status == "pass" and track_mesh_mapping_complete)
    )
    if track_graph_configured:
        checks.append(
            {
                "id": "track_graph.pass",
                "passed": track_graph_ready,
                "path": str(track_graph_audit_path),
                "status": track_graph_status or "missing",
            }
        )
        checks.append(
            {
                "id": "track_graph_mesh.pass",
                "passed": track_mesh_ready,
                "path": str(track_mesh_audit_path),
                "status": track_mesh_status or "missing",
                "registry_mapping_complete": track_mesh_mapping_complete,
            }
        )

    registry_path = project.workspace_path("asset_registry")
    registry_summary: dict[str, Any] | None = None
    registry_errors: list[str] = []
    blocking_release_assets: list[str] = []
    if registry_path.is_file():
        registry = load_json(registry_path)
        registry_errors = validate_registry_value(registry)
        registry_summary = summarize_registry(registry)
        blocking_release_assets = sorted(
            str(asset.get("id"))
            for asset in registry.get("assets", [])
            if asset.get("status") != "accepted"
        )
    checks.append(
        {
            "id": "registry.valid",
            "passed": registry_path.is_file() and not registry_errors,
            "path": str(registry_path),
            "errors": registry_errors,
        }
    )

    crs = project.value["project"].get("crs", {})
    check_point_reference = crs.get("independent_check_points")
    check_point_path = project.resolve(check_point_reference) if check_point_reference else None
    absolute_precision_ready = bool(
        crs.get("epsg")
        and crs.get("vertical_datum")
        and check_point_path
        and check_point_path.is_file()
    )
    precision_limitations: list[str] = []
    if not crs.get("epsg") or not crs.get("vertical_datum"):
        precision_limitations.append(
            "CRS and/or vertical datum is missing; report internal fit only."
        )
    if not check_point_path or not check_point_path.is_file():
        precision_limitations.append(
            "Independent check points are missing; do not claim absolute world accuracy."
        )
    result = {
        "schema_version": "railway.quality-report.v1",
        "project_id": project.project_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "asset_summary": registry_summary,
        "release_ready": bool(registry_summary and registry_summary["asset_count"])
        and not registry_errors
        and not blocking_release_assets
        and track_graph_ready
        and track_mesh_ready,
        "blocking_release_asset_count": len(blocking_release_assets),
        "blocking_release_asset_ids": blocking_release_assets[:100],
        "absolute_precision_ready": absolute_precision_ready,
        "limitations": precision_limitations,
    }
    write_json(project.workspace_path("reports") / "quality_report.json", result)
    return result
