from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import open3d as o3d

from .io import load_json, write_json
from .web_glb_export import inspect_web_glb


def accept_web_preview(
    *,
    base_url: str,
    preview_directory: str | Path,
    final_qa_path: str | Path,
    output_path: str | Path,
    browser_visual_review_completed: bool = False,
) -> dict[str, Any]:
    preview = Path(preview_directory).resolve()
    required_urls = {
        "index": "/",
        "javascript": "/assets/candidate-app.js",
        "stylesheet": "/assets/candidate-app.css",
        "project_config": "/project.json",
        "glb": "/model.glb",
        "registry": "/asset_registry.json",
        "mesh_audit": "/mesh_audit.json",
    }
    http: dict[str, dict[str, Any]] = {}
    for name, suffix in required_urls.items():
        request = urllib.request.Request(base_url.rstrip("/") + suffix, method="HEAD")
        with urllib.request.urlopen(request, timeout=15) as response:
            http[name] = {
                "status": int(response.status),
                "content_length": int(response.headers.get("Content-Length", "0")),
            }

    glb_path = preview / "model.glb"
    registry = load_json(preview / "asset_registry.json")
    mesh_audit = load_json(preview / "mesh_audit.json")
    project = load_json(preview / "project.json")
    final_qa = load_json(final_qa_path)
    inspection = inspect_web_glb(glb_path)
    mesh = o3d.io.read_triangle_mesh(str(glb_path))
    glb_names = set(inspection["node_names"])
    registered_nodes = {
        str(asset.get("geometry", {}).get("node"))
        for asset in registry.get("assets", [])
        if asset.get("geometry", {}).get("node")
    }
    missing_registered_nodes = sorted(registered_nodes - glb_names)
    checks = {
        "all_http_resources_200": all(value["status"] == 200 for value in http.values()),
        "glb_structural_validation_passed": inspection["node_count"] > 0
        and inspection["triangle_count"] > 0,
        "open3d_independent_parse_passed": not mesh.is_empty()
        and len(mesh.triangles) == inspection["triangle_count"]
        and mesh.has_vertex_normals(),
        "registered_geometry_nodes_present": not missing_registered_nodes,
        "mesh_audit_passed": bool(mesh_audit.get("passed")),
        "final_scene_qa_passed": bool(final_qa.get("summary", {}).get("passed")),
        "project_uses_final_glb_and_registry": project.get("model_glb_url") == "/model.glb"
        and any(
            value.get("url") == "/asset_registry.json"
            for value in project.get("registry_sources", [])
        ),
    }
    technical_passed = all(checks.values())
    report = {
        "schema_version": "railway.local-web-preview-acceptance.v2",
        "base_url": base_url,
        "preview_directory": str(preview),
        "http": http,
        "checks": checks,
        "glb_inspection": inspection,
        "independent_parse": {
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.triangles),
            "has_vertex_normals": mesh.has_vertex_normals(),
            "empty": mesh.is_empty(),
            "bounds_minimum": mesh.get_min_bound().tolist(),
            "bounds_maximum": mesh.get_max_bound().tolist(),
        },
        "registry": {
            "asset_count": len(registry.get("assets", [])),
            "registered_geometry_node_count": len(registered_nodes),
            "missing_registered_nodes": missing_registered_nodes,
        },
        "technical_passed": technical_passed,
        "browser_visual_review_completed": browser_visual_review_completed,
        "passed": technical_passed and browser_visual_review_completed,
        "status": (
            "technical_and_visual_web_acceptance_passed"
            if technical_passed and browser_visual_review_completed
            else "technical_load_and_resource_consistency_passed_visual_browser_review_pending"
            if technical_passed
            else "technical_web_acceptance_failed"
        ),
        "limitations": (
            []
            if browser_visual_review_completed
            else [
                "No controllable browser was connected to this Codex session, so automated canvas screenshot, picking and visual interaction review remain pending.",
                "Open the local URL and confirm overview, catenary layer, asset search and picking once before presentation.",
            ]
        ),
    }
    write_json(Path(output_path), report)
    return report

