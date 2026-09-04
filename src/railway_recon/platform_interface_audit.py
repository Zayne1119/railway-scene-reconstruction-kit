from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ProjectConfig
from .io import load_json, write_json
from .platform_mesh import platform_sections


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "platform-interface-audit.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "p50": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(array.max()),
    }


def _platform_at(
    sections: list[dict[str, float]], longitudinal: float, cross: float
) -> dict[str, float] | None:
    s_values = np.asarray([item["longitudinal_m"] for item in sections])
    if longitudinal < float(s_values.min()) or longitudinal > float(s_values.max()):
        return None
    low_cross = float(np.interp(longitudinal, s_values, [item["low_cross_m"] for item in sections]))
    high_cross = float(
        np.interp(longitudinal, s_values, [item["high_cross_m"] for item in sections])
    )
    low_z = float(np.interp(longitudinal, s_values, [item["low_z_m"] for item in sections]))
    high_z = float(np.interp(longitudinal, s_values, [item["high_z_m"] for item in sections]))
    interpolation = (cross - low_cross) / max(high_cross - low_cross, 1e-9)
    elevation = low_z + np.clip(interpolation, 0.0, 1.0) * (high_z - low_z)
    return {
        "low_cross_m": low_cross,
        "high_cross_m": high_cross,
        "elevation_m": float(elevation),
    }


def audit_platform_interfaces_data(
    platform: dict[str, Any],
    vertical: dict[str, Any],
    canopy: dict[str, Any],
    gate: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    component_id = str(gate["platform_component_id"])
    component = next(item for item in platform["platform_components"] if item["id"] == component_id)
    sections = platform_sections(component)
    rails = platform.get("rail_lines", [])
    rail_clearances: list[float] = []
    platform_heights: list[float] = []
    edge_cross_positions: list[float] = []
    rail_samples: list[dict[str, Any]] = []
    for section in sections:
        edge_cross = (
            section["high_cross_m"] if component["side"] == "left" else section["low_cross_m"]
        )
        edge_z = section["high_z_m"] if component["side"] == "left" else section["low_z_m"]
        nearest = min(rails, key=lambda rail: abs(float(rail["cross_position_m"]) - edge_cross))
        clearance = abs(edge_cross - float(nearest["cross_position_m"]))
        height = edge_z - float(nearest["median_z_m"])
        rail_clearances.append(clearance)
        platform_heights.append(height)
        edge_cross_positions.append(edge_cross)
        rail_samples.append(
            {
                "longitudinal_m": section["longitudinal_m"],
                "platform_edge_cross_m": edge_cross,
                "platform_edge_z_m": edge_z,
                "nearest_rail_id": nearest["id"],
                "edge_to_nearest_rail_center_m": clearance,
                "platform_above_nearest_rail_m": height,
            }
        )
    lateral_jumps = np.abs(np.diff(edge_cross_positions)).tolist()
    rail_interface = {
        "sample_count": len(rail_samples),
        "edge_to_nearest_rail_center_m": _statistics(rail_clearances),
        "platform_above_nearest_rail_m": _statistics(platform_heights),
        "adjacent_edge_lateral_jump_m": _statistics(lateral_jumps),
        "maximum_allowed_adjacent_edge_jump_m": float(
            settings["maximum_adjacent_platform_edge_jump_m"]
        ),
        "samples": rail_samples,
        "status": (
            "pass_internal_continuity_absolute_clearance_not_certified"
            if max(lateral_jumps, default=0.0)
            <= float(settings["maximum_adjacent_platform_edge_jump_m"])
            else "fail_platform_edge_discontinuity"
        ),
        "limitations": [
            "No design alignment or surveyed platform-clearance target is available.",
            "Reported distances are internal point-cloud/model relationships, not regulatory clearance certification.",
        ],
    }

    contact_tolerance = float(settings["vertical_base_contact_tolerance_m"])
    vertical_interfaces: list[dict[str, Any]] = []
    for candidate in vertical.get("candidates", []):
        longitudinal = float(candidate["longitudinal_position_m"])
        cross = float(candidate["cross_position_m"])
        footprint_radius = float(candidate.get("footprint_m", 0.0)) / 2.0
        local = _platform_at(sections, longitudinal, cross)
        if local is None:
            continue
        footprint_intersects = (
            cross + footprint_radius >= local["low_cross_m"]
            and cross - footprint_radius <= local["high_cross_m"]
        )
        if not footprint_intersects:
            continue
        center_inside = local["low_cross_m"] <= cross <= local["high_cross_m"]
        base_offset = float(candidate["minimum_z"] - local["elevation_m"])
        if abs(base_offset) <= contact_tolerance:
            contact = "base_contacts_platform_within_tolerance"
        elif base_offset > contact_tolerance:
            contact = "vertical_candidate_suspended_above_platform"
        else:
            contact = "vertical_candidate_penetrates_below_platform_top"
        semantic_conflict = (
            candidate["predicted_class"] == "false_positive"
            and center_inside
            and contact == "base_contacts_platform_within_tolerance"
        )
        vertical_interfaces.append(
            {
                "candidate_id": candidate["id"],
                "predicted_class": candidate["predicted_class"],
                "confidence": candidate["confidence"],
                "longitudinal_position_m": longitudinal,
                "cross_position_m": cross,
                "footprint_m": float(candidate.get("footprint_m", 0.0)),
                "center_inside_platform": center_inside,
                "footprint_intersects_platform": footprint_intersects,
                "platform_elevation_m": local["elevation_m"],
                "candidate_base_elevation_m": float(candidate["minimum_z"]),
                "base_vertical_offset_m": base_offset,
                "geometry_contact": contact,
                "semantic_conflict": semantic_conflict,
                "status": (
                    "semantic_reclassification_required"
                    if semantic_conflict
                    else "geometry_interface_review_required"
                ),
            }
        )

    stable_ids = set(canopy.get("stable_column_ids", []))
    stable_over_platform = [
        item for item in vertical_interfaces if item["candidate_id"] in stable_ids
    ]
    roof_overlaps: list[str] = []
    component_s = component["longitudinal_range_m"]
    component_c = component["cross_range_m"]
    for roof in canopy.get("roof_components", []):
        roof_s = roof["longitudinal_range_m"]
        roof_c = roof["cross_range_m"]
        s_overlap = min(float(component_s[1]), float(roof_s[1])) - max(
            float(component_s[0]), float(roof_s[0])
        )
        c_overlap = min(float(component_c[1]), float(roof_c[1])) - max(
            float(component_c[0]), float(roof_c[0])
        )
        if s_overlap > 0 and c_overlap > 0:
            roof_overlaps.append(roof["id"])
    canopy_interface = {
        "stable_column_count_in_source": len(stable_ids),
        "stable_column_count_intersecting_platform": len(stable_over_platform),
        "roof_component_ids_overlapping_platform": roof_overlaps,
        "status": (
            "candidate_canopy_geometry_overlaps_platform"
            if stable_over_platform or roof_overlaps
            else "right_platform_canopy_interface_unresolved_no_stable_overlap"
        ),
    }

    catenary_intersections = [
        item for item in vertical_interfaces if item["predicted_class"] == "catenary_support"
    ]
    sign_evidence = [
        disposition
        for disposition in gate.get("dispositions", [])
        if "sign_or_fence" in str(disposition.get("conclusion", ""))
    ]
    issues: list[dict[str, Any]] = []
    if rail_interface["status"].startswith("fail"):
        issues.append(
            {
                "id": "PLATFORM-INTERFACE-ISSUE-001",
                "severity": "high",
                "type": "platform_edge_discontinuity",
                "action": "review_shared_sections_before_scene_integration",
            }
        )
    conflicts = [item for item in vertical_interfaces if item["semantic_conflict"]]
    for conflict in conflicts:
        issues.append(
            {
                "id": f"PLATFORM-INTERFACE-ISSUE-{len(issues) + 1:03d}",
                "severity": "high",
                "type": "vertical_geometry_contacts_platform_but_is_classified_false_positive",
                "candidate_id": conflict["candidate_id"],
                "action": "reopen_vertical_semantic_review_before_deleting_or_modeling_asset",
            }
        )
    if not stable_over_platform and not roof_overlaps:
        issues.append(
            {
                "id": f"PLATFORM-INTERFACE-ISSUE-{len(issues) + 1:03d}",
                "severity": "high",
                "type": "right_platform_canopy_support_and_roof_not_resolved",
                "action": "run_targeted_canopy_recovery_on_right_platform_or_leave_canopy_absent",
            }
        )
    issues.append(
        {
            "id": f"PLATFORM-INTERFACE-ISSUE-{len(issues) + 1:03d}",
            "severity": "medium",
            "type": "platform_wall_geometry_not_available",
            "action": "extract_vertical_platform_face_before_full_scene_merge",
        }
    )
    if sign_evidence:
        issues.append(
            {
                "id": f"PLATFORM-INTERFACE-ISSUE-{len(issues) + 1:03d}",
                "severity": "medium",
                "type": "sign_or_fence_has_photo_evidence_but_no_asset_geometry",
                "action": "retain_evidence_only_until_sign_or_fence_geometry_is_reconstructed",
            }
        )
    return {
        "schema_version": "railway.platform-interface-audit.v1",
        "project_id": platform.get("project_id"),
        "segment_id": platform["segment_id"],
        "platform_component_id": component_id,
        "rail_interface": rail_interface,
        "vertical_interface_count": len(vertical_interfaces),
        "vertical_interfaces": vertical_interfaces,
        "semantic_conflict_count": len(conflicts),
        "canopy_interface": canopy_interface,
        "catenary_support_intersection_count": len(catenary_intersections),
        "catenary_support_intersections": catenary_intersections,
        "sign_or_fence_evidence_count": len(sign_evidence),
        "sign_or_fence_evidence": sign_evidence,
        "issue_count": len(issues),
        "issues": issues,
        "settings": settings,
        "status": "review_required" if issues else "pass",
    }


def build_platform_interface_graph(report: dict[str, Any]) -> dict[str, Any]:
    platform_id = report["platform_component_id"]
    nodes: list[dict[str, Any]] = [{"id": platform_id, "node_type": "platform_surface_candidate"}]
    edges: list[dict[str, Any]] = []
    for interface in report["vertical_interfaces"]:
        nodes.append(
            {
                "id": interface["candidate_id"],
                "node_type": "vertical_platform_interface",
                "predicted_class": interface["predicted_class"],
                "semantic_conflict": interface["semantic_conflict"],
                "status": interface["status"],
            }
        )
        edges.append(
            {
                "id": f"PLATFORM-INTERFACE-EDGE-{len(edges) + 1:03d}",
                "type": "vertical_footprint_intersects_platform",
                "source_id": interface["candidate_id"],
                "target_id": platform_id,
                "base_vertical_offset_m": interface["base_vertical_offset_m"],
                "status": interface["status"],
            }
        )
    for issue in report["issues"]:
        nodes.append({**issue, "node_type": "platform_interface_issue"})
        edges.append(
            {
                "id": f"PLATFORM-INTERFACE-EDGE-{len(edges) + 1:03d}",
                "type": "platform_has_interface_issue",
                "source_id": platform_id,
                "target_id": issue["id"],
                "status": "review_required",
            }
        )
    return {
        "schema_version": "railway.platform-interface-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_model_or_registry_write",
    }


def _render_diagnostic(
    report: dict[str, Any], platform: dict[str, Any], vertical: dict[str, Any], output: Path
) -> None:
    component = next(
        item
        for item in platform["platform_components"]
        if item["id"] == report["platform_component_id"]
    )
    sections = platform_sections(component)
    image = Image.new("RGB", (1800, 1000), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 24),
        f"{report['segment_id']} PLATFORM INTERFACE AUDIT | issues {report['issue_count']}",
        fill="#eaf9ff",
        font=font,
    )
    draw.text(
        (30, 48),
        "Cyan=platform | grey=rails | red=semantic conflict | green=stable canopy | yellow=catenary",
        fill="#ff9f31",
        font=font,
    )
    panel = (30, 90, 1350, 950)
    sidebar = (1380, 90, 1770, 950)
    draw.rectangle(panel, outline="#28505e", width=2)
    draw.rectangle(sidebar, outline="#28505e", width=2)
    s_min = sections[0]["longitudinal_m"]
    s_max = sections[-1]["longitudinal_m"]
    c_values = [item["low_cross_m"] for item in sections] + [
        item["high_cross_m"] for item in sections
    ]
    c_values.extend(float(item["cross_position_m"]) for item in platform.get("rail_lines", []))
    c_values.extend(
        float(item["cross_position_m"])
        for item in vertical.get("candidates", [])
        if s_min <= float(item["longitudinal_position_m"]) <= s_max
        and -30.0 <= float(item["cross_position_m"]) <= 15.0
    )
    c_min, c_max = min(c_values), max(c_values)

    def pixel(longitudinal: float, cross: float) -> tuple[int, int]:
        x = panel[0] + 40 + int((longitudinal - s_min) / max(s_max - s_min, 1e-9) * 1240)
        y = panel[3] - 40 - int((cross - c_min) / max(c_max - c_min, 1e-9) * 780)
        return x, y

    polygon = [pixel(item["longitudinal_m"], item["low_cross_m"]) for item in sections]
    polygon.extend(
        pixel(item["longitudinal_m"], item["high_cross_m"]) for item in reversed(sections)
    )
    draw.polygon(polygon, fill="#184c57", outline="#73e6ff")
    for rail in platform.get("rail_lines", []):
        draw.line(
            (
                *pixel(s_min, float(rail["cross_position_m"])),
                *pixel(s_max, float(rail["cross_position_m"])),
            ),
            fill="#83949a",
            width=2,
        )
    stable = set(report.get("canopy_stable_ids", []))
    interface_by_id = {item["candidate_id"]: item for item in report["vertical_interfaces"]}
    for candidate in vertical.get("candidates", []):
        longitudinal = float(candidate["longitudinal_position_m"])
        cross = float(candidate["cross_position_m"])
        if not (s_min <= longitudinal <= s_max and c_min <= cross <= c_max):
            continue
        interface = interface_by_id.get(candidate["id"])
        if interface and interface["semantic_conflict"]:
            color, radius = "#ff4f62", 7
        elif candidate["id"] in stable:
            color, radius = "#b7ff3c", 5
        elif candidate["predicted_class"] == "catenary_support":
            color, radius = "#ffd44a", 6
        else:
            color, radius = "#7f8f94", 3
        x, y = pixel(longitudinal, cross)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    rail_stats = report["rail_interface"]["edge_to_nearest_rail_center_m"]
    lines = [
        f"rail clearance p50 {rail_stats['p50']:.3f} m",
        f"vertical interfaces {report['vertical_interface_count']}",
        f"semantic conflicts {report['semantic_conflict_count']}",
        f"stable canopy overlap {report['canopy_interface']['stable_column_count_intersecting_platform']}",
        f"roof overlap {len(report['canopy_interface']['roof_component_ids_overlapping_platform'])}",
        f"issues {report['issue_count']}",
    ]
    for index, text in enumerate(lines):
        draw.text((sidebar[0] + 20, sidebar[1] + 30 + index * 30), text, fill="#c1d5dc", font=font)
    for index, issue in enumerate(report["issues"]):
        draw.text(
            (sidebar[0] + 20, sidebar[1] + 250 + index * 70),
            f"{issue['severity'].upper()} {issue['type']}",
            fill="#ff9f31" if issue["severity"] == "medium" else "#ff4f62",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def audit_platform_interfaces(
    project: ProjectConfig,
    segment_id: str,
    mesh_build_report_path: str | Path | None = None,
    platform_report_path: str | Path | None = None,
    vertical_report_path: str | Path | None = None,
    canopy_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    mesh_build_path = (
        project.resolve(mesh_build_report_path)
        if mesh_build_report_path
        else reports / f"{segment_id}_platform_mesh_build.json"
    )
    if not mesh_build_path.is_file():
        raise FileNotFoundError(mesh_build_path)
    mesh_build = load_json(mesh_build_path)
    platform_path = (
        project.resolve(platform_report_path)
        if platform_report_path
        else Path(mesh_build["platform_surface_report"])
    )
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    canopy_path = (
        project.resolve(canopy_report_path)
        if canopy_report_path
        else reports / f"{segment_id}_canopy_structure.json"
    )
    gate_path = Path(mesh_build["platform_mesh_gate"])
    for path in (platform_path, vertical_path, canopy_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    configured = project.value.get("algorithms", {}).get("platform_interface_audit")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    platform = load_json(platform_path)
    vertical = load_json(vertical_path)
    canopy = load_json(canopy_path)
    report = audit_platform_interfaces_data(
        platform, vertical, canopy, load_json(gate_path), settings
    )
    report["canopy_stable_ids"] = canopy.get("stable_column_ids", [])
    output_report = reports / f"{segment_id}_platform_interface_audit.json"
    output_graph = reports / f"{segment_id}_platform_interface_graph.json"
    output_image = reports / f"{segment_id}_platform_interface_audit.png"
    for path in (output_report, output_graph, output_image):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
    report.update(
        {
            "platform_mesh_build_report": str(mesh_build_path),
            "platform_surface_report": str(platform_path),
            "vertical_hypotheses_report": str(vertical_path),
            "canopy_structure_report": str(canopy_path),
            "platform_mesh_gate": str(gate_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "output_report": str(output_report),
            "output_graph": str(output_graph),
            "output_diagnostic": str(output_image),
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_platform_interface_graph(report))
    _render_diagnostic(report, platform, vertical, output_image)
    return report
