from __future__ import annotations

import json
import os
import re
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .algorithms.mesh import ObjWriter
from .config import ProjectConfig
from .geometry import CorridorFrame
from .io import load_json, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath("platform-mesh.default.json")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_prefix(value: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", value.upper())


def validate_platform_mesh_gate(gate: dict[str, Any], segment_id: str, component_id: str) -> None:
    if gate.get("schema_version") != "railway.platform-mesh-gate.v1":
        raise ValueError("Unsupported platform mesh gate schema")
    if gate.get("segment_id") != segment_id:
        raise ValueError("Platform mesh gate belongs to a different segment")
    if gate.get("platform_component_id") != component_id:
        raise ValueError("Platform mesh gate belongs to a different platform component")
    if int(gate.get("unresolved_gap_count", -1)) != 0:
        raise ValueError("Platform mesh gate still contains unresolved gaps")
    if not str(gate.get("mesh_gate", "")).startswith("passed_"):
        raise ValueError("Platform mesh gate has not passed")


def _fits_at_boundary(component: dict[str, Any], longitudinal: float) -> list[dict[str, Any]]:
    fits = list(component["fit_segments"])
    tolerance = 1e-8
    selected = [
        fit
        for fit in fits
        if float(fit["longitudinal_range_m"][0]) - tolerance
        <= longitudinal
        <= float(fit["longitudinal_range_m"][1]) + tolerance
    ]
    if selected:
        return selected
    return [
        min(
            fits,
            key=lambda fit: abs(
                longitudinal
                - (float(fit["longitudinal_range_m"][0]) + float(fit["longitudinal_range_m"][1]))
                / 2.0
            ),
        )
    ]


def _section_elevation(fits: list[dict[str, Any]], longitudinal: float, cross: float) -> float:
    values = []
    for fit in fits:
        a, b, d = (float(value) for value in fit["plane_z_equals_a_s_plus_b_c_plus_d"])
        values.append(a * longitudinal + b * cross + d)
    return float(np.mean(values))


def platform_sections(component: dict[str, Any]) -> list[dict[str, float]]:
    """Build shared section boundaries so adjacent 5 m fits cannot crack or overlap."""
    component_start, component_end = (float(value) for value in component["longitudinal_range_m"])
    boundaries = {component_start, component_end}
    for fit in component["fit_segments"]:
        for value in fit["longitudinal_range_m"]:
            value = float(value)
            if component_start < value < component_end:
                boundaries.add(value)
    result: list[dict[str, float]] = []
    for longitudinal in sorted(boundaries):
        fits = _fits_at_boundary(component, longitudinal)
        rail_cross = float(np.mean([float(fit["rail_side_edge_cross_m"]) for fit in fits]))
        outer_cross_values = []
        for fit in fits:
            cross_range = [float(value) for value in fit["observed_cross_range_m"]]
            outer_cross_values.append(
                cross_range[0] if component["side"] == "left" else cross_range[1]
            )
        outer_cross = float(np.mean(outer_cross_values))

        low_cross, high_cross = sorted((rail_cross, outer_cross))
        result.append(
            {
                "longitudinal_m": longitudinal,
                "low_cross_m": low_cross,
                "high_cross_m": high_cross,
                "low_z_m": _section_elevation(fits, longitudinal, low_cross),
                "high_z_m": _section_elevation(fits, longitudinal, high_cross),
                "contributing_fit_count": float(len(fits)),
            }
        )
    if len(result) < 2:
        raise ValueError("Platform component needs at least two shared sections")
    if any(
        current["longitudinal_m"] <= previous["longitudinal_m"]
        for previous, current in pairwise(result)
    ):
        raise ValueError("Platform shared sections are not strictly increasing")
    return result


def build_platform_component_mesh(
    component: dict[str, Any], frame: CorridorFrame, thickness_m: float
) -> tuple[
    np.ndarray,
    list[tuple[int, ...]],
    np.ndarray,
    list[tuple[int, ...]],
    list[dict[str, float]],
]:
    if thickness_m <= 0:
        raise ValueError("Platform inferred thickness must be positive")
    sections = platform_sections(component)
    top_vertices: list[list[float]] = []
    for section in sections:
        for cross_key, z_key in (("low_cross_m", "low_z_m"), ("high_cross_m", "high_z_m")):
            xy = frame.world_xy(
                np.asarray([section["longitudinal_m"]]),
                np.asarray([section[cross_key]]),
            )[0]
            top_vertices.append([float(xy[0]), float(xy[1]), section[z_key]])
    top = np.asarray(top_vertices, dtype=np.float64)
    top_faces = [
        (2 * index, 2 * index + 2, 2 * index + 3, 2 * index + 1)
        for index in range(len(sections) - 1)
    ]

    bottom = top.copy()
    bottom[:, 2] -= thickness_m
    shell = np.vstack((top, bottom))
    bottom_offset = len(top)
    shell_faces: list[tuple[int, ...]] = []
    for index in range(len(sections) - 1):
        low_a, high_a = 2 * index, 2 * index + 1
        low_b, high_b = 2 * index + 2, 2 * index + 3
        bottom_low_a, bottom_high_a = bottom_offset + low_a, bottom_offset + high_a
        bottom_low_b, bottom_high_b = bottom_offset + low_b, bottom_offset + high_b
        shell_faces.extend(
            (
                (bottom_low_a, bottom_high_a, bottom_high_b, bottom_low_b),
                (low_a, bottom_low_a, bottom_low_b, low_b),
                (high_a, high_b, bottom_high_b, bottom_high_a),
            )
        )
    final_low, final_high = len(top) - 2, len(top) - 1
    shell_faces.extend(
        (
            (0, 1, bottom_offset + 1, bottom_offset),
            (
                final_low,
                bottom_offset + final_low,
                bottom_offset + final_high,
                final_high,
            ),
        )
    )
    return top, top_faces, shell, shell_faces, sections


def _write_materials(path: Path) -> None:
    content = """# Evidence-aware platform materials
newmtl PlatformTopObserved
Kd 0.32 0.68 0.75
Ks 0.08 0.08 0.08
Ns 10

newmtl PlatformVolumeInferred
Kd 0.32 0.38 0.40
Ks 0.03 0.03 0.03
Ns 4
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _render_diagnostic(
    component: dict[str, Any], sections: list[dict[str, float]], output: Path
) -> None:
    image = Image.new("RGB", (1600, 800), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 24), f"{component['id']} continuous platform candidate mesh", fill="#eaf9ff", font=font
    )
    draw.text(
        (30, 48),
        "Cyan: shared-section observed top. Grey volume: rule-inferred thickness. No point gaps are cut.",
        fill="#ff9f31",
        font=font,
    )
    panel = (50, 100, 1550, 650)
    draw.rectangle(panel, outline="#28505e", width=2)
    s_min = sections[0]["longitudinal_m"]
    s_max = sections[-1]["longitudinal_m"]
    c_min = min(section["low_cross_m"] for section in sections)
    c_max = max(section["high_cross_m"] for section in sections)

    def pixel(longitudinal: float, cross: float) -> tuple[int, int]:
        x = panel[0] + 35 + int((longitudinal - s_min) / max(s_max - s_min, 1e-9) * 1430)
        y = panel[3] - 35 - int((cross - c_min) / max(c_max - c_min, 1e-9) * 480)
        return x, y

    polygon = [pixel(item["longitudinal_m"], item["low_cross_m"]) for item in sections]
    polygon.extend(
        pixel(item["longitudinal_m"], item["high_cross_m"]) for item in reversed(sections)
    )
    draw.polygon(polygon, fill="#205b68", outline="#73e6ff")
    for section in sections:
        draw.line(
            (
                *pixel(section["longitudinal_m"], section["low_cross_m"]),
                *pixel(section["longitudinal_m"], section["high_cross_m"]),
            ),
            fill="#8ba4ab",
            width=1,
        )
    draw.text(
        (50, 690),
        f"sections={len(sections)} | spans={len(sections) - 1} | S={s_min:.3f}..{s_max:.3f} m | unresolved openings=0",
        fill="#b7ff3c",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def build_platform_mesh(
    project: ProjectConfig,
    segment_id: str,
    mesh_gate_path: str | Path,
    overwrite: bool = False,
    platform_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    platform_path = (
        project.resolve(platform_report_path)
        if platform_report_path
        else reports / f"{segment_id}_platform_surface.json"
    )
    gate_path = project.resolve(mesh_gate_path)
    for path in (platform_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    platform = load_json(platform_path)
    gate = load_json(gate_path)
    component_id = str(gate.get("platform_component_id"))
    component = next(
        (item for item in platform["platform_components"] if item["id"] == component_id),
        None,
    )
    if component is None:
        raise ValueError(f"Mesh gate component is absent from platform report: {component_id}")
    validate_platform_mesh_gate(gate, segment_id, component_id)

    configured = project.value.get("algorithms", {}).get("platform_mesh")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    frame = CorridorFrame.from_json(platform["frame"])
    top, top_faces, shell, shell_faces, sections = build_platform_component_mesh(
        component, frame, float(settings["inferred_platform_thickness_m"])
    )

    output_dir = project.workspace_path("exports") / segment_id / "platform_candidate"
    obj_path = output_dir / "platform_candidate.obj"
    mtl_path = output_dir / "platform_candidate.mtl"
    origin_path = output_dir / "model_origin.json"
    report_path = reports / f"{segment_id}_platform_mesh_build.json"
    audit_path = reports / f"{segment_id}_platform_mesh_audit.json"
    diagnostic_path = reports / f"{segment_id}_platform_mesh.png"
    for path in (obj_path, mtl_path, origin_path, report_path, audit_path, diagnostic_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    origin = np.asarray(
        [(top[:, 0].min() + top[:, 0].max()) / 2.0, (top[:, 1].min() + top[:, 1].max()) / 2.0, 0.0]
    )
    writer = ObjWriter(origin, material_library=mtl_path.name)
    prefix = _safe_prefix(segment_id)
    top_id = f"{prefix}-PLATFORM-001-TOP"
    volume_id = f"{prefix}-PLATFORM-001-VOLUME"
    writer.add_mesh(top_id, top, top_faces, "PlatformTopObserved")
    writer.add_mesh(volume_id, shell, shell_faces, "PlatformVolumeInferred")
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(origin_path, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})

    mesh_audit = audit_obj(obj_path)
    mesh_audit.update(
        {
            "shared_section_count": len(sections),
            "span_count": len(sections) - 1,
            "confirmed_opening_count": int(gate["confirmed_opening_count"]),
            "unresolved_gap_count": int(gate["unresolved_gap_count"]),
            "internal_section_crack_count_by_construction": 0,
            "status": "pass" if mesh_audit["passed"] else "fail",
        }
    )
    write_json(audit_path, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Generated platform OBJ failed mesh audit")

    registry_path = project.workspace_path("asset_registry")
    registry = (
        load_json(registry_path) if registry_path.is_file() else new_registry(project.project_id)
    )
    asset_ids = {top_id, volume_id}
    conflicts = asset_ids & {str(item["id"]) for item in registry.get("assets", [])}
    if conflicts and not overwrite:
        raise ValueError(f"Asset registry already contains platform mesh ids: {sorted(conflicts)}")
    registry["assets"] = [
        item for item in registry.get("assets", []) if str(item["id"]) not in asset_ids
    ]
    registry["assets"].extend(
        [
            {
                "id": top_id,
                "type": "platform_surface",
                "subtype": "segmented_point_cloud_fit",
                "status": "candidate",
                "chainage_m": None,
                "evidence_level": "observed",
                "confidence": float(settings["observed_top_confidence"]),
                "sources": [
                    {"kind": "point_cloud", "reference": str(platform_path)},
                    {"kind": "manual_review", "reference": str(gate_path)},
                ],
                "parameters": {
                    "shared_section_count": len(sections),
                    "longitudinal_range_m": component["longitudinal_range_m"],
                    "cross_range_m": component["cross_range_m"],
                    "gap_cutout_count": 0,
                },
                "geometry": {"node": top_id, "file": str(obj_path)},
                "limitations": [
                    "Candidate top surface follows segmented point-cloud fits.",
                    "Only the observed right-side platform component is represented.",
                ],
            },
            {
                "id": volume_id,
                "type": "platform_volume",
                "subtype": "rule_inferred_skirt_and_bottom",
                "status": "candidate",
                "chainage_m": None,
                "evidence_level": "rule_inferred",
                "confidence": float(settings["inferred_volume_confidence"]),
                "sources": [
                    {"kind": "rule", "reference": "platform_mesh.inferred_platform_thickness_m"},
                    {"kind": "manual_review", "reference": str(gate_path)},
                ],
                "parameters": {
                    "thickness_m": float(settings["inferred_platform_thickness_m"]),
                    "top_asset_id": top_id,
                },
                "geometry": {"node": volume_id, "file": str(obj_path)},
                "limitations": [
                    "Side walls and bottom are visualization geometry, not measured structural thickness.",
                    "Do not use inferred volume for construction or clearance measurements.",
                ],
            },
        ]
    )
    relation_id = f"{prefix}-PLATFORM-001-TOP-SUPPORTED-BY-VOLUME"
    registry["relations"] = [
        item for item in registry.get("relations", []) if item.get("id") != relation_id
    ]
    registry["relations"].append(
        {"id": relation_id, "type": "supported_by", "from": top_id, "to": volume_id}
    )
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Generated platform registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)
    _render_diagnostic(component, sections, diagnostic_path)

    report = {
        "schema_version": "railway.platform-mesh-build.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "platform_component_id": component_id,
        "platform_surface_report": str(platform_path),
        "platform_mesh_gate": str(gate_path),
        "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "output_origin": str(origin_path),
        "output_diagnostic": str(diagnostic_path),
        "output_mesh_audit": str(audit_path),
        "asset_registry": str(registry_path),
        "asset_ids": [top_id, volume_id],
        "shared_section_count": len(sections),
        "span_count": len(sections) - 1,
        "vertex_count": writer.vertex_count,
        "face_count": writer.face_count,
        "confirmed_opening_count": int(gate["confirmed_opening_count"]),
        "unresolved_gap_count": int(gate["unresolved_gap_count"]),
        "left_platform_action": "not_generated_insufficient_point_cloud_support",
        "status": "candidate_platform_mesh_generated_review_required",
    }
    write_json(report_path, report)
    return report
