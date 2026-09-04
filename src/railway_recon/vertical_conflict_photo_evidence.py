from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import load_camera_rows
from .canopy_photo_evidence import (
    _crop_equirectangular,
    _local_to_world,
    _permutation_matrix,
    _wrapped_delta,
)
from .config import ProjectConfig
from .io import load_json, write_json
from .projection import _canonical_vectors, project_equirectangular


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "vertical-conflict-photo-evidence.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def candidate_markers(candidate: dict[str, Any]) -> tuple[np.ndarray, tuple[str, ...]]:
    longitudinal = float(candidate["longitudinal_position_m"])
    cross = float(candidate["cross_position_m"])
    bottom = float(candidate["minimum_z"])
    top = float(candidate["maximum_z"])
    radius = max(float(candidate.get("footprint_m", 0.0)) / 2.0, 0.05)
    local = np.asarray(
        [
            [longitudinal, cross, bottom],
            [longitudinal, cross, (bottom + top) / 2.0],
            [longitudinal, cross, top],
            [longitudinal - radius, cross - radius, bottom],
            [longitudinal + radius, cross - radius, bottom],
            [longitudinal + radius, cross + radius, bottom],
            [longitudinal - radius, cross + radius, bottom],
        ],
        dtype=np.float64,
    )
    return local, (
        "base",
        "middle",
        "top",
        "footprint_0",
        "footprint_1",
        "footprint_2",
        "footprint_3",
    )


def _project_markers(
    local: np.ndarray,
    frame: dict[str, Any],
    camera_xyz: np.ndarray,
    camera: dict[str, Any],
    convention: dict[str, Any],
    permutation: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    world = _local_to_world(local, frame)
    canonical = _canonical_vectors(
        world - camera_xyz,
        np.asarray(camera["consensus_base_rotation"], dtype=np.float64),
        str(convention["pose_direction"]),
        permutation,
    )
    u, v = project_equirectangular(canonical, width, height)
    return u, v, float(np.linalg.norm(world[1] - camera_xyz))


def _candidate_crop(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, list[float]], float]:
    source_width = image.shape[1]
    center_u = float(u[1])
    center_v = float((v[0] + v[2]) / 2.0)
    crop, crop_center = _crop_equirectangular(
        image,
        center_u,
        center_v,
        int(settings["crop_width_px"]),
        int(settings["crop_height_px"]),
    )
    crop_markers = {
        name: [
            crop_center[0] + _wrapped_delta(float(u[index]), center_u, source_width),
            crop_center[1] + float(v[index] - center_v),
        ]
        for index, name in enumerate(
            (
                "base",
                "middle",
                "top",
                "footprint_0",
                "footprint_1",
                "footprint_2",
                "footprint_3",
            )
        )
    }
    line_length = math.dist(crop_markers["base"], crop_markers["top"])
    return crop, crop_markers, line_length


def analyze_vertical_conflict_photo_evidence_data(
    interface: dict[str, Any],
    vertical: dict[str, Any],
    consensus: dict[str, Any],
    camera_rows: list[dict[str, str]],
    settings: dict[str, Any],
    image_loader: Any,
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    candidate_by_id = {item["id"]: item for item in vertical["candidates"]}
    conflict_ids = [
        item["candidate_id"]
        for item in interface["vertical_interfaces"]
        if item["semantic_conflict"]
    ]
    convention = consensus["best_shared_convention"]
    permutation = _permutation_matrix(str(convention["canonical_axes_from_local"]))
    rows = {int(row["index"]): row for row in camera_rows}
    results: list[dict[str, Any]] = []
    crops: dict[str, dict[int, np.ndarray]] = {}
    for candidate_id in conflict_ids:
        if candidate_id not in candidate_by_id:
            raise ValueError(f"Interface candidate missing from vertical report: {candidate_id}")
        candidate = candidate_by_id[candidate_id]
        local, _ = candidate_markers(candidate)
        views: list[dict[str, Any]] = []
        candidate_crops: dict[int, np.ndarray] = {}
        for camera in consensus["per_camera"]:
            camera_index = int(camera["camera_index"])
            if camera_index not in rows:
                raise ValueError(f"Camera {camera_index} missing from camera CSV")
            image = np.asarray(image_loader(Path(camera["photo"])).convert("RGB"))
            height, width = image.shape[:2]
            xyz = np.asarray(
                [float(rows[camera_index][key]) for key in ("x", "y", "z")],
                dtype=np.float64,
            )
            u, v, distance = _project_markers(
                local,
                vertical["frame"],
                xyz,
                camera,
                convention,
                permutation,
                width,
                height,
            )
            crop, markers, line_length = _candidate_crop(image, u, v, settings)
            trusted = bool(camera["local_best_matches_consensus"])
            reviewable = distance <= float(
                settings["maximum_camera_distance_m"]
            ) and line_length >= float(settings["minimum_projected_height_px"])
            views.append(
                {
                    "camera_index": camera_index,
                    "photo": camera["photo"],
                    "trusted_for_consensus": trusted,
                    "camera_distance_m": distance,
                    "projected_height_px": line_length,
                    "reviewable_resolution": reviewable,
                    "projected_uv": {
                        "base": [float(u[0]), float(v[0])],
                        "top": [float(u[2]), float(v[2])],
                    },
                    "crop_markers": markers,
                    "status": "photo_review_window_not_semantic_confirmation",
                }
            )
            candidate_crops[camera_index] = crop
        trusted_reviewable = [
            view
            for view in views
            if view["trusted_for_consensus"] and view["reviewable_resolution"]
        ]
        results.append(
            {
                "candidate_id": candidate_id,
                "original_predicted_class": candidate["predicted_class"],
                "original_confidence": candidate["confidence"],
                "longitudinal_position_m": float(candidate["longitudinal_position_m"]),
                "cross_position_m": float(candidate["cross_position_m"]),
                "minimum_z_m": float(candidate["minimum_z"]),
                "maximum_z_m": float(candidate["maximum_z"]),
                "height_m": float(candidate["maximum_z"] - candidate["minimum_z"]),
                "footprint_m": float(candidate.get("footprint_m", 0.0)),
                "trusted_reviewable_view_count": len(trusted_reviewable),
                "photo_review_result": (
                    "multi_view_photo_review_available"
                    if len(trusted_reviewable) >= int(settings["minimum_trusted_reviewable_views"])
                    else "insufficient_trusted_photo_review_views"
                ),
                "views": views,
                "status": "manual_semantic_review_required",
            }
        )
        crops[candidate_id] = candidate_crops
    return (
        {
            "schema_version": "railway.vertical-conflict-photo-evidence.v1",
            "project_id": interface.get("project_id"),
            "segment_id": interface["segment_id"],
            "candidate_count": len(results),
            "camera_count": len(consensus["per_camera"]),
            "trusted_camera_count": sum(
                bool(item["local_best_matches_consensus"]) for item in consensus["per_camera"]
            ),
            "candidates": results,
            "settings": settings,
            "status": "photo_review_packages_generated_manual_semantic_review_required",
            "limitations": [
                "Projection identifies where to review; it does not classify the object.",
                "Temporary machinery, signs, columns and vegetation may have similar vertical geometry.",
                "This stage never changes geometry, the vertical report, or the asset registry.",
            ],
        },
        crops,
    )


def build_vertical_conflict_photo_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        nodes.append(
            {
                "id": candidate["candidate_id"],
                "node_type": "vertical_semantic_conflict",
                "photo_review_result": candidate["photo_review_result"],
                "status": candidate["status"],
            }
        )
        for view in candidate["views"]:
            view_id = f"{candidate['candidate_id']}-CAM-{view['camera_index']}"
            nodes.append(
                {
                    "id": view_id,
                    "node_type": "vertical_conflict_photo_observation",
                    "camera_index": view["camera_index"],
                    "trusted_for_consensus": view["trusted_for_consensus"],
                    "reviewable_resolution": view["reviewable_resolution"],
                    "status": view["status"],
                }
            )
            edges.append(
                {
                    "id": f"VERTICAL-CONFLICT-PHOTO-EDGE-{len(edges) + 1:03d}",
                    "type": "candidate_has_photo_review_window",
                    "source_id": candidate["candidate_id"],
                    "target_id": view_id,
                    "status": "candidate_evidence_only",
                }
            )
    return {
        "schema_version": "railway.vertical-conflict-photo-evidence-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_model_registry_or_prediction_write",
    }


def _draw_cross(draw: ImageDraw.ImageDraw, point: list[float], color: str) -> None:
    x, y = point
    draw.line((x - 9, y, x + 9, y), fill=color, width=3)
    draw.line((x, y - 9, x, y + 9), fill=color, width=3)


def _render_candidate_sheet(
    candidate: dict[str, Any], crops: dict[int, np.ndarray], output: Path, settings: dict[str, Any]
) -> None:
    tile_width = int(settings["sheet_tile_width_px"])
    tile_height = int(settings["sheet_tile_height_px"])
    sheet = Image.new("RGB", (tile_width * len(candidate["views"]), tile_height + 70), "#06141d")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (12, 10),
        f"{candidate['candidate_id']} | original={candidate['original_predicted_class']} | H={candidate['height_m']:.2f} m | footprint={candidate['footprint_m']:.2f} m",
        fill="#eaf9ff",
        font=font,
    )
    draw.text(
        (12, 34),
        "orange=full vertical extent | magenta=base footprint | green=base | cyan=top",
        fill="#9bb1b9",
        font=font,
    )
    for tile_index, view in enumerate(candidate["views"]):
        annotated = Image.fromarray(crops[int(view["camera_index"])], mode="RGB")
        layer = ImageDraw.Draw(annotated)
        markers = view["crop_markers"]
        layer.line((*markers["base"], *markers["top"]), fill="#ff9f31", width=5)
        footprint = [markers[f"footprint_{index}"] for index in range(4)]
        layer.line([*footprint, footprint[0]], fill="#ff38a5", width=4)
        _draw_cross(layer, markers["base"], "#b7ff3c")
        _draw_cross(layer, markers["top"], "#24d7ff")
        annotated.thumbnail((tile_width, tile_height - 42), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "#06141d")
        tile.paste(annotated, ((tile_width - annotated.width) // 2, 37))
        tile_draw = ImageDraw.Draw(tile)
        trust = "trusted" if view["trusted_for_consensus"] else "reference"
        resolution = "reviewable" if view["reviewable_resolution"] else "low-res"
        tile_draw.text(
            (8, 8),
            f"cam {view['camera_index']} | {trust} | {resolution} | {view['camera_distance_m']:.1f} m | {view['projected_height_px']:.0f} px",
            fill="#b7ff3c" if view["reviewable_resolution"] else "#9bb1b9",
            font=font,
        )
        sheet.paste(tile, (tile_index * tile_width, 70))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def _render_summary(report: dict[str, Any], output: Path) -> None:
    image_height = max(700, 130 + len(report["candidates"]) * 180)
    image = Image.new("RGB", (1600, image_height), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 22),
        f"{report['segment_id']} VERTICAL SEMANTIC CONFLICT PHOTO REVIEW",
        fill="#eaf9ff",
        font=font,
    )
    draw.text(
        (30, 46),
        "Projection windows support review only; no class, geometry or registry record is changed.",
        fill="#ff9f31",
        font=font,
    )
    for index, candidate in enumerate(report["candidates"]):
        y = 100 + index * 180
        draw.rectangle((40, y, 1560, y + 145), outline="#28505e", width=2)
        draw.text((60, y + 18), candidate["candidate_id"], fill="#b7ff3c", font=font)
        draw.text(
            (350, y + 18),
            f"S {candidate['longitudinal_position_m']:.2f} | C {candidate['cross_position_m']:.2f} | H {candidate['height_m']:.2f} m",
            fill="#c1d5dc",
            font=font,
        )
        draw.text(
            (60, y + 50),
            f"trusted reviewable views {candidate['trusted_reviewable_view_count']} | {candidate['photo_review_result']}",
            fill="#c1d5dc",
            font=font,
        )
        for view_index, view in enumerate(candidate["views"]):
            x = 60 + view_index * 250
            color = "#b7ff3c" if view["reviewable_resolution"] else "#52676e"
            draw.text(
                (x, y + 92),
                f"cam {view['camera_index']} / {view['projected_height_px']:.0f}px / {view['camera_distance_m']:.0f}m",
                fill=color,
                font=font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_vertical_conflict_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    projection_consensus_path: str | Path,
    interface_report_path: str | Path | None = None,
    vertical_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    interface_path = (
        project.resolve(interface_report_path)
        if interface_report_path
        else reports / f"{segment_id}_platform_interface_audit.json"
    )
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    consensus_path = project.resolve(projection_consensus_path)
    for path in (interface_path, vertical_path, consensus_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    configured = project.value.get("algorithms", {}).get("vertical_conflict_photo_evidence")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    consensus = load_json(consensus_path)
    cache: dict[Path, Image.Image] = {}

    def load_image(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path]

    report, crops = analyze_vertical_conflict_photo_evidence_data(
        load_json(interface_path),
        load_json(vertical_path),
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    output_report = reports / f"{segment_id}_vertical_conflict_photo_evidence.json"
    output_graph = reports / f"{segment_id}_vertical_conflict_photo_evidence_graph.json"
    output_summary = reports / f"{segment_id}_vertical_conflict_photo_evidence.png"
    output_directory = reports / f"{segment_id}_vertical_conflict_photo_evidence"
    for path in (output_report, output_graph, output_summary):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for candidate in report["candidates"]:
        sheet = output_directory / f"{candidate['candidate_id']}.jpg"
        _render_candidate_sheet(candidate, crops[candidate["candidate_id"]], sheet, settings)
        candidate["evidence_sheet"] = str(sheet)
    report.update(
        {
            "interface_audit_report": str(interface_path),
            "vertical_hypotheses_report": str(vertical_path),
            "projection_consensus_report": str(consensus_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "output_report": str(output_report),
            "output_graph": str(output_graph),
            "output_summary": str(output_summary),
            "output_evidence_directory": str(output_directory),
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_vertical_conflict_photo_graph(report))
    _render_summary(report, output_summary)
    for image in cache.values():
        image.close()
    return report


def analyze_selected_vertical_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    candidate_ids: list[str],
    projection_consensus_path: str | Path,
    output_name: str,
    vertical_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build immutable photo-review sheets for explicitly selected hypotheses."""
    if not candidate_ids:
        raise ValueError("At least one vertical candidate ID is required")
    if Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be one file-safe basename")
    unique_ids = list(dict.fromkeys(str(value) for value in candidate_ids))
    reports = project.workspace_path("reports")
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    consensus_path = project.resolve(projection_consensus_path)
    for path in (vertical_path, consensus_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    configured = project.value.get("algorithms", {}).get(
        "vertical_conflict_photo_evidence"
    )
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    vertical = load_json(vertical_path)
    available_ids = {str(item["id"]) for item in vertical.get("candidates", [])}
    missing = sorted(set(unique_ids) - available_ids)
    if missing:
        raise ValueError(f"Selected vertical candidates are absent: {missing}")
    consensus = load_json(consensus_path)
    interface = {
        "project_id": project.project_id,
        "segment_id": segment_id,
        "vertical_interfaces": [
            {"candidate_id": candidate_id, "semantic_conflict": True}
            for candidate_id in unique_ids
        ],
    }
    cache: dict[Path, Image.Image] = {}

    def load_image(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path]

    report, crops = analyze_vertical_conflict_photo_evidence_data(
        interface,
        vertical,
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    output_report = reports / f"{output_name}.json"
    output_graph = reports / f"{output_name}_graph.json"
    output_summary = reports / f"{output_name}.png"
    output_directory = reports / output_name
    for path in (output_report, output_graph, output_summary):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for candidate in report["candidates"]:
        sheet = output_directory / f"{candidate['candidate_id']}.jpg"
        _render_candidate_sheet(candidate, crops[candidate["candidate_id"]], sheet, settings)
        candidate["evidence_sheet"] = str(sheet)
    report.update(
        {
            "selection_mode": "explicit_candidate_ids",
            "selected_candidate_ids": unique_ids,
            "vertical_hypotheses_report": str(vertical_path),
            "projection_consensus_report": str(consensus_path),
            "settings_source": (
                str(resolved_settings) if resolved_settings else "bundled_default"
            ),
            "output_report": str(output_report),
            "output_graph": str(output_graph),
            "output_summary": str(output_summary),
            "output_evidence_directory": str(output_directory),
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_vertical_conflict_photo_graph(report))
    _render_summary(report, output_summary)
    for image in cache.values():
        image.close()
    return report
