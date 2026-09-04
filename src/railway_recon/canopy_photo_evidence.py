from __future__ import annotations

import json
import math
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel

from .camera import load_camera_rows
from .config import ProjectConfig
from .io import load_json, write_json
from .projection import _canonical_vectors, project_equirectangular, signed_permutations


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath("canopy-photo-evidence.default.json")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _permutation_matrix(label: str) -> np.ndarray:
    try:
        return dict(signed_permutations())[label]
    except KeyError as exc:
        raise ValueError(f"Unknown canonical-axis permutation: {label}") from exc


def _local_to_world(values: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    local = np.asarray(values, dtype=np.float64)
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = origin + local[:, 0, None] * along + local[:, 1, None] * cross
    return np.column_stack((xy, local[:, 2]))


def _wrapped_delta(value: float, center: float, period: int) -> float:
    return float((value - center + period / 2.0) % period - period / 2.0)


def _crop_equirectangular(
    image: np.ndarray, center_u: float, center_v: float, width: int, height: int
) -> tuple[np.ndarray, tuple[float, float]]:
    source_height, source_width = image.shape[:2]
    start_x = math.floor(center_u - width / 2.0)
    start_y = math.floor(center_v - height / 2.0)
    x_indexes = np.mod(np.arange(start_x, start_x + width), source_width)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    source_y0 = max(0, start_y)
    source_y1 = min(source_height, start_y + height)
    if source_y1 > source_y0:
        destination_y0 = source_y0 - start_y
        destination_y1 = destination_y0 + source_y1 - source_y0
        result[destination_y0:destination_y1] = image[source_y0:source_y1][:, x_indexes]
    return result, (width / 2.0, center_v - start_y)


def directional_edge_evidence(
    crop: np.ndarray,
    center: tuple[float, float],
    transverse_start: tuple[float, float],
    transverse_end: tuple[float, float],
    roi_radius_px: int,
) -> dict[str, float | int | bool]:
    """Measure image edges aligned with the projected transverse direction."""
    vector = np.asarray(transverse_end) - np.asarray(transverse_start)
    span = float(np.linalg.norm(vector))
    if span <= 1e-9:
        return _invalid_edge(span)
    tangent = vector / span
    normal = np.asarray([-tangent[1], tangent[0]])
    gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]) / 255.0
    gx = sobel(gray, axis=1, mode="reflect") / 8.0
    gy = sobel(gray, axis=0, mode="reflect") / 8.0
    directional = np.abs(gx * normal[0] + gy * normal[1])
    tangential = np.abs(gx * tangent[0] + gy * tangent[1])
    cx, cy = round(center[0]), round(center[1])
    x0, x1 = max(0, cx - roi_radius_px), min(crop.shape[1], cx + roi_radius_px + 1)
    y0, y1 = max(0, cy - roi_radius_px), min(crop.shape[0], cy + roi_radius_px + 1)
    if x1 <= x0 or y1 <= y0:
        return _invalid_edge(span)
    directional_roi = directional[y0:y1, x0:x1]
    tangential_roi = tangential[y0:y1, x0:x1]
    d_sum, t_sum = float(directional_roi.sum()), float(tangential_roi.sum())
    return {
        "valid": True,
        "projected_transverse_span_px": span,
        "directional_response": float(np.percentile(directional_roi, 90.0)),
        "directional_orientation_ratio": d_sum / max(d_sum + t_sum, 1e-12),
        "roi_pixel_count": int(directional_roi.size),
    }


def _invalid_edge(span: float) -> dict[str, float | int | bool]:
    return {
        "valid": False,
        "projected_transverse_span_px": span,
        "directional_response": 0.0,
        "directional_orientation_ratio": 0.0,
        "roi_pixel_count": 0,
    }


def _project_markers(
    local_points: np.ndarray,
    frame: dict[str, Any],
    camera: dict[str, Any],
    row: dict[str, str],
    convention: dict[str, Any],
    permutation: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    world = _local_to_world(local_points, frame)
    camera_xyz = np.asarray([float(row[key]) for key in ("x", "y", "z")])
    canonical = _canonical_vectors(
        world - camera_xyz,
        np.asarray(camera["consensus_base_rotation"]),
        str(convention["pose_direction"]),
        permutation,
    )
    u, v = project_equirectangular(canonical, width, height)
    return u, v, float(np.linalg.norm(world[0] - camera_xyz))


def analyze_canopy_photo_evidence_data(
    canopy: dict[str, Any],
    vertical: dict[str, Any],
    consensus: dict[str, Any],
    camera_rows: list[dict[str, str]],
    settings: dict[str, Any],
    image_loader: Callable[[Path], Image.Image],
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    convention = consensus["best_shared_convention"]
    permutation = _permutation_matrix(str(convention["canonical_axes_from_local"]))
    rows = {int(row["index"]): row for row in camera_rows}
    candidate_ids = {item["id"] for item in vertical["candidates"]}
    half_span = float(settings["transverse_probe_half_span_m"])
    below = float(settings["column_probe_below_top_m"])
    columns: list[dict[str, Any]] = []
    crops: dict[str, dict[int, np.ndarray]] = {}
    for contact in canopy["column_roof_contacts"]:
        column_id = str(contact["column_id"])
        if column_id not in candidate_ids:
            raise ValueError(f"Canopy column missing from vertical report: {column_id}")
        top = np.asarray(contact["column_top"], dtype=np.float64)
        roof = np.asarray(contact["nearest_roof_point"], dtype=np.float64)
        local = np.asarray(
            [
                top,
                [top[0], top[1], top[2] - below],
                [top[0], top[1] - half_span, top[2]],
                [top[0], top[1] + half_span, top[2]],
                roof,
            ]
        )
        views: list[dict[str, Any]] = []
        column_crops: dict[int, np.ndarray] = {}
        for camera in consensus["per_camera"]:
            index = int(camera["camera_index"])
            if index not in rows:
                raise ValueError(f"Camera {index} missing from consensus camera CSV")
            image = np.asarray(image_loader(Path(camera["photo"])).convert("RGB"))
            height, width = image.shape[:2]
            u, v, distance = _project_markers(
                local, canopy["frame"], camera, rows[index], convention, permutation, width, height
            )
            crop, center = _crop_equirectangular(
                image,
                float(u[0]),
                float(v[0]),
                int(settings["crop_width_px"]),
                int(settings["crop_height_px"]),
            )
            mx = [center[0] + _wrapped_delta(float(value), float(u[0]), width) for value in u]
            my = [center[1] + float(value - v[0]) for value in v]
            marker_names = (
                "column_top",
                "column_below_top",
                "transverse_start",
                "transverse_end",
                "nearest_roof_point",
            )
            markers = {name: [mx[i], my[i]] for i, name in enumerate(marker_names)}
            evidence = directional_edge_evidence(
                crop,
                center,
                tuple(markers["transverse_start"]),
                tuple(markers["transverse_end"]),
                int(settings["edge_roi_radius_px"]),
            )
            trusted = bool(camera["local_best_matches_consensus"])
            valid = (
                bool(evidence["valid"])
                and distance <= float(settings["maximum_camera_distance_m"])
                and float(evidence["projected_transverse_span_px"])
                >= float(settings["minimum_transverse_probe_span_px"])
            )
            supported = (
                valid
                and float(evidence["directional_response"])
                >= float(settings["directional_response_threshold"])
                and float(evidence["directional_orientation_ratio"])
                >= float(settings["directional_orientation_ratio_threshold"])
            )
            views.append(
                {
                    "camera_index": index,
                    "photo": camera["photo"],
                    "camera_distance_m": distance,
                    "trusted_for_consensus": trusted,
                    "projection_valid": valid,
                    "projected_uv": {
                        "column_top": [float(u[0]), float(v[0])],
                        "nearest_roof_point": [float(u[4]), float(v[4])],
                    },
                    "crop_markers": markers,
                    **evidence,
                    "edge_supported": supported,
                    "status": "photo_edge_observation_not_beam_confirmation",
                }
            )
            column_crops[index] = crop
        trusted_views = [
            view for view in views if view["trusted_for_consensus"] and view["projection_valid"]
        ]
        supported_views = [view for view in trusted_views if view["edge_supported"]]
        if len(trusted_views) < int(settings["minimum_trusted_views"]):
            result = "insufficient_trusted_photo_views"
        elif len(supported_views) >= int(settings["minimum_supported_views"]):
            result = "multi_view_transverse_edge_supported_not_beam_confirmed"
        else:
            result = "multi_view_transverse_edge_not_supported"
        columns.append(
            {
                "column_id": column_id,
                "longitudinal_position_m": float(top[0]),
                "cross_position_m": float(top[1]),
                "column_top_z_m": float(top[2]),
                "trusted_valid_view_count": len(trusted_views),
                "trusted_supported_view_count": len(supported_views),
                "multi_view_result": result,
                "views": views,
                "status": "automatic_photo_hypothesis_review_required",
            }
        )
        crops[column_id] = column_crops
    result_counts: dict[str, int] = {}
    for column in columns:
        key = str(column["multi_view_result"])
        result_counts[key] = result_counts.get(key, 0) + 1
    report = {
        "schema_version": "railway.canopy-photo-evidence.v1",
        "project_id": canopy.get("project_id"),
        "segment_id": canopy["segment_id"],
        "camera_count": len(consensus["per_camera"]),
        "trusted_camera_count": sum(
            bool(item["local_best_matches_consensus"]) for item in consensus["per_camera"]
        ),
        "column_count": len(columns),
        "result_counts": result_counts,
        "columns": columns,
        "settings": settings,
        "status": "photo_edge_hypotheses_generated_manual_review_required",
        "limitations": [
            "Directional image edges can be caused by roof eaves, shadows, signs, or beams.",
            "Projection uses the shared pose convention without bundle-adjustment bias correction.",
            "This stage never writes asset registry records or model geometry.",
        ],
    }
    return report, crops


def build_canopy_photo_evidence_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for column in report["columns"]:
        nodes.append(
            {
                "id": column["column_id"],
                "node_type": "canopy_column_hypothesis",
                "photo_evidence_result": column["multi_view_result"],
            }
        )
        for view in column["views"]:
            view_id = f"PHOTO-EDGE-{column['column_id']}-{view['camera_index']}"
            nodes.append(
                {
                    "id": view_id,
                    "node_type": "directional_photo_edge_observation",
                    "camera_index": view["camera_index"],
                    "trusted_for_consensus": view["trusted_for_consensus"],
                    "edge_supported": view["edge_supported"],
                    "status": "photo_observation_not_beam_confirmation",
                }
            )
            edges.append(
                {
                    "id": f"CANOPY-PHOTO-EDGE-{len(edges) + 1:03d}",
                    "type": "column_top_has_photo_edge_observation",
                    "source_id": column["column_id"],
                    "target_id": view_id,
                    "status": "candidate_evidence_only",
                }
            )
    return {
        "schema_version": "railway.canopy-photo-evidence-graph.v1",
        "project_id": report["project_id"],
        "segment_id": report["segment_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_asset_registry_or_model_write",
    }


def _draw_cross(draw: ImageDraw.ImageDraw, xy: list[float], color: str) -> None:
    x, y = xy
    draw.line((x - 8, y, x + 8, y), fill=color, width=3)
    draw.line((x, y - 8, x, y + 8), fill=color, width=3)


def _render_column_sheet(
    column: dict[str, Any], crops: dict[int, np.ndarray], output: Path, settings: dict[str, Any]
) -> None:
    tile_w, tile_h = int(settings["sheet_tile_width_px"]), int(settings["sheet_tile_height_px"])
    sheet = Image.new("RGB", (tile_w * len(column["views"]), tile_h + 60), "#06141d")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 9),
        f"{column['column_id']} | {column['multi_view_result']} | supported {column['trusted_supported_view_count']}",
        fill="#eaf9ff",
        font=ImageFont.load_default(),
    )
    draw.text(
        (12, 30),
        "cyan=transverse | orange=column | green=top | magenta=nearest roof",
        fill="#90aab3",
        font=ImageFont.load_default(),
    )
    for tile_index, view in enumerate(column["views"]):
        annotated = Image.fromarray(crops[int(view["camera_index"])], mode="RGB")
        layer = ImageDraw.Draw(annotated)
        markers = view["crop_markers"]
        layer.line(
            (*markers["transverse_start"], *markers["transverse_end"]), fill="#24d7ff", width=3
        )
        layer.line((*markers["column_below_top"], *markers["column_top"]), fill="#ff9f31", width=3)
        _draw_cross(layer, markers["column_top"], "#b7ff3c")
        _draw_cross(layer, markers["nearest_roof_point"], "#e98cff")
        annotated.thumbnail((tile_w, tile_h - 40), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "#06141d")
        tile.paste(annotated, ((tile_w - annotated.width) // 2, 34))
        td = ImageDraw.Draw(tile)
        support = "EDGE" if view["edge_supported"] else "NO EDGE"
        trust = "trusted" if view["trusted_for_consensus"] else "reference"
        td.text(
            (7, 7),
            f"cam {view['camera_index']} | {trust} | {support} | R90 {view['directional_response']:.3f} O {view['directional_orientation_ratio']:.2f}",
            fill="#b7ff3c" if view["edge_supported"] else "#9bb1b9",
            font=ImageFont.load_default(),
        )
        sheet.paste(tile, (tile_index * tile_w, 60))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def _render_summary(columns: list[dict[str, Any]], output: Path, settings: dict[str, Any]) -> None:
    image = Image.new("RGB", (1500, 940), "#06141d")
    draw = ImageDraw.Draw(image)
    draw.text(
        (30, 20),
        "CANOPY COLUMN-TOP PHOTO EVIDENCE / MULTI-VIEW EDGE RESPONSE",
        fill="#eaf9ff",
        font=ImageFont.load_default(),
    )
    draw.text(
        (30, 42),
        "Edge support is an observation, not proof of a beam or permission to write geometry.",
        fill="#ff9f31",
        font=ImageFont.load_default(),
    )
    threshold = float(settings["directional_response_threshold"])
    for i, column in enumerate(columns):
        row, col = divmod(i, 3)
        x0, y0 = 30 + col * 490, 80 + row * 280
        draw.rectangle((x0, y0, x0 + 470, y0 + 260), outline="#28505e", width=2)
        draw.text(
            (x0 + 12, y0 + 10), column["column_id"], fill="#b7ff3c", font=ImageFont.load_default()
        )
        draw.text(
            (x0 + 12, y0 + 30),
            f"trusted {column['trusted_valid_view_count']} | supported {column['trusted_supported_view_count']}",
            fill="#d0e2e8",
            font=ImageFont.load_default(),
        )
        for j, view in enumerate(column["views"]):
            bx = x0 + 18 + j * 85
            bh = int(
                min(float(view["directional_response"]) / max(threshold, 1e-9), 2.0) / 2.0 * 145
            )
            color = "#b7ff3c" if view["edge_supported"] else "#2c7184"
            if not view["trusted_for_consensus"]:
                color = "#765f8f"
            draw.rectangle((bx, y0 + 210 - bh, bx + 34, y0 + 210), fill=color)
            draw.text(
                (bx, y0 + 219),
                str(view["camera_index"]),
                fill="#8ca5ad",
                font=ImageFont.load_default(),
            )
        draw.text(
            (x0 + 12, y0 + 244),
            column["multi_view_result"],
            fill="#7fa0aa",
            font=ImageFont.load_default(),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_canopy_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    projection_consensus_path: str | Path,
    overwrite: bool = False,
    canopy_report_path: str | Path | None = None,
    vertical_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    canopy_path = (
        project.resolve(canopy_report_path)
        if canopy_report_path
        else reports / f"{segment_id}_canopy_structure.json"
    )
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    consensus_path = project.resolve(projection_consensus_path)
    for source in (canopy_path, vertical_path, consensus_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    configured = project.value.get("algorithms", {}).get("canopy_photo_evidence")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    output_report = reports / f"{segment_id}_canopy_photo_evidence.json"
    output_graph = reports / f"{segment_id}_canopy_photo_evidence_graph.json"
    output_summary = reports / f"{segment_id}_canopy_photo_evidence.png"
    output_directory = reports / f"{segment_id}_canopy_photo_evidence"
    if not overwrite:
        for path in (output_report, output_graph, output_summary):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
    consensus = load_json(consensus_path)
    cache: dict[Path, Image.Image] = {}

    def load_image(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path]

    report, crops = analyze_canopy_photo_evidence_data(
        load_json(canopy_path),
        load_json(vertical_path),
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    for column in report["columns"]:
        sheet = output_directory / f"{column['column_id']}.jpg"
        _render_column_sheet(column, crops[column["column_id"]], sheet, settings)
        column["evidence_sheet"] = str(sheet)
    report.update(
        {
            "canopy_structure_report": str(canopy_path),
            "vertical_hypotheses_report": str(vertical_path),
            "projection_consensus_report": str(consensus_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "output_report": str(output_report),
            "output_candidate_graph": str(output_graph),
            "output_summary_image": str(output_summary),
            "output_evidence_directory": str(output_directory),
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_canopy_photo_evidence_graph(report))
    _render_summary(report["columns"], output_summary, settings)
    for image in cache.values():
        image.close()
    return report
