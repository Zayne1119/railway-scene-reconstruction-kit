from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import sobel

from .camera import load_camera_rows
from .canopy_photo_evidence import _local_to_world, _permutation_matrix, _wrapped_delta
from .config import ProjectConfig
from .io import load_json, write_json
from .projection import _canonical_vectors, project_equirectangular


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "platform-photo-evidence.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _platform_segment_points(
    component: dict[str, Any], segment: dict[str, Any], settings: dict[str, Any]
) -> dict[str, np.ndarray]:
    s0, s1 = (float(value) for value in segment["longitudinal_range_m"])
    c0, c1 = (float(value) for value in segment["observed_cross_range_m"])
    rail_edge = float(segment["rail_side_edge_cross_m"])
    outer_edge = c0 if component["side"] == "left" else c1
    direction = -1.0 if component["side"] == "left" else 1.0
    safety_cross = rail_edge + direction * float(settings["safety_line_probe_offset_m"])
    coefficients = np.asarray(segment["plane_z_equals_a_s_plus_b_c_plus_d"], dtype=np.float64)

    def line(cross: float) -> np.ndarray:
        s_values = np.linspace(s0, s1, int(settings["line_projection_samples"]))
        z_values = coefficients[0] * s_values + coefficients[1] * cross + coefficients[2]
        return np.column_stack((s_values, np.full(s_values.size, cross), z_values))

    return {
        "rail_edge": line(rail_edge),
        "outer_edge": line(outer_edge),
        "safety_probe": line(safety_cross),
    }


def _platform_gap_boundary(
    component: dict[str, Any], gap: dict[str, Any], settings: dict[str, Any]
) -> np.ndarray:
    """Return a closed local-coordinate boundary on the nearest fitted platform plane."""
    s0, s1 = (float(value) for value in gap["longitudinal_range_m"])
    c0, c1 = (float(value) for value in gap["cross_range_m"])
    center_s = (s0 + s1) / 2.0
    fit = min(
        component["fit_segments"],
        key=lambda item: abs(
            center_s
            - (float(item["longitudinal_range_m"][0]) + float(item["longitudinal_range_m"][1]))
            / 2.0
        ),
    )
    samples = max(int(settings["gap_edge_projection_samples"]), 2)
    boundary_sc = np.vstack(
        (
            np.column_stack((np.linspace(s0, s1, samples), np.full(samples, c0))),
            np.column_stack((np.full(samples, s1), np.linspace(c0, c1, samples))),
            np.column_stack((np.linspace(s1, s0, samples), np.full(samples, c1))),
            np.column_stack((np.full(samples, s0), np.linspace(c1, c0, samples))),
        )
    )
    coefficients = np.asarray(fit["plane_z_equals_a_s_plus_b_c_plus_d"], dtype=np.float64)
    z = coefficients[0] * boundary_sc[:, 0] + coefficients[1] * boundary_sc[:, 1] + coefficients[2]
    return np.column_stack((boundary_sc, z))


def _wrapped_projection_span(u: np.ndarray, v: np.ndarray, width: int) -> tuple[float, float]:
    unwrapped = np.asarray([float(u[0])], dtype=np.float64)
    for value in u[1:]:
        unwrapped = np.append(
            unwrapped,
            unwrapped[-1] + _wrapped_delta(float(value), float(unwrapped[-1]), width),
        )
    return float(np.ptp(unwrapped)), float(np.ptp(v))


def _project_local(
    local_points: np.ndarray,
    frame: dict[str, Any],
    camera_xyz: np.ndarray,
    camera: dict[str, Any],
    convention: dict[str, Any],
    permutation: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    world = _local_to_world(local_points, frame)
    canonical = _canonical_vectors(
        world - camera_xyz,
        np.asarray(camera["consensus_base_rotation"], dtype=np.float64),
        str(convention["pose_direction"]),
        permutation,
    )
    return project_equirectangular(canonical, width, height)


def _gradient_fields(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = (
        image[:, :, 0].astype(np.float64) * 0.299
        + image[:, :, 1].astype(np.float64) * 0.587
        + image[:, :, 2].astype(np.float64) * 0.114
    ) / 255.0
    return sobel(gray, axis=1, mode="reflect") / 8.0, sobel(gray, axis=0, mode="reflect") / 8.0


def _sample_polyline_band(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    band_radius: int,
) -> np.ndarray:
    height, width = field.shape
    values: list[np.ndarray] = []
    for offset in range(-band_radius, band_radius + 1):
        x = np.mod(np.rint(u + normal[0] * offset).astype(np.int64), width)
        y = np.clip(np.rint(v + normal[1] * offset).astype(np.int64), 0, height - 1)
        values.append(field[y, x])
    return np.concatenate(values)


def polyline_edge_evidence(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    band_radius: int,
) -> dict[str, float | int | bool]:
    if len(u) < 2:
        return {
            "valid": False,
            "projected_length_px": 0.0,
            "directional_response": 0.0,
            "directional_orientation_ratio": 0.0,
            "sample_count": 0,
        }
    width = gradient_x.shape[1]
    unwrapped = np.asarray([float(u[0])], dtype=np.float64)
    for value in u[1:]:
        unwrapped = np.append(
            unwrapped,
            unwrapped[-1] + _wrapped_delta(float(value), float(unwrapped[-1]), width),
        )
    dx = float(unwrapped[-1] - unwrapped[0])
    dy = float(v[-1] - v[0])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return {
            "valid": False,
            "projected_length_px": length,
            "directional_response": 0.0,
            "directional_orientation_ratio": 0.0,
            "sample_count": 0,
        }
    tangent = np.asarray([dx / length, dy / length])
    normal = np.asarray([-tangent[1], tangent[0]])
    normal_field = np.abs(gradient_x * normal[0] + gradient_y * normal[1])
    tangent_field = np.abs(gradient_x * tangent[0] + gradient_y * tangent[1])
    normal_values = _sample_polyline_band(
        normal_field, np.mod(unwrapped, width), v, normal, band_radius
    )
    tangent_values = _sample_polyline_band(
        tangent_field, np.mod(unwrapped, width), v, normal, band_radius
    )
    normal_sum = float(normal_values.sum())
    tangent_sum = float(tangent_values.sum())
    return {
        "valid": True,
        "projected_length_px": length,
        "directional_response": float(np.percentile(normal_values, 90.0)),
        "directional_orientation_ratio": normal_sum / max(normal_sum + tangent_sum, 1e-12),
        "sample_count": int(normal_values.size),
    }


def polyline_warm_color_fraction(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    band_radius: int,
) -> float:
    height, width = image.shape[:2]
    samples: list[np.ndarray] = []
    for offset in range(-band_radius, band_radius + 1):
        x = np.mod(np.rint(u).astype(np.int64), width)
        y = np.clip(np.rint(v + offset).astype(np.int64), 0, height - 1)
        samples.append(image[y, x].astype(np.float64))
    rgb = np.concatenate(samples, axis=0)
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    warm = (
        (red >= float(80))
        & (green >= float(45))
        & (red >= green * 0.9)
        & (green >= blue * 1.15)
        & (red >= blue * 1.35)
    )
    return float(np.mean(warm)) if warm.size else 0.0


def _draw_wrapped_polyline(
    draw: ImageDraw.ImageDraw,
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    color: str,
    line_width: int,
) -> None:
    for index in range(len(u) - 1):
        if abs(float(u[index + 1] - u[index])) <= width / 2.0:
            draw.line(
                (float(u[index]), float(v[index]), float(u[index + 1]), float(v[index + 1])),
                fill=color,
                width=line_width,
            )


def analyze_platform_photo_evidence_data(
    platform: dict[str, Any],
    consensus: dict[str, Any],
    camera_rows: list[dict[str, str]],
    settings: dict[str, Any],
    image_loader: Any,
) -> tuple[dict[str, Any], dict[int, Image.Image]]:
    convention = consensus["best_shared_convention"]
    permutation = _permutation_matrix(str(convention["canonical_axes_from_local"]))
    rows = {int(row["index"]): row for row in camera_rows}
    analysis_width = int(settings["analysis_image_width_px"])
    camera_context: dict[int, dict[str, Any]] = {}
    overlays: dict[int, Image.Image] = {}
    for camera in consensus["per_camera"]:
        index = int(camera["camera_index"])
        if index not in rows:
            raise ValueError(f"Camera {index} missing from consensus camera CSV")
        original = image_loader(Path(camera["photo"])).convert("RGB")
        analysis_height = round(original.height * analysis_width / original.width)
        analysis = original.resize((analysis_width, analysis_height), Image.Resampling.LANCZOS)
        array = np.asarray(analysis)
        gx, gy = _gradient_fields(array)
        camera_context[index] = {
            "camera": camera,
            "xyz": np.asarray([float(rows[index][key]) for key in ("x", "y", "z")]),
            "image": array,
            "gradient_x": gx,
            "gradient_y": gy,
        }
        overlays[index] = analysis.copy()

    components: list[dict[str, Any]] = []
    for component in platform["platform_components"]:
        segment_records: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(component["fit_segments"], start=1):
            segment_id = f"{component['id']}-FIT-{segment_index:03d}"
            local_lines = _platform_segment_points(component, segment, settings)
            views: list[dict[str, Any]] = []
            for camera_index, context in camera_context.items():
                height, width = context["image"].shape[:2]
                projected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for name, local_points in local_lines.items():
                    projected[name] = _project_local(
                        local_points,
                        platform["frame"],
                        context["xyz"],
                        context["camera"],
                        convention,
                        permutation,
                        width,
                        height,
                    )
                edge_u, edge_v = projected["rail_edge"]
                edge = polyline_edge_evidence(
                    context["gradient_x"],
                    context["gradient_y"],
                    edge_u,
                    edge_v,
                    int(settings["edge_band_radius_px"]),
                )
                safety_u, safety_v = projected["safety_probe"]
                warm_fraction = polyline_warm_color_fraction(
                    context["image"],
                    safety_u,
                    safety_v,
                    int(settings["warm_color_band_radius_px"]),
                )
                projection_valid = bool(edge["valid"]) and float(
                    edge["projected_length_px"]
                ) >= float(settings["minimum_projected_segment_length_px"])
                trusted = bool(context["camera"]["local_best_matches_consensus"])
                edge_supported = (
                    projection_valid
                    and float(edge["directional_response"])
                    >= float(settings["edge_response_threshold"])
                    and float(edge["directional_orientation_ratio"])
                    >= float(settings["edge_orientation_ratio_threshold"])
                )
                warm_supported = projection_valid and warm_fraction >= float(
                    settings["warm_color_fraction_threshold"]
                )
                views.append(
                    {
                        "camera_index": camera_index,
                        "trusted_for_consensus": trusted,
                        "projection_valid": projection_valid,
                        **edge,
                        "rail_edge_supported": edge_supported,
                        "safety_probe_warm_color_fraction": warm_fraction,
                        "safety_probe_warm_color_supported": warm_supported,
                        "status": "photo_observation_not_asset_confirmation",
                    }
                )
                layer = ImageDraw.Draw(overlays[camera_index])
                _draw_wrapped_polyline(layer, *projected["rail_edge"], width, "#23d5ff", 2)
                _draw_wrapped_polyline(layer, *projected["outer_edge"], width, "#a98cff", 2)
                _draw_wrapped_polyline(layer, *projected["safety_probe"], width, "#ff9f31", 2)
            trusted_views = [
                view for view in views if view["trusted_for_consensus"] and view["projection_valid"]
            ]
            edge_support = [view for view in trusted_views if view["rail_edge_supported"]]
            warm_support = [
                view for view in trusted_views if view["safety_probe_warm_color_supported"]
            ]
            segment_records.append(
                {
                    "segment_id": segment_id,
                    "longitudinal_range_m": segment["longitudinal_range_m"],
                    "rail_side_edge_cross_m": segment["rail_side_edge_cross_m"],
                    "trusted_valid_view_count": len(trusted_views),
                    "trusted_rail_edge_support_count": len(edge_support),
                    "trusted_warm_color_support_count": len(warm_support),
                    "rail_edge_result": (
                        "multi_view_edge_response_supported_not_boundary_confirmed"
                        if len(edge_support) >= int(settings["minimum_supported_views"])
                        else "multi_view_edge_response_not_supported"
                    ),
                    "safety_probe_result": (
                        "multi_view_warm_color_supported_not_safety_line_confirmed"
                        if len(warm_support) >= int(settings["minimum_supported_views"])
                        else "multi_view_warm_color_not_supported"
                    ),
                    "views": views,
                    "status": "automatic_photo_hypothesis_review_required",
                }
            )
        gap_records: list[dict[str, Any]] = []
        for gap in component.get("interior_gap_candidates", []):
            local_boundary = _platform_gap_boundary(component, gap, settings)
            views: list[dict[str, Any]] = []
            for camera_index, context in camera_context.items():
                height, width = context["image"].shape[:2]
                u, v = _project_local(
                    local_boundary,
                    platform["frame"],
                    context["xyz"],
                    context["camera"],
                    convention,
                    permutation,
                    width,
                    height,
                )
                span_width, span_height = _wrapped_projection_span(u, v, width)
                reviewable = max(span_width, span_height) >= float(
                    settings["minimum_projected_gap_span_px"]
                )
                trusted = bool(context["camera"]["local_best_matches_consensus"])
                views.append(
                    {
                        "camera_index": camera_index,
                        "trusted_for_consensus": trusted,
                        "projected_width_px": span_width,
                        "projected_height_px": span_height,
                        "reviewable_resolution": reviewable,
                        "status": "photo_review_window_not_gap_confirmation",
                    }
                )
                color = (
                    "#ff9f31"
                    if gap["gap_classification"]
                    == "periodic_occlusion_pattern_not_opening_candidate"
                    else "#ff315d"
                )
                _draw_wrapped_polyline(
                    ImageDraw.Draw(overlays[camera_index]), u, v, width, color, 4
                )
            gap_records.append(
                {
                    "gap_id": gap["id"],
                    "gap_classification": gap["gap_classification"],
                    "mesh_action": gap["mesh_action"],
                    "trusted_reviewable_view_count": sum(
                        view["trusted_for_consensus"] and view["reviewable_resolution"]
                        for view in views
                    ),
                    "views": views,
                    "status": "projected_for_manual_review_not_asset_confirmation",
                }
            )
        components.append(
            {
                "component_id": component["id"],
                "side": component["side"],
                "segment_count": len(segment_records),
                "segments": segment_records,
                "interior_gap_evidence": gap_records,
            }
        )
    edge_supported_count = sum(
        segment["rail_edge_result"].startswith("multi_view_edge_response_supported")
        for component in components
        for segment in component["segments"]
    )
    safety_supported_count = sum(
        segment["safety_probe_result"].startswith("multi_view_warm_color_supported")
        for component in components
        for segment in component["segments"]
    )
    edge_intervals: list[dict[str, Any]] = []
    for component in components:
        supported_ranges = [
            segment["longitudinal_range_m"]
            for segment in component["segments"]
            if segment["rail_edge_result"].startswith("multi_view_edge_response_supported")
        ]
        merged: list[list[float]] = []
        for start, end in sorted(supported_ranges):
            if merged and float(start) <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], float(end))
            else:
                merged.append([float(start), float(end)])
        edge_intervals.append(
            {
                "component_id": component["component_id"],
                "photo_supported_intervals_m": merged,
            }
        )
    report = {
        "schema_version": "railway.platform-photo-evidence.v1",
        "project_id": platform.get("project_id"),
        "segment_id": platform["segment_id"],
        "camera_count": len(camera_context),
        "trusted_camera_count": sum(
            bool(item["local_best_matches_consensus"]) for item in consensus["per_camera"]
        ),
        "component_count": len(components),
        "fit_segment_count": sum(item["segment_count"] for item in components),
        "multi_view_edge_supported_segment_count": edge_supported_count,
        "multi_view_edge_supported_intervals": edge_intervals,
        "multi_view_warm_color_supported_segment_count": safety_supported_count,
        "interior_gap_review_window_count": sum(
            len(component["interior_gap_evidence"]) for component in components
        ),
        "unresolved_gap_review_window_count": sum(
            item["gap_classification"] == "unique_enclosed_gap_review_required"
            for component in components
            for item in component["interior_gap_evidence"]
        ),
        "components": components,
        "settings": settings,
        "status": "platform_photo_hypotheses_generated_manual_review_required",
        "limitations": [
            "A strong image edge is not by itself proof of the physical platform boundary.",
            "Warm-color response can be caused by machinery, signs, ballast, or facade materials.",
            "Red gap windows support manual review only; they do not confirm an opening.",
            "Orange periodic gap windows are retained as occlusion evidence and excluded from mesh cutouts.",
            "No mesh or asset registry record is written by this stage.",
        ],
    }
    return report, overlays


def build_platform_photo_evidence_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for component in report["components"]:
        nodes.append(
            {
                "id": component["component_id"],
                "node_type": "platform_surface_hypothesis",
            }
        )
        for segment in component["segments"]:
            nodes.append(
                {
                    "id": segment["segment_id"],
                    "node_type": "platform_segment_photo_evidence",
                    "rail_edge_result": segment["rail_edge_result"],
                    "safety_probe_result": segment["safety_probe_result"],
                    "status": segment["status"],
                }
            )
            edges.append(
                {
                    "id": f"PLATFORM-PHOTO-EDGE-{len(edges) + 1:03d}",
                    "type": "platform_has_segment_photo_evidence",
                    "source_id": component["component_id"],
                    "target_id": segment["segment_id"],
                    "status": "candidate_evidence_only",
                }
            )
            for view in segment["views"]:
                view_id = f"{segment['segment_id']}-CAM-{view['camera_index']}"
                nodes.append(
                    {
                        "id": view_id,
                        "node_type": "platform_segment_photo_observation",
                        "camera_index": view["camera_index"],
                        "trusted_for_consensus": view["trusted_for_consensus"],
                        "projection_valid": view["projection_valid"],
                        "rail_edge_supported": view["rail_edge_supported"],
                        "safety_probe_warm_color_supported": view[
                            "safety_probe_warm_color_supported"
                        ],
                        "status": view["status"],
                    }
                )
                edges.append(
                    {
                        "id": f"PLATFORM-PHOTO-EDGE-{len(edges) + 1:03d}",
                        "type": "segment_has_camera_observation",
                        "source_id": segment["segment_id"],
                        "target_id": view_id,
                        "status": "candidate_evidence_only",
                    }
                )
        for gap in component["interior_gap_evidence"]:
            nodes.append(
                {
                    "id": gap["gap_id"],
                    "node_type": "platform_gap_photo_review_window",
                    "gap_classification": gap["gap_classification"],
                    "mesh_action": gap["mesh_action"],
                    "status": gap["status"],
                }
            )
            edges.append(
                {
                    "id": f"PLATFORM-PHOTO-EDGE-{len(edges) + 1:03d}",
                    "type": "platform_gap_has_photo_review_window",
                    "source_id": component["component_id"],
                    "target_id": gap["gap_id"],
                    "status": "candidate_evidence_only",
                }
            )
            for view in gap["views"]:
                view_id = f"{gap['gap_id']}-CAM-{view['camera_index']}"
                nodes.append(
                    {
                        "id": view_id,
                        "node_type": "platform_gap_photo_observation",
                        **view,
                    }
                )
                edges.append(
                    {
                        "id": f"PLATFORM-PHOTO-EDGE-{len(edges) + 1:03d}",
                        "type": "gap_has_camera_review_window",
                        "source_id": gap["gap_id"],
                        "target_id": view_id,
                        "status": "candidate_evidence_only",
                    }
                )
    return {
        "schema_version": "railway.platform-photo-evidence-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_asset_registry_or_model_write",
    }


def _render_summary(report: dict[str, Any], overlays: dict[int, Image.Image], output: Path) -> None:
    image = Image.new("RGB", (1800, 760), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 22),
        f"{report['segment_id']} PLATFORM PHOTO EVIDENCE | cyan=rail | purple=outer | orange=safety/periodic gap | red=unresolved gap",
        fill="#eaf9ff",
        font=font,
    )
    draw.text(
        (30, 45),
        "Projection overlays are evidence checks, not approved boundaries or safety-line assets.",
        fill="#ff9f31",
        font=font,
    )
    for order, (camera_index, overlay) in enumerate(overlays.items()):
        tile = overlay.copy()
        tile.thumbnail((560, 280), Image.Resampling.LANCZOS)
        row, column = divmod(order, 3)
        x = 30 + column * 590
        y = 85 + row * 325
        image.paste(tile, (x, y + 20))
        draw.text((x, y), f"CAMERA {camera_index}", fill="#b7ff3c", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_platform_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    projection_consensus_path: str | Path,
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
    consensus_path = project.resolve(projection_consensus_path)
    for source in (platform_path, consensus_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    configured = project.value.get("algorithms", {}).get("platform_photo_evidence")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    output_report = reports / f"{segment_id}_platform_photo_evidence.json"
    output_graph = reports / f"{segment_id}_platform_photo_evidence_graph.json"
    output_summary = reports / f"{segment_id}_platform_photo_evidence.png"
    output_directory = reports / f"{segment_id}_platform_photo_evidence"
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

    report, overlays = analyze_platform_photo_evidence_data(
        load_json(platform_path),
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    overlay_paths: list[str] = []
    for camera_index, overlay in overlays.items():
        path = output_directory / f"camera_{camera_index}_platform_overlay.jpg"
        overlay.save(path, quality=92)
        overlay_paths.append(str(path))
    report.update(
        {
            "platform_surface_report": str(platform_path),
            "projection_consensus_report": str(consensus_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "output_report": str(output_report),
            "output_candidate_graph": str(output_graph),
            "output_summary_image": str(output_summary),
            "output_overlay_directory": str(output_directory),
            "output_overlay_images": overlay_paths,
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_platform_photo_evidence_graph(report))
    _render_summary(report, overlays, output_summary)
    for image in cache.values():
        image.close()
    return report
