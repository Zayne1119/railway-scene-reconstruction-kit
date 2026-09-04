from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .camera import load_camera_rows
from .config import ProjectConfig
from .io import load_json, write_json
from .vertical_conflict_photo_evidence import (
    _render_candidate_sheet,
    _render_summary,
    _resource_settings,
    analyze_vertical_conflict_photo_evidence_data,
    build_vertical_conflict_photo_graph,
)


def build_small_asset_vertical_proxies(
    small_assets: dict[str, Any], platform: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Convert small-asset bounding boxes to photo-projection hypotheses.

    Long linear candidates are sampled at three longitudinal anchors so one central
    crop cannot incorrectly stand in for an entire fence or mixed residual strip.
    """
    proxies: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    for side in small_assets.get("sides", []):
        for candidate in side.get("candidates", []):
            minimum = [float(value) for value in candidate["minimum_s_c_z_m"]]
            maximum = [float(value) for value in candidate["maximum_s_c_z_m"]]
            centroid = [float(value) for value in candidate["centroid_s_c_z_m"]]
            span = [float(value) for value in candidate["span_s_c_z_m"]]
            fractions = (0.1, 0.5, 0.9) if span[0] > 5.0 else (0.5,)
            for anchor_index, fraction in enumerate(fractions, start=1):
                suffix = f"-ANCHOR-{anchor_index:02d}" if len(fractions) > 1 else ""
                proxy_id = f"{candidate['id']}{suffix}"
                longitudinal = minimum[0] + fraction * span[0]
                proxies.append(
                    {
                        "id": proxy_id,
                        "predicted_class": candidate["candidate_class"],
                        "confidence": "geometry_candidate_only",
                        "longitudinal_position_m": longitudinal,
                        "cross_position_m": centroid[1],
                        "minimum_z": minimum[2],
                        "maximum_z": maximum[2],
                        "footprint_m": max(0.1, min(max(span[0], span[1]), 2.0)),
                    }
                )
                provenance[proxy_id] = {
                    "source_candidate_id": candidate["id"],
                    "source_side": side["side"],
                    "source_candidate_class": candidate["candidate_class"],
                    "anchor_fraction": fraction,
                    "source_bbox_minimum_s_c_z_m": minimum,
                    "source_bbox_maximum_s_c_z_m": maximum,
                }
    return (
        {
            "schema_version": "railway.vertical-hypotheses.photo-proxy.v1",
            "project_id": small_assets.get("project_id"),
            "segment_id": small_assets["segment_id"],
            "frame": platform["frame"],
            "candidates": proxies,
        },
        provenance,
    )


def analyze_small_asset_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    small_asset_report_path: str | Path,
    projection_consensus_path: str | Path,
    *,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    small_path = project.resolve(small_asset_report_path)
    consensus_path = project.resolve(projection_consensus_path)
    for path in (small_path, consensus_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    small_assets = load_json(small_path)
    if small_assets.get("segment_id") != segment_id:
        raise ValueError("Small-asset report segment does not match requested segment")
    platform_path = project.resolve(small_assets["platform_report"])
    if not platform_path.is_file():
        raise FileNotFoundError(platform_path)
    vertical, provenance = build_small_asset_vertical_proxies(
        small_assets, load_json(platform_path)
    )
    if not vertical["candidates"]:
        raise ValueError("Small-asset report contains no candidates")
    interface = {
        "project_id": project.project_id,
        "segment_id": segment_id,
        "vertical_interfaces": [
            {"candidate_id": item["id"], "semantic_conflict": True}
            for item in vertical["candidates"]
        ],
    }
    configured = project.value.get("algorithms", {}).get(
        "vertical_conflict_photo_evidence"
    )
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
        interface,
        vertical,
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    output_name = f"{segment_id}_small_asset_photo_evidence"
    output_report = reports / f"{output_name}.json"
    output_graph = reports / f"{output_name}_graph.json"
    output_summary = reports / f"{output_name}.png"
    output_directory = reports / output_name
    for path in (output_report, output_graph, output_summary):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for candidate in report["candidates"]:
        candidate.update(provenance[candidate["candidate_id"]])
        sheet = output_directory / f"{candidate['candidate_id']}.jpg"
        _render_candidate_sheet(
            candidate, crops[candidate["candidate_id"]], sheet, settings
        )
        candidate["evidence_sheet"] = str(sheet)
    report.update(
        {
            "schema_version": "railway.small-asset-photo-evidence.v1",
            "small_asset_candidate_report": str(small_path),
            "platform_report": str(platform_path),
            "projection_consensus_report": str(consensus_path),
            "settings_source": (
                str(resolved_settings) if resolved_settings else "bundled_default"
            ),
            "source_candidate_count": int(small_assets["candidate_count"]),
            "photo_proxy_count": len(vertical["candidates"]),
            "output_report": str(output_report),
            "output_graph": str(output_graph),
            "output_summary": str(output_summary),
            "output_evidence_directory": str(output_directory),
            "status": "photo_review_packages_generated_manual_semantic_review_required",
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_vertical_conflict_photo_graph(report))
    _render_summary(report, output_summary)
    for image in cache.values():
        image.close()
    return report
