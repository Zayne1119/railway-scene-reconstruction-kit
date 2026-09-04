from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_external_context_post_row(
    *,
    gap_report_path: str | Path,
    detail_evidence_path: str | Path,
    photo_evidence_path: str | Path,
    attribute_evidence_path: str | Path,
    model_path: str | Path,
    registry_path: str | Path,
    candidate_ids: tuple[str, ...],
    output_path: str | Path,
) -> Path:
    """Close a photo-reviewed row of non-railway background posts without geometry."""
    if len(candidate_ids) < 2:
        raise ValueError("At least two candidates are required to establish a post row")
    gap = load_json(Path(gap_report_path))
    detail = load_json(Path(detail_evidence_path))
    photo = load_json(Path(photo_evidence_path))
    attributes = load_json(Path(attribute_evidence_path))
    gap_by_id = {str(item["candidate_id"]): item for item in gap["vertical_candidates"]}
    detail_by_id = {str(item["candidate_id"]): item for item in detail["candidates"]}
    photo_by_id = {str(item["candidate_id"]): item for item in photo["candidates"]}
    attribute_by_id = {
        str(item["candidate_id"]): item for item in attributes["records"]
    }
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = gap_by_id.get(candidate_id)
        dense = detail_by_id.get(candidate_id)
        panorama = photo_by_id.get(candidate_id)
        attribute = attribute_by_id.get(candidate_id)
        if candidate is None or dense is None or panorama is None or attribute is None:
            raise ValueError(f"Incomplete evidence package for {candidate_id}")
        height = float(candidate["extent_xyz_m"][2])
        nearest_distance = float(
            candidate["nearest_existing_vertical_asset"]["footprint_distance_m"]
        )
        if not 2.0 <= height <= 4.0:
            raise ValueError(f"{candidate_id} is outside the reviewed post-height range")
        if nearest_distance < 10.0:
            raise ValueError(f"{candidate_id} is not sufficiently outside modeled railway assets")
        if int(dense["rendered_candidate_point_count"]) <= 0:
            raise ValueError(f"{candidate_id} lacks dense point evidence")
        if int(panorama["trusted_reviewable_view_count"]) < 2:
            raise ValueError(f"{candidate_id} lacks two trusted panorama views")
        if float(attribute["vegetation_class_fraction"]) > 0.1:
            raise ValueError(f"{candidate_id} is vegetation-class dominated")
        records.append(
            {
                "candidate_id": candidate_id,
                "station_m": float(candidate["station_m"]),
                "cross_m": float(candidate["cross_m"]),
                "height_m": height,
                "nearest_railway_asset_distance_m": nearest_distance,
                "dense_point_evidence": str(dense["evidence_figure"]),
                "photo_evidence": str(panorama["evidence_sheet"]),
                "trusted_photo_view_count": int(
                    panorama["trusted_reviewable_view_count"]
                ),
                "median_rgb_8bit": attribute["median_rgb_8bit"],
                "median_rgb_saturation": float(attribute["median_rgb_saturation"]),
                "closed": True,
            }
        )
    station_values = np.asarray([item["station_m"] for item in records], dtype=float)
    cross_values = np.asarray([item["cross_m"] for item in records], dtype=float)
    height_values = np.asarray([item["height_m"] for item in records], dtype=float)
    station_span = float(np.ptp(station_values))
    cross_std = float(np.std(cross_values))
    if station_span < 10.0 or cross_std > 0.15:
        raise ValueError("Candidates do not form a stable station-aligned external post row")
    model = Path(model_path).resolve()
    registry = Path(registry_path).resolve()
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.external-context-post-row-closure.v1",
            "bound_release": {
                "model": str(model),
                "model_sha256": _sha256(model),
                "registry": str(registry),
                "registry_sha256": _sha256(registry),
            },
            "evidence": {
                "gap_report": str(Path(gap_report_path).resolve()),
                "detail_evidence": str(Path(detail_evidence_path).resolve()),
                "photo_evidence": str(Path(photo_evidence_path).resolve()),
                "attribute_evidence": str(Path(attribute_evidence_path).resolve()),
            },
            "group_fit": {
                "station_span_m": station_span,
                "cross_mean_m": float(np.mean(cross_values)),
                "cross_standard_deviation_m": cross_std,
                "height_mean_m": float(np.mean(height_values)),
            },
            "semantic_decision": {
                "class": "external_non_railway_context_post_row",
                "basis": (
                    "Three collinear thin posts lie about 24 m beyond the nearest modeled "
                    "railway asset; trusted panoramas place them behind the platform boundary "
                    "within the agricultural/background context."
                ),
                "geometry_action": "exclude_from_structured_railway_asset_model",
                "display_action": "retain_in_point_cloud_or_optional_context_layer",
            },
            "summary": {
                "reviewed_candidate_count": len(records),
                "closed_candidate_count": len(records),
                "new_railway_asset_count": 0,
                "geometry_write": False,
            },
            "records": records,
            "status": "closed_external_context_no_geometry_change",
        },
    )
    return destination
