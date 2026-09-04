from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import load_json, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_rgb_vegetation_residuals(
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
    """Close RGB-supported vegetation residuals without creating fake vertical assets."""
    if not candidate_ids:
        raise ValueError("At least one candidate id is required")
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
        if attribute.get("interpretation") != "rgb_vegetation_like_review_no_build":
            raise ValueError(f"{candidate_id} is not RGB vegetation-like")
        green_fraction = float(attribute["green_dominant_fraction"])
        excess_green = float(attribute["median_normalized_excess_green"])
        if green_fraction < 0.70 or excess_green < 0.08:
            raise ValueError(f"{candidate_id} lacks strong RGB vegetation support")
        if int(dense["rendered_candidate_point_count"]) <= 0:
            raise ValueError(f"{candidate_id} lacks dense point evidence")
        if int(panorama["trusted_reviewable_view_count"]) < 2:
            raise ValueError(f"{candidate_id} lacks two trusted panorama views")
        records.append(
            {
                "candidate_id": candidate_id,
                "station_m": float(candidate["station_m"]),
                "cross_m": float(candidate["cross_m"]),
                "height_m": float(candidate["extent_xyz_m"][2]),
                "green_dominant_fraction": green_fraction,
                "median_normalized_excess_green": excess_green,
                "median_rgb_8bit": attribute["median_rgb_8bit"],
                "dense_point_evidence": str(dense["evidence_figure"]),
                "photo_evidence": str(panorama["evidence_sheet"]),
                "semantic_class": "vegetation_or_landscape_residual",
                "geometry_action": "exclude_from_structured_asset_model",
                "display_action": "retain_in_point_cloud_or_optional_landscape_layer",
                "closed": True,
            }
        )
    model = Path(model_path).resolve()
    registry = Path(registry_path).resolve()
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.rgb-vegetation-residual-closure.v1",
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
            "summary": {
                "reviewed_candidate_count": len(records),
                "closed_candidate_count": len(records),
                "new_asset_count": 0,
                "geometry_write": False,
            },
            "records": records,
            "status": "closed_rgb_vegetation_no_geometry_change",
        },
    )
    return destination
