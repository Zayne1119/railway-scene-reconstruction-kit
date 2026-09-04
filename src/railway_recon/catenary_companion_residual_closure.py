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


def close_catenary_companion_residuals(
    *,
    gap_report_path: str | Path,
    detail_evidence_path: str | Path,
    photo_evidence_path: str | Path,
    model_path: str | Path,
    registry_path: str | Path,
    candidate_ids: tuple[str, ...],
    output_path: str | Path,
) -> Path:
    """Close reviewed mast-aligned companion traces without changing geometry."""
    if not candidate_ids:
        raise ValueError("At least one candidate id is required")
    gap_report = load_json(Path(gap_report_path))
    detail_evidence = load_json(Path(detail_evidence_path))
    photo_evidence = load_json(Path(photo_evidence_path))
    gap_by_id = {
        str(item["candidate_id"]): item
        for item in gap_report.get("vertical_candidates", [])
    }
    detail_by_id = {
        str(item["candidate_id"]): item
        for item in detail_evidence.get("candidates", [])
    }
    photo_by_id = {
        str(item["candidate_id"]): item
        for item in photo_evidence.get("candidates", [])
    }
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = gap_by_id.get(candidate_id)
        detail = detail_by_id.get(candidate_id)
        photo = photo_by_id.get(candidate_id)
        if candidate is None or detail is None or photo is None:
            raise ValueError(f"Incomplete evidence package for {candidate_id}")
        if candidate.get("semantic_review_hint") != (
            "station_aligned_catenary_companion_trace_or_attachment"
        ):
            raise ValueError(f"{candidate_id} is not a companion-trace review candidate")
        aligned = candidate.get("aligned_existing_catenary_asset")
        if not isinstance(aligned, dict):
            raise TypeError(f"{candidate_id} has no aligned catenary asset")
        if int(photo.get("trusted_reviewable_view_count", 0)) < 2:
            raise ValueError(f"{candidate_id} lacks two trusted photo views")
        if int(detail.get("rendered_candidate_point_count", 0)) <= 0:
            raise ValueError(f"{candidate_id} lacks dense point evidence")
        records.append(
            {
                "candidate_id": candidate_id,
                "station_m": float(candidate["station_m"]),
                "cross_m": float(candidate["cross_m"]),
                "height_m": float(candidate["extent_xyz_m"][2]),
                "vertical_occupancy": float(candidate["vertical_occupancy"]),
                "aligned_asset_id": str(aligned["asset_id"]),
                "station_delta_m": float(aligned["station_delta_m"]),
                "cross_delta_m": float(aligned["cross_delta_m"]),
                "dense_point_evidence": str(detail["evidence_figure"]),
                "photo_evidence": str(photo["evidence_sheet"]),
                "trusted_photo_view_count": int(
                    photo["trusted_reviewable_view_count"]
                ),
                "review_interpretation": (
                    "discontinuous mast-station companion trace; no second independent "
                    "mast is visible in trusted panorama views"
                ),
                "geometry_action": "retain_existing_mast_no_new_vertical_asset",
                "detail_disposition": (
                    "treat as attachment_or_scan_companion_residual; revisit only with "
                    "higher-resolution calibrated evidence"
                ),
                "closed": True,
            }
        )
    model = Path(model_path).resolve()
    registry = Path(registry_path).resolve()
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.catenary-companion-residual-closure.v1",
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
            },
            "summary": {
                "reviewed_candidate_count": len(records),
                "closed_candidate_count": len(records),
                "new_asset_count": 0,
                "moved_existing_asset_count": 0,
                "geometry_write": False,
            },
            "records": records,
            "status": "closed_no_geometry_change",
        },
    )
    return destination
