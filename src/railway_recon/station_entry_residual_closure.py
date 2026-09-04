from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .io import load_json, write_json

BLUE_EDGE = "existing_station_entry_blue_surface_edge"
SPARSE_CONTEXT = "sparse_station_entry_context_residual"
_ALLOWED_DECISIONS = {BLUE_EDGE, SPARSE_CONTEXT}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_peak_review(
    surface_analysis: dict[str, Any],
    *,
    station_m: float,
    cross_m: float,
    tolerance_m: float = 0.25,
) -> dict[str, Any]:
    matching_segment: dict[str, Any] | None = None
    for segment in surface_analysis.get("segmented_cross_histogram_peaks", []):
        bounds = segment.get("station_range_m", [])
        if len(bounds) == 2 and float(bounds[0]) <= station_m <= float(bounds[1]):
            matching_segment = segment
            break
    if matching_segment is None:
        return {
            "segment_available": False,
            "continuous_peak_near_candidate": False,
            "nearest_peak_cross_distance_m": None,
        }
    peak_distances = [
        abs(cross_m - float(sum(peak["cross_range_m"]) / 2.0))
        for peak in matching_segment.get("cross_peaks", [])
        if len(peak.get("cross_range_m", [])) == 2
    ]
    nearest = min(peak_distances) if peak_distances else None
    return {
        "segment_available": True,
        "segment_station_range_m": matching_segment["station_range_m"],
        "continuous_peak_near_candidate": nearest is not None and nearest <= tolerance_m,
        "nearest_peak_cross_distance_m": nearest,
        "peak_tolerance_m": tolerance_m,
    }


def close_station_entry_surface_residuals(
    *,
    gap_report_path: str | Path,
    detail_evidence_path: str | Path,
    photo_evidence_path: str | Path,
    attribute_evidence_path: str | Path,
    surface_analysis_path: str | Path,
    model_path: str | Path,
    registry_path: str | Path,
    decisions: dict[str, str],
    output_path: str | Path,
) -> Path:
    """Close sparse station-entry edge residuals without inventing pole assets."""
    if not decisions:
        raise ValueError("At least one candidate decision is required")
    unknown = set(decisions.values()) - _ALLOWED_DECISIONS
    if unknown:
        raise ValueError(f"Unsupported station-entry residual decisions: {sorted(unknown)}")

    gap = load_json(Path(gap_report_path))
    detail = load_json(Path(detail_evidence_path))
    photo = load_json(Path(photo_evidence_path))
    attributes = load_json(Path(attribute_evidence_path))
    surface_analysis = load_json(Path(surface_analysis_path))
    gap_by_id = {str(item["candidate_id"]): item for item in gap["vertical_candidates"]}
    detail_by_id = {str(item["candidate_id"]): item for item in detail["candidates"]}
    photo_by_id = {str(item["candidate_id"]): item for item in photo["candidates"]}
    attribute_by_id = {
        str(item["candidate_id"]): item for item in attributes["records"]
    }

    records: list[dict[str, Any]] = []
    for candidate_id, decision in decisions.items():
        candidate = gap_by_id.get(candidate_id)
        dense = detail_by_id.get(candidate_id)
        panorama = photo_by_id.get(candidate_id)
        attribute = attribute_by_id.get(candidate_id)
        if candidate is None or dense is None or panorama is None or attribute is None:
            raise ValueError(f"Incomplete evidence package for {candidate_id}")
        station_m = float(candidate["station_m"])
        cross_m = float(candidate["cross_m"])
        if not 95.0 <= station_m <= 125.0 or not 14.5 <= cross_m <= 25.5:
            raise ValueError(f"{candidate_id} is outside the station-entry review envelope")
        full_point_count = int(attribute["full_resolution_point_count"])
        candidate_point_count = int(dense["rendered_candidate_point_count"])
        if full_point_count > 150 or candidate_point_count > 150:
            raise ValueError(f"{candidate_id} is too dense for sparse-edge closure")
        trusted_views = int(panorama["trusted_reviewable_view_count"])
        if trusted_views < 2:
            raise ValueError(f"{candidate_id} lacks two trusted panorama views")

        rgb = [float(value) for value in attribute["median_rgb_8bit"]]
        saturation = float(attribute["median_rgb_saturation"])
        surface_review = _surface_peak_review(
            surface_analysis,
            station_m=station_m,
            cross_m=cross_m,
        )
        if surface_review["continuous_peak_near_candidate"]:
            raise ValueError(
                f"{candidate_id} has a continuous point-cloud surface peak and must be fitted"
            )

        if decision == BLUE_EDGE:
            blue_margin = rgb[2] - max(rgb[0], rgb[1])
            if saturation < 0.50 or blue_margin < 15.0:
                raise ValueError(f"{candidate_id} lacks blue station-entry edge support")
            semantic_class = "station_entry_blue_facade_or_door_edge_residual"
            basis = (
                "Trusted panoramas place the trace on the blue station-entry enclosure; "
                "source RGB is strongly blue, while the dense cloud contains only a thin "
                "edge and no continuous new facade plane."
            )
            geometry_action = (
                "retain_with_existing_station_entry_group_no_new_standalone_asset"
            )
        else:
            if full_point_count > 100 or saturation > 0.25:
                raise ValueError(f"{candidate_id} is not a sparse neutral-colour residual")
            semantic_class = "sparse_station_entry_outer_context_residual"
            basis = (
                "The trace contains too few disconnected neutral-colour points to support "
                "a pole, wall or sign; panorama projection overlaps the station-entry region "
                "but does not establish a separate object."
            )
            geometry_action = "withhold_geometry_until_continuous_surface_evidence_exists"

        records.append(
            {
                "candidate_id": candidate_id,
                "station_m": station_m,
                "cross_m": cross_m,
                "height_m": float(candidate["extent_xyz_m"][2]),
                "full_resolution_point_count": full_point_count,
                "rendered_candidate_point_count": candidate_point_count,
                "trusted_photo_view_count": trusted_views,
                "median_rgb_8bit": rgb,
                "median_rgb_saturation": saturation,
                "surface_peak_review": surface_review,
                "dense_point_evidence": str(dense["evidence_figure"]),
                "photo_evidence": str(panorama["evidence_sheet"]),
                "semantic_class": semantic_class,
                "basis": basis,
                "geometry_action": geometry_action,
                "closed": True,
            }
        )

    model = Path(model_path).resolve()
    registry = Path(registry_path).resolve()
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.station-entry-residual-closure.v1",
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
                "surface_analysis": str(Path(surface_analysis_path).resolve()),
            },
            "summary": {
                "reviewed_candidate_count": len(records),
                "closed_candidate_count": len(records),
                "new_asset_count": 0,
                "geometry_write": False,
            },
            "records": records,
            "status": "closed_station_entry_residuals_no_geometry_change",
        },
    )
    return destination
