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


def build_no_geometry_evidence_closure(
    *,
    model_path: str | Path,
    registry_path: str | Path,
    mesh_audit_path: str | Path,
    vertical_decisions_path: str | Path,
    canopy_audit_path: str | Path,
    platform_support_path: str | Path,
    platform_seam_audit_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Bind a no-geometry review round to one immutable model and its QA evidence."""
    model = Path(model_path).resolve()
    registry = Path(registry_path).resolve()
    mesh_audit = load_json(Path(mesh_audit_path))
    vertical = load_json(Path(vertical_decisions_path))
    canopy = load_json(Path(canopy_audit_path))
    support = load_json(Path(platform_support_path))
    seams = load_json(Path(platform_seam_audit_path))

    vertical_summary = vertical["summary"]
    if int(vertical_summary["unresolved_candidate_count"]) != 0:
        raise ValueError("Vertical candidates remain unresolved")
    if int(vertical_summary["geometry_add_count"]) != 0:
        raise ValueError("This closure is only valid for a no-geometry review round")
    if not bool(mesh_audit["passed"]):
        raise ValueError("Bound source model failed mesh audit")
    if not bool(seams["passed"]):
        raise ValueError("Platform seams failed")

    platform_surfaces: list[dict[str, Any]] = []
    for record in support.get("objects", []):
        if record.get("asset_type") != "platform_surface":
            continue
        delta = record.get("nearest_delta_median_xyz_m") or [None, None, None]
        platform_surfaces.append(
            {
                "object_name": record["object_name"],
                "point_distance_p90_m": float(record["p90_m"]),
                "coverage_at_0_10m": float(record["coverage_at_0_10m"]),
                "median_vertical_delta_m": (
                    None if delta[2] is None else float(delta[2])
                ),
                "disposition": record["disposition"],
            }
        )

    canopy_failed = [
        {
            "object_name": item["object_name"],
            "vertical_correction_at_center_m": float(
                item["vertical_correction_at_center_m"]
            ),
            "plane_angle_change_deg": float(item["plane_angle_change_deg"]),
            "failed_gates": sorted(
                name for name, passed in item["gates"].items() if not passed
            ),
        }
        for item in canopy["records"]
        if not item["passed"]
    ]

    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.no-geometry-evidence-closure.v1",
            "bound_release": {
                "model": str(model),
                "model_sha256": _sha256(model),
                "registry": str(registry),
                "registry_sha256": _sha256(registry),
            },
            "vertical_review": {
                "source": str(Path(vertical_decisions_path).resolve()),
                **vertical_summary,
                "passed": True,
            },
            "canopy_review": {
                "source": str(Path(canopy_audit_path).resolve()),
                "roof_object_count": int(canopy["roof_object_count"]),
                "passed_count": int(canopy["passed_count"]),
                "withheld_count": len(canopy_failed),
                "withheld_surfaces": canopy_failed,
                "disposition": "retain_current_geometry_for_failed_local_planes",
            },
            "platform_review": {
                "support_source": str(Path(platform_support_path).resolve()),
                "seam_source": str(Path(platform_seam_audit_path).resolve()),
                "surface_records": platform_surfaces,
                "seams": [
                    {
                        "seam_station_m": float(item["seam_station_m"]),
                        "p90_m": float(
                            item["symmetric_nearest_edge_distance"]["p90_m"]
                        ),
                        "maximum_m": float(
                            item["symmetric_nearest_edge_distance"]["maximum_m"]
                        ),
                        "passed": bool(item["passed"]),
                    }
                    for item in seams["seams"]
                ],
                "passed": True,
                "disposition": "retain_current_geometry_no_global_platform_shift",
            },
            "mesh_review": {
                "source": str(Path(mesh_audit_path).resolve()),
                "passed": True,
                "object_count": int(mesh_audit["object_count"]),
                "triangle_count": int(
                    mesh_audit["triangle_count_after_fan_triangulation"]
                ),
                "duplicate_face_count": int(mesh_audit["duplicate_face_count"]),
                "degenerate_triangle_count": int(
                    mesh_audit["degenerate_triangle_count"]
                ),
            },
            "geometry_write": False,
            "status": "evidence_closed_model_hash_unchanged",
            "next_action": (
                "Run a constrained multi-segment roof-group fit for the three withheld "
                "right-canopy local planes; do not alter the accepted release until that "
                "candidate passes seam and fixed-view QA."
            ),
        },
    )
    return destination

