from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .algorithms.mesh import ObjWriter, oriented_box
from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value


def analyze_catenary_section_data(
    candidate: dict[str, Any],
    frame: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    neighbourhood_radius_m: float = 1.2,
    height_bin_m: float = 0.5,
    minimum_bin_points: int = 30,
    maximum_shaft_bin_width_m: float = 0.65,
) -> dict[str, Any]:
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    relative = np.column_stack((x, y)) - origin
    longitudinal = relative @ along
    transverse = relative @ cross
    candidate_s = float(candidate["longitudinal_position_m"])
    candidate_c = float(candidate["cross_position_m"])
    minimum_z = float(candidate["minimum_z"])
    maximum_z = float(candidate["maximum_z"])
    radial = np.hypot(longitudinal - candidate_s, transverse - candidate_c)
    selected = (
        (radial <= neighbourhood_radius_m)
        & (z >= minimum_z - 1e-8)
        & (z <= maximum_z + 1e-8)
    )
    selected_s = longitudinal[selected]
    selected_c = transverse[selected]
    selected_z = z[selected]
    if selected_z.size < minimum_bin_points:
        raise ValueError("Insufficient points around reviewed catenary candidate")
    bins: list[dict[str, Any]] = []
    z_low = minimum_z
    while z_low < maximum_z - 1e-8:
        z_high = min(z_low + height_bin_m, maximum_z)
        in_bin = (selected_z >= z_low) & (
            selected_z <= z_high if z_high >= maximum_z - 1e-8 else selected_z < z_high
        )
        if int(np.count_nonzero(in_bin)) >= minimum_bin_points:
            values_s = selected_s[in_bin]
            values_c = selected_c[in_bin]
            s05, s50, s95 = np.percentile(values_s, (5.0, 50.0, 95.0))
            c05, c50, c95 = np.percentile(values_c, (5.0, 50.0, 95.0))
            bins.append(
                {
                    "z_low_m": z_low,
                    "z_high_m": z_high,
                    "point_count": int(values_s.size),
                    "along_p05_m": float(s05),
                    "along_median_m": float(s50),
                    "along_p95_m": float(s95),
                    "along_width_p90_m": float(s95 - s05),
                    "cross_p05_m": float(c05),
                    "cross_median_m": float(c50),
                    "cross_p95_m": float(c95),
                    "cross_width_p90_m": float(c95 - c05),
                }
            )
        z_low = z_high
    shaft_bins = [
        item
        for item in bins
        if float(item["z_low_m"]) >= minimum_z + height_bin_m - 1e-8
        and float(item["along_width_p90_m"]) <= maximum_shaft_bin_width_m
        and float(item["cross_width_p90_m"]) <= maximum_shaft_bin_width_m
    ]
    if len(shaft_bins) < 3:
        raise ValueError("Insufficient slender height bins for a stable mast section")
    estimated = {
        "longitudinal_position_m": float(
            np.median([item["along_median_m"] for item in shaft_bins])
        ),
        "cross_position_m": float(
            np.median([item["cross_median_m"] for item in shaft_bins])
        ),
        "along_width_p90_m": float(
            np.median([item["along_width_p90_m"] for item in shaft_bins])
        ),
        "cross_width_p90_m": float(
            np.median([item["cross_width_p90_m"] for item in shaft_bins])
        ),
        "profile_recommendation": "rectangular_or_H_section_not_round_pole",
        "supporting_height_bin_count": len(shaft_bins),
    }
    return {
        "candidate_id": candidate["id"],
        "neighbourhood_radius_m": neighbourhood_radius_m,
        "point_count": int(selected_z.size),
        "candidate_vertical_range_m": [minimum_z, maximum_z],
        "estimated_shaft": estimated,
        "height_bins": bins,
        "section_gate": "passed",
        "limitations": [
            "The section dimensions are robust point-envelope estimates, not a product designation.",
            "Photo evidence is still required to distinguish a physical mast from a coincident edge.",
        ],
    }


def audit_catenary_candidate_section(
    project: ProjectConfig,
    segment_id: str,
    candidate_id: str,
    *,
    vertical_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    vertical = load_json(vertical_path)
    candidate = next(
        (
            item
            for item in vertical.get("candidates", [])
            if str(item.get("id")) == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"Reviewed candidate is missing: {candidate_id}")
    source = Path(vertical["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    resolved_output = (
        project.resolve(output_path)
        if output_path
        else reports / f"{segment_id}_catenary_candidate_section_audit.json"
    )
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {resolved_output}")
    cloud = laspy.read(source)
    report = analyze_catenary_section_data(
        candidate,
        vertical["frame"],
        np.asarray(cloud.x, dtype=np.float64),
        np.asarray(cloud.y, dtype=np.float64),
        np.asarray(cloud.z, dtype=np.float64),
    )
    report.update(
        {
            "schema_version": "railway.catenary-candidate-section-audit.v1",
            "project_id": project.project_id,
            "segment_id": segment_id,
            "source": str(source),
            "source_vertical_report": str(vertical_path),
            "status": "observed_section_audit_passed_photo_semantic_review_required",
        }
    )
    write_json(resolved_output, report)
    return report


def _signed_area(polygon: np.ndarray) -> float:
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * float(
        np.sum(polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1])
    )


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    denominator = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denominator) <= 1.0e-12:
        return False
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denominator)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denominator)
    return u >= -1.0e-10 and v >= -1.0e-10 and u + v <= 1.0 + 1.0e-10


def _triangulate_polygon(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    if len(polygon) < 3:
        raise ValueError("A polygon needs at least three vertices")
    if _signed_area(polygon) < 0.0:
        order = list(reversed(range(len(polygon))))
    else:
        order = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    while len(order) > 3:
        ear_found = False
        for position, current in enumerate(order):
            previous = order[position - 1]
            following = order[(position + 1) % len(order)]
            a, b, c = polygon[[previous, current, following]]
            first = b - a
            second = c - b
            cross = float(first[0] * second[1] - first[1] * second[0])
            if cross <= 1.0e-12:
                continue
            triangle = np.asarray([a, b, c], dtype=np.float64)
            if any(
                _point_in_triangle(polygon[index], triangle)
                for index in order
                if index not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            del order[position]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Unable to triangulate catenary mast section")
    triangles.append(tuple(order))
    return triangles


def h_section_prism(
    center_xy: np.ndarray,
    along_xy: np.ndarray,
    cross_xy: np.ndarray,
    along_width_m: float,
    cross_depth_m: float,
    flange_thickness_m: float,
    web_thickness_m: float,
    bottom_z_m: float,
    top_z_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    if not (
        0.0 < web_thickness_m < along_width_m
        and 0.0 < 2.0 * flange_thickness_m < cross_depth_m
        and bottom_z_m < top_z_m
    ):
        raise ValueError("Invalid H-section dimensions")
    w = along_width_m
    d = cross_depth_m
    tf = flange_thickness_m
    tw = web_thickness_m
    section = np.asarray(
        [
            [-w / 2.0, -d / 2.0],
            [w / 2.0, -d / 2.0],
            [w / 2.0, -d / 2.0 + tf],
            [tw / 2.0, -d / 2.0 + tf],
            [tw / 2.0, d / 2.0 - tf],
            [w / 2.0, d / 2.0 - tf],
            [w / 2.0, d / 2.0],
            [-w / 2.0, d / 2.0],
            [-w / 2.0, d / 2.0 - tf],
            [-tw / 2.0, d / 2.0 - tf],
            [-tw / 2.0, -d / 2.0 + tf],
            [-w / 2.0, -d / 2.0 + tf],
        ],
        dtype=np.float64,
    )
    xy = (
        center_xy[None, :]
        + section[:, 0, None] * along_xy[None, :]
        + section[:, 1, None] * cross_xy[None, :]
    )
    lower = np.column_stack((xy, np.full(len(xy), bottom_z_m)))
    upper = np.column_stack((xy, np.full(len(xy), top_z_m)))
    vertices = np.vstack((lower, upper))
    cap_triangles = _triangulate_polygon(section)
    count = len(section)
    faces: list[tuple[int, ...]] = []
    faces.extend(tuple(reversed(face)) for face in cap_triangles)
    faces.extend(tuple(index + count for index in face) for face in cap_triangles)
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, following + count, index + count))
    return vertices, faces


def _write_materials(path: Path) -> None:
    content = """# Site B evidence-driven catenary materials
newmtl CatenaryMastSteel
Kd 0.43 0.49 0.52
Ks 0.22 0.22 0.22
Ns 35

newmtl CatenaryFoundationConcrete
Kd 0.48 0.47 0.44
Ks 0.04 0.04 0.04
Ns 5
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def audit_catenary_track_clearance(
    graph: dict[str, Any],
    section_audit: dict[str, Any],
    track_build: dict[str, Any],
) -> dict[str, Any]:
    """Check that the observed mast foundation envelope does not intersect model track beds."""
    estimated = section_audit["estimated_shaft"]
    center_cross = float(estimated["cross_position_m"])
    first_bin = section_audit["height_bins"][0]
    foundation_width = float(np.clip(first_bin["cross_width_p90_m"], 0.75, 1.25))
    foundation_range = [
        center_cross - foundation_width / 2.0,
        center_cross + foundation_width / 2.0,
    ]
    bed_width = float(track_build["track_bed"]["bottom_width_m"])
    checks: list[dict[str, Any]] = []
    for observation in graph.get("observations", []):
        track_center = float(observation["lateral_offset_m"])
        bed_range = [track_center - bed_width / 2.0, track_center + bed_width / 2.0]
        if foundation_range[1] < bed_range[0]:
            clearance = bed_range[0] - foundation_range[1]
        elif bed_range[1] < foundation_range[0]:
            clearance = foundation_range[0] - bed_range[1]
        else:
            clearance = -(
                min(foundation_range[1], bed_range[1])
                - max(foundation_range[0], bed_range[0])
            )
        checks.append(
            {
                "track_id": observation["global_track_id"],
                "track_center_cross_m": track_center,
                "track_bed_cross_range_m": bed_range,
                "foundation_cross_range_m": foundation_range,
                "signed_foundation_to_bed_clearance_m": clearance,
                "status": "pass" if clearance >= 0.0 else "intersection",
            }
        )
    passed = bool(checks) and all(item["status"] == "pass" for item in checks)
    return {
        "schema_version": "railway.catenary-track-clearance-audit.v1",
        "candidate_id": section_audit["candidate_id"],
        "foundation_center_cross_m": center_cross,
        "foundation_cross_width_m": foundation_width,
        "track_bed_width_m": bed_width,
        "checks": checks,
        "minimum_signed_clearance_m": min(
            (float(item["signed_foundation_to_bed_clearance_m"]) for item in checks),
            default=None,
        ),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "limitations": [
            "This is a model-geometry intersection test, not an electrification design clearance check.",
            "Absolute surveying accuracy is not claimed without verified control points.",
        ],
    }


def audit_catenary_track_clearance_files(
    project: ProjectConfig,
    graph_path: str | Path,
    section_audit_path: str | Path,
    track_build_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_graph = project.resolve(graph_path)
    resolved_section = project.resolve(section_audit_path)
    resolved_track_build = project.resolve(track_build_path)
    resolved_output = project.resolve(output_path)
    for path in (resolved_graph, resolved_section, resolved_track_build):
        if not path.is_file():
            raise FileNotFoundError(path)
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {resolved_output}")
    section = load_json(resolved_section)
    report = audit_catenary_track_clearance(
        load_json(resolved_graph), section, load_json(resolved_track_build)
    )
    report.update(
        {
            "project_id": project.project_id,
            "segment_id": section.get("segment_id"),
            "source_track_graph": str(resolved_graph),
            "source_section_audit": str(resolved_section),
            "source_track_build": str(resolved_track_build),
        }
    )
    write_json(resolved_output, report)
    return report


def build_catenary_candidate_mesh(
    project: ProjectConfig,
    vertical_report_path: str | Path,
    semantic_review_path: str | Path,
    section_audit_path: str | Path,
    *,
    output_dir: str | Path,
    report_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    vertical_path = project.resolve(vertical_report_path)
    review_path = project.resolve(semantic_review_path)
    section_path = project.resolve(section_audit_path)
    vertical = load_json(vertical_path)
    review = load_json(review_path)
    section = load_json(section_path)
    if review.get("schema_version") != "railway.catenary-candidate-semantic-review.v1":
        raise ValueError("Expected railway.catenary-candidate-semantic-review.v1")
    decisions = [
        item
        for item in review.get("decisions", [])
        if item.get("reviewed_class") == "catenary_support"
        and item.get("model_action") == "build_observed_h_section_mast_and_foundation"
    ]
    if not decisions:
        raise ValueError("No reviewed catenary support is approved for mesh generation")
    candidate_by_id = {item["id"]: item for item in vertical.get("candidates", [])}
    frame = vertical["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along_xy = np.asarray(frame["along_xy"], dtype=np.float64)
    cross_xy = np.asarray(frame["cross_xy"], dtype=np.float64)
    output_root = project.resolve(output_dir)
    reports = project.resolve(report_dir)
    output_obj = output_root / "catenary_candidate.obj"
    output_mtl = output_root / "catenary_candidate.mtl"
    output_origin = output_root / "model_origin.json"
    output_registry = reports / "catenary_candidate_assets.json"
    output_audit = reports / "catenary_candidate_mesh_audit.json"
    output_report = reports / "catenary_candidate_mesh_build.json"
    outputs = (output_obj, output_mtl, output_origin, output_registry, output_audit, output_report)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite catenary outputs: {existing}")

    origin = np.asarray([frame_origin[0], frame_origin[1], 0.0], dtype=np.float64)
    writer = ObjWriter(origin, material_library=output_mtl.name)
    registry = new_registry(project.project_id)
    for sequence, decision in enumerate(decisions, start=1):
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in candidate_by_id:
            raise ValueError(f"Reviewed candidate is missing: {candidate_id}")
        candidate = candidate_by_id[candidate_id]
        estimated = section["estimated_shaft"]
        s = float(estimated["longitudinal_position_m"])
        c = float(estimated["cross_position_m"])
        center_xy = frame_origin + along_xy * s + cross_xy * c
        along_width = float(np.clip(estimated["along_width_p90_m"], 0.18, 0.40))
        cross_depth = float(np.clip(estimated["cross_width_p90_m"], 0.18, 0.45))
        base_z = float(candidate["minimum_z"])
        top_z = float(candidate["maximum_z"])
        first_bin = section["height_bins"][0]
        foundation_along = float(np.clip(first_bin["along_width_p90_m"], 0.75, 1.25))
        foundation_cross = float(np.clip(first_bin["cross_width_p90_m"], 0.75, 1.25))
        foundation_top = min(base_z + 0.50, top_z - 0.50)
        mast_bottom = foundation_top - 0.08
        mast_id = f"CATENARY-MAST-{sequence:04d}"
        foundation_id = f"{mast_id}-FOUNDATION"
        foundation_vertices, foundation_faces = oriented_box(
            np.asarray([center_xy[0], center_xy[1], base_z]),
            np.asarray([along_xy[0], along_xy[1], 0.0]),
            foundation_cross,
            foundation_along,
            base_z,
            foundation_top,
        )
        writer.add_mesh(
            foundation_id,
            foundation_vertices,
            foundation_faces,
            "CatenaryFoundationConcrete",
        )
        mast_vertices, mast_faces = h_section_prism(
            center_xy,
            along_xy,
            cross_xy,
            along_width,
            cross_depth,
            min(max(cross_depth * 0.10, 0.018), cross_depth * 0.20),
            min(max(along_width * 0.08, 0.016), along_width * 0.16),
            mast_bottom,
            top_z,
        )
        writer.add_mesh(mast_id, mast_vertices, mast_faces, "CatenaryMastSteel")
        sources = [
            {"kind": "point_cloud", "reference": f"{vertical_path}#{candidate_id}"},
            {"kind": "panorama", "reference": str(decision["evidence_sheet"])},
            {"kind": "rule", "reference": str(section_path)},
        ]
        registry["assets"].extend(
            [
                {
                    "id": foundation_id,
                    "type": "catenary_foundation",
                    "status": "candidate",
                    "evidence_level": "photo_interpreted",
                    "confidence": float(decision["confidence"]),
                    "sources": sources,
                    "parameters": {
                        "source_candidate_id": candidate_id,
                        "along_width_m": foundation_along,
                        "cross_width_m": foundation_cross,
                        "height_m": foundation_top - base_z,
                    },
                    "geometry": {"file": str(output_obj), "node": foundation_id},
                    "limitations": ["Foundation outline is a robust lower-bin envelope."],
                },
                {
                    "id": mast_id,
                    "type": "catenary_mast",
                    "subtype": "H_section",
                    "status": "candidate",
                    "evidence_level": "photo_interpreted",
                    "confidence": float(decision["confidence"]),
                    "sources": sources,
                    "parameters": {
                        "source_candidate_id": candidate_id,
                        "longitudinal_position_m": s,
                        "cross_position_m": c,
                        "height_m": top_z - mast_bottom,
                        "along_width_m": along_width,
                        "cross_depth_m": cross_depth,
                        "profile": "point_section_fitted_H_section",
                    },
                    "geometry": {"file": str(output_obj), "node": mast_id},
                    "limitations": [
                        "Exact flange/web product designation is unresolved.",
                        "Cantilever, insulator and wire geometry are not included in this asset stage.",
                    ],
                },
            ]
        )
        registry["relations"].append(
            {
                "id": f"REL-{foundation_id}-SUPPORTS-{mast_id}",
                "type": "supports",
                "from": foundation_id,
                "to": mast_id,
            }
        )
    writer.write(output_obj)
    _write_materials(output_mtl)
    write_json(output_origin, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Catenary candidate registry is invalid: " + "; ".join(errors))
    write_json(output_registry, registry)
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Catenary candidate OBJ failed mesh audit")
    report = {
        "schema_version": "railway.catenary-candidate-mesh-build.v1",
        "project_id": project.project_id,
        "status": "candidate_mesh_built_review_required",
        "source_vertical_report": str(vertical_path),
        "source_semantic_review": str(review_path),
        "source_section_audit": str(section_path),
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_registry": str(output_registry),
        "output_mesh_audit": str(output_audit),
        "model_sha256": sha256_file(output_obj),
        "asset_count": len(registry["assets"]),
        "relation_count": len(registry["relations"]),
        "limitations": [
            "Only the reviewed mast shaft and observed foundation envelope are generated.",
            "Cantilever, insulators and conductors require a separate line-to-mast evidence gate.",
        ],
    }
    write_json(output_report, report)
    return report
