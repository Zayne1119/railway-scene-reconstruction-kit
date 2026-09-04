from __future__ import annotations

import copy
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .algorithms.mesh import ObjWriter
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import parse_obj_model, sample_object_surfaces
from .registry import new_registry, summarize_registry, validate_registry_value
from .station_entry_surface_analysis import analyse_station_entry_surfaces
from .targeted_canopy_integration import merge_candidate_objs


def _local_xyz_to_world(
    local: np.ndarray,
    frame: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = origin[None, :] + values[:, 0, None] * along + values[:, 1, None] * cross
    return np.column_stack((xy, values[:, 2]))


def _panel_prism_local(
    outline_station_z: np.ndarray,
    *,
    cross_m: float,
    thickness_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    outline = np.asarray(outline_station_z, dtype=np.float64)
    if outline.ndim != 2 or outline.shape[1] != 2 or len(outline) < 3:
        raise ValueError("outline_station_z must contain at least three [station, z] points")
    front = np.column_stack((outline[:, 0], np.full(len(outline), cross_m), outline[:, 1]))
    back = front.copy()
    back[:, 1] += thickness_m
    vertices = np.vstack((front, back))
    count = len(outline)
    faces: list[tuple[int, ...]] = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following, count + index))
    return vertices, faces


def _roof_prism_local(
    *,
    station_start_m: float,
    station_end_m: float,
    cross_start_m: float,
    cross_end_m: float,
    start_top_z_m: float,
    end_top_z_m: float,
    thickness_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    top = np.asarray(
        (
            (station_start_m, cross_start_m, start_top_z_m),
            (station_end_m, cross_start_m, end_top_z_m),
            (station_end_m, cross_end_m, end_top_z_m),
            (station_start_m, cross_end_m, start_top_z_m),
        ),
        dtype=np.float64,
    )
    bottom = top.copy()
    bottom[:, 2] -= thickness_m
    vertices = np.vstack((bottom, top))
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def _write_materials(path: Path) -> None:
    path.write_text(
        """# Supplemental observed station-entry materials
newmtl SupplementalEntryCladding
Ka 0.20 0.22 0.22
Kd 0.72 0.75 0.74
Ks 0.12 0.12 0.12
Ns 22.0
d 1.0
illum 2

newmtl SupplementalEntryBlueDoor
Ka 0.02 0.06 0.10
Kd 0.03 0.18 0.42
Ks 0.12 0.18 0.26
Ns 36.0
d 1.0
illum 2

newmtl SupplementalEntryGlass
Ka 0.03 0.08 0.10
Kd 0.15 0.48 0.58
Ks 0.45 0.60 0.65
Ns 90.0
d 0.62
illum 2

newmtl SupplementalEntryFrame
Ka 0.08 0.10 0.11
Kd 0.20 0.27 0.29
Ks 0.35 0.40 0.42
Ns 62.0
d 1.0
illum 2
""",
        encoding="utf-8",
        newline="\n",
    )


def _load_station_entry_cloud(
    cloud: Path,
    frame: dict[str, Any],
    *,
    station_bounds_m: tuple[float, float] = (97.5, 124.5),
    cross_bounds_m: tuple[float, float] = (15.0, 20.7),
    z_bounds_m: tuple[float, float] = (20.0, 27.5),
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    parts: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                (
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
            )
            local = xyz[:, :2] - origin
            station = local @ along
            lateral = local @ cross
            selected = (
                (station >= station_bounds_m[0])
                & (station <= station_bounds_m[1])
                & (lateral >= cross_bounds_m[0])
                & (lateral <= cross_bounds_m[1])
                & (xyz[:, 2] >= z_bounds_m[0])
                & (xyz[:, 2] <= z_bounds_m[1])
            )
            if np.any(selected):
                parts.append(xyz[selected])
    return np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64)


def _support_metrics(
    addon_obj: Path,
    model_origin: np.ndarray,
    cloud_points: np.ndarray,
    observed_nodes: set[str],
) -> dict[str, dict[str, float | int | bool]]:
    model = parse_obj_model(addon_obj)
    samples = sample_object_surfaces(
        model,
        origin_xyz=model_origin,
        maximum_samples_per_object=8_000,
    )
    tree = cKDTree(cloud_points)
    records: dict[str, dict[str, float | int | bool]] = {}
    for node, points in samples.items():
        if node not in observed_nodes:
            continue
        distances, _ = tree.query(points, workers=-1)
        p90 = float(np.percentile(distances, 90))
        coverage = float(np.mean(distances <= 0.30))
        records[node] = {
            "sample_count": len(points),
            "p50_m": float(np.percentile(distances, 50)),
            "p90_m": p90,
            "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
            "coverage_at_0_30m": coverage,
            "passed_review_gate": p90 <= 0.45 and coverage >= 0.65,
        }
    return records


def build_supplemental_station_entry(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    surface_analysis_path: str | Path,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    photo_evidence_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cloud = Path(cloud_path).resolve()
    source_model = Path(source_obj).resolve()
    source_origin_path = Path(source_origin).resolve()
    source_registry_path = Path(source_registry).resolve()
    evidence_directory = Path(photo_evidence_directory).resolve()
    frame = load_json(Path(frame_report_path))["frame"]
    surface_analysis_file = Path(surface_analysis_path).resolve()
    if not surface_analysis_file.is_file():
        analyse_station_entry_surfaces(
            cloud_path=cloud,
            frame_report_path=frame_report_path,
            output_path=surface_analysis_file,
        )
    analysis = load_json(surface_analysis_file)
    planes = {item["name"]: item for item in analysis["plane_bands"]}
    segmented = analysis["segmented_cross_histogram_peaks"]

    def segment_cross_peak(
        segment_index: int,
        minimum_cross_m: float,
        maximum_cross_m: float,
    ) -> float:
        matches = [
            item
            for item in segmented[segment_index]["cross_peaks"]
            if minimum_cross_m <= float(np.mean(item["cross_range_m"])) <= maximum_cross_m
        ]
        if not matches:
            raise ValueError("Required segmented facade peak was not detected")
        return float(np.mean(max(matches, key=lambda item: item["point_count"])["cross_range_m"]))

    platform_wall_peaks = [
        float(np.mean(item["cross_range_m"]))
        for item in segmented[1]["cross_peaks"]
        if 15.0 <= float(np.mean(item["cross_range_m"])) <= 16.2
        and int(item["point_count"]) >= 1_000
    ]
    if len(platform_wall_peaks) < 2:
        raise ValueError("Both faces of the platform-side wall were not detected")
    platform_cross = min(platform_wall_peaks)
    platform_back_cross = max(platform_wall_peaks)
    outer_cross = segment_cross_peak(0, 19.0, 21.0)
    outer_back_cross = float(
        np.mean(
            max(
                (
                    item
                    for item in segmented[0]["cross_peaks"]
                    if outer_cross + 0.10
                    <= float(np.mean(item["cross_range_m"]))
                    <= outer_cross + 0.50
                ),
                key=lambda item: item["point_count"],
            )["cross_range_m"]
        )
    )
    slope_platform_cross = segment_cross_peak(2, 15.0, 16.2)
    slope_outer_cross = segment_cross_peak(2, 19.0, 21.0)
    base_z = 21.13
    upper_top_z = float(
        np.median(
            [
                item["z_percentiles_m"]["p90"]
                for item in planes["platform_side_facade"]["station_profile"]
                if 102.0 <= float(item["station_range_m"][0]) < 109.0
            ]
        )
    )
    slope_records = [
        item
        for item in planes["platform_side_facade"]["station_profile"]
        if 115.0 <= float(item["station_range_m"][0]) < 124.0
    ]
    slope_station = np.asarray(
        [float(np.mean(item["station_range_m"])) for item in slope_records],
        dtype=np.float64,
    )
    slope_z = np.asarray(
        [float(item["z_percentiles_m"]["p50"]) for item in slope_records],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(slope_station, slope_z, 1)
    slope_start = 115.0
    slope_end = 123.45
    slope_start_z = float(slope * slope_start + intercept)
    slope_end_z = float(max(base_z + 0.22, slope * slope_end + intercept))

    origin_value = load_json(source_origin_path)
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    addon_obj = output / "supplemental_station_entry.obj"
    addon_mtl = output / "supplemental_station_entry.mtl"
    addon_origin = output / "supplemental_station_entry_origin.json"
    writer = ObjWriter(model_origin, material_library=addon_mtl.name)

    emitted: list[dict[str, Any]] = []

    def add_panel(
        node: str,
        outline: tuple[tuple[float, float], ...],
        cross_m: float,
        thickness_m: float,
        material: str,
        asset_type: str,
        evidence_level: str,
        confidence: float,
    ) -> None:
        local_vertices, faces = _panel_prism_local(
            np.asarray(outline, dtype=np.float64),
            cross_m=cross_m,
            thickness_m=thickness_m,
        )
        writer.add_mesh(node, _local_xyz_to_world(local_vertices, frame), faces, material)
        emitted.append(
            {
                "id": node,
                "type": asset_type,
                "evidence_level": evidence_level,
                "confidence": confidence,
                "station_range_m": [
                    float(np.min(local_vertices[:, 0])),
                    float(np.max(local_vertices[:, 0])),
                ],
                "cross_range_m": [
                    float(np.min(local_vertices[:, 1])),
                    float(np.max(local_vertices[:, 1])),
                ],
            }
        )

    add_panel(
        "SUPPLEMENTAL-STATION-ENTRY-01-UPPER-CLADDING",
        ((101.55, 23.62), (110.55, 23.62), (110.55, upper_top_z), (101.55, upper_top_z)),
        platform_cross,
        platform_back_cross - platform_cross,
        "SupplementalEntryCladding",
        "station_entry_cladding",
        "observed",
        0.86,
    )
    door_edges = np.linspace(102.0, 110.5, 5)
    for index, (start, end) in enumerate(pairwise(door_edges), 1):
        add_panel(
            f"SUPPLEMENTAL-STATION-ENTRY-01-BLUE-DOOR-{index:02d}",
            ((float(start), base_z), (float(end), base_z), (float(end), 23.58), (float(start), 23.58)),
            platform_cross - 0.015,
            0.055,
            "SupplementalEntryBlueDoor",
            "station_entry_door",
            "photo_interpreted",
            0.78,
        )
    add_panel(
        "SUPPLEMENTAL-STATION-ENTRY-01-OUTER-WALL-A",
        ((98.1, 20.72), (103.25, 20.72), (103.25, upper_top_z), (98.1, upper_top_z)),
        outer_cross,
        outer_back_cross - outer_cross,
        "SupplementalEntryCladding",
        "station_entry_outer_wall",
        "observed",
        0.84,
    )

    pane_edges = np.linspace(slope_start, slope_end, 6)
    for index, (start, end) in enumerate(pairwise(pane_edges), 1):
        start_top = float(slope * start + intercept)
        end_top = float(max(base_z + 0.22, slope * end + intercept))
        add_panel(
            f"SUPPLEMENTAL-STATION-ENTRY-01-GLASS-{index:02d}",
            ((float(start), base_z), (float(end), base_z), (float(end), end_top), (float(start), start_top)),
            slope_platform_cross,
            0.055,
            "SupplementalEntryGlass",
            "station_entry_glazing",
            "photo_interpreted",
            0.82,
        )
        add_panel(
            f"SUPPLEMENTAL-STATION-ENTRY-01-OUTER-GLASS-{index:02d}",
            ((float(start), base_z), (float(end), base_z), (float(end), end_top), (float(start), start_top)),
            slope_outer_cross,
            0.055,
            "SupplementalEntryGlass",
            "station_entry_glazing",
            "photo_interpreted",
            0.78,
        )

    for index, station in enumerate(pane_edges, 1):
        top_z = float(max(base_z + 0.22, slope * station + intercept))
        add_panel(
            f"SUPPLEMENTAL-STATION-ENTRY-01-GLASS-MULLION-{index:02d}",
            (
                (float(station - 0.025), base_z),
                (float(station + 0.025), base_z),
                (float(station + 0.025), top_z),
                (float(station - 0.025), top_z),
            ),
            slope_platform_cross - 0.035,
            0.035,
            "SupplementalEntryFrame",
            "station_entry_glazing_frame",
            "photo_interpreted",
            0.76,
        )
    add_panel(
        "SUPPLEMENTAL-STATION-ENTRY-01-GLASS-TOP-FRAME",
        (
            (slope_start, slope_start_z - 0.10),
            (slope_end, slope_end_z - 0.10),
            (slope_end, slope_end_z),
            (slope_start, slope_start_z),
        ),
        slope_platform_cross - 0.035,
        0.035,
        "SupplementalEntryFrame",
        "station_entry_glazing_frame",
        "photo_interpreted",
        0.76,
    )
    writer.write(addon_obj)
    _write_materials(addon_mtl)
    write_json(addon_origin, origin_value)

    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [(source_model, source_origin_path), (addon_obj, addon_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )

    cloud_points = _load_station_entry_cloud(cloud, frame)
    observed_nodes = {
        item["id"]
        for item in emitted
        if item["evidence_level"] == "observed"
    }
    support = _support_metrics(addon_obj, model_origin, cloud_points, observed_nodes)
    support_passed = sum(bool(item["passed_review_gate"]) for item in support.values())

    registry = copy.deepcopy(load_json(source_registry_path))
    for asset in registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    photo_references = [
        str(evidence_directory / f"GAP-VERTICAL-{suffix}.jpg")
        for suffix in ("0019", "0020", "0021", "0022", "0025", "0027", "0028", "0041", "0042")
    ]
    new_assets: list[dict[str, Any]] = []
    for record in emitted:
        evidence_level = str(record["evidence_level"])
        sources: list[dict[str, str]] = [
            {"kind": "point_cloud", "reference": str(surface_analysis_file)}
        ]
        if record["type"] in {
            "station_entry_door",
            "station_entry_glazing",
            "station_entry_glazing_frame",
        }:
            sources.append({"kind": "panorama", "reference": photo_references[0]})
        asset = {
            "id": record["id"],
            "type": record["type"],
            "subtype": "visible_station_entry_component",
            "status": "candidate",
            "chainage_m": float(np.mean(record["station_range_m"])),
            "evidence_level": evidence_level,
            "confidence": record["confidence"],
            "sources": sources,
            "parameters": {
                "station_range_m": record["station_range_m"],
                "cross_range_m": record["cross_range_m"],
                "geometry_method": "grouped_dense_cloud_plane_fit_with_photo_semantics",
            },
            "geometry": {"file": str(output_obj), "node": record["id"]},
            "limitations": [
                "Only visible and point-supported entrance surfaces are represented.",
                "Interior structure and the unobserved rear enclosure are intentionally excluded.",
            ],
        }
        if record["id"] in support:
            asset["parameters"]["point_support"] = support[record["id"]]
        new_assets.append(asset)
        registry["assets"].append(asset)
    registry["release_id"] = output_name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    additions = new_registry(str(registry.get("project_id", "site-b")))
    additions["assets"] = copy.deepcopy(new_assets)
    additions["summary"] = summarize_registry(additions)
    addition_errors = validate_registry_value(additions)
    if addition_errors:
        raise ValueError("Station-entry assets are invalid: " + "; ".join(addition_errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Supplemental station-entry candidate failed mesh audit")
    report = output / "supplemental_station_entry_report.json"
    write_json(
        report,
        {
            "schema_version": "railway.supplemental-station-entry.v1",
            "source_obj": str(source_model),
            "output_obj": str(output_obj),
            "surface_analysis": str(surface_analysis_file),
            "fit": {
                "platform_side_cross_m": platform_cross,
                "platform_side_back_cross_m": platform_back_cross,
                "outer_cross_m": outer_cross,
                "outer_back_cross_m": outer_back_cross,
                "sloped_platform_side_cross_m": slope_platform_cross,
                "sloped_outer_side_cross_m": slope_outer_cross,
                "base_z_m": base_z,
                "upper_top_z_m": upper_top_z,
                "sloped_top_line": {"slope": float(slope), "intercept": float(intercept)},
            },
            "new_assets": new_assets,
            "point_support": support,
            "point_support_gate_summary": {
                "passed_node_count": support_passed,
                "evaluated_node_count": len(support),
                "all_passed": support_passed == len(support),
            },
            "merge": merge,
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_built_fixed_view_and_gap_regression_review_required",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": report,
    }
