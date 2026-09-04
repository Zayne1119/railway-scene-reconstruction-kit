from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .canopy_second_row_candidate import clone_obj_objects
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value


def robust_column_target(points_station_cross_z: np.ndarray) -> dict[str, float]:
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    if len(points) < 100:
        raise ValueError("Insufficient observed shaft points")
    minimum_z = float(np.min(points[:, 2]))
    maximum_z = float(np.max(points[:, 2]))
    shaft = points[
        (points[:, 2] >= minimum_z + 0.35)
        & (points[:, 2] <= minimum_z + 0.72 * (maximum_z - minimum_z))
    ]
    if len(shaft) < 80:
        raise ValueError("Insufficient mid-shaft points")
    station_low, station_high = np.percentile(shaft[:, 0], (3, 97))
    cross_low, cross_high = np.percentile(shaft[:, 1], (3, 97))
    return {
        "station_m": float((station_low + station_high) / 2.0),
        "cross_m": float((cross_low + cross_high) / 2.0),
        "observed_along_extent_m": float(station_high - station_low),
        "observed_cross_extent_m": float(cross_high - cross_low),
        "base_z_m": float(np.percentile(points[:, 2], 0.5)),
        "maximum_observed_z_m": float(np.percentile(points[:, 2], 99.5)),
        "source_point_count": len(points),
        "shaft_point_count": len(shaft),
    }


def _load_candidate_points(
    cloud_path: Path,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    chunk_size: int = 2_000_000,
) -> dict[str, np.ndarray]:
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    collected: dict[str, list[np.ndarray]] = {
        str(item["candidate_id"]): [] for item in candidates
    }
    with laspy.open(cloud_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            world = np.column_stack((x, y, z))
            local_xy = world[:, :2] - origin_xy
            station = local_xy @ along
            lateral = local_xy @ cross
            for candidate in candidates:
                minimum = np.asarray(candidate["minimum_xyz_m"], dtype=np.float64)
                maximum = np.asarray(candidate["maximum_xyz_m"], dtype=np.float64)
                selected = np.all(
                    (world >= minimum[None, :]) & (world <= maximum[None, :]), axis=1
                )
                if np.any(selected):
                    collected[str(candidate["candidate_id"])].append(
                        np.column_stack((station[selected], lateral[selected], z[selected]))
                    )
    return {
        key: np.vstack(value) if value else np.empty((0, 3), dtype=np.float64)
        for key, value in collected.items()
    }


def _object_center_and_base(
    model: Any, object_name: str, origin: np.ndarray
) -> tuple[np.ndarray, float]:
    indexes = object_vertex_indices(model, object_name)
    if not len(indexes):
        raise ValueError(f"Template object is absent: {object_name}")
    vertices = model.vertices[indexes] + origin
    return (np.min(vertices, axis=0) + np.max(vertices, axis=0))[:2] / 2.0, float(
        np.min(vertices[:, 2])
    )


def _sample_object_surface(
    model: Any,
    object_name: str,
    origin: np.ndarray,
    *,
    spacing_m: float = 0.04,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    rng = np.random.default_rng(20260902)
    for face in model.faces_by_object.get(object_name, []):
        for index in range(1, len(face) - 1):
            triangle = model.vertices[[face[0], face[index], face[index + 1]]]
            area = float(
                np.linalg.norm(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                )
                / 2.0
            )
            if area <= 1.0e-12:
                continue
            count = max(1, int(np.ceil(area / (spacing_m * spacing_m * 0.5))))
            root = np.sqrt(rng.random(count))
            second = rng.random(count)
            samples.append(
                (1.0 - root)[:, None] * triangle[0]
                + (root * (1.0 - second))[:, None] * triangle[1]
                + (root * second)[:, None] * triangle[2]
            )
    if not samples:
        raise ValueError(f"No surface samples for endpoint column: {object_name}")
    return np.vstack(samples) + origin


def _candidate_point_to_mesh_support(
    *,
    model: Any,
    origin: np.ndarray,
    object_name: str,
    points_station_cross_z: np.ndarray,
    frame: dict[str, Any],
) -> dict[str, Any]:
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    minimum_z = float(np.min(points[:, 2]))
    maximum_z = float(np.max(points[:, 2]))
    shaft = points[
        (points[:, 2] >= minimum_z + 0.35)
        & (points[:, 2] <= minimum_z + 0.72 * (maximum_z - minimum_z))
    ]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    world_xy = (
        frame_origin[None, :]
        + shaft[:, 0, None] * along[None, :]
        + shaft[:, 1, None] * cross[None, :]
    )
    world = np.column_stack((world_xy, shaft[:, 2]))
    surface = _sample_object_surface(model, object_name, origin)
    distances, _ = cKDTree(surface).query(world, workers=-1)
    p90 = float(np.percentile(distances, 90))
    coverage = float(np.mean(distances <= 0.20))
    gates = {
        "p90_within_0_20m": p90 <= 0.20,
        "coverage_at_0_20m_at_least_0_90": coverage >= 0.90,
    }
    return {
        "shaft_point_count": len(shaft),
        "model_surface_sample_count": len(surface),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": p90,
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_10m": float(np.mean(distances <= 0.10)),
        "coverage_at_0_20m": coverage,
        "gates": gates,
        "passed": all(gates.values()),
        "direction": "observed_candidate_shaft_points_to_dense_column_mesh_surface",
    }


def _row_grid_target(
    *,
    model: Any,
    origin: np.ndarray,
    registry: dict[str, Any],
    frame: dict[str, Any],
    source_root: str,
    observed_station_m: float,
    observed_cross_m: float,
) -> dict[str, float]:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    source_xy, _ = _object_center_and_base(model, source_root, origin)
    source_delta = source_xy - frame_origin
    source_cross = float(source_delta @ cross)
    row: list[tuple[float, float]] = []
    for asset in registry.get("assets", []):
        if asset.get("type") != "canopy_column":
            continue
        name = str(asset.get("geometry", {}).get("node", asset.get("id", "")))
        if name not in model.faces_by_object:
            continue
        center_xy, _ = _object_center_and_base(model, name, origin)
        delta = center_xy - frame_origin
        station_m = float(delta @ along)
        cross_m = float(delta @ cross)
        if abs(cross_m - source_cross) <= 1.0:
            row.append((station_m, cross_m))
    if len(row) < 4:
        raise ValueError(f"Too few accepted columns to extrapolate row for {source_root}")
    row.sort()
    stations = np.asarray([item[0] for item in row], dtype=np.float64)
    differences = np.diff(stations)
    spacing = float(np.median(differences[(differences >= 7.0) & (differences <= 11.0)]))
    if not 7.0 <= spacing <= 11.0:
        raise ValueError(f"Invalid canopy-column spacing for {source_root}: {spacing}")
    if observed_station_m < float(stations[0]):
        target_station = float(stations[0] - spacing)
    elif observed_station_m > float(stations[-1]):
        target_station = float(stations[-1] + spacing)
    else:
        raise ValueError("Endpoint-column candidate lies inside the accepted row range")
    target_cross = float(np.median([item[1] for item in row]))
    station_offset = observed_station_m - target_station
    cross_offset = observed_cross_m - target_cross
    if abs(station_offset) > 0.75 or abs(cross_offset) > 0.60:
        raise ValueError(
            f"Point evidence disagrees with row extrapolation for {source_root}: "
            f"station {station_offset:.3f} m, cross {cross_offset:.3f} m"
        )
    return {
        "grid_station_m": target_station,
        "grid_cross_m": target_cross,
        "row_spacing_m": spacing,
        "row_column_count": len(row),
        "observed_to_grid_station_offset_m": station_offset,
        "observed_to_grid_cross_offset_m": cross_offset,
    }


def build_supplemental_endpoint_columns(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gap_report = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    candidates = [
        item
        for item in gap_report["vertical_candidates"]
        if item.get("priority") == "P0"
        and item.get("asset_relation") == "new_standalone_vertical_candidate"
    ]
    if len(candidates) != 3:
        raise ValueError(f"Expected three endpoint-column candidates; got {len(candidates)}")
    model = parse_obj_model(source_obj)
    origin_value = load_json(Path(source_origin))
    origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    assets = {str(item["id"]): item for item in registry.get("assets", [])}
    point_groups = _load_candidate_points(Path(cloud_path), candidates, frame)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    root_names = {
        "GAP-VERTICAL-0001": "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-START",
        "GAP-VERTICAL-0002": "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-END",
        "GAP-VERTICAL-0003": "SUPPLEMENTAL-OPPOSITE-CANOPY-END-COLUMN",
    }
    clone_specs: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        fit = robust_column_target(point_groups[candidate_id])
        source_root = str(candidate["nearest_existing_vertical_asset"]["asset_id"])
        if source_root not in assets:
            raise ValueError(f"Column template is absent from registry: {source_root}")
        target_root = root_names[candidate_id]
        grid = _row_grid_target(
            model=model,
            origin=origin,
            registry=registry,
            frame=frame,
            source_root=source_root,
            observed_station_m=fit["station_m"],
            observed_cross_m=fit["cross_m"],
        )
        target_xy = (
            frame_origin
            + grid["grid_station_m"] * along
            + grid["grid_cross_m"] * cross
        )
        source_xy, source_base_z = _object_center_and_base(model, source_root, origin)
        translation_xyz = np.asarray(
            [
                target_xy[0] - source_xy[0],
                target_xy[1] - source_xy[1],
                fit["base_z_m"] - source_base_z,
            ],
            dtype=np.float64,
        )
        associated = sorted(
            name
            for name in model.faces_by_object
            if name == source_root or name.startswith(f"{source_root}-")
        )
        if not associated:
            raise ValueError(f"Column template assembly is absent: {source_root}")
        for source_name in associated:
            suffix = source_name[len(source_root) :]
            clone_specs.append(
                {
                    "source_object": source_name,
                    "clone_object": f"{target_root}{suffix}",
                    "translation_local_xyz_m": translation_xyz.tolist(),
                    "candidate_id": candidate_id,
                    "source_root": source_root,
                    "target_root": target_root,
                    "station_m": grid["grid_station_m"],
                    "cross_m": grid["grid_cross_m"],
                }
            )
        fit_records.append(
            {
                "candidate_id": candidate_id,
                "asset_id": target_root,
                "source_template_asset_id": source_root,
                **fit,
                **grid,
                "translation_xyz_m": translation_xyz.tolist(),
                "component_count": len(associated),
            }
        )
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    clone_obj_objects(
        source_obj,
        output_obj,
        clone_specs,
        output_mtl_name=output_mtl.name,
    )
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(source_origin, output_origin)
    output_model = parse_obj_model(output_obj)
    for record in fit_records:
        record["point_to_mesh_support"] = _candidate_point_to_mesh_support(
            model=output_model,
            origin=origin,
            object_name=str(record["asset_id"]),
            points_station_cross_z=point_groups[str(record["candidate_id"])],
            frame=frame,
        )
    if not all(record["point_to_mesh_support"]["passed"] for record in fit_records):
        raise ValueError("At least one endpoint column failed observed point-to-mesh support")

    updated_registry = copy.deepcopy(registry)
    for asset in updated_registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    added_assets: list[dict[str, Any]] = []
    added_relations: list[dict[str, Any]] = []
    for spec in clone_specs:
        source = copy.deepcopy(assets[str(spec["source_object"])])
        source["id"] = str(spec["clone_object"])
        source["status"] = "candidate"
        source["evidence_level"] = "observed"
        source["confidence"] = 0.94 if source.get("type") == "canopy_column" else 0.88
        source["chainage_m"] = float(spec["station_m"])
        source["geometry"] = {"file": str(output_obj), "node": str(spec["clone_object"])}
        source.setdefault("parameters", {}).update(
            {
                "source_template_asset_id": str(spec["source_object"]),
                "supplemental_candidate_id": str(spec["candidate_id"]),
                "cross_position_m": float(spec["cross_m"]),
                "geometry_method": "observed_center_plus_nearest_accepted_assembly",
            }
        )
        source["sources"] = [
            {"kind": "point_cloud", "reference": str(Path(cloud_path).resolve())},
            {
                "kind": "rule",
                "reference": str(Path(gap_report_path).resolve()),
                "note": "Reverse gap audit plus dense coloured three-view confirmation.",
            },
        ]
        source["limitations"] = [
            "Observed position and base elevation; section and hidden connection reuse the nearest accepted row template."
        ]
        added_assets.append(source)
        updated_registry["assets"].append(source)
    for record in fit_records:
        root = str(record["asset_id"])
        component_ids = [
            str(item["clone_object"])
            for item in clone_specs
            if item["target_root"] == root and item["clone_object"] != root
        ]
        previous = root
        for component_id in component_ids:
            relation = {
                "id": f"REL-{previous}-SUPPORTS-{component_id}",
                "type": "supports",
                "from": previous,
                "to": component_id,
            }
            added_relations.append(relation)
            updated_registry.setdefault("relations", []).append(relation)
            previous = component_id
    updated_registry["release_id"] = output_name
    updated_registry["updated_at"] = datetime.now(UTC).isoformat()
    updated_registry["summary"] = summarize_registry(updated_registry)
    inherited_errors = validate_registry_value(registry)
    addition_registry = new_registry(str(registry.get("project_id", "site-b")))
    addition_registry["assets"] = copy.deepcopy(added_assets)
    addition_registry["relations"] = copy.deepcopy(added_relations)
    addition_registry["summary"] = summarize_registry(addition_registry)
    addition_errors = validate_registry_value(addition_registry)
    if addition_errors:
        raise ValueError("Endpoint-column additions are invalid: " + "; ".join(addition_errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, updated_registry)
    mesh_audit = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Endpoint-column candidate failed mesh audit")
    output_report = output / "supplemental_endpoint_column_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.supplemental-endpoint-columns.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "fit_records": fit_records,
            "added_asset_count": len(added_assets),
            "added_relation_count": len(added_relations),
            "new_assets_schema_valid": True,
            "inherited_registry_validation_error_count": len(inherited_errors),
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_requires_point_support_and_fixed_view_review",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
