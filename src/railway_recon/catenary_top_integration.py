from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .algorithms.mesh import ObjWriter
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value
from .targeted_canopy_integration import merge_candidate_objs

Mesh = tuple[np.ndarray, list[tuple[int, ...]]]


def tube_between(
    start: np.ndarray, end: np.ndarray, radius_m: float, *, sides: int = 8
) -> Mesh:
    start_value = np.asarray(start, dtype=np.float64)
    end_value = np.asarray(end, dtype=np.float64)
    axis = end_value - start_value
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-6 or radius_m <= 0.0 or sides < 6:
        raise ValueError("Invalid tube dimensions")
    direction = axis / length
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(direction @ helper)) > 0.90:
        helper = np.asarray([1.0, 0.0, 0.0])
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    offsets = radius_m * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    vertices = np.vstack((start_value + offsets, end_value + offsets))
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sides)))]
    faces.append(tuple(sides + index for index in range(sides)))
    for index in range(sides):
        following = (index + 1) % sides
        faces.append((index, following, following + sides, index + sides))
    return vertices, faces


def ribbed_insulator(start: np.ndarray, end: np.ndarray, *, sides: int = 10) -> Mesh:
    start_value = np.asarray(start, dtype=np.float64)
    end_value = np.asarray(end, dtype=np.float64)
    axis = end_value - start_value
    length = float(np.linalg.norm(axis))
    if length <= 0.12:
        raise ValueError("Insulator axis is too short")
    direction = axis / length
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(direction @ helper)) > 0.90:
        helper = np.asarray([1.0, 0.0, 0.0])
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    positions = np.asarray([0.0, 0.12, 0.20, 0.30, 0.38, 0.48, 0.56, 0.68, 0.76, 0.88, 1.0])
    radii = np.asarray([0.045, 0.050, 0.105, 0.052, 0.105, 0.052, 0.105, 0.052, 0.105, 0.050, 0.045])
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rings = []
    for position, radius in zip(positions, radii, strict=True):
        center = start_value + position * axis
        rings.append(
            center
            + radius
            * (
                np.cos(angles)[:, None] * first
                + np.sin(angles)[:, None] * second
            )
        )
    vertices = np.vstack(rings)
    faces: list[tuple[int, ...]] = [tuple(reversed(range(sides)))]
    for ring in range(len(rings) - 1):
        a = ring * sides
        b = (ring + 1) * sides
        for index in range(sides):
            following = (index + 1) % sides
            faces.append((a + index, a + following, b + following, b + index))
    final = (len(rings) - 1) * sides
    faces.append(tuple(final + index for index in range(sides)))
    return vertices, faces


def combine_meshes(meshes: list[Mesh]) -> Mesh:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, ...]] = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(mesh_vertices)
        faces.extend(tuple(index + offset for index in face) for face in mesh_faces)
        offset += len(mesh_vertices)
    if not vertices:
        raise ValueError("At least one mesh is required")
    return np.vstack(vertices), faces


def _point_on_component(
    station_m: float,
    cross_m: float,
    direction: int,
    u_m: float,
    z_m: float,
    frame_origin: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    xy = frame_origin + station_m * along + (cross_m + direction * u_m) * cross
    return np.asarray([xy[0], xy[1], z_m], dtype=np.float64)


def _sample_segment(start: np.ndarray, end: np.ndarray, count: int = 80) -> np.ndarray:
    return np.linspace(start, end, count)


def _support(samples_world: np.ndarray, local_points_world: np.ndarray) -> dict[str, Any]:
    distances, _ = cKDTree(local_points_world).query(samples_world, workers=-1)
    gates = {
        "p90_within_0_16m": float(np.percentile(distances, 90)) <= 0.16,
        "coverage_at_0_20m_at_least_0_90": float(np.mean(distances <= 0.20)) >= 0.90,
    }
    return {
        "sample_count": len(samples_world),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_10m": float(np.mean(distances <= 0.10)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _material_file(path: Path) -> None:
    path.write_text(
        """# Point-supported catenary top equipment
newmtl PointFittedCantileverSteel
Ka 0.06 0.15 0.18
Kd 0.10 0.62 0.75
Ks 0.28 0.30 0.32
Ns 42.0
d 1.0
illum 2

newmtl PointFittedPositionerSteel
Ka 0.08 0.12 0.13
Kd 0.22 0.48 0.55
Ks 0.32 0.34 0.35
Ns 48.0
d 1.0
illum 2

newmtl PointFittedInsulator
Ka 0.12 0.11 0.07
Kd 0.63 0.54 0.31
Ks 0.18 0.18 0.16
Ns 28.0
d 1.0
illum 2
""",
        encoding="utf-8",
        newline="\n",
    )


def integrate_catenary_top_equipment(
    *,
    evidence_report_path: str | Path,
    local_points_path: str | Path,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    evidence_file = Path(evidence_report_path).resolve()
    points_file = Path(local_points_path).resolve()
    source_model = Path(source_obj).resolve()
    source_origin_file = Path(source_origin).resolve()
    source_registry_file = Path(source_registry).resolve()
    frame_file = Path(frame_report_path).resolve()
    for path in (
        evidence_file,
        points_file,
        source_model,
        source_origin_file,
        source_registry_file,
        frame_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    evidence = load_json(evidence_file)
    eligible = [item for item in evidence["records"] if item["evidence"]["passed"]]
    if len(eligible) != int(evidence["summary"]["build_eligible_count"]):
        raise ValueError("Evidence summary does not match eligible mast records")
    default_frame = load_json(frame_file)["frame"]
    selection_by_candidate: dict[str, dict[str, Any]] = {}
    selection_report = evidence.get("selection_report")
    if selection_report:
        selection = load_json(Path(selection_report))
        selection_by_candidate = {
            str(item["candidate_id"]): item
            for item in selection.get("candidates", [])
            if item.get("candidate_id")
        }
    origin_value = load_json(source_origin_file)
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    point_archive = np.load(points_file)

    addon_obj = output / "point_supported_catenary_top_equipment.obj"
    addon_mtl = output / "point_supported_catenary_top_equipment.mtl"
    addon_origin = output / "point_supported_catenary_top_equipment_origin.json"
    writer = ObjWriter(model_origin, material_library=addon_mtl.name)
    records: list[dict[str, Any]] = []
    component_assets: list[dict[str, Any]] = []
    component_relations: list[dict[str, Any]] = []
    for mast in eligible:
        asset_id = str(mast["asset_id"])
        station = float(mast["station_m"])
        lateral = float(mast["cross_m"])
        direction = int(mast["inward_cross_direction"])
        result = mast["evidence"]
        frame = default_frame
        frame_source = str(frame_file)
        source_candidate_id = str(mast.get("source_candidate_id", ""))
        selection_candidate = selection_by_candidate.get(source_candidate_id)
        if selection_candidate and selection_candidate.get("report"):
            candidate_report = Path(str(selection_candidate["report"])).resolve()
            frame = load_json(candidate_report)["frame"]
            frame_source = str(candidate_report)
        frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
        along = np.asarray(frame["along_xy"], dtype=np.float64)
        cross = np.asarray(frame["cross_xy"], dtype=np.float64)

        def point(
            u_z: list[float],
            station_value: float = station,
            lateral_value: float = lateral,
            direction_value: int = direction,
            frame_origin_value: np.ndarray = frame_origin,
            along_value: np.ndarray = along,
            cross_value: np.ndarray = cross,
        ) -> np.ndarray:
            return _point_on_component(
                station_value,
                lateral_value,
                direction_value,
                float(u_z[0]),
                float(u_z[1]),
                frame_origin_value,
                along_value,
                cross_value,
            )

        local = point_archive[asset_id.replace("-", "_")]
        local_world = np.column_stack(
            (
                frame_origin[None, :]
                + (station + local[:, 0])[:, None] * along[None, :]
                + (lateral + direction * local[:, 1])[:, None] * cross[None, :],
                local[:, 2],
            )
        )
        main_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for trace in result["linear_traces"]:
            if (
                abs(float(trace["slope_dz_du"])) <= 0.18
                and float(trace["transverse_span_m"]) >= 1.50
                and float(trace["u_range_m"][0]) <= 0.90
            ):
                trace_start, trace_end = (
                    point(value) for value in trace["endpoints_u_z_m"]
                )
                trace_support = _support(
                    _sample_segment(trace_start, trace_end), local_world
                )
                if trace_support["passed"]:
                    main_candidates.append((trace, trace_support))
        if not main_candidates:
            raise ValueError(f"Eligible mast lacks a point-supported main trace: {asset_id}")
        main, main_trace_support = min(
            main_candidates, key=lambda item: float(item[1]["p90_m"])
        )
        _, main_end = (point(value) for value in main["endpoints_u_z_m"])
        shaft_surface_u = 0.14
        connected_start = point(
            [
                shaft_surface_u,
                float(main["slope_dz_du"]) * shaft_surface_u
                + float(main["intercept_z_m"]),
            ]
        )
        cantilever_segments = [(connected_start, main_end)]
        traces = list(result["selected_brace_or_positioner_traces"])
        segment_candidates: list[dict[str, Any]] = []
        for trace in traces:
            start, end = (point(value) for value in trace["endpoints_u_z_m"])
            trace_support = _support(_sample_segment(start, end), local_world)
            segment_candidates.append(
                {
                    "source": str(trace["trace_id"]),
                    "start": start,
                    "end": end,
                    "support": trace_support,
                }
            )
        secondary = result["secondary_structure_evidence"]
        vertex = point(secondary["vertex_u_z_m"])
        fallback_support = _support(_sample_segment(vertex, main_end), local_world)
        segment_candidates.append(
            {
                "source": "secondary_structure_vertex_to_main_end",
                "start": vertex,
                "end": main_end,
                "support": fallback_support,
            }
        )
        passing_segments = sorted(
            (item for item in segment_candidates if item["support"]["passed"]),
            key=lambda item: float(item["support"]["p90_m"]),
        )
        if not passing_segments:
            raise ValueError(f"No point-supported brace or positioner segment: {asset_id}")
        if len(passing_segments) >= 2:
            cantilever_segments.append(
                (passing_segments[0]["start"], passing_segments[0]["end"])
            )
            positioner_segment = passing_segments[1]
        else:
            positioner_segment = passing_segments[0]
        cantilever_mesh = combine_meshes(
            [tube_between(start, end, 0.055) for start, end in cantilever_segments]
        )
        cantilever_id = f"{asset_id}-CANTILEVER"
        writer.add_mesh(cantilever_id, *cantilever_mesh, "PointFittedCantileverSteel")

        positioner_start = positioner_segment["start"]
        positioner_end = positioner_segment["end"]
        positioner_id = f"{asset_id}-POSITIONER"
        writer.add_mesh(
            positioner_id,
            *tube_between(positioner_start, positioner_end, 0.037),
            "PointFittedPositionerSteel",
        )

        attachments = sorted(
            result["attachment_clusters"],
            key=lambda item: float(item["z_span_m"]) * np.sqrt(float(item["point_count"])),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        for attachment in attachments:
            if not chosen or all(
                abs(float(attachment["u_m"]) - float(item["u_m"])) >= 0.28
                for item in chosen
            ):
                chosen.append(attachment)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            raise ValueError(f"Eligible mast lacks two separated attachment clusters: {asset_id}")
        insulator_records: list[dict[str, Any]] = []
        for number, attachment in enumerate(chosen, start=1):
            center_u = float(attachment["u_m"])
            center_z = float(attachment["z_m"])
            half_length = float(np.clip(attachment["z_span_m"] * 0.35, 0.14, 0.24))
            start = point([center_u - half_length, center_z])
            end = point([center_u + half_length, center_z])
            node = f"{asset_id}-INSULATOR-{number:02d}"
            writer.add_mesh(node, *ribbed_insulator(start, end), "PointFittedInsulator")
            insulator_records.append(
                {"node": node, "axis_start_world": start.tolist(), "axis_end_world": end.tolist()}
            )

        support = {
            "cantilever": _support(
                np.vstack([_sample_segment(start, end) for start, end in cantilever_segments]),
                local_world,
            ),
            "positioner": _support(
                _sample_segment(positioner_start, positioner_end), local_world
            ),
        }
        for number, item in enumerate(insulator_records, start=1):
            support[f"insulator_{number:02d}"] = _support(
                _sample_segment(
                    np.asarray(item["axis_start_world"]),
                    np.asarray(item["axis_end_world"]),
                    40,
                ),
                local_world,
            )
        if not all(item["passed"] for item in support.values()):
            raise ValueError(f"Generated top equipment failed local point support: {asset_id}")
        component_ids = [cantilever_id, positioner_id] + [item["node"] for item in insulator_records]
        records.append(
            {
                "mast_asset_id": asset_id,
                "frame_source": frame_source,
                "component_ids": component_ids,
                "main_trace": main,
                "main_trace_point_support": main_trace_support,
                "accepted_secondary_segments": [
                    {"source": item["source"], "point_support": item["support"]}
                    for item in passing_segments[:2]
                ],
                "positioner_source": positioner_segment["source"],
                "insulator_attachments": chosen,
                "point_support": support,
            }
        )
        type_by_id = {
            cantilever_id: "catenary_cantilever",
            positioner_id: "catenary_positioner",
            **{item["node"]: "catenary_insulator" for item in insulator_records},
        }
        for component_id in component_ids:
            component_assets.append(
                {
                    "id": component_id,
                    "type": type_by_id[component_id],
                    "subtype": "point_supported_simplified_node",
                    "status": "accepted",
                    "chainage_m": station,
                    "evidence_level": "observed",
                    "confidence": 0.90,
                    "sources": [
                        {
                            "kind": "point_cloud",
                            "reference": f"{evidence_file}#{asset_id}",
                            "note": "Local station/cross/elevation trace passed component evidence gates.",
                        }
                    ],
                    "parameters": {
                        "parent_mast_asset_id": asset_id,
                        "geometry_method": "local_point_trace_fit_without_catalogue_profile_claim",
                    },
                    "geometry": {"file": "", "node": component_id},
                    "limitations": [
                        "Topology and placement are point supported; product catalogue dimensions remain unresolved."
                    ],
                }
            )
            component_relations.append(
                {
                    "id": f"REL-{asset_id}-HAS-{component_id}",
                    "type": "has_component",
                    "from": asset_id,
                    "to": component_id,
                }
            )

    writer.write(addon_obj)
    _material_file(addon_mtl)
    write_json(addon_origin, origin_value)
    output_obj = output / f"{output.name}.obj"
    output_mtl = output / f"{output.name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [(source_model, source_origin_file), (addon_obj, addon_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )

    registry = copy.deepcopy(load_json(source_registry_file))
    component_ids = {asset["id"] for asset in component_assets}
    existing_ids = {str(asset["id"]) for asset in registry.get("assets", [])}
    if component_ids & existing_ids:
        raise ValueError(f"Component asset ids already exist: {sorted(component_ids & existing_ids)}")
    for asset in registry.get("assets", []):
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            geometry["file"] = str(output_obj)
        if asset.get("id") in {item["mast_asset_id"] for item in records}:
            asset.setdefault("parameters", {})["geometry_scope"] = (
                "observed_shaft_plus_point_supported_top_equipment"
            )
            asset.setdefault("sources", []).append(
                {"kind": "point_cloud", "reference": str(evidence_file)}
            )
    for asset in component_assets:
        asset["geometry"]["file"] = str(output_obj)
    registry.setdefault("assets", []).extend(component_assets)
    registry.setdefault("relations", []).extend(component_relations)
    registry["release_id"] = output.name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    addition = new_registry(str(registry.get("project_id", "site-b")))
    addition["assets"] = copy.deepcopy(component_assets)
    addition["summary"] = summarize_registry(addition)
    addition_errors = validate_registry_value(addition)
    if addition_errors:
        raise ValueError("New catenary component registry records are invalid: " + "; ".join(addition_errors))
    final_ids = {str(asset["id"]) for asset in registry.get("assets", [])}
    if any(
        relation["from"] not in final_ids or relation["to"] not in final_ids
        for relation in component_relations
    ):
        raise ValueError("New catenary relations contain an unknown final-registry endpoint")
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh = audit_obj(output_obj)
    output_mesh = output / "mesh_audit.json"
    write_json(output_mesh, mesh)
    if not mesh["passed"]:
        raise ValueError("Integrated catenary top equipment failed mesh audit")
    report_path = output / "catenary_top_equipment_integration_report.json"
    if evidence.get("inputs", {}).get("inventory"):
        low_confidence = [
            item["asset_id"]
            for item in load_json(Path(evidence["inputs"]["inventory"]))["records"]
            if item["priority"] == "P1_confirm_foundation_and_top_equipment"
        ]
    else:
        low_confidence = [
            str(item["asset_id"])
            for item in evidence.get("records", [])
            if not bool(item.get("evidence", {}).get("passed"))
        ]
    report = {
        "schema_version": "railway.catenary-top-equipment-integration.v1",
        "inputs": {
            "evidence_report": str(evidence_file),
            "source_obj": str(source_model),
            "source_obj_sha256": sha256_file(source_model),
        },
        "records": records,
        "added_mast_assembly_count": len(records),
        "added_component_asset_count": len(component_assets),
        "explicitly_withheld_low_confidence_masts": low_confidence,
        "merge": merge,
        "mesh_audit_passed": True,
        "inherited_registry_validation_error_count": len(
            validate_registry_value(load_json(source_registry_file))
        ),
        "new_registry_records_valid": True,
        "output_obj_sha256": sha256_file(output_obj),
        "status": "point_supported_top_equipment_integrated_low_confidence_masts_withheld",
    }
    write_json(report_path, report)
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh,
        "report": report_path,
    }
