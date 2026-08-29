from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..camera import camera_trajectory, load_camera_rows
from ..config import ProjectConfig
from ..io import load_json, sha256_file, write_json
from ..mesh_audit import audit_obj
from ..registry import new_registry, summarize_registry, validate_registry_value
from ..track_graph import (
    RouteSampler,
    audit_track_graph,
    load_track_graph_settings,
    validate_track_graph_bindings,
)
from .mesh import ObjWriter, oriented_box, rail_profile, sweep_mesh, write_track_materials


def _chainage_samples(
    start_m: float, end_m: float, step_m: float, boundaries: list[float]
) -> np.ndarray:
    if end_m <= start_m:
        raise ValueError("TrackGraph mesh interval must have positive length")
    if step_m <= 0:
        raise ValueError("track_build.polyline_step_m must be positive")
    regular = np.arange(start_m, end_m, step_m, dtype=np.float64)
    values = np.concatenate(
        (
            regular,
            np.asarray([start_m, end_m, *boundaries], dtype=np.float64),
        )
    )
    values = values[(values >= start_m) & (values <= end_m)]
    return np.unique(np.round(values, 9))


def _interpolated_track_value(
    observations: list[dict[str, Any]],
    chainages: np.ndarray,
    field: str,
    start_field: str | None = None,
    end_field: str | None = None,
) -> np.ndarray:
    ordered = sorted(observations, key=lambda item: float(item["chainage_start_m"]))
    if start_field is not None and end_field is not None:
        controls = [
            (
                float(item["chainage_start_m"]),
                float(item.get(start_field, item[field])),
            )
            for item in ordered
        ] + [
            (
                float(item["chainage_end_m"]),
                float(item.get(end_field, item[field])),
            )
            for item in ordered
        ]
    else:
        controls = [
            (float(ordered[0]["chainage_start_m"]), float(ordered[0][field])),
            *[
                (
                    (float(item["chainage_start_m"]) + float(item["chainage_end_m"]))
                    / 2.0,
                    float(item[field]),
                )
                for item in ordered
            ],
            (float(ordered[-1]["chainage_end_m"]), float(ordered[-1][field])),
        ]
    grouped: dict[float, list[float]] = {}
    for control_chainage, control_value in controls:
        grouped.setdefault(control_chainage, []).append(control_value)
    x = np.asarray(sorted(grouped), dtype=np.float64)
    y = np.asarray(
        [float(np.mean(grouped[control_chainage])) for control_chainage in x],
        dtype=np.float64,
    )
    return np.interp(chainages, x, y)


def _track_centerline(
    route: RouteSampler,
    observations: list[dict[str, Any]],
    step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    start = min(float(item["chainage_start_m"]) for item in observations)
    end = max(float(item["chainage_end_m"]) for item in observations)
    boundaries = [
        value
        for item in observations
        for value in (float(item["chainage_start_m"]), float(item["chainage_end_m"]))
    ]
    chainages = _chainage_samples(start, end, step_m, boundaries)
    rail_z = _interpolated_track_value(
        observations,
        chainages,
        "rail_top_z_m",
        "rail_top_start_z_m",
        "rail_top_end_z_m",
    )
    xyz = np.empty((len(chainages), 3), dtype=np.float64)
    fitted_world_fields = {
        "center_world_start_x_m",
        "center_world_start_y_m",
        "center_world_x_m",
        "center_world_y_m",
        "center_world_end_x_m",
        "center_world_end_y_m",
    }
    if all(fitted_world_fields.issubset(item) for item in observations):
        xyz[:, 0] = _interpolated_track_value(
            observations,
            chainages,
            "center_world_x_m",
            "center_world_start_x_m",
            "center_world_end_x_m",
        )
        xyz[:, 1] = _interpolated_track_value(
            observations,
            chainages,
            "center_world_y_m",
            "center_world_start_y_m",
            "center_world_end_y_m",
        )
    else:
        lateral = _interpolated_track_value(
            observations,
            chainages,
            "lateral_offset_m",
            "lateral_offset_start_m",
            "lateral_offset_end_m",
        )
        for index, (chainage, offset) in enumerate(zip(chainages, lateral)):
            route_point, _, normal = route.sample(float(chainage))
            xyz[index, :2] = route_point[:2] + float(offset) * normal
    xyz[:, 2] = rail_z
    delta = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], delta > 1e-9))
    xyz = xyz[keep]
    chainages = chainages[keep]
    if len(xyz) < 2:
        raise ValueError("TrackGraph produced fewer than two distinct centerline points")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("TrackGraph produced non-finite centerline coordinates")
    return chainages, xyz


def _centerline_normals(centerline: np.ndarray) -> np.ndarray:
    tangent = np.gradient(centerline[:, :2], axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Track centerline contains an undefined tangent")
    tangent /= norms[:, None]
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def _merge_meshes(
    meshes: list[tuple[np.ndarray, list[tuple[int, ...]]]],
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    if not meshes:
        raise ValueError("Cannot merge an empty mesh collection")
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, ...]] = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(mesh_vertices)
        faces.extend(tuple(offset + index for index in face) for face in mesh_faces)
        offset += len(mesh_vertices)
    return np.vstack(vertices), faces


def _sample_centerline(
    chainages: np.ndarray, centerline: np.ndarray, values: np.ndarray
) -> np.ndarray:
    result = np.empty((len(values), 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(values, chainages, centerline[:, axis])
    return result


def _track_evidence_intervals(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return contiguous, non-overlapping evidence intervals for one track.

    Rail geometry must never be registered as wholly observed when even a short
    part of its controlling TrackGraph is inferred.  The intervals are derived
    from the observations (the same controls used by the centreline) and are
    intentionally not filled across gaps.
    """
    ordered = sorted(observations, key=lambda item: float(item["chainage_start_m"]))
    intervals: list[dict[str, Any]] = []
    for observation in ordered:
        start = float(observation["chainage_start_m"])
        end = float(observation["chainage_end_m"])
        if end <= start:
            raise ValueError(
                f"TrackGraph observation has a non-positive interval: {observation.get('id')}"
            )
        evidence = str(observation.get("evidence_level", "observed"))
        source_id = str(observation.get("id", "unknown"))
        if (
            intervals
            and intervals[-1]["evidence_level"] == evidence
            and abs(float(intervals[-1]["chainage_end_m"]) - start) <= 1e-6
        ):
            intervals[-1]["chainage_end_m"] = end
            intervals[-1]["source_observation_ids"].append(source_id)
        else:
            intervals.append(
                {
                    "chainage_start_m": start,
                    "chainage_end_m": end,
                    "evidence_level": evidence,
                    "source_observation_ids": [source_id],
                }
            )
    return intervals


def _polyline_interval(
    chainages: np.ndarray,
    polyline: np.ndarray,
    start_m: float,
    end_m: float,
) -> np.ndarray:
    if end_m <= start_m:
        raise ValueError("Polyline evidence interval must have positive length")
    internal = chainages[(chainages > start_m + 1e-9) & (chainages < end_m - 1e-9)]
    values = np.concatenate(
        (np.asarray([start_m]), internal, np.asarray([end_m]))
    )
    return _sample_centerline(chainages, polyline, values)


def _hermite_turnout_centerline(
    branch_chainages: np.ndarray,
    branch_centerline: np.ndarray,
    target_chainages: np.ndarray,
    target_centerline: np.ndarray,
    merge_chainage_m: float,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a tangent-continuous, explicitly inferred turnout connector."""
    start_m = float(branch_chainages[-1])
    end_m = float(merge_chainage_m)
    if end_m <= start_m:
        raise ValueError("Turnout merge chainage must follow the branch endpoint")
    if end_m > float(target_chainages[-1]) + 1e-9:
        raise ValueError("Turnout merge chainage exceeds the target track")
    if step_m <= 0:
        raise ValueError("Turnout connector step must be positive")

    start = branch_centerline[-1].astype(np.float64)
    end = _sample_centerline(
        target_chainages,
        target_centerline,
        np.asarray([end_m], dtype=np.float64),
    )[0]
    branch_delta_m = max(
        float(branch_chainages[-1] - branch_chainages[-2]), 1e-9
    )
    start_derivative = (
        branch_centerline[-1] - branch_centerline[-2]
    ) / branch_delta_m
    tangent_half_width = min(0.5, max(0.05, (end_m - start_m) / 20.0))
    target_samples = _sample_centerline(
        target_chainages,
        target_centerline,
        np.asarray(
            [
                max(float(target_chainages[0]), end_m - tangent_half_width),
                min(float(target_chainages[-1]), end_m + tangent_half_width),
            ],
            dtype=np.float64,
        ),
    )
    target_sample_span = min(float(target_chainages[-1]), end_m + tangent_half_width) - max(
        float(target_chainages[0]), end_m - tangent_half_width
    )
    if target_sample_span <= 1e-9:
        raise ValueError("Target tangent is undefined at the turnout merge")
    end_derivative = (target_samples[1] - target_samples[0]) / target_sample_span

    chainages = _chainage_samples(start_m, end_m, step_m, [])
    t = ((chainages - start_m) / (end_m - start_m))[:, None]
    t2 = t * t
    t3 = t2 * t
    scale = end_m - start_m
    centerline = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * start
        + (t3 - 2.0 * t2 + t) * start_derivative * scale
        + (-2.0 * t3 + 3.0 * t2) * end
        + (t3 - t2) * end_derivative * scale
    )
    centerline[0] = start
    centerline[-1] = end
    horizontal_step = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    if np.any(horizontal_step <= 1e-9):
        raise ValueError("Turnout connector contains a duplicate or reversed sample")
    return chainages, centerline


def _sample_tangent(
    chainages: np.ndarray, centerline: np.ndarray, chainage_m: float
) -> np.ndarray:
    epsilon = min(0.10, max(0.01, float(chainages[-1] - chainages[0]) / 1000.0))
    values = np.asarray(
        [max(float(chainages[0]), chainage_m - epsilon), min(float(chainages[-1]), chainage_m + epsilon)]
    )
    points = _sample_centerline(chainages, centerline, values)
    tangent = points[1] - points[0]
    if float(np.linalg.norm(tangent[:2])) <= 1e-12:
        raise ValueError(f"Sleeper tangent is undefined at chainage {chainage_m:.3f}")
    return tangent


def _sleeper_chainages(
    start_m: float, end_m: float, spacing_m: float, phase_m: float
) -> np.ndarray:
    if spacing_m <= 0:
        raise ValueError("Sleeper spacing must be positive")
    first_index = math.ceil((start_m - phase_m) / spacing_m)
    first = phase_m + first_index * spacing_m
    values = np.arange(first, end_m + spacing_m * 0.25, spacing_m, dtype=np.float64)
    values = values[(values > start_m + 1e-6) & (values < end_m - 1e-6)]
    if not len(values):
        return np.asarray([(start_m + end_m) / 2.0], dtype=np.float64)
    return values


def _asset(
    asset_id: str,
    asset_type: str,
    subtype: str,
    evidence_level: str,
    confidence: float,
    chainage_m: float,
    graph_reference: str,
    parameters: dict[str, Any],
    output_obj: Path,
    *,
    node: str | None = None,
    limitations: list[str] | None = None,
    source_kind: str = "point_cloud",
) -> dict[str, Any]:
    geometry: dict[str, Any] = {"file": str(output_obj)}
    if node is not None:
        geometry["node"] = node
    return {
        "id": asset_id,
        "type": asset_type,
        "subtype": subtype,
        "status": "candidate",
        "chainage_m": chainage_m,
        "evidence_level": evidence_level,
        "confidence": confidence,
        "sources": [{"kind": source_kind, "reference": graph_reference}],
        "parameters": parameters,
        "geometry": geometry,
        "limitations": limitations or [],
    }


def _replace_registry_assets(
    project: ProjectConfig,
    assets: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    registry_path = project.workspace_path("asset_registry")
    registry = load_json(registry_path) if registry_path.is_file() else new_registry(project.project_id)
    incoming = {str(item["id"]) for item in assets}
    known = {str(item["id"]) for item in registry.get("assets", [])}
    conflicts = sorted(incoming & known)
    if conflicts and not overwrite:
        raise ValueError(f"Asset registry already contains TrackGraph mesh ids: {conflicts}")
    if conflicts:
        registry["assets"] = [item for item in registry["assets"] if item["id"] not in incoming]
        registry["relations"] = [
            item
            for item in registry.get("relations", [])
            if item.get("from") not in incoming and item.get("to") not in incoming
        ]
    relation_ids = {str(item["id"]) for item in relations}
    registry["relations"] = [
        item for item in registry.get("relations", []) if str(item.get("id")) not in relation_ids
    ]
    registry["assets"].extend(assets)
    registry["relations"].extend(relations)
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Generated TrackGraph registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)


def build_track_graph_mesh(
    project: ProjectConfig,
    graph_value: str | Path | None = None,
    audit_value: str | Path | None = None,
    overwrite: bool = False,
    output_dir_value: str | Path | None = None,
    report_dir_value: str | Path | None = None,
    update_registry: bool = True,
) -> dict[str, Any]:
    graph_path = (
        Path(graph_value).resolve()
        if graph_value is not None
        else project.workspace_path("derived") / "track_graph.json"
    )
    graph_audit_path = (
        Path(audit_value).resolve()
        if audit_value is not None
        else project.workspace_path("reports") / "track_graph_audit.json"
    )
    for source in (graph_path, graph_audit_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    graph = load_json(graph_path)
    recorded_audit = load_json(graph_audit_path)
    graph_hash = sha256_file(graph_path)
    if recorded_audit.get("graph_sha256") != graph_hash:
        raise ValueError("TrackGraph hash does not match its audit; rebuild or revalidate it")
    if recorded_audit.get("status") != "pass":
        raise ValueError(f"TrackGraph audit is not pass: {recorded_audit.get('status')}")
    binding_failures = validate_track_graph_bindings(project, graph)
    if binding_failures:
        raise ValueError(f"TrackGraph project bindings are stale: {binding_failures}")
    current_audit = audit_track_graph(graph, load_track_graph_settings(project))
    if current_audit["status"] != "pass":
        raise ValueError("TrackGraph no longer passes current project settings")

    config = load_json(project.resolve(project.value["algorithms"]["track_build"]))
    global_config = config.get("global_mesh", {})
    if bool(global_config.get("require_observed_edges", True)):
        unsupported = [
            edge["id"]
            for edge in graph.get("edges", [])
            if edge.get("evidence_level") != "observed"
        ]
        if unsupported:
            raise ValueError(
                "TrackGraph contains non-observed edges without an approved mesh policy: "
                + ", ".join(unsupported[:10])
            )

    camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    track_graph_settings = load_track_graph_settings(project)
    route = RouteSampler(
        camera_trajectory(load_camera_rows(camera_path)),
        float(track_graph_settings.get("route_frame_tangent_window_m", 5.0)),
    )
    output_dir = (
        Path(output_dir_value).resolve()
        if output_dir_value is not None
        else project.workspace_path("exports") / "track_graph" / "track"
    )
    report_dir = (
        Path(report_dir_value).resolve()
        if report_dir_value is not None
        else project.workspace_path("reports")
    )
    obj_path = output_dir / "track_graph_track.obj"
    mtl_path = output_dir / "railway_model.mtl"
    origin_path = output_dir / "model_origin.json"
    report_path = report_dir / "track_graph_mesh_build.json"
    mesh_audit_path = report_dir / "track_graph_mesh_audit.json"
    asset_sidecar_path = report_dir / "track_graph_mesh_assets.json"
    for path in (
        obj_path,
        mtl_path,
        origin_path,
        report_path,
        mesh_audit_path,
        asset_sidecar_path,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    observations_by_track: dict[str, list[dict[str, Any]]] = {}
    for observation in graph["observations"]:
        observations_by_track.setdefault(str(observation["global_track_id"]), []).append(
            observation
        )
    if not observations_by_track:
        raise ValueError("TrackGraph has no observations")

    step = float(config["polyline_step_m"])
    track_geometry: list[dict[str, Any]] = []
    for track in graph["tracks"]:
        track_id = str(track["id"])
        members = observations_by_track.get(track_id, [])
        if not members:
            raise ValueError(f"TrackGraph track has no observations: {track_id}")
        chainages, centerline = _track_centerline(route, members, step)
        evidence_intervals = _track_evidence_intervals(members)
        track_geometry.append(
            {
                "track": track,
                "observations": members,
                "chainages": chainages,
                "centerline": centerline,
                "evidence_intervals": evidence_intervals,
            }
        )

    maximum_step = max(step * float(global_config.get("maximum_centerline_step_ratio", 2.0)), step + 0.05)
    maximum_deflection = float(global_config.get("maximum_centerline_deflection_deg", 15.0))
    centerline_failures: list[dict[str, Any]] = []
    for value in track_geometry:
        delta = np.diff(value["centerline"][:, :2], axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(lengths[:, None], 1e-12)
        dot = np.einsum("ij,ij->i", unit[:-1], unit[1:]) if len(unit) > 1 else np.asarray([])
        deflection = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) if len(dot) else np.asarray([])
        value["centerline_quality"] = {
            "maximum_step_m": float(lengths.max()) if len(lengths) else 0.0,
            "maximum_deflection_deg": float(deflection.max()) if len(deflection) else 0.0,
            "backtracking_transition_count": int(np.count_nonzero(dot <= 0.0)) if len(dot) else 0,
        }
        failure: dict[str, Any] = {"track_id": value["track"]["id"]}
        if len(lengths) and float(lengths.max()) > maximum_step:
            failure["maximum_step_m"] = float(lengths.max())
            failure["maximum_allowed_step_m"] = maximum_step
        if len(deflection) and float(deflection.max()) > maximum_deflection:
            failure["maximum_deflection_deg"] = float(deflection.max())
            failure["maximum_allowed_deflection_deg"] = maximum_deflection
        if len(dot) and bool(np.any(dot <= 0.0)):
            failure["backtracking_transition_count"] = int(np.count_nonzero(dot <= 0.0))
        if len(failure) > 1:
            centerline_failures.append(failure)
    if centerline_failures:
        raise ValueError(f"TrackGraph centerline smoothness gate failed: {centerline_failures}")

    first = min(track_geometry, key=lambda item: float(item["chainages"][0]))
    first_point = first["centerline"][0]
    origin = np.asarray([first_point[0], first_point[1], 0.0], dtype=np.float64)
    writer = ObjWriter(origin)
    assets: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    build_stats: list[dict[str, Any]] = []
    visible_node_ids: list[str] = []
    profile_settings = config.get("rail_profile", {})
    profile = rail_profile(profile_settings)
    rail_head_width = float(profile_settings.get("head_width_m", 0.073))
    rail_profile_height = abs(float(np.min(profile[:, 1])))
    configured_contact = float(config["sleeper"]["top_below_rail_m"])
    contact_error = abs(configured_contact - rail_profile_height)
    maximum_contact_error = float(global_config.get("maximum_rail_sleeper_contact_error_m", 0.001))
    if contact_error > maximum_contact_error:
        raise ValueError(
            "Rail profile and sleeper top do not contact: "
            f"error {contact_error:.6f} m exceeds {maximum_contact_error:.6f} m"
        )
    gauge_default = float(config["gauge_m"])
    rail_center_spacing_default = gauge_default + rail_head_width
    track_geometry_by_id = {
        str(item["track"]["id"]): item for item in track_geometry
    }

    for value in track_geometry:
        track = value["track"]
        track_id = str(track["id"])
        chainages = value["chainages"]
        centerline = value["centerline"]
        evidence_intervals = value["evidence_intervals"]
        normals = _centerline_normals(centerline)
        observed_gauge = float(track["median_gauge_m"])
        observed_rail_center_spacing = float(
            track.get(
                "median_rail_center_spacing_m",
                observed_gauge + rail_head_width,
            )
        )
        gauge = gauge_default if bool(config.get("constrain_nominal_gauge", True)) else observed_gauge
        rail_center_spacing = (
            rail_center_spacing_default
            if bool(config.get("constrain_nominal_gauge", True))
            else observed_rail_center_spacing
        )
        left = centerline.copy()
        right = centerline.copy()
        left[:, :2] += normals * (rail_center_spacing / 2.0)
        right[:, :2] -= normals * (rail_center_spacing / 2.0)
        midpoint = float((chainages[0] + chainages[-1]) / 2.0)
        length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
        graph_reference = f"track_graph:{graph_hash}"

        component_ids: list[str] = []
        for side, line in (("LEFT", left), ("RIGHT", right)):
            for interval_index, interval in enumerate(evidence_intervals, start=1):
                evidence = str(interval["evidence_level"])
                if len(evidence_intervals) == 1 and evidence == "observed":
                    asset_id = f"{track_id}-RAIL-{side}"
                else:
                    label = "OBSERVED" if evidence == "observed" else "INFERRED"
                    asset_id = f"{track_id}-RAIL-{side}-{label}-{interval_index:03d}"
                interval_line = _polyline_interval(
                    chainages,
                    line,
                    float(interval["chainage_start_m"]),
                    float(interval["chainage_end_m"]),
                )
                vertices, faces = sweep_mesh(interval_line, profile)
                material = "RailSteel" if evidence == "observed" else "RailSteelInferred"
                writer.add_mesh(asset_id, vertices, faces, material)
                visible_node_ids.append(asset_id)
                component_ids.append(asset_id)
                inferred = evidence != "observed"
                confidence = float(
                    config[
                        "default_inferred_confidence"
                        if inferred
                        else "default_observed_confidence"
                    ]
                )
                limitations = ["Profile is configurable; verify the project rail section"]
                if inferred:
                    limitations.append(
                        "Orange interval is rule-inferred and is not an independently observed rail fit"
                    )
                assets.append(
                    _asset(
                        asset_id,
                        "rail",
                        side.lower(),
                        evidence,
                        confidence,
                        float(
                            (
                                float(interval["chainage_start_m"])
                                + float(interval["chainage_end_m"])
                            )
                            / 2.0
                        ),
                        graph_reference,
                        {
                            "profile": profile_settings.get(
                                "name", "generic_60kg_envelope"
                            ),
                            "gauge_m": gauge,
                            "rail_center_spacing_m": rail_center_spacing,
                            "track_id": track_id,
                            "chainage_start_m": float(interval["chainage_start_m"]),
                            "chainage_end_m": float(interval["chainage_end_m"]),
                            "source_observation_ids": interval[
                                "source_observation_ids"
                            ],
                        },
                        obj_path,
                        node=asset_id,
                        limitations=limitations,
                        source_kind="rule" if inferred else "point_cloud",
                    )
                )

        sleeper = config["sleeper"]
        sleeper_values = _sleeper_chainages(
            float(chainages[0]),
            float(chainages[-1]),
            float(sleeper["spacing_m"]),
            float(global_config.get("sleeper_phase_chainage_m", 0.0)),
        )
        sleeper_centers = _sample_centerline(chainages, centerline, sleeper_values)
        sleeper_meshes: list[tuple[np.ndarray, list[tuple[int, ...]]]] = []
        for sleeper_chainage, sleeper_center in zip(sleeper_values, sleeper_centers):
            tangent = _sample_tangent(chainages, centerline, float(sleeper_chainage))
            top_z = float(sleeper_center[2]) - float(sleeper["top_below_rail_m"])
            sleeper_meshes.append(
                oriented_box(
                    sleeper_center,
                    tangent,
                    float(sleeper["length_m"]),
                    float(sleeper["width_m"]),
                    top_z - float(sleeper["height_m"]),
                    top_z,
                )
            )
        sleeper_id = f"{track_id}-SLEEPERS"
        sleeper_vertices, sleeper_faces = _merge_meshes(sleeper_meshes)
        writer.add_mesh(sleeper_id, sleeper_vertices, sleeper_faces, "SleeperConcrete")
        visible_node_ids.append(sleeper_id)
        component_ids.append(sleeper_id)
        assets.append(
            _asset(
                sleeper_id,
                "sleeper_group",
                "global_phase_regularized",
                "rule_inferred",
                float(config["default_inferred_confidence"]),
                midpoint,
                graph_reference,
                {
                    **sleeper,
                    "count": len(sleeper_values),
                    "phase_chainage_m": float(global_config.get("sleeper_phase_chainage_m", 0.0)),
                    "first_chainage_m": float(sleeper_values[0]),
                    "last_chainage_m": float(sleeper_values[-1]),
                },
                obj_path,
                node=sleeper_id,
                limitations=["Sleeper locations are globally regularized, not individually surveyed"],
            )
        )

        bed = config["track_bed"]
        bed_profile = np.asarray(
            [
                [-float(bed["bottom_width_m"]) / 2.0, -float(bed["top_below_rail_m"]) - float(bed["depth_m"])],
                [float(bed["bottom_width_m"]) / 2.0, -float(bed["top_below_rail_m"]) - float(bed["depth_m"])],
                [float(bed["top_width_m"]) / 2.0, -float(bed["top_below_rail_m"])],
                [-float(bed["top_width_m"]) / 2.0, -float(bed["top_below_rail_m"])],
            ],
            dtype=np.float64,
        )
        bed_id = f"{track_id}-BED"
        bed_vertices, bed_faces = sweep_mesh(centerline, bed_profile)
        writer.add_mesh(bed_id, bed_vertices, bed_faces, "Ballast")
        visible_node_ids.append(bed_id)
        component_ids.append(bed_id)
        assets.append(
            _asset(
                bed_id,
                "track_bed",
                "parametric_ballast",
                "rule_inferred",
                float(config["default_inferred_confidence"]),
                midpoint,
                graph_reference,
                {**bed, "track_id": track_id},
                obj_path,
                node=bed_id,
                limitations=["Ballast section is a configurable approximation"],
            )
        )

        has_inferred_rail = any(
            interval["evidence_level"] != "observed"
            for interval in evidence_intervals
        )
        aggregate_evidence = "rule_inferred" if has_inferred_rail else "observed"
        aggregate_confidence = float(
            config[
                "default_inferred_confidence"
                if has_inferred_rail
                else "default_observed_confidence"
            ]
        )
        aggregate_limitations = [
            "Aggregate asset; components remain candidate until Mesh review"
        ]
        if has_inferred_rail:
            aggregate_limitations.append(
                "Track includes orange rule-inferred rail intervals; inspect parameters.evidence_intervals"
            )
        assets.append(
            _asset(
                track_id,
                "track",
                "standard_gauge",
                aggregate_evidence,
                aggregate_confidence,
                midpoint,
                graph_reference,
                {
                    "gauge_m": gauge,
                    "observed_gauge_m": observed_gauge,
                    "rail_center_spacing_m": rail_center_spacing,
                    "observed_rail_center_spacing_m": observed_rail_center_spacing,
                    "length_m": length,
                    "chainage_start_m": float(chainages[0]),
                    "chainage_end_m": float(chainages[-1]),
                    "component_ids": component_ids,
                    "evidence_intervals": evidence_intervals,
                },
                obj_path,
                limitations=aggregate_limitations,
                source_kind="rule" if has_inferred_rail else "point_cloud",
            )
        )
        for component_id in component_ids:
            relations.append(
                {
                    "id": f"REL-{track_id}-{component_id}",
                    "type": "contains",
                    "from": track_id,
                    "to": component_id,
                }
            )
        spacing_deltas = np.diff(sleeper_values)
        build_stats.append(
            {
                "track_id": track_id,
                "chainage_start_m": float(chainages[0]),
                "chainage_end_m": float(chainages[-1]),
                "length_m": length,
                "centerline_point_count": len(centerline),
                "sleeper_count": len(sleeper_values),
                "sleeper_minimum_spacing_m": float(spacing_deltas.min()) if len(spacing_deltas) else None,
                "sleeper_maximum_spacing_m": float(spacing_deltas.max()) if len(spacing_deltas) else None,
                "visible_node_ids": component_ids,
                "centerline_quality": value["centerline_quality"],
                "evidence_intervals": evidence_intervals,
            }
        )

    turnout_stats: list[dict[str, Any]] = []
    connector_config = list(global_config.get("turnout_connectors", []))
    if connector_config and not bool(
        global_config.get("allow_rule_inferred_turnout_connectors", False)
    ):
        raise ValueError(
            "Turnout connectors are configured without an explicit inferred-geometry policy"
        )
    known_connector_ids: set[str] = set()
    for connector in connector_config:
        connector_id = str(connector["id"])
        if connector_id in known_connector_ids:
            raise ValueError(f"Duplicate turnout connector id: {connector_id}")
        known_connector_ids.add(connector_id)
        branch_track_id = str(connector["branch_track_id"])
        target_track_id = str(connector["target_track_id"])
        if branch_track_id == target_track_id:
            raise ValueError("Turnout connector branch and target must differ")
        if branch_track_id not in track_geometry_by_id:
            raise ValueError(f"Unknown turnout branch track: {branch_track_id}")
        if target_track_id not in track_geometry_by_id:
            raise ValueError(f"Unknown turnout target track: {target_track_id}")
        reason = str(connector.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"Turnout connector requires a reason: {connector_id}")
        branch = track_geometry_by_id[branch_track_id]
        target = track_geometry_by_id[target_track_id]
        branch_end_observation = max(
            branch["observations"], key=lambda item: float(item["chainage_end_m"])
        )
        if branch_end_observation.get("end_boundary_reason") != "confirmed_turnout":
            raise ValueError(
                f"Turnout branch lacks a confirmed_turnout boundary: {branch_track_id}"
            )
        merge_chainage = float(connector["merge_chainage_m"])
        connector_length = merge_chainage - float(branch["chainages"][-1])
        maximum_length = float(
            global_config.get("maximum_rule_inferred_turnout_length_m", 150.0)
        )
        if connector_length > maximum_length:
            raise ValueError(
                f"Turnout connector exceeds maximum inferred length: {connector_id}"
            )
        connector_chainages, connector_centerline = _hermite_turnout_centerline(
            branch["chainages"],
            branch["centerline"],
            target["chainages"],
            target["centerline"],
            merge_chainage,
            float(connector.get("step_m", min(step, 0.25))),
        )
        connector_normals = _centerline_normals(connector_centerline)
        connector_left = connector_centerline.copy()
        connector_right = connector_centerline.copy()
        connector_left[:, :2] += connector_normals * (
            rail_center_spacing_default / 2.0
        )
        connector_right[:, :2] -= connector_normals * (
            rail_center_spacing_default / 2.0
        )
        confidence = float(
            connector.get("confidence", config["default_inferred_confidence"])
        )
        if not 0.0 <= confidence <= float(config["default_inferred_confidence"]):
            raise ValueError(
                f"Turnout connector confidence exceeds inferred policy: {connector_id}"
            )
        connector_component_ids: list[str] = []
        for side, line in (("LEFT", connector_left), ("RIGHT", connector_right)):
            asset_id = f"{connector_id}-RAIL-{side}"
            vertices, faces = sweep_mesh(line, profile)
            writer.add_mesh(asset_id, vertices, faces, "RailSteelInferred")
            visible_node_ids.append(asset_id)
            connector_component_ids.append(asset_id)
            assets.append(
                _asset(
                    asset_id,
                    "rail",
                    f"turnout_connector_{side.lower()}",
                    "rule_inferred",
                    confidence,
                    float(
                        (float(connector_chainages[0]) + float(connector_chainages[-1]))
                        / 2.0
                    ),
                    graph_reference,
                    {
                        "connector_id": connector_id,
                        "branch_track_id": branch_track_id,
                        "target_track_id": target_track_id,
                        "chainage_start_m": float(connector_chainages[0]),
                        "chainage_end_m": float(connector_chainages[-1]),
                        "profile": profile_settings.get(
                            "name", "generic_60kg_envelope"
                        ),
                        "rail_center_spacing_m": rail_center_spacing_default,
                        "inference_reason": reason,
                    },
                    obj_path,
                    node=asset_id,
                    limitations=[
                        "Orange display-completion rail inferred from a confirmed turnout boundary",
                        "No surveyed switch blades, frog, guard rails, turnout sleepers or ballast transition",
                    ],
                    source_kind="rule",
                )
            )
        assets.append(
            _asset(
                connector_id,
                "turnout",
                "evidence_bounded_display_connector",
                "rule_inferred",
                confidence,
                float(
                    (float(connector_chainages[0]) + float(connector_chainages[-1]))
                    / 2.0
                ),
                graph_reference,
                {
                    "branch_track_id": branch_track_id,
                    "target_track_id": target_track_id,
                    "chainage_start_m": float(connector_chainages[0]),
                    "chainage_end_m": float(connector_chainages[-1]),
                    "component_ids": connector_component_ids,
                    "inference_reason": reason,
                },
                obj_path,
                limitations=[
                    "Display completion only; turnout type and exact railwork are unresolved",
                    "Must remain visually distinct from point-cloud-observed rail",
                ],
                source_kind="rule",
            )
        )
        for component_id in connector_component_ids:
            relations.append(
                {
                    "id": f"REL-{connector_id}-{component_id}",
                    "type": "contains",
                    "from": connector_id,
                    "to": component_id,
                }
            )
        relations.extend(
            [
                {
                    "id": f"REL-{branch_track_id}-{connector_id}",
                    "type": "connects_to",
                    "from": branch_track_id,
                    "to": connector_id,
                },
                {
                    "id": f"REL-{connector_id}-{target_track_id}",
                    "type": "connects_to",
                    "from": connector_id,
                    "to": target_track_id,
                },
            ]
        )
        connector_delta = np.diff(connector_centerline[:, :2], axis=0)
        connector_unit = connector_delta / np.maximum(
            np.linalg.norm(connector_delta, axis=1)[:, None], 1e-12
        )
        connector_dot = (
            np.einsum("ij,ij->i", connector_unit[:-1], connector_unit[1:])
            if len(connector_unit) > 1
            else np.asarray([])
        )
        maximum_connector_deflection = (
            float(
                np.degrees(
                    np.arccos(np.clip(connector_dot, -1.0, 1.0))
                ).max()
            )
            if len(connector_dot)
            else 0.0
        )
        turnout_stats.append(
            {
                "id": connector_id,
                "branch_track_id": branch_track_id,
                "target_track_id": target_track_id,
                "chainage_start_m": float(connector_chainages[0]),
                "chainage_end_m": float(connector_chainages[-1]),
                "centerline_length_m": float(
                    np.linalg.norm(np.diff(connector_centerline, axis=0), axis=1).sum()
                ),
                "maximum_deflection_deg": maximum_connector_deflection,
                "confidence": confidence,
                "evidence_level": "rule_inferred",
                "visible_node_ids": connector_component_ids,
                "reason": reason,
            }
        )

    writer.write(obj_path)
    write_track_materials(mtl_path)
    write_json(
        origin_path,
        {
            "origin_xyz": origin.tolist(),
            "units": "metre",
            "axis": "Z-up",
            "canonical_axis": "camera_trajectory_chainage",
            "source_track_graph_sha256": graph_hash,
        },
    )
    mesh_audit = audit_obj(obj_path)
    mesh_audit["status"] = "pass" if mesh_audit["passed"] else "fail"
    mesh_audit["source_track_graph_sha256"] = graph_hash
    mesh_audit["visible_node_ids"] = visible_node_ids
    mesh_audit["visible_node_count"] = len(visible_node_ids)
    actual_nodes = set(mesh_audit["object_names"])
    expected_nodes = set(visible_node_ids)
    mesh_audit["registry_mapping_complete"] = {
        asset["id"] for asset in assets
    }.issuperset(actual_nodes)
    mesh_audit["visible_node_set_matches_expected"] = actual_nodes == expected_nodes
    write_json(mesh_audit_path, mesh_audit)
    if (
        not mesh_audit["passed"]
        or not mesh_audit["registry_mapping_complete"]
        or not mesh_audit["visible_node_set_matches_expected"]
    ):
        raise ValueError("Generated TrackGraph OBJ failed automatic Mesh audit")

    asset_sidecar = new_registry(project.project_id)
    asset_sidecar["assets"] = assets
    asset_sidecar["relations"] = relations
    asset_sidecar["summary"] = summarize_registry(asset_sidecar)
    sidecar_errors = validate_registry_value(asset_sidecar)
    if sidecar_errors:
        raise ValueError(
            "Generated TrackGraph asset sidecar is invalid: "
            + "; ".join(sidecar_errors)
        )
    write_json(asset_sidecar_path, asset_sidecar)

    if update_registry:
        _replace_registry_assets(project, assets, relations, overwrite)
    report = {
        "schema_version": "railway.track-graph-mesh-build.v1",
        "project_id": project.project_id,
        "status": "review_required",
        "automatic_checks_passed": True,
        "source_track_graph": str(graph_path),
        "source_track_graph_sha256": graph_hash,
        "source_track_graph_audit": str(graph_audit_path),
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "origin": str(origin_path),
        "mesh_audit": str(mesh_audit_path),
        "asset_registry_sidecar": str(asset_sidecar_path),
        "vertex_count": writer.vertex_count,
        "face_count": writer.face_count,
        "visible_node_count": len(visible_node_ids),
        "rail_profile_height_m": rail_profile_height,
        "maximum_rail_foot_to_sleeper_contact_error_m": contact_error,
        "registry_updated": update_registry,
        "asset_count_generated": len(assets),
        "relation_count_generated": len(relations),
        "asset_count_added": len(assets) if update_registry else 0,
        "relation_count_added": len(relations) if update_registry else 0,
        "tracks": build_stats,
        "turnout_connectors": turnout_stats,
        "limitations": [
            "Automatic topology and OBJ checks passed, but fixed-view Mesh review is still required.",
            "Sleepers and ballast are rule-inferred parametric assets.",
            "Rule-inferred rail intervals are separate orange mesh objects and registry assets.",
            "Configured turnout connectors are orange display completions; exact switch railwork remains unresolved.",
        ],
    }
    write_json(report_path, report)
    return report
