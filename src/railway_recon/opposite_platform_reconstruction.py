from __future__ import annotations

import copy
import json
import os
import re
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .algorithms.mesh import ObjWriter
from .config import ProjectConfig
from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .platform_density_consensus import evaluate_platform_density_consensus
from .platform_mesh import build_platform_component_mesh
from .platform_surface import analyze_platform_surface_data
from .registry import new_registry, summarize_registry, validate_registry_value


def _resource_json(name: str) -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(name)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_prefix(value: str) -> str:
    return re.sub(r"[^A-Z0-9_-]", "-", value.upper())


def _replace_component_id(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_component_id(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_component_id(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _fit_gap_maximum(fits: list[dict[str, Any]]) -> float:
    ordered = sorted(fits, key=lambda item: float(item["longitudinal_range_m"][0]))
    return max(
        (
            max(
                0.0,
                float(current["longitudinal_range_m"][0])
                - float(previous["longitudinal_range_m"][1]),
            )
            for previous, current in pairwise(ordered)
        ),
        default=0.0,
    )


def local_owned_interval_from_plan(
    ownership_plan: dict[str, Any],
    segment_id: str,
    local_frame: CorridorFrame,
) -> tuple[float, float]:
    """Convert a shared corridor ownership interval into a segment-local interval."""
    segment = next(
        (
            item
            for item in ownership_plan.get("segments", [])
            if str(item.get("segment_id")) == segment_id
        ),
        None,
    )
    if segment is None:
        raise ValueError(f"Segment is not present in ownership plan: {segment_id}")
    frame_value = ownership_plan.get("frame", {})
    cross_key = "cross_xy" if "cross_xy" in frame_value else "lateral_xy"
    if cross_key not in frame_value:
        raise ValueError("Ownership plan frame requires cross_xy or lateral_xy")
    master_frame = CorridorFrame(
        np.asarray(frame_value["origin_xy"], dtype=np.float64),
        np.asarray(frame_value["along_xy"], dtype=np.float64),
        np.asarray(frame_value[cross_key], dtype=np.float64),
    )
    master_interval = np.asarray(segment["owned_interval_m"], dtype=np.float64)
    endpoints = master_frame.world_xy(master_interval, np.zeros(2, dtype=np.float64))
    local_station, _ = local_frame.project(endpoints[:, 0], endpoints[:, 1])
    return float(np.min(local_station)), float(np.max(local_station))


def prepare_opposite_platform_component(
    candidates: list[dict[str, Any]],
    *,
    side: str,
    owned_interval_m: tuple[float, float],
    selected_configuration_id: str | None,
    component_id: str,
    minimum_accepted_configurations: int = 3,
    maximum_rail_side_spread_m: float = 1.0,
    maximum_median_elevation_spread_m: float = 0.10,
    maximum_boundary_extrapolation_m: float = 0.50,
    maximum_fit_gap_m: float = 0.25,
    rail_side_clamp_half_width_m: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn a multi-density platform hypothesis into one conservative component.

    The unstable far edge is intersected across accepted configurations, while
    the rail-side edge and elevation must remain stable.  This deliberately
    allows a density consensus to be useful even when only the unobservable far
    platform edge fails the generic all-fields consensus gate.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    owned_low, owned_high = (float(value) for value in owned_interval_m)
    if owned_low >= owned_high:
        raise ValueError("owned_interval_m must have positive length")
    if not candidates:
        raise ValueError("At least one density configuration is required")

    consensus = evaluate_platform_density_consensus(
        candidates,
        side=side,
        minimum_accepted_configurations=minimum_accepted_configurations,
        maximum_median_elevation_spread_m=maximum_median_elevation_spread_m,
    )
    accepted = [item for item in consensus["configurations"] if item["accepted_for_consensus"]]
    if len(accepted) < minimum_accepted_configurations:
        raise ValueError("Too few accepted platform density configurations")

    rail_values = np.asarray([float(item["rail_side_edge_cross_m"]) for item in accepted])
    outer_values = np.asarray([float(item["outer_edge_cross_m"]) for item in accepted])
    elevation_values = np.asarray([float(item["median_elevation_m"]) for item in accepted])
    rail_spread = float(np.ptp(rail_values))
    elevation_spread = float(np.ptp(elevation_values))
    if rail_spread > maximum_rail_side_spread_m:
        raise ValueError("Platform rail-side boundary is unstable across density settings")
    if elevation_spread > maximum_median_elevation_spread_m:
        raise ValueError("Platform elevation is unstable across density settings")
    stable_rail = float(np.median(rail_values))
    conservative_outer = (
        float(np.max(outer_values)) if side == "left" else float(np.min(outer_values))
    )
    if side == "left" and conservative_outer >= stable_rail:
        raise ValueError("Conservative left-platform width is not positive")
    if side == "right" and conservative_outer <= stable_rail:
        raise ValueError("Conservative right-platform width is not positive")

    accepted_ids = {str(item["configuration_id"]) for item in accepted}
    if selected_configuration_id is None:
        selected_configuration_id = str(
            max(accepted, key=lambda item: int(item["occupied_cell_count"]))["configuration_id"]
        )
    if selected_configuration_id not in accepted_ids:
        raise ValueError("Selected platform configuration did not pass the density gate")
    selected_candidate = next(
        item for item in candidates if str(item["configuration_id"]) == selected_configuration_id
    )
    selected_components = [
        item
        for item in selected_candidate["report"].get("platform_components", [])
        if item.get("side") == side
    ]
    if not selected_components:
        raise ValueError("Selected density report has no component on requested side")
    source = max(selected_components, key=lambda item: int(item.get("occupied_cell_count", 0)))
    component = copy.deepcopy(source)
    component = _replace_component_id(component, str(source["id"]), component_id)

    clipped_fits: list[dict[str, Any]] = []
    for source_fit in component.get("fit_segments", []):
        fit = copy.deepcopy(source_fit)
        source_low, source_high = (float(value) for value in fit["longitudinal_range_m"])
        low = max(source_low, owned_low)
        high = min(source_high, owned_high)
        if high - low <= 1e-8:
            continue
        fit["longitudinal_range_m"] = [low, high]
        observed = [float(value) for value in fit["observed_cross_range_m"]]
        if side == "left":
            observed[0] = max(observed[0], conservative_outer)
        else:
            observed[1] = min(observed[1], conservative_outer)
        fit["observed_cross_range_m"] = observed
        fit["rail_side_edge_cross_m"] = float(
            np.clip(
                float(fit["rail_side_edge_cross_m"]),
                stable_rail - rail_side_clamp_half_width_m,
                stable_rail + rail_side_clamp_half_width_m,
            )
        )
        clipped_fits.append(fit)
    if not clipped_fits:
        raise ValueError("Selected platform has no fit segment inside its owned interval")
    clipped_fits.sort(key=lambda item: float(item["longitudinal_range_m"][0]))
    start_gap = float(clipped_fits[0]["longitudinal_range_m"][0]) - owned_low
    end_gap = owned_high - float(clipped_fits[-1]["longitudinal_range_m"][1])
    maximum_gap = _fit_gap_maximum(clipped_fits)
    if max(start_gap, end_gap) > maximum_boundary_extrapolation_m:
        raise ValueError("Platform evidence is too far from an ownership boundary")
    if maximum_gap > maximum_fit_gap_m:
        raise ValueError("Platform fit contains an unsupported longitudinal gap")
    clipped_fits[0]["longitudinal_range_m"][0] = owned_low
    clipped_fits[-1]["longitudinal_range_m"][1] = owned_high
    component["fit_segments"] = clipped_fits
    component["longitudinal_range_m"] = [owned_low, owned_high]
    component["cross_range_m"] = (
        [conservative_outer, stable_rail] if side == "left" else [stable_rail, conservative_outer]
    )

    checks = {
        "enough_density_configurations": len(accepted) >= minimum_accepted_configurations,
        "rail_side_stable": rail_spread <= maximum_rail_side_spread_m,
        "elevation_stable": elevation_spread <= maximum_median_elevation_spread_m,
        "boundary_support_close_enough": max(start_gap, end_gap)
        <= maximum_boundary_extrapolation_m,
        "no_unsupported_longitudinal_gap": maximum_gap <= maximum_fit_gap_m,
        "positive_conservative_width": abs(stable_rail - conservative_outer) > 0,
    }
    gate = {
        "schema_version": "railway.opposite-platform-consensus-gate.v1",
        "side": side,
        "owned_interval_m": [owned_low, owned_high],
        "selected_configuration_id": selected_configuration_id,
        "accepted_configuration_count": len(accepted),
        "generic_consensus_passed": bool(consensus["passed"]),
        "generic_consensus": consensus,
        "conservative_outer_edge_cross_m": conservative_outer,
        "stable_rail_side_edge_cross_m": stable_rail,
        "rail_side_edge_spread_m": rail_spread,
        "median_elevation_spread_m": elevation_spread,
        "start_boundary_extrapolation_m": start_gap,
        "end_boundary_extrapolation_m": end_gap,
        "maximum_internal_fit_gap_m": maximum_gap,
        "checks": checks,
        "passed": all(checks.values()),
        "status": "pass_conservative_opposite_platform_component",
        "outer_edge_policy": "intersection_of_accepted_density_extents",
        "limitations": [
            "The unstable far edge is clipped to the common observed extent.",
            "Only short boundary extrapolation inside the configured tolerance is allowed.",
            "The platform volume thickness remains rule-inferred visualization geometry.",
        ],
    }
    return component, gate


def _write_materials(path: Path) -> None:
    content = """# Opposite-platform consensus materials
newmtl OppositePlatformTopObserved
Kd 0.54 0.61 0.62
Ks 0.07 0.07 0.07
Ns 9

newmtl OppositePlatformVolumeInferred
Kd 0.31 0.36 0.37
Ks 0.03 0.03 0.03
Ns 4
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def reconstruct_opposite_platform(
    project: ProjectConfig,
    segment_id: str,
    point_cloud_path: str | Path,
    vertical_report_path: str | Path,
    output_dir: str | Path,
    *,
    side: str,
    owned_interval_m: tuple[float, float] | None = None,
    ownership_plan_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a density sweep and emit one evidence-gated opposite platform mesh."""
    source = project.resolve(point_cloud_path)
    vertical_path = project.resolve(vertical_report_path)
    ownership_path = project.resolve(ownership_plan_path) if ownership_plan_path else None
    settings_source = project.resolve(settings_path) if settings_path else None
    for path in (source, vertical_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (owned_interval_m is None) == (ownership_path is None):
        raise ValueError("Configure exactly one of owned_interval_m or ownership_plan_path")
    if ownership_path is not None and not ownership_path.is_file():
        raise FileNotFoundError(ownership_path)
    settings = (
        load_json(settings_source)
        if settings_source
        else _resource_json("opposite-platform-reconstruction.default.json")
    )
    if settings.get("schema_version") != ("railway.opposite-platform-reconstruction-settings.v1"):
        raise ValueError("Unsupported opposite-platform reconstruction settings")
    base = _resource_json("platform-surface.default.json")
    density_configurations = list(settings.get("density_configurations", []))
    if len(density_configurations) < int(settings["minimum_accepted_configurations"]):
        raise ValueError("Settings do not contain enough density configurations")

    output = project.resolve(output_dir)
    obj_path = output / "opposite_platform.obj"
    mtl_path = output / "opposite_platform.mtl"
    origin_path = output / "model_origin.json"
    registry_path = output / "asset_registry.json"
    audit_path = output / "mesh_audit.json"
    gate_path = output / "opposite_platform_gate.json"
    density_path = output / "density_sweep.json"
    build_path = output / "opposite_platform_build.json"
    outputs = (
        obj_path,
        mtl_path,
        origin_path,
        registry_path,
        audit_path,
        gate_path,
        density_path,
        build_path,
    )
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    vertical = load_json(vertical_path)
    frame = CorridorFrame.from_json(vertical["frame"])
    if ownership_path is not None:
        owned_interval_m = local_owned_interval_from_plan(
            load_json(ownership_path), segment_id, frame
        )
    assert owned_interval_m is not None
    cloud = laspy.read(source)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    station, _ = frame.project(x, y)
    owned_low, owned_high = (float(value) for value in owned_interval_m)
    padding = float(settings["analysis_padding_m"])
    mask = (station >= owned_low - padding) & (station <= owned_high + padding)
    if not np.any(mask):
        raise ValueError("No point-cloud samples intersect the owned interval")

    candidates: list[dict[str, Any]] = []
    for configuration in density_configurations:
        identifier = str(configuration["id"])
        values = dict(base)
        values.update({key: value for key, value in configuration.items() if key != "id"})
        report = analyze_platform_surface_data(vertical, x[mask], y[mask], z[mask], values)
        candidates.append({"configuration_id": identifier, "settings": values, "report": report})

    prefix = f"{_safe_prefix(segment_id)}-{side.upper()}-OPPOSITE"
    component_id = f"{prefix}-PLATFORM-CANDIDATE-001"
    component, gate = prepare_opposite_platform_component(
        candidates,
        side=side,
        owned_interval_m=(owned_low, owned_high),
        selected_configuration_id=settings.get("selected_configuration_id"),
        component_id=component_id,
        minimum_accepted_configurations=int(settings["minimum_accepted_configurations"]),
        maximum_rail_side_spread_m=float(settings["maximum_rail_side_spread_m"]),
        maximum_median_elevation_spread_m=float(settings["maximum_median_elevation_spread_m"]),
        maximum_boundary_extrapolation_m=float(settings["maximum_boundary_extrapolation_m"]),
        maximum_fit_gap_m=float(settings["maximum_fit_gap_m"]),
        rail_side_clamp_half_width_m=float(settings["rail_side_clamp_half_width_m"]),
    )
    if not gate["passed"]:
        raise ValueError("Opposite-platform consensus gate did not pass")

    thickness = float(settings["inferred_platform_thickness_m"])
    top, top_faces, shell, shell_faces, sections = build_platform_component_mesh(
        component, frame, thickness
    )
    origin = np.asarray(
        [
            (float(np.min(top[:, 0])) + float(np.max(top[:, 0]))) / 2.0,
            (float(np.min(top[:, 1])) + float(np.max(top[:, 1]))) / 2.0,
            0.0,
        ],
        dtype=np.float64,
    )
    writer = ObjWriter(origin, material_library=mtl_path.name)
    top_id = f"{prefix}-PLATFORM-TOP"
    volume_id = f"{prefix}-PLATFORM-VOLUME"
    writer.add_mesh(top_id, top, top_faces, "OppositePlatformTopObserved")
    writer.add_mesh(volume_id, shell, shell_faces, "OppositePlatformVolumeInferred")
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(
        origin_path,
        {
            "schema_version": "railway.model-origin.v1",
            "origin_xyz": origin.tolist(),
            "units": "metre",
            "axis": "Z-up",
        },
    )

    gate.update(
        {
            "project_id": project.project_id,
            "segment_id": segment_id,
            "point_cloud": str(source),
            "vertical_report": str(vertical_path),
            "ownership_plan": str(ownership_path) if ownership_path else None,
            "platform_component_id": component_id,
        }
    )
    write_json(gate_path, gate)
    write_json(
        density_path,
        {
            "schema_version": "railway.opposite-platform-density-sweep.v1",
            "project_id": project.project_id,
            "segment_id": segment_id,
            "side": side,
            "owned_interval_m": [owned_low, owned_high],
            "configurations": candidates,
        },
    )
    audit = audit_obj(obj_path)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(audit_path, audit)
    if not audit["passed"]:
        raise ValueError("Opposite-platform OBJ failed mesh audit")

    registry = new_registry(project.project_id)
    registry["assets"] = [
        {
            "id": top_id,
            "type": "platform_surface",
            "subtype": "multi_density_conservative_fit",
            "status": "candidate",
            "chainage_m": None,
            "evidence_level": "observed",
            "confidence": float(settings["observed_top_confidence"]),
            "sources": [
                {"kind": "point_cloud", "reference": str(source)},
                {"kind": "rule", "reference": str(gate_path)},
            ],
            "parameters": {
                "side": side,
                "owned_interval_m": [owned_low, owned_high],
                "shared_section_count": len(sections),
                "outer_edge_policy": gate["outer_edge_policy"],
            },
            "geometry": {"file": str(obj_path), "node": top_id},
            "limitations": gate["limitations"],
        },
        {
            "id": volume_id,
            "type": "platform_volume",
            "subtype": "rule_inferred_skirt_and_bottom",
            "status": "candidate",
            "chainage_m": None,
            "evidence_level": "rule_inferred",
            "confidence": float(settings["inferred_volume_confidence"]),
            "sources": [{"kind": "rule", "reference": str(gate_path)}],
            "parameters": {"thickness_m": thickness, "top_asset_id": top_id},
            "geometry": {"file": str(obj_path), "node": volume_id},
            "limitations": ["Hidden side walls and thickness are visualization geometry."],
        },
    ]
    registry["relations"] = [
        {
            "id": f"{top_id}-SUPPORTED-BY-VOLUME",
            "type": "supported_by",
            "from": top_id,
            "to": volume_id,
        }
    ]
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Opposite-platform registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)

    build = {
        "schema_version": "railway.opposite-platform-reconstruction.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "side": side,
        "source_policy": "read_only",
        "point_cloud": str(source),
        "point_cloud_sha256": sha256_file(source),
        "ownership_plan": str(ownership_path) if ownership_path else None,
        "settings_source": str(settings_source) if settings_source else "bundled_default",
        "owned_interval_m": [owned_low, owned_high],
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "output_origin": str(origin_path),
        "output_registry": str(registry_path),
        "output_mesh_audit": str(audit_path),
        "output_gate": str(gate_path),
        "output_density_sweep": str(density_path),
        "asset_count": 2,
        "shared_section_count": len(sections),
        "vertex_count": writer.vertex_count,
        "face_count": writer.face_count,
        "model_sha256": sha256_file(obj_path),
        "status": "opposite_platform_candidate_generated_scope_gated",
    }
    write_json(build_path, build)
    return build
