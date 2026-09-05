"""Offline, persistent onboarding demo; never a real-station quality benchmark.

Only track candidates are extracted from generated synthetic points. Platforms,
columns, beams and roof are authored mathematical demonstration layouts. The
existing single-segment track builder and reviewed-layout mesh builder are
reused without changing production defaults or claiming real observations.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import laspy
import numpy as np

from .algorithms.rail_candidates import detect_rail_candidates
from .algorithms.reviewed_scene import build_reviewed_scene
from .algorithms.track import build_parametric_track
from .config import initialize_project, load_project
from .io import load_json, write_json
from .mesh_assembly import assemble_candidate_meshes
from .mesh_audit import audit_obj
from .registry import summarize_registry, validate_registry_value
from .segments import crop_segments, plan_segments
from .web_glb_export import export_obj_to_web_glb, inspect_web_glb

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEMO_SETTINGS = {
    "schema_version": "railway.onboarding-demo-settings.v1",
    "namespace": "independent_authored_onboarding_v1_not_research_test_allocation",
    "seed": 731912,
    "length_m": 40.0,
    "rail_center_spacing_m": 1.508,
    "rail_top_z_m": 0.30,
    "nominal_clear_gauge_m": 1.435,
    "point_step_m": 0.125,
    "ground_points": 800,
    "authored_platform_x_m": [2.0, 38.0],
    "authored_platform_y_m": [2.8, 7.8],
    "authored_column_x_m": [8.0, 16.0, 24.0, 32.0],
    "track_route": "existing_single_segment_parametric_demo_not_full_production_track_graph",
    "station_route": "authored_synthetic_layout_not_extracted_from_point_cloud",
}
WEB_FILES = (
    "project.json", "scene.glb", "asset_registry.json", "mesh_audit.json",
    "model_origin.json", "track_build.json", "README.txt",
)
DISCLOSURE = (
    "SYNTHETIC DEMO: independently authored mathematical input, not a real station, "
    "customer model, survey result, accuracy benchmark or automatic station extraction. "
    "Track candidates come from synthetic points; platform, columns, beams and roof "
    "are authored demonstration layouts. All assets remain demonstration candidates."
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _source_bindings() -> dict[str, str]:
    """Bind the complete static local import closure and packaged JSON settings."""
    pending = ["railway_recon.onboarding_demo"]
    modules, paths = set(), set()
    source_root = REPOSITORY_ROOT / "src"
    while pending:
        module = pending.pop()
        if module in modules or not module.startswith("railway_recon"):
            continue
        modules.add(module)
        path = source_root.joinpath(*module.split(".")).with_suffix(".py")
        package = False
        if not path.is_file():
            path = source_root.joinpath(*module.split("."), "__init__.py")
            package = True
        if not path.is_file():
            continue
        paths.add(path)
        for parent_count in range(1, len(module.split("."))):
            initializer = source_root.joinpath(*module.split(".")[:parent_count], "__init__.py")
            if initializer.is_file():
                paths.add(initializer)
        parts = module.split(".") if package else module.split(".")[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                pending.extend(item.name for item in node.names if item.name.startswith("railway_recon"))
            elif isinstance(node, ast.ImportFrom):
                base = parts[: len(parts) - node.level + 1] if node.level else []
                target = ".".join([*base, *([node.module] if node.module else [])])
                if target.startswith("railway_recon"):
                    pending.append(target)
                    pending.extend(target + "." + item.name for item in node.names if item.name != "*")
    paths.add(source_root / "railway_recon" / "__init__.py")
    paths.update((source_root / "railway_recon" / "resources").glob("*.json"))
    paths.update(REPOSITORY_ROOT / name for name in (
        "pyproject.toml", "uv.lock", "scripts/build_onboarding_demo.py"
    ))
    return {path.relative_to(REPOSITORY_ROOT).as_posix(): _sha(path) for path in sorted(paths)}


def _environment() -> dict:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "laspy", "Pillow", "jsonschema")}


def _destination(value: str | Path) -> Path:
    path = Path(value)
    if ".." in path.parts or any(part in {".git", ".venv", "node_modules"} for part in path.parts):
        raise ValueError("Output path must not contain traversal or protected directories")
    resolved = path.resolve()
    if resolved == REPOSITORY_ROOT or resolved in REPOSITORY_ROOT.parents or resolved.parent == resolved:
        raise ValueError("Output must be a dedicated new child directory, not a broad root")
    if path.absolute() != resolved:
        raise ValueError("Output must not use symlinked or redirected directories")
    return resolved


def _safe_artifact(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Manifest artifact must be a safe relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("Manifest artifact path traversal is forbidden")
    result = root.joinpath(*path.parts)
    if result.resolve() != result or not result.resolve().is_relative_to(root):
        raise ValueError("Manifest artifact points outside its owned directory")
    return result


def _records(root: Path, *, exclude: str) -> list[dict]:
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != exclude]


def _verify_inventory(root: Path, records: list[dict], manifest_name: str) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("Demo manifest has no artifact inventory")
    expected = set()
    for record in records:
        relative = record["path"]
        path = _safe_artifact(root, relative)
        if relative in expected or relative == manifest_name:
            raise ValueError("Demo artifact inventory is duplicated or self-referential")
        expected.add(relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
            raise ValueError(f"Demo artifact changed or missing: {relative}; use a new version path")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected | {manifest_name}:
        raise ValueError("Demo directory contains unowned or missing files; use a new version path")


def verify_onboarding_demo(output: str | Path) -> dict:
    """Reuse only an intact source/settings/environment/output-bound demo."""
    root = _destination(output)
    manifest_path = root / "demo_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError("Existing directory is not a completed owned demo; use a new version path")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != "railway.onboarding-demo-manifest.v1" or manifest.get("status") != "completed":
        raise ValueError("Demo manifest is not a completed supported run")
    if manifest.get("settings") != DEMO_SETTINGS or manifest.get("settings_sha256") != hashlib.sha256(_json_bytes(DEMO_SETTINGS)).hexdigest():
        raise ValueError("Demo settings changed; use a new version path")
    if manifest.get("source_sha256") != _source_bindings() or manifest.get("packages") != _environment():
        raise ValueError("Demo source or dependency environment changed; use a new version path")
    _verify_inventory(root, manifest["files"], "demo_manifest.json")
    if manifest_path.read_bytes() != raw:
        raise ValueError("Demo manifest changed during verification")
    return manifest


def _build_cloud(path: Path) -> dict:
    rng = np.random.default_rng(DEMO_SETTINGS["seed"])
    along = np.arange(0.0, DEMO_SETTINGS["length_m"] + 0.001, DEMO_SETTINGS["point_step_m"])
    rails = []
    for side in (-1, 1):
        rails.append(np.column_stack((np.repeat(along, 3),
            np.full(len(along) * 3, side * DEMO_SETTINGS["rail_center_spacing_m"] / 2)
            + np.tile([-0.012, 0.0, 0.012], len(along)),
            np.full(len(along) * 3, DEMO_SETTINGS["rail_top_z_m"]))))
    count = DEMO_SETTINGS["ground_points"]
    ground = np.column_stack((rng.uniform(0, DEMO_SETTINGS["length_m"], count),
                              rng.uniform(-4, 4, count), rng.normal(0, 0.008, count)))
    xyz = np.vstack([*rails, ground])
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.creation_date = date(2020, 1, 1)
    header.generating_software = "Authored synthetic demo v1"
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz.T
    cloud.classification = np.zeros(len(xyz), dtype=np.uint8)
    cloud.red = cloud.green = cloud.blue = np.full(len(xyz), 30000, dtype=np.uint16)
    cloud.write(path)
    return {"point_count": len(xyz), "sha256": _sha(path), "synthetic_demo": True,
            "numeric_classification": "all_zero_no_reference_truth_labels",
            "input_role": "mathematical_points_for_onboarding_not_field_data"}


def _authored_station_layout() -> dict:
    assets, relations = [], []

    def add(identifier: str, kind: str, geometry: dict) -> None:
        assets.append({"id": identifier, "type": kind, "status": "candidate",
                       "evidence_level": "rule_inferred", "confidence": 0.0,
                       "sources": [{"kind": "rule", "reference": "authored-synthetic-onboarding-layout-v1"}],
                       "parameters": {"synthetic_demo": True, "extracted_from_points": False,
                                      "confidence_semantics": "not_measured_or_calibrated"},
                       "geometry": geometry, "limitations": [DISCLOSURE]})

    add("DEMO-PLATFORM", "platform", {"kind": "extruded_polygon_xy",
        "points": [[2, 2.8], [38, 2.8], [38, 7.8], [2, 7.8]], "bottom_z": 0.0, "top_z": 1.0})
    for index, x in enumerate(DEMO_SETTINGS["authored_column_x_m"], start=1):
        column, beam = f"DEMO-CANOPY-COLUMN-{index:02d}", f"DEMO-CANOPY-BEAM-{index:02d}"
        column_top = 5.065 + (5.5 - 3.0) * 0.15 / 4.7 - 0.125
        add(column, "canopy_column", {"kind": "box", "center": [x, 5.5, (1.0 + column_top) / 2.0],
                                     "size": [0.35, 0.35, column_top - 1.0]})
        add(beam, "canopy_beam", {"kind": "beam", "start": [x, 3.0, 5.065],
                                 "end": [x, 7.7, 5.215], "width": 0.24, "height": 0.25})
        relations.extend([{"id": f"REL-COLUMN-BEAM-{index:02d}", "type": "supports", "from": column, "to": beam},
                          {"id": f"REL-BEAM-ROOF-{index:02d}", "type": "supports", "from": beam, "to": "DEMO-CANOPY-ROOF"}])
    add("DEMO-CANOPY-ROOF", "canopy_roof", {"kind": "panel",
        "points": [[3, 3.0, 5.25], [37, 3.0, 5.25], [37, 7.7, 5.4], [3, 7.7, 5.4]], "thickness": 0.12})
    return {"schema_version": "railway.reviewed-scene-layout.v1", "origin_xyz": [0, 0, 0],
            "synthetic_demo": True, "layout_role": "authored_not_automatically_extracted",
            "disclosure": DISCLOSURE, "assets": assets, "relations": relations}


def _viewer_registry(nodes: list[str], layout: dict) -> dict:
    authored = {"STATION--" + asset["id"]: asset for asset in layout["assets"]}
    assets = []
    for node in nodes:
        if node in authored:
            kind = authored[node]["type"]
            route = "authored_synthetic_layout_not_extracted_from_points"
        elif "-RAIL-" in node:
            kind, route = "rail", "candidate_fit_to_generated_synthetic_points"
        elif "-SLEEPER-" in node:
            kind, route = "sleeper", "parametric_rule_from_synthetic_track_candidate"
        elif node.endswith("-BED"):
            kind, route = "track_bed", "parametric_rule_from_synthetic_track_candidate"
        else:
            raise ValueError("Unexpected unregistered demo geometry object")
        from_points = kind == "rail"
        assets.append({"id": node, "type": kind, "status": "candidate",
                       "evidence_level": "observed" if from_points else "rule_inferred", "confidence": 0.0,
                       "sources": [{"kind": "point_cloud" if from_points else "rule",
                                    "reference": "synthetic-onboarding-v1", "note": DISCLOSURE}],
                       "parameters": {"synthetic_demo": True, "generation_route": route,
                                      "confidence_semantics": "not_measured_or_calibrated"},
                       "geometry": {"node": node, "file": "/demo/scene.glb"}, "limitations": [DISCLOSURE]})
    relations = [{**relation, "from": "STATION--" + relation["from"], "to": "STATION--" + relation["to"],
                  "synthetic_authored_relation": True} for relation in layout["relations"]]
    registry = {"schema_version": "railway.asset-registry.v1", "project_id": "onboarding-demo",
                "synthetic_demo": True, "disclosure": DISCLOSURE, "assets": assets, "relations": relations}
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Invalid demo registry: " + "; ".join(errors))
    return registry


def _viewer_payloads(root: Path, mesh: dict, track: dict, candidates: dict, layout: dict) -> dict:
    viewer = root / "viewer"
    inspection = inspect_web_glb(viewer / "scene.glb")
    registry = _viewer_registry(inspection["node_names"], layout)
    _write_new(viewer / "asset_registry.json", _json_bytes(registry))
    selected_mesh = {key: value for key, value in mesh.items() if key != "path" and not isinstance(value, Path)}
    selected_mesh.pop("source", None)
    selected_mesh["synthetic_demo"], selected_mesh["disclosure"] = True, DISCLOSURE
    # The public viewer receives numeric QA and object names, never local machine paths.
    selected_mesh = {key: value for key, value in selected_mesh.items()
                     if not (isinstance(value, str) and (str(root) in value or ":\\" in value))}
    _write_new(viewer / "mesh_audit.json", _json_bytes(selected_mesh))
    origin = load_json(root / "workspace" / "exports" / "demo_scene" / "model_origin.json")
    _write_new(viewer / "model_origin.json", _json_bytes({"origin_xyz": origin["origin_xyz"],
               "units": "metre", "axis": "Z-up", "synthetic_demo": True}))
    track_report = {"schema_version": "railway.onboarding-demo-track.v1", "synthetic_demo": True,
                    "disclosure": DISCLOSURE, "rail_pair_count": candidates["rail_pair_count"],
                    "tracks": track["tracks"], "include_inferred_gap_hypotheses": False}
    _write_new(viewer / "track_build.json", _json_bytes(track_report))
    config = {"title": "SYNTHETIC DEMO · 合成演示（非真实站场 / 非测量成果）",
              "synthetic_demo": True, "disclosure": DISCLOSURE,
              "model_glb_url": "/demo/scene.glb", "full_model_glb_url": "/demo/scene.glb",
              "origin_url": "/demo/model_origin.json", "mesh_audit_url": "/demo/mesh_audit.json",
              "track_report_url": "/demo/track_build.json",
              "registry_sources": [{"namespace": "DEMO", "url": "/demo/asset_registry.json"}]}
    _write_new(viewer / "project.json", _json_bytes(config))
    _write_new(viewer / "README.txt", (DISCLOSURE + "\nMount this directory at /demo/ and open ?demo=1.\n").encode("utf-8"))
    return {"inspection": inspection, "asset_count": len(registry["assets"]), "config": "viewer/project.json"}


def _build_new_demo(root: Path, sources: dict, packages: dict) -> dict:
    # Acquire this exact directory before writing any failure marker. A race
    # creating an unrelated directory must not turn it into an owned demo.
    root.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": "railway.onboarding-demo-state.v1", "status": "running",
             "synthetic_demo": True, "events": [], "disclosure": DISCLOSURE}
    stage = "initialize"
    try:
        initialize_project(root, "onboarding-demo", "SYNTHETIC onboarding demonstration")
        _write_new(root / "demo_settings.json", _json_bytes(DEMO_SETTINGS))
        project_config = load_json(root / "project.json")
        project_config["segmentation"].update(length_m=40.0, corridor_half_width_m=10.0,
                                               z_below_camera_m=5.0, z_above_camera_m=8.0)
        project_config["algorithms"].pop("track_graph", None)
        project_config["reconstruction"]["manual_review_required"] = False
        project_config["project"]["name"] = "SYNTHETIC DEMO - not a real station or survey"
        write_json(root / "project.json", project_config)
        detector = load_json(root / "rail_detection.json")
        detector.update(z_mode="absolute", minimum_z=0.20, maximum_z=0.40,
                        minimum_peak_prominence=0.01, median_filter_bins=1, gaussian_sigma_bins=0.6)
        write_json(root / "rail_detection.json", detector)
        project = load_project(root / "project.json")

        def complete(name: str, details: dict | None = None) -> None:
            state["events"].append({"stage": name, "status": "complete", **(details or {})})
            write_json(root / "run_state.json", state)

        complete(stage)
        stage = "generate_independent_synthetic_inputs"
        cloud = _build_cloud(root / "input" / "pointcloud" / "site.laz")
        with (root / "input" / "cameras.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
            for index, x in enumerate((0, 10, 20, 30, 40)):
                writer.writerow([index, index, f"synthetic-camera-{index:02d}-no-image", x, 0, 2])
        _write_new(root / "synthetic_input_provenance.json", _json_bytes({**cloud, "disclosure": DISCLOSURE}))
        complete(stage, {"point_count": cloud["point_count"]})
        stage = "segment_and_extract_synthetic_track"
        plan = plan_segments(project)
        if len(plan["segments"]) != 1:
            raise ValueError("Onboarding fixture must be exactly one synthetic segment")
        segment = plan["segments"][0]["id"]
        crop_segments(project, {segment})
        candidates = detect_rail_candidates(project, segment)
        if candidates["rail_pair_count"] != 1:
            raise ValueError("Expected exactly one rail pair in the deterministic synthetic input")
        complete(stage, {"rail_pair_count": candidates["rail_pair_count"]})
        stage = "build_existing_parametric_track"
        track = build_parametric_track(project, segment)
        if not audit_obj(Path(track["output_obj"]))["passed"]:
            raise ValueError("Synthetic track failed mesh integrity audit")
        complete(stage)
        stage = "build_authored_synthetic_station_layout"
        layout = _authored_station_layout()
        layout_path = root / "authored_station_layout.json"
        _write_new(layout_path, _json_bytes(layout))
        station = build_reviewed_scene(project, layout_path)
        complete(stage, {"authored_not_extracted": True, "asset_count": len(layout["assets"])})
        stage = "assemble_audit_and_export_glb"
        assembled = assemble_candidate_meshes(project, root / "workspace" / "exports" / "demo_scene",
                                              "scene", [("TRACK", Path(track["output_obj"])),
                                                        ("STATION", Path(station["output_obj"]))])
        mesh = load_json(Path(assembled["output_mesh_audit"]))
        converted = export_obj_to_web_glb(assembled["output_obj"], root / "viewer" / "scene.glb",
                                          report_path=root / "workspace" / "reports" / "demo_glb_conversion.json")
        if not converted["passed"] or converted["triangle_count"] < 1:
            raise ValueError("Demo GLB is empty or failed validation")
        viewer = _viewer_payloads(root, mesh, track, candidates, layout)
        complete(stage, {"triangles": converted["triangle_count"], "objects": viewer["asset_count"]})
        if sources != _source_bindings() or packages != _environment():
            raise ValueError("Demo source or dependencies changed during construction")
        state["status"] = "completed_synthetic_demo_only"
        write_json(root / "run_state.json", state)
        manifest = {"schema_version": "railway.onboarding-demo-manifest.v1", "status": "completed",
                    "synthetic_demo": True, "customer_data_used": False, "network_used": False,
                    "real_station_extraction_demonstrated": False, "survey_accuracy_claimed": False,
                    "settings": copy.deepcopy(DEMO_SETTINGS),
                    "settings_sha256": hashlib.sha256(_json_bytes(DEMO_SETTINGS)).hexdigest(),
                    "source_sha256": sources, "packages": packages, "input": cloud,
                    "geometry": viewer["inspection"], "asset_count": viewer["asset_count"],
                    "disclosure": DISCLOSURE, "files": _records(root, exclude="demo_manifest.json")}
        _write_new(root / "demo_manifest.json", _json_bytes(manifest))
        return manifest
    except Exception as error:
        if root.is_dir() and not (root / "run_failure.json").exists():
            _write_new(root / "run_failure.json", _json_bytes({"status": "failed", "stage": stage,
                       "exception_type": type(error).__name__, "recovery": "preserve_outputs_and_use_new_version_path"}))
        raise


def publish_onboarding_demo(output: str | Path, web_output: str | Path) -> dict:
    """Copy only verified synthetic viewer artifacts into a dedicated new directory."""
    root, target = _destination(output), _destination(web_output)
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Project and web output directories must not overlap")
    verify_onboarding_demo(root)
    source_manifest_sha = _sha(root / "demo_manifest.json")
    source_payloads = {name: (root / "viewer" / name).read_bytes() for name in WEB_FILES}
    if target.exists():
        manifest_path = target / "demo_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("Web directory is not an owned demo; use a new version path")
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get("schema_version") != "railway.onboarding-demo-web-manifest.v1"
                or manifest.get("status") != "completed"
                or manifest.get("source_demo_manifest_sha256") != source_manifest_sha):
            raise ValueError("Web demo belongs to different source/settings; use a new version path")
        _verify_inventory(target, manifest["files"], "demo_manifest.json")
        if any((target / name).read_bytes() != raw for name, raw in source_payloads.items()):
            raise ValueError("Published demo differs from verified source payloads")
        return {"status": "verified_existing", "file_count": len(WEB_FILES)}
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw in source_payloads.items():
            _write_new(target / name, raw)
        verify_onboarding_demo(root)
        if _sha(root / "demo_manifest.json") != source_manifest_sha:
            raise ValueError("Source demo changed during publication")
        manifest = {"schema_version": "railway.onboarding-demo-web-manifest.v1", "status": "completed",
                    "synthetic_demo": True, "network_upload_performed": False,
                    "source_demo_manifest_sha256": source_manifest_sha,
                    "disclosure": DISCLOSURE, "files": _records(target, exclude="demo_manifest.json")}
        _write_new(target / "demo_manifest.json", _json_bytes(manifest))
        return {"status": "written_local_synthetic_viewer_bundle", "file_count": len(WEB_FILES)}
    except Exception as error:
        _write_new(target / "run_failure.json", _json_bytes({"status": "failed", "exception_type": type(error).__name__}))
        raise


def build_onboarding_demo(output: str | Path | None = None, web_output: str | Path | None = None) -> dict:
    """Build or verify a deterministic demo; never overwrite an unrelated directory."""
    root = _destination(output if output is not None else REPOSITORY_ROOT / "projects" / "onboarding-demo")
    if web_output is not None:
        target = _destination(web_output)
        if target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError("Project and web output directories must not overlap")
        if target.exists() and not (target / "demo_manifest.json").is_file():
            raise FileExistsError("Web output contains unowned files; choose a dedicated new directory")
    reused = root.exists()
    manifest = verify_onboarding_demo(root) if reused else _build_new_demo(root, _source_bindings(), _environment())
    result = {"status": "verified_existing" if reused else "built_synthetic_demo",
              "output": str(root), "synthetic_demo": True, "point_count": manifest["input"]["point_count"],
              "asset_count": manifest["asset_count"], "triangle_count": manifest["geometry"]["triangle_count"],
              "model_sha256": _sha(root / "viewer" / "scene.glb"), "disclosure": DISCLOSURE}
    if web_output is not None:
        result["web"] = publish_onboarding_demo(root, web_output)
        result["web_output"] = str(_destination(web_output))
        result["viewer_query"] = "?demo=1"
    return result
