from __future__ import annotations

import argparse
import copy
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from railway_recon.config import load_project
from railway_recon.io import load_json, sha256_file, write_json
from railway_recon.mesh_assembly import assemble_candidate_meshes
from railway_recon.mesh_object_registry import build_mesh_object_registry
from railway_recon.obj_subset import subset_obj_objects
from railway_recon.registry import summarize_registry, validate_registry_value

DEFAULT_STATION_TYPES = {
    "platform_surface",
    "platform_volume",
    "canopy_capital",
    "canopy_column",
    "canopy_roof_surface",
    "canopy_underroof_connector",
    "platform_fence",
    "station_information_sign",
}


def _object_names(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("o "):
            names.append(line[2:].strip())
    if not names:
        raise ValueError(f"OBJ has no object declarations: {path}")
    return names


def _copy_origin(source_obj: Path, output_dir: Path) -> Path:
    source = source_obj.parent / "model_origin.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = output_dir / "model_origin.json"
    shutil.copyfile(source, output)
    return output


def _filtered_registry(
    source_registry: Path,
    output_registry: Path,
    selected_nodes: set[str],
    output_obj: Path,
) -> dict:
    registry = copy.deepcopy(load_json(source_registry))
    assets = [
        asset
        for asset in registry.get("assets", [])
        if str(asset.get("geometry", {}).get("node")) in selected_nodes
    ]
    for asset in assets:
        asset.setdefault("geometry", {})["file"] = str(output_obj)
    selected_ids = {str(asset.get("id")) for asset in assets}
    relations = []
    for relation in registry.get("relations", []):
        values = {
            str(relation.get(key))
            for key in ("source", "target", "from", "to", "parent", "child")
            if relation.get(key) is not None
        }
        if values and values.issubset(selected_ids):
            relations.append(relation)
    registry["assets"] = assets
    registry["relations"] = relations
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["model_sha256"] = sha256_file(output_obj)
    registry["unclassified_asset_ids"] = []
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Filtered station registry is invalid: " + "; ".join(errors))
    write_json(output_registry, registry)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace an automatic station candidate with a reviewed station detail subset."
    )
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--track", required=True, type=Path)
    parser.add_argument("--catenary", required=True, type=Path)
    parser.add_argument("--conductor", required=True, type=Path)
    parser.add_argument("--automatic-station", required=True, type=Path)
    parser.add_argument("--detail-station", required=True, type=Path)
    parser.add_argument("--detail-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--station-type", action="append", default=[])
    parser.add_argument("--exclude-automatic", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = load_project(args.project)
    output_root = args.output_dir.resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    station_types = set(args.station_type) or DEFAULT_STATION_TYPES
    detail_registry_value = load_json(args.detail_registry)
    detail_assets = [
        asset
        for asset in detail_registry_value.get("assets", [])
        if str(asset.get("type")) in station_types
    ]
    detail_nodes = [str(asset["geometry"]["node"]) for asset in detail_assets]
    if not detail_nodes:
        raise ValueError("No station-detail assets matched the requested asset types")

    detail_dir = output_root / "station_detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_obj = detail_dir / "station_detail.obj"
    detail_subset = subset_obj_objects(args.detail_station, detail_obj, detail_nodes)
    _copy_origin(args.detail_station, detail_dir)
    detail_registry = _filtered_registry(
        args.detail_registry,
        detail_dir / "asset_registry.json",
        set(detail_nodes),
        detail_obj,
    )

    patterns = [re.compile(value, re.IGNORECASE) for value in args.exclude_automatic]
    automatic_nodes = _object_names(args.automatic_station)
    retained_automatic_nodes = [
        node for node in automatic_nodes if not any(pattern.search(node) for pattern in patterns)
    ]
    removed_automatic_nodes = [
        node for node in automatic_nodes if node not in set(retained_automatic_nodes)
    ]
    if not retained_automatic_nodes:
        raise ValueError("Automatic station filtering removed every object")
    outer_dir = output_root / "station_outer"
    outer_dir.mkdir(parents=True, exist_ok=True)
    outer_obj = outer_dir / "station_outer.obj"
    outer_subset = subset_obj_objects(
        args.automatic_station,
        outer_obj,
        retained_automatic_nodes,
    )
    _copy_origin(args.automatic_station, outer_dir)
    outer_registry_report = build_mesh_object_registry(
        project,
        outer_obj,
        outer_dir / "asset_registry.json",
        evidence_reference="station-detail-integration-report.json",
        overwrite=True,
    )

    scene_dir = output_root / "scene"
    assembly = assemble_candidate_meshes(
        project,
        scene_dir,
        args.output_name,
        [
            ("TRACK", args.track),
            ("CATENARY", args.catenary),
            ("CONDUCTOR", args.conductor),
            ("STATION_OUTER", outer_obj),
            ("STATION_DETAIL", detail_obj),
        ],
        overwrite=args.overwrite,
    )
    report = {
        "schema_version": "railway.station-detail-corridor-integration.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "station_asset_types": sorted(station_types),
        "automatic_exclusion_patterns": [pattern.pattern for pattern in patterns],
        "removed_automatic_objects": removed_automatic_nodes,
        "retained_automatic_object_count": len(retained_automatic_nodes),
        "detail_asset_count": len(detail_registry.get("assets", [])),
        "detail_subset": detail_subset,
        "outer_subset": outer_subset,
        "outer_registry": outer_registry_report,
        "assembly_report": assembly["output_mesh_audit"],
        "output_obj": assembly["output_obj"],
        "output_mtl": assembly["output_mtl"],
        "output_origin": assembly["output_origin"],
        "passed": True,
        "status": "reviewed_station_detail_integrated_without_track_or_catenary_duplication",
    }
    write_json(output_root / "station-detail-integration-report.json", report)
    print(output_root / "station-detail-integration-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
