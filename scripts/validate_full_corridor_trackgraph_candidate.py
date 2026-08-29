"""Validate a full-scene TrackGraph replacement candidate and bind its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def glb_document(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"glTF" or struct.unpack_from("<I", data, 4)[0] != 2:
        raise ValueError(f"Invalid GLB header: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing GLB JSON chunk: {path}")
    return json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))


def check(identifier: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--mesh-audit", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--track-mesh-audit", type=Path, required=True)
    parser.add_argument("--track-sidecar", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version", default="v9.6-trackgraph-visual-candidate-01"
    )
    args = parser.parse_args()

    document = glb_document(args.glb)
    registry = read_json(args.registry)
    integration = read_json(args.integration)
    mesh = read_json(args.mesh_audit)
    relationships = read_json(args.relationships)
    track_mesh = read_json(args.track_mesh_audit)
    sidecar = read_json(args.track_sidecar)
    views = read_json(args.views)
    embedded = document.get("extras", {}).get("assetRegistry", [])
    summary = mesh["summary"]
    track_nodes = {
        str(asset["geometry"]["node"])
        for asset in sidecar.get("assets", [])
        if isinstance(asset.get("geometry"), dict) and asset["geometry"].get("node")
    }
    registry_ids = {str(asset["id"]) for asset in registry}
    new_track_ids = {
        identifier
        for identifier in registry_ids
        if identifier.startswith(("TRACK-", "TURNOUT-"))
    }
    legacy_track_ids = {
        str(asset["id"])
        for asset in registry
        if str(asset.get("type")) in {"Rail", "SleeperArray", "TrackBed"}
        and not str(asset["id"]).startswith(("TRACK-", "TURNOUT-"))
    }
    view_files = [args.views.parent / Path(shot["image"]).name for shot in views.get("shots", [])]
    checks = [
        check("integration.status", integration.get("status") == "automatic_checks_passed_visual_review_required", integration.get("status")),
        check("integration.removed_legacy_assets", integration.get("integration", {}).get("removed_asset_count") == 57, integration.get("integration", {}).get("removed_asset_count")),
        check("integration.added_track_objects", integration.get("integration", {}).get("added_object_count") == 32, integration.get("integration", {}).get("added_object_count")),
        check("registry.count", len(registry) == 743, len(registry)),
        check("registry.indices", [int(asset["index"]) for asset in registry] == list(range(len(registry))), [registry[0]["index"], registry[-1]["index"]]),
        check("registry.embedded", embedded == registry, len(embedded)),
        check("registry.no_legacy_track", not legacy_track_ids, sorted(legacy_track_ids)[:20]),
        check("registry.track_nodes", new_track_ids == track_nodes, {"registry": len(new_track_ids), "sidecar": len(track_nodes)}),
        check("mesh.asset_count", summary.get("asset_count") == 743, summary.get("asset_count")),
        check("mesh.object_count", summary.get("object_count") == 755, summary.get("object_count")),
        check("mesh.triangle_count", summary.get("source_triangle_count") == 631146, summary.get("source_triangle_count")),
        check("mesh.degenerate", summary.get("visible_degenerate_triangle_count") == 0, summary.get("visible_degenerate_triangle_count")),
        check("mesh.winding", summary.get("visible_winding_normal_conflict_triangle_count") == 0, summary.get("visible_winding_normal_conflict_triangle_count")),
        check("mesh.normals", summary.get("visible_primitive_without_normals_count") == 0 and summary.get("visible_nonfinite_normal_count") == 0, {"missing": summary.get("visible_primitive_without_normals_count"), "nonfinite": summary.get("visible_nonfinite_normal_count")}),
        check("mesh.duplicates", summary.get("visible_exact_duplicate_triangle_count") == 0 and summary.get("visible_near_duplicate_triangle_count_at_0_1mm") == 0, {"exact": summary.get("visible_exact_duplicate_triangle_count"), "near": summary.get("visible_near_duplicate_triangle_count_at_0_1mm")}),
        check("mesh.opaque_single_sided", summary.get("opaque_double_sided_material_count") == 0, summary.get("opaque_double_sided_material_count")),
        check("relationships", relationships.get("status") == "pass" and relationships.get("failure_count") == 0 and relationships.get("relationship_count") == 99, {k: relationships.get(k) for k in ("status", "relationship_count", "failure_count")}),
        check("track_mesh", track_mesh.get("status") == "pass" and track_mesh.get("visible_node_count") == 32, {"status": track_mesh.get("status"), "nodes": track_mesh.get("visible_node_count")}),
        check("fixed_views", len(view_files) == 8 and all(path.is_file() for path in view_files), [str(path) for path in view_files]),
        check("interchange.fbx", args.glb.with_suffix(".fbx").is_file(), str(args.glb.with_suffix(".fbx"))),
        check("interchange.obj", args.glb.with_suffix(".obj").is_file(), str(args.glb.with_suffix(".obj"))),
        check("interchange.mtl", args.glb.with_suffix(".mtl").is_file(), str(args.glb.with_suffix(".mtl"))),
    ]
    failures = [item for item in checks if not item["passed"]]
    result = {
        "schema_version": "railway.trackgraph-full-corridor-validation.v1",
        "version": args.version,
        "status": "conditional_pass_display_candidate" if not failures else "fail",
        "checks": checks,
        "summary": {
            "checks": len(checks),
            "passed": len(checks) - len(failures),
            "failures": len(failures),
            "asset_count": len(registry),
            "object_count": summary.get("object_count"),
            "triangle_count": summary.get("source_triangle_count"),
            "glb_sha256": sha256(args.glb),
        },
        "release_boundary": {
            "allowed": "complete-scene visual review, Web preview and UE rendering candidate",
            "blocked": "survey-grade turnout geometry or dimensional claims for the orange rule-inferred rails",
            "reason": "turnout sleepers, blades, frogs, guard rails and independent control are unavailable",
        },
        "failures": failures,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
