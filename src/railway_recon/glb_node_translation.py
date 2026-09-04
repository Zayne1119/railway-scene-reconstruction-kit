from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json
from .recovery_binding import _read_glb, _write_glb


def translate_named_glb_nodes(
    *,
    source_glb: str | Path,
    output_glb: str | Path,
    translations: dict[str, tuple[float, float, float]],
    registry_path: str | Path,
    candidate_obj_path: str | Path,
    report_path: str | Path,
    release_id: str,
) -> Path:
    """Apply reviewed rigid node translations without re-encoding mesh buffers."""
    source = Path(source_glb).resolve()
    output = Path(output_glb).resolve()
    registry_source = Path(registry_path).resolve()
    candidate_obj = Path(candidate_obj_path).resolve()
    report = Path(report_path).resolve()
    for path in (output, report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite GLB promotion output: {path}")
    if not translations:
        raise ValueError("At least one node translation is required")

    document, remainder, _ = _read_glb(source)
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("GLB document has no nodes list")
    nodes_by_name: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            nodes_by_name.setdefault(node["name"], []).append(node)

    applied: list[dict[str, Any]] = []
    for name, delta_value in translations.items():
        matching = nodes_by_name.get(name, [])
        if len(matching) != 1:
            raise ValueError(f"Expected one GLB node named {name}, got {len(matching)}")
        node = matching[0]
        if "matrix" in node:
            raise ValueError(f"Matrix-authored node cannot be translated safely: {name}")
        delta = [float(value) for value in delta_value]
        if len(delta) != 3:
            raise ValueError(f"Node translation must have three components: {name}")
        before = [float(value) for value in node.get("translation", [0.0, 0.0, 0.0])]
        after = [before[index] + delta[index] for index in range(3)]
        node["translation"] = after
        applied.append(
            {
                "node_name": name,
                "translation_before_m": before,
                "translation_delta_m": delta,
                "translation_after_m": after,
            }
        )

    registry = load_json(registry_source)
    assets = registry.get("assets") if isinstance(registry, dict) else None
    if not isinstance(assets, list):
        raise TypeError("Candidate registry must contain an assets list")
    node_names = {str(node.get("name")) for node in nodes if isinstance(node, dict)}
    asset_ids = {str(asset.get("id")) for asset in assets if isinstance(asset, dict)}
    if node_names != asset_ids:
        raise ValueError("Candidate registry IDs do not exactly match GLB node names")

    embedded_assets = document.get("extras", {}).get("assetRegistry")
    if not isinstance(embedded_assets, list):
        raise TypeError("Source GLB must contain its Web asset registry")
    embedded_ids = {
        str(asset.get("id")) for asset in embedded_assets if isinstance(asset, dict)
    }
    if embedded_ids != node_names:
        raise ValueError("Source Web registry IDs do not exactly match GLB node names")
    candidate_by_id = {
        str(asset["id"]): asset for asset in assets if isinstance(asset, dict)
    }
    translated_by_id = {item["node_name"]: item for item in applied}
    web_assets = deepcopy(embedded_assets)
    for web_asset in web_assets:
        asset_id = str(web_asset.get("id"))
        translation = translated_by_id.get(asset_id)
        if translation is None:
            continue
        candidate_asset = candidate_by_id[asset_id]
        delta = translation["translation_delta_m"]
        if isinstance(web_asset.get("center"), list):
            web_asset["center"] = [
                float(value) + float(delta[index])
                for index, value in enumerate(web_asset["center"])
            ]
        bounds = web_asset.get("bounds")
        if isinstance(bounds, dict):
            for key in ("min", "max"):
                if isinstance(bounds.get(key), list):
                    bounds[key] = [
                        float(value) + float(delta[index])
                        for index, value in enumerate(bounds[key])
                    ]
        web_asset["status"] = candidate_asset.get("status")
        web_asset["evidenceScope"] = candidate_asset.get("evidence_level")
        web_asset["geometryRevision"] = {
            "releaseId": release_id,
            "operation": candidate_asset.get("parameters", {}).get(
                "run09_geometry_operation"
            ),
            "translationDeltaM": delta,
        }
        web_asset.setdefault("parameters", {}).update(
            deepcopy(candidate_asset.get("parameters", {}))
        )
        source_items = web_asset.setdefault("source", [])
        existing_references = {
            str(item.get("reference"))
            for item in source_items
            if isinstance(item, dict)
        }
        for item in candidate_asset.get("sources", []):
            if str(item.get("reference")) not in existing_references:
                source_items.append(deepcopy(item))
                existing_references.add(str(item.get("reference")))

    document.setdefault("extras", {})["assetRegistry"] = web_assets
    document["extras"]["assetCount"] = len(web_assets)
    document["extras"]["sourceRelease"] = release_id
    document["extras"]["geometryRevision"] = {
        "operation": "reviewed_rigid_node_translation",
        "candidateObjSha256": sha256_file(candidate_obj),
        "translatedNodes": applied,
    }
    _write_glb(document, remainder, output)

    output_document, output_remainder, _ = _read_glb(output)
    if output_remainder != remainder:
        raise RuntimeError("GLB binary geometry chunk changed during node translation")
    output_nodes = {
        str(node.get("name")): node
        for node in output_document.get("nodes", [])
        if isinstance(node, dict)
    }
    for item in applied:
        if output_nodes[item["node_name"]].get("translation") != item[
            "translation_after_m"
        ]:
            raise RuntimeError(f"Translated node failed round-trip: {item['node_name']}")

    write_json(
        report,
        {
            "schema_version": "railway.glb-node-translation.v1",
            "status": "pass",
            "release_id": release_id,
            "source_glb": str(source),
            "source_glb_sha256": sha256_file(source),
            "candidate_obj": str(candidate_obj),
            "candidate_obj_sha256": sha256_file(candidate_obj),
            "candidate_registry": str(registry_source),
            "candidate_registry_sha256": sha256_file(registry_source),
            "output_glb": str(output),
            "output_glb_sha256": sha256_file(output),
            "asset_count": len(web_assets),
            "node_count": len(nodes),
            "translated_nodes": applied,
            "binary_geometry_chunk_unchanged": True,
            "limitations": [
                "This promotion is valid only because candidate OBJ topology is identical to the source GLB.",
                "Known pre-existing canopy ownership seams remain a separate repair item.",
            ],
        },
    )
    return report
