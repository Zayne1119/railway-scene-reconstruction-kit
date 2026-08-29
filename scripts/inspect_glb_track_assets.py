from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

TRACK_TOKENS = ("track", "rail", "sleeper", "ballast", "turnout", "frog", "guardrail")


def read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a GLB: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing GLB JSON chunk: {path}")
    return json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))


def text_matches(*values: object) -> bool:
    value = " ".join(str(item).lower() for item in values if item is not None)
    return any(token in value for token in TRACK_TOKENS)


def overlaps(asset: dict[str, Any], start_m: float, end_m: float) -> bool:
    chainage = asset.get("chainageRangeM")
    if isinstance(chainage, list) and len(chainage) == 2:
        return float(chainage[1]) >= start_m and float(chainage[0]) <= end_m
    center = asset.get("chainage_m")
    return center is not None and start_m <= float(center) <= end_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-m", type=float, default=0.0)
    parser.add_argument("--end-m", type=float, default=200.0)
    args = parser.parse_args()
    document = read_glb_json(args.source.resolve())
    registry = list(document.get("extras", {}).get("assetRegistry", []))
    assets = [
        asset
        for asset in registry
        if text_matches(
            asset.get("id"),
            asset.get("type"),
            asset.get("name"),
            asset.get("professional"),
            asset.get("integrationLayer"),
        )
        and overlaps(asset, args.start_m, args.end_m)
    ]
    asset_ids = {str(asset.get("id")) for asset in assets}
    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(document.get("nodes", [])):
        if "mesh" not in node:
            continue
        extras = node.get("extras", {})
        asset_id = str(extras.get("assetId") or node.get("name") or "")
        layer = str(extras.get("v63Layer") or extras.get("v63_layer") or "")
        if asset_id in asset_ids or text_matches(asset_id, node.get("name"), layer):
            nodes.append(
                {
                    "node_index": index,
                    "name": node.get("name"),
                    "asset_id": asset_id,
                    "layer": layer,
                    "mesh_index": node.get("mesh"),
                    "registered_in_target_range": asset_id in asset_ids,
                }
            )
    result = {
        "schema_version": "railway.glb-track-inventory.v1",
        "source": str(args.source.resolve()),
        "target_chainage_range_m": [args.start_m, args.end_m],
        "embedded_registry_count": len(registry),
        "matched_asset_count": len(assets),
        "matched_node_count": len(nodes),
        "assets": assets,
        "nodes": nodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "embedded_registry_count": len(registry),
                "matched_asset_count": len(assets),
                "matched_node_count": len(nodes),
                "matched_types": sorted({str(asset.get("type")) for asset in assets}),
                "matched_layers": sorted(
                    {
                        str(asset.get("integrationLayer"))
                        for asset in assets
                        if asset.get("integrationLayer")
                    }
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
