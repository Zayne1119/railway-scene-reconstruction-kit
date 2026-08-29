from __future__ import annotations

import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


def _registry_assets(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("assets"), list):
        return value["assets"]
    raise ValueError("Scene registry must be a list or contain an assets list")


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes, bytes]:
    data = path.read_bytes()
    if data[:4] != b"glTF" or len(data) < 20:
        raise ValueError(f"Not a GLB: {path}")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    json_bytes = data[20 : 20 + json_length]
    document = json.loads(json_bytes.decode("utf-8").rstrip("\x00 "))
    return document, data[20 + json_length :], data


def _write_glb(document: dict[str, Any], remainder: bytes, output: Path) -> None:
    json_bytes = json.dumps(
        document, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total_length = 12 + 8 + len(json_bytes) + len(remainder)
    header = b"glTF" + struct.pack("<II", 2, total_length)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + json_chunk + remainder)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bind_recovery_registry(
    registry: list[dict[str, Any]],
    recovery: dict[str, Any],
    recovery_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    recovered = [
        item
        for item in recovery.get("recoveries", [])
        if item.get("status") == "fixed_center_evidence_recovered"
    ]
    unresolved = [
        str(item.get("observation_id"))
        for item in recovery.get("recoveries", [])
        if item.get("status") != "fixed_center_evidence_recovered"
    ]
    if unresolved:
        raise ValueError(f"Recovery still has unresolved observations: {unresolved}")
    if not recovered:
        raise ValueError("Recovery report contains no recovered observations")

    result = deepcopy(registry)
    assets_by_observation: dict[str, list[str]] = {}
    for item in recovered:
        observation_id = str(item["observation_id"])
        matching = []
        for asset in result:
            parameters = asset.get("parameters")
            if not isinstance(parameters, dict):
                continue
            source_ids = [str(value) for value in parameters.get("source_observation_ids", [])]
            if observation_id not in source_ids or str(asset.get("type")) != "Rail":
                continue
            matching.append(asset)
        if len(matching) != 2:
            raise ValueError(
                f"Expected two rail assets for {observation_id}, got {len(matching)}"
            )
        assets_by_observation[observation_id] = [str(asset["id"]) for asset in matching]
        evidence = {
            "observationId": observation_id,
            "status": "fixed_center_evidence_recovered",
            "geometryChanged": False,
            "support": deepcopy(item["support"]),
            "recoveryReportSha256": recovery_sha256,
        }
        for asset in matching:
            asset.setdefault("targetedRecoveryEvidence", []).append(evidence)
            parameters = asset.setdefault("parameters", {})
            parameters["targeted_recovery_status"] = (
                "fixed_center_evidence_recovered"
            )
            parameters["targeted_recovery_geometry_changed"] = False
            parameters["targeted_recovery_report_sha256"] = recovery_sha256
            sources = asset.setdefault("sources", [])
            reference = f"targeted-recovery:{recovery_sha256}:{observation_id}"
            if not any(str(value.get("reference")) == reference for value in sources):
                sources.append(
                    {
                        "kind": "point_cloud_fixed_center_support",
                        "reference": reference,
                    }
                )
    return result, assets_by_observation


def bind_targeted_recovery_to_scene(
    source_glb_value: str | Path,
    registry_value: str | Path,
    recovered_graph_value: str | Path,
    recovery_value: str | Path,
    output_glb_value: str | Path,
    output_registry_value: str | Path,
    report_value: str | Path,
    *,
    profile: str,
) -> Path:
    source_glb = Path(source_glb_value).resolve()
    registry_path = Path(registry_value).resolve()
    recovered_graph_path = Path(recovered_graph_value).resolve()
    recovery_path = Path(recovery_value).resolve()
    output_glb = Path(output_glb_value).resolve()
    output_registry = Path(output_registry_value).resolve()
    report_path = Path(report_value).resolve()
    for output in (output_glb, output_registry, report_path):
        if output.exists():
            raise FileExistsError(f"Recovery binding output already exists: {output}")

    recovery = load_json(recovery_path)
    graph = load_json(recovered_graph_path)
    if recovery.get("schema_version") != "railway.targeted-rail-recovery.v1":
        raise ValueError("Unsupported targeted recovery report")
    if graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError("Recovery binding requires TrackGraph v1")
    binding = graph.get("targeted_pair_recovery")
    if not isinstance(binding, dict) or binding.get("geometry_changed") is not False:
        raise ValueError("Recovered graph does not prove geometry_changed=false")
    recovery_sha256 = sha256_file(recovery_path)
    if str(binding.get("recovery_report_sha256")) != recovery_sha256:
        raise ValueError("Recovered graph is not bound to the requested recovery report")

    document, remainder, source_bytes = _read_glb(source_glb)
    sidecar_value = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    sidecar_assets = _registry_assets(sidecar_value)
    embedded_assets = document.get("extras", {}).get("assetRegistry")
    if not isinstance(embedded_assets, list):
        raise TypeError("Source GLB has no embedded asset registry")
    sidecar_ids = [str(asset["id"]) for asset in sidecar_assets]
    embedded_ids = [str(asset["id"]) for asset in embedded_assets]
    if sidecar_ids != embedded_ids:
        raise ValueError("Sidecar registry does not match the embedded GLB registry")

    updated_assets, assets_by_observation = bind_recovery_registry(
        sidecar_assets, recovery, recovery_sha256
    )
    document.setdefault("extras", {})["assetRegistry"] = updated_assets
    document["extras"]["targetedRecoveryBinding"] = {
        "profile": profile,
        "geometryChanged": False,
        "recoveryReportSha256": recovery_sha256,
        "recoveredGraphSha256": sha256_file(recovered_graph_path),
        "observationAssets": assets_by_observation,
    }
    _write_glb(document, remainder, output_glb)
    output_value: Any = updated_assets
    if isinstance(sidecar_value, dict):
        output_value = deepcopy(sidecar_value)
        output_value["assets"] = updated_assets
        output_value["targetedRecoveryBinding"] = document["extras"][
            "targetedRecoveryBinding"
        ]
    write_json(output_registry, output_value)

    output_document, output_remainder, output_bytes = _read_glb(output_glb)
    if output_remainder != remainder:
        raise RuntimeError("GLB binary geometry chunk changed during evidence binding")
    if output_document.get("extras", {}).get("assetRegistry") != updated_assets:
        raise RuntimeError("Updated registry was not embedded in the output GLB")
    report = {
        "schema_version": "railway.targeted-recovery-scene-binding.v1",
        "status": "pass",
        "profile": profile,
        "geometry_changed": False,
        "source_glb": str(source_glb),
        "source_glb_sha256": _sha256_bytes(source_bytes),
        "output_glb": str(output_glb),
        "output_glb_sha256": _sha256_bytes(output_bytes),
        "binary_geometry_chunk_sha256_before": _sha256_bytes(remainder),
        "binary_geometry_chunk_sha256_after": _sha256_bytes(output_remainder),
        "registry_asset_count": len(updated_assets),
        "recovery_report": str(recovery_path),
        "recovery_report_sha256": recovery_sha256,
        "recovered_graph": str(recovered_graph_path),
        "recovered_graph_sha256": sha256_file(recovered_graph_path),
        "observation_assets": assets_by_observation,
        "limitations": [
            "This revision binds fixed-center point support; it does not move geometry.",
            "Owner override remains distinct from independent review.",
        ],
    }
    write_json(report_path, report)
    return report_path
