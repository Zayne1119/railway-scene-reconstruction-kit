"""Fail-closed lineage and hash audit for non-deliverable release candidates.

The audit deliberately treats ``formal_release=false`` and
``delivery_allowed=false`` as the expected candidate state.  It verifies that
the candidate is internally self-consistent without promoting or packaging it.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file


@dataclass(frozen=True)
class CandidateReleasePolicy:
    """Project-independent contract for one candidate release closure."""

    required_manifest_roles: tuple[str, ...] = (
        "compact_model",
        "registry",
        "hq_assetized_glb",
        "web_assetized_glb",
        "fixed_views",
        "delivery_gate",
    )
    required_gate_roles: tuple[str, ...] = ("model", "registry", "fixed_views")
    embedded_registry_glb_roles: tuple[str, ...] = (
        "hq_assetized_glb",
        "web_assetized_glb",
    )
    required_stage_ids: tuple[str, ...] = ()
    expected_glb_release_status: str = "candidate_not_formal_release"
    require_parent_artifact_hashes: bool = False
    require_parent_state_preservation: bool = False
    allowed_parent_asset_state_change_ids: tuple[str, ...] = ()
    allowed_parent_relation_state_change_ids: tuple[str, ...] = ()


_MANIFEST_TO_GATE_ROLE = {
    "compact_model": "model",
    "registry": "registry",
    "pointcloud": "pointcloud",
    "heatmap": "heatmap",
    "fixed_views": "fixed_views",
    "web_status": "web",
}
_PROMOTION_TOKENS = {"accepted", "approved", "final", "formal", "released"}
_PROMOTED_EXACT_STATES = _PROMOTION_TOKENS | {"supported"}
_FAILED_STAGE_TOKENS = {"abort", "aborted", "blocked", "error", "fail", "failed", "invalid"}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _status_is_promoted(value: Any, *, evidence: bool = False) -> bool:
    """Reject formal state, while allowing provenance text such as point-supported."""

    if not _nonempty_text(value):
        return False
    normalized = value.strip().lower()
    if evidence and normalized in _PROMOTED_EXACT_STATES:
        return True
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", normalized)))
    return bool(tokens & _PROMOTION_TOKENS)


def _stage_status_failed(value: Any) -> bool:
    if not _nonempty_text(value):
        return True
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", value.strip().lower())))
    return bool(tokens & _FAILED_STAGE_TOKENS)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_in_release(value: Any, root: Path) -> Path | None:
    if not _nonempty_text(value):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_glb_document(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("not a GLB 2 container")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length != len(data):
        raise ValueError("invalid GLB header or byte length")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A or 20 + json_length > len(data):
        raise ValueError("missing or truncated GLB JSON chunk")
    value = json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))
    if not isinstance(value, dict):
        raise TypeError("GLB JSON chunk is not an object")
    return value


def _read_json(path: Path, fail: Any, *, code: str) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        fail(code, path=str(path), error=str(error))
        return None


def _artifact_map(
    records: Any,
    *,
    owner: str,
    release_id: Any,
    root: Path,
    fail: Any,
) -> dict[str, tuple[dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    if not isinstance(records, list):
        fail(f"{owner}_artifacts_not_list")
        return result
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            fail(f"{owner}_artifact_not_object", index=index)
            continue
        role = record.get("role")
        if not _nonempty_text(role):
            fail(f"{owner}_artifact_role_missing", index=index)
            continue
        if role in result:
            fail(f"{owner}_artifact_role_duplicate", role=role)
            continue
        if record.get("release_id") != release_id:
            fail(
                f"{owner}_artifact_release_id_mismatch",
                role=role,
                actual=record.get("release_id"),
                expected=release_id,
            )
        path = _resolve_in_release(record.get("path"), root)
        if path is None:
            fail(f"{owner}_artifact_path_missing", role=role)
            continue
        result[role] = (record, path)
        if not _inside(path, root):
            fail(f"{owner}_artifact_outside_release", role=role, path=str(path))
            continue
        if not path.is_file():
            fail(f"{owner}_artifact_missing", role=role, path=str(path))
            continue
        expected_hash = record.get("sha256")
        actual_hash = sha256_file(path)
        if not isinstance(expected_hash, str) or _HEX_SHA256.fullmatch(expected_hash) is None:
            fail(f"{owner}_artifact_sha256_invalid", role=role, actual=expected_hash)
        elif expected_hash != actual_hash:
            fail(
                f"{owner}_artifact_sha256_mismatch",
                role=role,
                expected=expected_hash,
                actual=actual_hash,
            )
        expected_size = record.get("size_bytes")
        if expected_size is not None and expected_size != path.stat().st_size:
            fail(
                f"{owner}_artifact_size_mismatch",
                role=role,
                expected=expected_size,
                actual=path.stat().st_size,
            )
    return result


def _audit_fixed_views(
    path: Path,
    *,
    release_id: str,
    model_path: Path,
    registry_path: Path,
    fail: Any,
) -> dict[str, Any]:
    manifest = _read_json(path, fail, code="fixed_views_invalid_json")
    if manifest is None:
        return {"view_count": 0, "group_counts": {}}
    if manifest.get("release_id") != release_id:
        fail("fixed_views_release_id_mismatch", actual=manifest.get("release_id"))
    if manifest.get("formal_release") is not False:
        fail("fixed_views_formal_release_not_false")
    bindings = (
        ("source_obj", "source_obj_sha256", model_path),
        ("source_registry", "source_registry_sha256", registry_path),
    )
    for path_key, hash_key, expected_path in bindings:
        bound_path = _resolve_in_release(manifest.get(path_key), path.parent)
        if bound_path != expected_path.resolve():
            fail(
                "fixed_views_source_path_mismatch",
                field=path_key,
                actual=str(bound_path) if bound_path else None,
                expected=str(expected_path.resolve()),
            )
        if manifest.get(hash_key) != sha256_file(expected_path):
            fail(
                "fixed_views_source_sha256_mismatch",
                field=hash_key,
                actual=manifest.get(hash_key),
                expected=sha256_file(expected_path),
            )
    groups: dict[str, int] = {}
    rows: list[tuple[str, Any]] = []
    for key, value in manifest.items():
        if not isinstance(value, list):
            continue
        if not any(
            isinstance(item, Mapping) and {"id", "image", "image_sha256"} & set(item)
            for item in value
        ):
            continue
        groups[key] = len(value)
        rows.extend((key, item) for item in value)
    if not rows:
        fail("fixed_views_empty")
    ids: list[str] = []
    images: list[str] = []
    for group, item in rows:
        if not isinstance(item, dict):
            fail("fixed_view_not_object", group=group)
            continue
        view_id = item.get("id")
        image_value = item.get("image")
        expected_hash = item.get("image_sha256")
        if not _nonempty_text(view_id):
            fail("fixed_view_id_missing", group=group)
        else:
            ids.append(view_id)
        if not _nonempty_text(image_value):
            fail("fixed_view_image_missing", group=group, view_id=view_id)
            continue
        images.append(image_value)
        image_path = (path.parent / image_value).resolve()
        if not _inside(image_path, path.parent.resolve()):
            fail("fixed_view_image_outside_manifest_directory", view_id=view_id)
        elif not image_path.is_file():
            fail("fixed_view_image_file_missing", view_id=view_id, path=str(image_path))
        elif expected_hash != sha256_file(image_path):
            fail(
                "fixed_view_image_sha256_mismatch",
                view_id=view_id,
                expected=expected_hash,
                actual=sha256_file(image_path),
            )
    if len(set(ids)) != len(ids):
        fail("fixed_view_id_duplicate")
    if len(set(images)) != len(images):
        fail("fixed_view_image_duplicate")
    return {"view_count": len(rows), "group_counts": groups}


def _audit_registry(
    path: Path,
    *,
    release_id: str,
    model_path: Path,
    expected_parent_release_id: str | None,
    fail: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    registry = _read_json(path, fail, code="registry_invalid_json")
    if registry is None:
        return {}, {}
    if registry.get("release_id") != release_id:
        fail("registry_release_id_mismatch", actual=registry.get("release_id"))
    if registry.get("formal_release") is not False:
        fail("registry_formal_release_not_false")
    if registry.get("model_sha256") != sha256_file(model_path):
        fail(
            "registry_model_sha256_mismatch",
            actual=registry.get("model_sha256"),
            expected=sha256_file(model_path),
        )
    policy = registry.get("candidate_state_policy")
    if not isinstance(policy, dict):
        fail("registry_candidate_state_policy_missing")
    else:
        if policy.get("formal_acceptance") is not False:
            fail("registry_formal_acceptance_not_false")
        if policy.get("delivery_allowed") is not False:
            fail("registry_delivery_allowed_not_false")
        if (
            expected_parent_release_id is not None
            and policy.get("source_release_id") != expected_parent_release_id
        ):
            fail(
                "registry_parent_release_id_mismatch",
                actual=policy.get("source_release_id"),
                expected=expected_parent_release_id,
            )
    assets = registry.get("assets")
    if not isinstance(assets, list) or not assets:
        fail("registry_assets_missing_or_empty")
        assets = []
    assets_by_id: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            fail("registry_asset_not_object", index=index)
            continue
        asset_id = asset.get("id")
        if not _nonempty_text(asset_id):
            fail("registry_asset_id_missing", index=index)
            continue
        if asset_id in assets_by_id:
            fail("registry_asset_id_duplicate", asset_id=asset_id)
        assets_by_id[asset_id] = asset
        if _status_is_promoted(asset.get("status")):
            fail("registry_asset_status_promoted", asset_id=asset_id, value=asset.get("status"))
        if _status_is_promoted(asset.get("evidence_status"), evidence=True):
            fail(
                "registry_asset_evidence_status_promoted",
                asset_id=asset_id,
                value=asset.get("evidence_status"),
            )
        for field in ("formal_release", "delivery_allowed"):
            if field in asset and asset.get(field) is not False:
                fail(
                    "registry_asset_candidate_boolean_not_false",
                    asset_id=asset_id,
                    field=field,
                    actual=asset.get(field),
                )
    relations = registry.get("relations")
    if not isinstance(relations, list):
        fail("registry_relations_not_list")
        relations = []
    relation_ids: set[str] = set()
    unbound_endpoints: list[dict[str, str]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            fail("registry_relation_not_object", index=index)
            continue
        relation_id = relation.get("id")
        if not _nonempty_text(relation_id):
            fail("registry_relation_id_missing", index=index)
            continue
        if relation_id in relation_ids:
            fail("registry_relation_id_duplicate", relation_id=relation_id)
        relation_ids.add(relation_id)
        if _status_is_promoted(relation.get("status")):
            fail(
                "registry_relation_status_promoted",
                relation_id=relation_id,
                value=relation.get("status"),
            )
        if _status_is_promoted(relation.get("evidence_status"), evidence=True):
            fail(
                "registry_relation_evidence_status_promoted",
                relation_id=relation_id,
                value=relation.get("evidence_status"),
            )
        for field in ("formal_release", "delivery_allowed"):
            if field in relation and relation.get(field) is not False:
                fail(
                    "registry_relation_candidate_boolean_not_false",
                    relation_id=relation_id,
                    field=field,
                    actual=relation.get(field),
                )
        for endpoint in ("from", "to"):
            endpoint_id = relation.get(endpoint)
            if _nonempty_text(endpoint_id) and endpoint_id not in assets_by_id:
                unbound_endpoints.append(
                    {"relation_id": relation_id, "endpoint": endpoint, "asset_id": endpoint_id}
                )
    return assets_by_id, {
        "asset_count": len(assets_by_id),
        "relation_count": len(relation_ids),
        # Historical registries may reference stable external IDs.  Surface this
        # debt without treating it as evidence of promotion or broken hashes.
        "unbound_relation_endpoint_count": len(unbound_endpoints),
        "unbound_relation_endpoints": unbound_endpoints,
    }


def _audit_embedded_registry_glb(
    path: Path,
    *,
    role: str,
    release_id: str,
    registry_assets: Mapping[str, dict[str, Any]],
    expected_release_status: str,
    fail: Any,
) -> dict[str, Any]:
    try:
        document = _load_glb_document(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        fail("glb_invalid", role=role, path=str(path), error=str(error))
        return {"role": role, "asset_count": 0, "mesh_count": 0}
    extras = document.get("extras")
    if not isinstance(extras, dict):
        fail("glb_extras_missing", role=role)
        extras = {}
    if extras.get("sourceRelease") != release_id:
        fail(
            "glb_release_id_mismatch",
            role=role,
            actual=extras.get("sourceRelease"),
            expected=release_id,
        )
    if extras.get("releaseStatus") != expected_release_status:
        fail(
            "glb_release_status_mismatch",
            role=role,
            actual=extras.get("releaseStatus"),
            expected=expected_release_status,
        )
    embedded = extras.get("assetRegistry")
    if not isinstance(embedded, list):
        fail("glb_asset_registry_missing", role=role)
        embedded = []
    if extras.get("assetCount") != len(registry_assets):
        fail(
            "glb_declared_asset_count_mismatch",
            role=role,
            actual=extras.get("assetCount"),
            expected=len(registry_assets),
        )
    embedded_by_id: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(embedded):
        if not isinstance(asset, dict) or not _nonempty_text(asset.get("id")):
            fail("glb_asset_invalid", role=role, index=index)
            continue
        asset_id = asset["id"]
        if asset_id in embedded_by_id:
            fail("glb_asset_id_duplicate", role=role, asset_id=asset_id)
        embedded_by_id[asset_id] = asset
    if set(embedded_by_id) != set(registry_assets):
        fail(
            "glb_asset_id_set_mismatch",
            role=role,
            missing=sorted(set(registry_assets) - set(embedded_by_id)),
            unknown=sorted(set(embedded_by_id) - set(registry_assets)),
        )
    for asset_id in sorted(set(embedded_by_id) & set(registry_assets)):
        embedded_asset = embedded_by_id[asset_id]
        source_asset = registry_assets[asset_id]
        parameters = embedded_asset.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        for field in ("status", "evidence_status"):
            if parameters.get(field) != source_asset.get(field):
                fail(
                    "glb_asset_state_mismatch",
                    role=role,
                    asset_id=asset_id,
                    field=field,
                    actual=parameters.get(field),
                    expected=source_asset.get(field),
                )
        if embedded_asset.get("type") != source_asset.get("type"):
            fail(
                "glb_asset_type_mismatch",
                role=role,
                asset_id=asset_id,
                actual=embedded_asset.get("type"),
                expected=source_asset.get("type"),
            )
    mesh_count = len(document.get("meshes", []))
    if mesh_count != len(registry_assets):
        fail(
            "glb_mesh_count_mismatch",
            role=role,
            actual=mesh_count,
            expected=len(registry_assets),
        )
    return {"role": role, "asset_count": len(embedded_by_id), "mesh_count": mesh_count}


def audit_candidate_release(
    release_directory: str | Path,
    manifest_path: str | Path,
    delivery_gate_path: str | Path,
    *,
    lineage_path: str | Path | None = None,
    expected_release_id: str | None = None,
    expected_parent_release_id: str | None = None,
    policy: CandidateReleasePolicy | None = None,
) -> dict[str, Any]:
    """Validate candidate lineage, immutable artifacts, stages, views, and GLBs."""

    contract = policy or CandidateReleasePolicy()
    root = Path(release_directory).resolve()
    manifest_file = Path(manifest_path).resolve()
    gate_file = Path(delivery_gate_path).resolve()
    failures: list[dict[str, Any]] = []

    def fail(code: str, **details: Any) -> None:
        failures.append({"code": code, **details})

    if not root.is_dir():
        fail("release_directory_missing", path=str(root))
    for role, path in (("manifest", manifest_file), ("delivery_gate", gate_file)):
        if not _inside(path, root):
            fail(f"{role}_outside_release", path=str(path))
        if not path.is_file():
            fail(f"{role}_missing", path=str(path))
    manifest = (
        _read_json(manifest_file, fail, code="manifest_invalid_json")
        if manifest_file.is_file()
        else None
    )
    gate = _read_json(gate_file, fail, code="delivery_gate_invalid_json") if gate_file.is_file() else None
    release_id = manifest.get("release_id") if manifest else None
    if not _nonempty_text(release_id):
        fail("manifest_release_id_missing")
        release_id = expected_release_id or ""
    if expected_release_id is not None and release_id != expected_release_id:
        fail("manifest_release_id_mismatch", actual=release_id, expected=expected_release_id)

    manifest_artifacts: dict[str, tuple[dict[str, Any], Path]] = {}
    gate_artifacts: dict[str, tuple[dict[str, Any], Path]] = {}
    if manifest is not None:
        for field in ("formal_release", "delivery_allowed", "package_generated", "published"):
            if manifest.get(field) is not False:
                fail("manifest_candidate_boolean_not_false", field=field, actual=manifest.get(field))
        if "candidate" not in str(manifest.get("release_class", "")).lower():
            fail("manifest_release_class_not_candidate", actual=manifest.get("release_class"))
        if manifest.get("status_promotion_count", 0) != 0:
            fail(
                "manifest_status_promotion_count_not_zero",
                actual=manifest.get("status_promotion_count"),
            )
        manifest_artifacts = _artifact_map(
            manifest.get("artifacts"),
            owner="manifest",
            release_id=release_id,
            root=root,
            fail=fail,
        )
        raw_manifest_artifacts = manifest.get("artifacts")
        actual_artifact_count = (
            len(raw_manifest_artifacts) if isinstance(raw_manifest_artifacts, list) else 0
        )
        if manifest.get("artifact_count") != actual_artifact_count:
            fail(
                "manifest_artifact_count_mismatch",
                declared=manifest.get("artifact_count"),
                actual=actual_artifact_count,
            )
        missing = sorted(set(contract.required_manifest_roles) - set(manifest_artifacts))
        if missing:
            fail("manifest_required_artifact_roles_missing", roles=missing)

    if gate is not None:
        if gate.get("release_id") != release_id:
            fail("delivery_gate_release_id_mismatch", actual=gate.get("release_id"), expected=release_id)
        if gate.get("formal_release") is not False:
            fail("delivery_gate_formal_release_not_false")
        if gate.get("delivery_allowed") is not False:
            fail("delivery_gate_delivery_allowed_not_false")
        if not _nonempty_text(gate.get("status")) or "blocked" not in gate["status"].lower():
            fail("delivery_gate_status_not_blocked", actual=gate.get("status"))
        reasons = gate.get("blocking_reasons")
        if not isinstance(reasons, list) or not reasons or any(not _nonempty_text(x) for x in reasons):
            fail("delivery_gate_blocking_reasons_missing")
        gate_artifacts = _artifact_map(
            gate.get("artifacts"), owner="gate", release_id=release_id, root=root, fail=fail
        )
        missing = sorted(set(contract.required_gate_roles) - set(gate_artifacts))
        if missing:
            fail("gate_required_artifact_roles_missing", roles=missing)
        stages = gate.get("critical_stages")
        if not isinstance(stages, list) or not stages:
            fail("critical_stages_missing")
            stages = []
        stage_ids: set[str] = set()
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                fail("critical_stage_not_object", index=index)
                continue
            stage_id = stage.get("id")
            if not _nonempty_text(stage_id):
                fail("critical_stage_id_missing", index=index)
                continue
            if stage_id in stage_ids:
                fail("critical_stage_id_duplicate", stage_id=stage_id)
            stage_ids.add(stage_id)
            if stage.get("critical") is not True:
                fail("critical_stage_not_critical", stage_id=stage_id)
            if stage.get("release_id") != release_id:
                fail("critical_stage_release_id_mismatch", stage_id=stage_id)
            if _stage_status_failed(stage.get("status")):
                fail("critical_stage_failed", stage_id=stage_id, status=stage.get("status"))
            report_path = _resolve_in_release(stage.get("report"), root)
            if report_path is None or not _inside(report_path, root):
                fail("critical_stage_report_outside_or_missing_path", stage_id=stage_id)
                continue
            if not report_path.is_file():
                fail("critical_stage_report_missing", stage_id=stage_id, path=str(report_path))
                continue
            actual_hash = sha256_file(report_path)
            if stage.get("sha256") != actual_hash:
                fail(
                    "critical_stage_report_sha256_mismatch",
                    stage_id=stage_id,
                    actual=actual_hash,
                    expected=stage.get("sha256"),
                )
            report = _read_json(report_path, fail, code="critical_stage_report_invalid_json")
            if report is None:
                continue
            if report.get("release_id") != release_id:
                fail("critical_stage_report_release_id_mismatch", stage_id=stage_id)
            if report.get("stage_id") != stage_id:
                fail("critical_stage_report_stage_id_mismatch", stage_id=stage_id)
            if report.get("critical") is not True:
                fail("critical_stage_report_not_critical", stage_id=stage_id)
            if report.get("formal_release") is not False:
                fail("critical_stage_report_formal_release_not_false", stage_id=stage_id)
            if report.get("status") != stage.get("status"):
                fail("critical_stage_status_mismatch", stage_id=stage_id)
            if _stage_status_failed(report.get("status")):
                fail(
                    "critical_stage_report_failed",
                    stage_id=stage_id,
                    status=report.get("status"),
                )
            source_path = _resolve_in_release(report.get("source_report"), root)
            if source_path is None or not _inside(source_path, root) or not source_path.is_file():
                fail("critical_stage_source_report_missing_or_outside", stage_id=stage_id)
            elif report.get("source_report_sha256") != sha256_file(source_path):
                fail("critical_stage_source_report_sha256_mismatch", stage_id=stage_id)
        missing_stages = sorted(set(contract.required_stage_ids) - stage_ids)
        if missing_stages:
            fail("required_critical_stages_missing", stage_ids=missing_stages)

    for manifest_role, gate_role in _MANIFEST_TO_GATE_ROLE.items():
        if manifest_role not in manifest_artifacts or gate_role not in gate_artifacts:
            continue
        manifest_record = manifest_artifacts[manifest_role][0]
        gate_record = gate_artifacts[gate_role][0]
        if (
            manifest_record.get("sha256") != gate_record.get("sha256")
            or manifest_artifacts[manifest_role][1] != gate_artifacts[gate_role][1]
        ):
            fail(
                "manifest_gate_artifact_binding_mismatch",
                manifest_role=manifest_role,
                gate_role=gate_role,
            )

    delivery_gate_entry = manifest_artifacts.get("delivery_gate")
    if delivery_gate_entry and delivery_gate_entry[1] != gate_file:
        fail(
            "manifest_delivery_gate_path_mismatch",
            actual=str(delivery_gate_entry[1]),
            expected=str(gate_file),
        )

    model_entry = manifest_artifacts.get("compact_model")
    registry_entry = manifest_artifacts.get("registry")
    fixed_entry = manifest_artifacts.get("fixed_views")
    registry_assets: dict[str, dict[str, Any]] = {}
    registry_summary: dict[str, Any] = {}
    fixed_summary: dict[str, Any] = {}
    if model_entry and registry_entry and model_entry[1].is_file() and registry_entry[1].is_file():
        registry_assets, registry_summary = _audit_registry(
            registry_entry[1],
            release_id=release_id,
            model_path=model_entry[1],
            expected_parent_release_id=expected_parent_release_id,
            fail=fail,
        )
        if (
            manifest is not None
            and manifest.get("asset_count") is not None
            and manifest.get("asset_count") != len(registry_assets)
        ):
            fail(
                "manifest_registry_asset_count_mismatch",
                actual=manifest.get("asset_count"),
                expected=len(registry_assets),
            )
        if (
            manifest is not None
            and manifest.get("relation_count") is not None
            and manifest.get("relation_count") != registry_summary.get("relation_count")
        ):
            fail(
                "manifest_registry_relation_count_mismatch",
                actual=manifest.get("relation_count"),
                expected=registry_summary.get("relation_count"),
            )
    if (
        fixed_entry
        and model_entry
        and registry_entry
        and fixed_entry[1].is_file()
        and model_entry[1].is_file()
        and registry_entry[1].is_file()
    ):
        fixed_summary = _audit_fixed_views(
            fixed_entry[1],
            release_id=release_id,
            model_path=model_entry[1],
            registry_path=registry_entry[1],
            fail=fail,
        )
        declared_fixed_counts = manifest.get("fixed_view_counts") if manifest else None
        if (
            isinstance(declared_fixed_counts, dict)
            and declared_fixed_counts != fixed_summary.get("group_counts")
        ):
            fail(
                "manifest_fixed_view_counts_mismatch",
                actual=declared_fixed_counts,
                expected=fixed_summary.get("group_counts"),
            )
        if (
            manifest is not None
            and manifest.get("fixed_view_count") is not None
            and manifest.get("fixed_view_count") != fixed_summary.get("view_count")
        ):
            fail(
                "manifest_fixed_view_count_mismatch",
                actual=manifest.get("fixed_view_count"),
                expected=fixed_summary.get("view_count"),
            )
    glb_summaries: list[dict[str, Any]] = []
    if registry_assets:
        for role in contract.embedded_registry_glb_roles:
            entry = manifest_artifacts.get(role)
            if entry and entry[1].is_file():
                glb_summaries.append(
                    _audit_embedded_registry_glb(
                        entry[1],
                        role=role,
                        release_id=release_id,
                        registry_assets=registry_assets,
                        expected_release_status=contract.expected_glb_release_status,
                        fail=fail,
                    )
                )

    lineage_file: Path | None = None
    if lineage_path is not None:
        lineage_file = Path(lineage_path).resolve()
    elif "integration" in manifest_artifacts:
        lineage_file = manifest_artifacts["integration"][1]
    elif "lineage" in manifest_artifacts:
        lineage_file = manifest_artifacts["lineage"][1]
    lineage: dict[str, Any] | None = None
    if lineage_file is None:
        fail("lineage_report_missing")
    elif not _inside(lineage_file, root) or not lineage_file.is_file():
        fail("lineage_report_missing_or_outside", path=str(lineage_file))
    else:
        lineage = _read_json(lineage_file, fail, code="lineage_report_invalid_json")
        if lineage is not None:
            if lineage.get("release_id") != release_id:
                fail("lineage_release_id_mismatch", actual=lineage.get("release_id"))
            if lineage.get("formal_release") is not False:
                fail("lineage_formal_release_not_false")
            if lineage.get("delivery_allowed") is not False:
                fail("lineage_delivery_allowed_not_false")
            source_parent = lineage.get("source_release_id")
            explicit_parent = lineage.get("parent_release_id")
            if (
                _nonempty_text(source_parent)
                and _nonempty_text(explicit_parent)
                and source_parent != explicit_parent
            ):
                fail(
                    "lineage_parent_fields_disagree",
                    source_release_id=source_parent,
                    parent_release_id=explicit_parent,
                )
            actual_parent = explicit_parent if _nonempty_text(explicit_parent) else source_parent
            if expected_parent_release_id is not None and actual_parent != expected_parent_release_id:
                fail(
                    "lineage_parent_release_id_mismatch",
                    actual=actual_parent,
                    expected=expected_parent_release_id,
                )
    manifest_lineage_entry = manifest_artifacts.get("integration") or manifest_artifacts.get(
        "lineage"
    )
    if (
        lineage_file is not None
        and manifest_lineage_entry
        and lineage_file != manifest_lineage_entry[1]
    ):
        fail(
            "manifest_lineage_path_mismatch",
            actual=str(manifest_lineage_entry[1]),
            expected=str(lineage_file),
        )

    parent_registry_document: dict[str, Any] | None = None
    if contract.require_parent_artifact_hashes or contract.require_parent_state_preservation:
        if expected_parent_release_id is None:
            fail("parent_release_id_required_for_dependency_hashes")
        elif lineage is None:
            fail("lineage_required_for_parent_dependency_hashes")
        else:
            parent_root = (root.parent / expected_parent_release_id).resolve()
            parent_manifest_path = parent_root / "candidate_release_manifest.json"
            parent_manifest = (
                _read_json(
                    parent_manifest_path,
                    fail,
                    code="parent_release_manifest_invalid_json",
                )
                if parent_manifest_path.is_file()
                else None
            )
            if parent_manifest is None:
                fail("parent_release_manifest_missing", path=str(parent_manifest_path))
            else:
                if parent_manifest.get("release_id") != expected_parent_release_id:
                    fail("parent_release_manifest_release_id_mismatch")
                parent_records = parent_manifest.get("artifacts")
                parent_by_role = {
                    record.get("role"): record
                    for record in parent_records
                    if isinstance(record, dict) and _nonempty_text(record.get("role"))
                } if isinstance(parent_records, list) else {}
                for role, lineage_field in (
                    ("compact_model", "parent_model_sha256"),
                    ("registry", "parent_registry_sha256"),
                ):
                    record = parent_by_role.get(role)
                    if record is None:
                        fail("parent_artifact_role_missing", role=role)
                        continue
                    parent_path = _resolve_in_release(record.get("path"), parent_root)
                    if parent_path is None or not _inside(parent_path, parent_root):
                        fail("parent_artifact_path_invalid", role=role)
                        continue
                    if not parent_path.is_file():
                        fail("parent_artifact_missing", role=role, path=str(parent_path))
                        continue
                    actual_hash = sha256_file(parent_path)
                    if record.get("sha256") != actual_hash:
                        fail("parent_manifest_artifact_sha256_mismatch", role=role)
                    if contract.require_parent_artifact_hashes and lineage.get(
                        lineage_field
                    ) != actual_hash:
                        fail(
                            "lineage_parent_artifact_sha256_mismatch",
                            role=role,
                            field=lineage_field,
                            actual=lineage.get(lineage_field),
                            expected=actual_hash,
                        )
                    if role == "registry":
                        parent_registry_document = _read_json(
                            parent_path,
                            fail,
                            code="parent_registry_invalid_json",
                        )

    if contract.require_parent_state_preservation:
        if parent_registry_document is None or registry_entry is None:
            fail("parent_state_preservation_inputs_missing")
        else:
            current_registry = _read_json(
                registry_entry[1], fail, code="current_registry_invalid_for_parent_comparison"
            )
            if current_registry is not None:
                for collection, allowed_ids in (
                    ("assets", set(contract.allowed_parent_asset_state_change_ids)),
                    ("relations", set(contract.allowed_parent_relation_state_change_ids)),
                ):
                    parent_rows = parent_registry_document.get(collection)
                    current_rows = current_registry.get(collection)
                    if not isinstance(parent_rows, list) or not isinstance(current_rows, list):
                        fail("parent_state_collection_invalid", collection=collection)
                        continue
                    parent_by_id = {
                        item.get("id"): item
                        for item in parent_rows
                        if isinstance(item, dict) and _nonempty_text(item.get("id"))
                    }
                    current_by_id = {
                        item.get("id"): item
                        for item in current_rows
                        if isinstance(item, dict) and _nonempty_text(item.get("id"))
                    }
                    removed = sorted(set(parent_by_id) - set(current_by_id))
                    if removed:
                        fail("parent_registry_ids_removed", collection=collection, ids=removed)
                    for identifier in sorted(set(parent_by_id) & set(current_by_id)):
                        for field in ("status", "evidence_status"):
                            before = parent_by_id[identifier].get(field)
                            after = current_by_id[identifier].get(field)
                            if before != after and identifier not in allowed_ids:
                                fail(
                                    "parent_registry_state_changed_without_allowlist",
                                    collection=collection,
                                    identifier=identifier,
                                    field=field,
                                    parent=before,
                                    candidate=after,
                                )

    return {
        "schema_version": "railway.candidate-release-lineage-audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "release_directory": str(root),
        "release_id": release_id,
        "expected_release_id": expected_release_id,
        "expected_parent_release_id": expected_parent_release_id,
        "formal_release": False,
        "delivery_allowed": False,
        "inputs": {
            "manifest": {
                "path": str(manifest_file),
                "sha256": sha256_file(manifest_file) if manifest_file.is_file() else None,
            },
            "delivery_gate": {
                "path": str(gate_file),
                "sha256": sha256_file(gate_file) if gate_file.is_file() else None,
            },
            "lineage": {
                "path": str(lineage_file) if lineage_file is not None else None,
                "sha256": (
                    sha256_file(lineage_file)
                    if lineage_file is not None and lineage_file.is_file()
                    else None
                ),
            },
        },
        "policy": {
            "required_manifest_roles": list(contract.required_manifest_roles),
            "required_gate_roles": list(contract.required_gate_roles),
            "embedded_registry_glb_roles": list(contract.embedded_registry_glb_roles),
            "required_stage_ids": list(contract.required_stage_ids),
            "expected_glb_release_status": contract.expected_glb_release_status,
            "require_parent_artifact_hashes": contract.require_parent_artifact_hashes,
            "require_parent_state_preservation": contract.require_parent_state_preservation,
            "allowed_parent_asset_state_change_ids": list(
                contract.allowed_parent_asset_state_change_ids
            ),
            "allowed_parent_relation_state_change_ids": list(
                contract.allowed_parent_relation_state_change_ids
            ),
        },
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "manifest_artifact_count": len(manifest_artifacts),
        "gate_artifact_count": len(gate_artifacts),
        "registry": registry_summary,
        "fixed_views": fixed_summary,
        "embedded_glbs": glb_summaries,
        "decision": (
            "candidate_lineage_and_hash_closure_passed_not_formal_release"
            if not failures
            else "candidate_release_rejected_fail_closed"
        ),
    }


def write_candidate_release_audit(
    output_path: str | Path,
    release_directory: str | Path,
    manifest_path: str | Path,
    delivery_gate_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write an immutable audit report."""

    report = audit_candidate_release(
        release_directory, manifest_path, delivery_gate_path, **kwargs
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed hash and lineage audit for a non-deliverable candidate"
    )
    parser.add_argument("--release-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--delivery-gate", type=Path, required=True)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--expected-release-id")
    parser.add_argument("--expected-parent-release-id")
    parser.add_argument("--required-stage", action="append", default=[])
    parser.add_argument("--require-parent-artifact-hashes", action="store_true")
    parser.add_argument("--require-parent-state-preservation", action="store_true")
    parser.add_argument("--allow-parent-asset-state-change", action="append", default=[])
    parser.add_argument("--allow-parent-relation-state-change", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    policy = CandidateReleasePolicy(
        required_stage_ids=tuple(args.required_stage),
        require_parent_artifact_hashes=args.require_parent_artifact_hashes,
        require_parent_state_preservation=args.require_parent_state_preservation,
        allowed_parent_asset_state_change_ids=tuple(args.allow_parent_asset_state_change),
        allowed_parent_relation_state_change_ids=tuple(args.allow_parent_relation_state_change),
    )
    try:
        report = write_candidate_release_audit(
            args.output,
            args.release_directory,
            args.manifest,
            args.delivery_gate,
            lineage_path=args.lineage,
            expected_release_id=args.expected_release_id,
            expected_parent_release_id=args.expected_parent_release_id,
            policy=policy,
        )
    except FileExistsError:
        print(f"Refusing to overwrite candidate release audit: {args.output.resolve()}", file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
