from __future__ import annotations

import copy
import re
from typing import Any

FORMAL_STATUSES = {
    "accepted",
    "accepted_candidate",
    "approved",
    "final",
    "released",
}
FORMAL_EVIDENCE_STATUSES = {
    "accepted",
    "approved",
    "final",
    "released",
    "supported",
}
DESTRUCTIVE_KEYS = {
    "asset_deletions",
    "delete_assets",
    "relation_deletions",
    "delete_relations",
}


def _identifier(value: dict[str, Any], *, label: str) -> str:
    identifier = str(value.get("id") or value.get("asset_id") or "")
    if not identifier:
        raise ValueError(f"{label} requires id or asset_id")
    return identifier


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"Invalid dotted update path: {dotted!r}")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot apply dotted update through non-object key {part!r}")
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _merge_mapping(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _assert_candidate_fields(value: dict[str, Any], *, context: str) -> None:
    status = str(value.get("status") or "").strip().lower()
    evidence_status = str(value.get("evidence_status") or "").strip().lower()
    status_tokens = set(filter(None, re.split(r"[^a-z0-9]+", status)))
    if status in FORMAL_STATUSES or status_tokens & {"accepted", "approved", "final", "released"}:
        raise ValueError(f"Candidate patch promotes status in {context}: {status}")
    if evidence_status in FORMAL_EVIDENCE_STATUSES:
        raise ValueError(
            f"Candidate patch promotes evidence_status in {context}: {evidence_status}"
        )


def _assert_candidate_tree(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        _assert_candidate_fields(value, context=context)
        if "formal_release" in value and value["formal_release"] is not False:
            raise ValueError(f"Candidate patch declares formal release in {context}")
        destructive = sorted(key for key in DESTRUCTIVE_KEYS if value.get(key))
        if destructive:
            raise ValueError(
                f"Candidate patch contains destructive keys in {context}: "
                + ", ".join(destructive)
            )
        for key, child in value.items():
            _assert_candidate_tree(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_candidate_tree(child, context=f"{context}[{index}]")


def assert_candidate_patch(patch: dict[str, Any]) -> None:
    """Reject deletion or acceptance semantics before a candidate patch is applied."""

    if patch.get("formal_release") is not False:
        raise ValueError("Candidate patch must declare formal_release=false")
    _assert_candidate_tree(patch, context="patch root")
    for collection in ("asset_updates", "asset_additions", "new_assets", "relations"):
        rows = patch.get(collection, [])
        if not isinstance(rows, list):
            raise TypeError(f"{collection} must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"{collection}[{index}] must be an object")
            if collection == "asset_updates":
                updates = row.get("set", {})
                if not isinstance(updates, dict):
                    raise TypeError(f"asset_updates[{index}].set must be an object")


def _relation_endpoints(relation: dict[str, Any]) -> tuple[str, str]:
    source = str(relation.get("source") or relation.get("from") or "")
    target = str(relation.get("target") or relation.get("to") or "")
    if not source or not target:
        raise ValueError(f"Relation requires source/from and target/to: {relation.get('id')}")
    if relation.get("source") and relation.get("from") not in {None, source}:
        raise ValueError(f"Relation source/from disagree: {relation.get('id')}")
    if relation.get("target") and relation.get("to") not in {None, target}:
        raise ValueError(f"Relation target/to disagree: {relation.get('id')}")
    return source, target


def apply_candidate_registry_patch(
    registry: dict[str, Any],
    patch: dict[str, Any],
    *,
    expected_target_release_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply an additive candidate patch without deleting or promoting assets.

    Reapplying an identical addition is a no-op. A colliding but different asset
    or relation fails instead of silently replacing the existing record.
    """

    assert_candidate_patch(patch)
    target_release = str(patch.get("target_release_id") or patch.get("base_release_id") or "")
    expected = expected_target_release_id or str(registry.get("release_id") or "")
    if expected and target_release != expected:
        raise ValueError(
            f"Candidate patch target release mismatch: {target_release!r} != {expected!r}"
        )

    result = copy.deepcopy(registry)
    assets = result.setdefault("assets", [])
    relations = result.setdefault("relations", [])
    if not isinstance(assets, list) or not isinstance(relations, list):
        raise TypeError("Registry assets and relations must be lists")
    assets_by_id = {_identifier(asset, label="registry asset"): asset for asset in assets}
    if len(assets_by_id) != len(assets):
        raise ValueError("Registry contains duplicate asset IDs")
    relations_by_id = {
        _identifier(relation, label="registry relation"): relation for relation in relations
    }
    if len(relations_by_id) != len(relations):
        raise ValueError("Registry contains duplicate relation IDs")

    updated_ids: list[str] = []
    added_ids: list[str] = []
    existing_asset_noops: list[str] = []
    for update in patch.get("asset_updates", []):
        asset_id = _identifier(update, label="asset update")
        asset = assets_by_id.get(asset_id)
        if asset is None:
            raise KeyError(f"Candidate patch update target is absent: {asset_id}")
        expected_node = update.get("match_geometry_node")
        if expected_node is not None and asset.get("geometry", {}).get("node") != expected_node:
            raise ValueError(f"Geometry-node precondition failed for {asset_id}")
        before = copy.deepcopy(asset)
        for key, value in update.get("set", {}).items():
            _set_dotted(asset, str(key), value)
        parameters = update.get("parameters_merge", {})
        if not isinstance(parameters, dict):
            raise TypeError("parameters_merge must be an object")
        _merge_mapping(asset.setdefault("parameters", {}), parameters)
        limitations = asset.setdefault("limitations", [])
        if not isinstance(limitations, list):
            raise TypeError(f"Asset limitations must be a list: {asset_id}")
        for limitation in update.get("limitations_append_unique", []):
            if limitation not in limitations:
                limitations.append(copy.deepcopy(limitation))
        # Validate the resulting root state without re-litigating historic nested
        # evidence already present in the base registry. All incoming nested
        # values were recursively checked above as part of the patch itself.
        _assert_candidate_fields(asset, context=f"updated asset {asset_id}")
        if asset != before:
            updated_ids.append(asset_id)

    additions = [*patch.get("asset_additions", []), *patch.get("new_assets", [])]
    for addition in additions:
        asset_id = _identifier(addition, label="asset addition")
        existing = assets_by_id.get(asset_id)
        if existing is not None:
            if existing != addition:
                raise ValueError(f"Candidate asset addition collides with existing asset: {asset_id}")
            existing_asset_noops.append(asset_id)
            continue
        copied = copy.deepcopy(addition)
        _assert_candidate_tree(copied, context=f"added asset {asset_id}")
        assets.append(copied)
        assets_by_id[asset_id] = copied
        added_ids.append(asset_id)

    added_relation_ids: list[str] = []
    existing_relation_noops: list[str] = []
    for relation in patch.get("relations", []):
        relation_id = _identifier(relation, label="relation addition")
        source, target = _relation_endpoints(relation)
        if source not in assets_by_id or target not in assets_by_id:
            raise ValueError(f"Candidate relation has unknown endpoint: {relation_id}")
        existing = relations_by_id.get(relation_id)
        if existing is not None:
            if existing != relation:
                raise ValueError(
                    f"Candidate relation addition collides with existing relation: {relation_id}"
                )
            existing_relation_noops.append(relation_id)
            continue
        copied = copy.deepcopy(relation)
        _assert_candidate_tree(copied, context=f"added relation {relation_id}")
        relations.append(copied)
        relations_by_id[relation_id] = copied
        added_relation_ids.append(relation_id)

    report = {
        "schema_version": "railway.candidate-registry-patch-application.v1",
        "target_release_id": target_release,
        "formal_release": False,
        "input_asset_count": len(registry.get("assets", [])),
        "output_asset_count": len(assets),
        "input_relation_count": len(registry.get("relations", [])),
        "output_relation_count": len(relations),
        "updated_asset_ids": updated_ids,
        "added_asset_ids": added_ids,
        "existing_asset_noop_ids": existing_asset_noops,
        "added_relation_ids": added_relation_ids,
        "existing_relation_noop_ids": existing_relation_noops,
        "destructive_change_count": 0,
        "status": "candidate_patch_applied_not_formal_release",
    }
    return result, report
