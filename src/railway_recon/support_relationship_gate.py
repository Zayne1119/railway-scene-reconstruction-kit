"""Fail-closed validation for candidate bearing and support relationships.

Geometry proximity can identify objects worth inspecting, but it cannot identify
the semantic owner of a bearing interface.  This module keeps those two facts
separate and rejects candidate support claims unless the owner, contact, and
evidence provenance are explicit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SupportRelationshipPolicy:
    """Thresholds and vocabulary for a candidate support-relationship gate."""

    maximum_absolute_contact_residual_m: float = 0.05
    minimum_direct_evidence_kinds: int = 1
    require_modeled_owner: bool = True
    owner_endpoint_by_relation: tuple[tuple[str, str], ...] = (
        ("rests_on", "target"),
        ("supported_by", "target"),
        ("bears_on", "target"),
        ("inserted_into", "target"),
        ("mast_direct_burial", "target"),
        ("mast_inserted_into_foundation", "target"),
        ("supports", "source"),
    )
    forbidden_status_tokens: tuple[str, ...] = (
        "accepted",
        "approved",
        "final",
        "formal",
        "released",
    )
    allowed_direct_evidence_kinds: tuple[str, ...] = (
        "as_built_record",
        "design_record",
        "full_density_point_cloud",
        "manual_panorama_review",
        "measured_interface",
        "registered_topology",
        "survey_control",
    )
    allowed_owner_identification_methods: tuple[str, ...] = (
        "as_built_record",
        "design_record",
        "measured_interface_correspondence",
        "observed_boundary",
        "registered_topology",
        "surveyed_interface_boundary",
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_absolute_contact_residual_m):
            raise ValueError("maximum_absolute_contact_residual_m must be finite")
        if self.maximum_absolute_contact_residual_m < 0.0:
            raise ValueError("maximum_absolute_contact_residual_m cannot be negative")
        if self.minimum_direct_evidence_kinds <= 0:
            raise ValueError("minimum_direct_evidence_kinds must be positive")
        endpoints = dict(self.owner_endpoint_by_relation)
        if not endpoints or any(value not in {"source", "target"} for value in endpoints.values()):
            raise ValueError("owner_endpoint_by_relation must map types to source or target")
        if not self.allowed_direct_evidence_kinds:
            raise ValueError("allowed_direct_evidence_kinds cannot be empty")
        if not self.allowed_owner_identification_methods:
            raise ValueError("allowed_owner_identification_methods cannot be empty")


def _identifier(value: Mapping[str, Any], *names: str) -> str:
    return next((str(value.get(name) or "").strip() for name in names if value.get(name)), "")


def _status_tokens(value: object) -> set[str]:
    return set(filter(None, re.split(r"[^a-z0-9]+", str(value or "").strip().lower())))


def _has_forbidden_status(value: object, policy: SupportRelationshipPolicy) -> bool:
    return bool(_status_tokens(value) & set(policy.forbidden_status_tokens))


def _modeled_geometry_present(asset: Mapping[str, Any]) -> bool:
    geometry = asset.get("geometry")
    if not isinstance(geometry, Mapping):
        return False
    return bool(str(geometry.get("node") or "").strip()) and bool(
        str(geometry.get("file") or "").strip()
    )


def _is_proximity_label(value: object) -> bool:
    tokens = _status_tokens(value)
    return bool(tokens & {"aabb", "distance", "nearest", "overlap", "proximity"})


def evaluate_support_relationship_claim(
    claim: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    *,
    policy: SupportRelationshipPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate one proposed support relation without mutating the registry.

    ``support_evidence`` must declare a direct contact observation, a finite
    signed contact residual, a resolved owner interpretation, an owner
    identification method, and one or more direct evidence kinds.  A nearest
    object, AABB overlap, or small distance remains diagnostic only.
    """

    active_policy = policy or SupportRelationshipPolicy()
    endpoint_map = dict(active_policy.owner_endpoint_by_relation)
    relation_id = _identifier(claim, "id", "claim_id") or "<missing-id>"
    relation_type = str(claim.get("type") or "").strip()
    source = _identifier(claim, "source", "from")
    target = _identifier(claim, "target", "to")
    owner_endpoint = endpoint_map.get(relation_type)
    owner_id = source if owner_endpoint == "source" else target if owner_endpoint == "target" else ""
    subject_id = target if owner_endpoint == "source" else source
    evidence = claim.get("support_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    failures: list[dict[str, str]] = []

    def fail(code: str, message: str) -> None:
        failures.append({"code": code, "message": message})

    if relation_id == "<missing-id>":
        fail("missing_relation_id", "Support claim requires a stable id.")
    if owner_endpoint is None:
        fail("unsupported_relation_type", f"Unsupported support relation type: {relation_type!r}.")
    if not subject_id:
        fail("missing_subject", "Support claim does not identify the supported asset.")
    if not owner_id:
        fail("missing_owner", "Support claim does not identify a bearing owner.")

    subject = assets.get(subject_id)
    owner = assets.get(owner_id)
    if subject_id and subject is None:
        fail("subject_not_registered", f"Supported asset is not registered: {subject_id}.")
    if owner_id and owner is None:
        fail("owner_not_registered", f"Bearing owner is not registered: {owner_id}.")
    if owner is not None and active_policy.require_modeled_owner and not _modeled_geometry_present(owner):
        fail("owner_geometry_missing", f"Bearing owner has no modeled geometry: {owner_id}.")

    active_statuses = {
        "relation.status": claim.get("status"),
        "relation.evidence_status": claim.get("evidence_status"),
        "subject.status": subject.get("status") if subject is not None else None,
        "subject.evidence_status": (
            subject.get("evidence_status") if subject is not None else None
        ),
        "owner.status": owner.get("status") if owner is not None else None,
        "owner.evidence_status": owner.get("evidence_status") if owner is not None else None,
    }
    if not str(claim.get("status") or "").strip():
        fail("relation_status_missing", "Candidate support claim requires an explicit status.")
    if not str(claim.get("evidence_status") or "").strip():
        fail(
            "relation_evidence_status_missing",
            "Candidate support claim requires an explicit evidence_status.",
        )
    polluted = [
        name for name, value in active_statuses.items() if _has_forbidden_status(value, active_policy)
    ]
    if polluted:
        fail(
            "accepted_or_formal_status_pollution",
            "Candidate support claim contains accepted/formal state in " + ", ".join(polluted) + ".",
        )

    if not isinstance(claim.get("support_evidence"), Mapping):
        fail("support_evidence_missing", "Support claim requires a support_evidence object.")
    if evidence.get("owner_semantics_resolved") is not True:
        fail("owner_semantics_unresolved", "Bearing-owner semantics are not explicitly resolved.")
    if evidence.get("direct_contact_observed") is not True:
        fail("direct_contact_evidence_missing", "Direct contact was not explicitly observed.")
    method = str(evidence.get("owner_identification_method") or "").strip()
    if not method:
        fail("owner_identification_method_missing", "Owner identification method is absent.")
    elif method.lower() not in set(active_policy.allowed_owner_identification_methods):
        fail(
            "owner_identification_method_not_allowed",
            f"Owner identification method is not allowlisted: {method!r}.",
        )
    if evidence.get("proximity_only") is True or _is_proximity_label(method):
        fail(
            "proximity_only_evidence",
            "Nearest-distance, AABB, or overlap evidence cannot establish bearing ownership.",
        )

    residual = evidence.get("contact_residual_m")
    if isinstance(residual, bool) or not isinstance(residual, (int, float)) or not math.isfinite(residual):
        fail("contact_residual_missing", "A finite signed contact_residual_m is required.")
        residual_value = None
    else:
        residual_value = float(residual)
        if abs(residual_value) > active_policy.maximum_absolute_contact_residual_m:
            fail(
                "contact_residual_exceeds_tolerance",
                f"Absolute contact residual {abs(residual_value):.6f} m exceeds "
                f"{active_policy.maximum_absolute_contact_residual_m:.6f} m.",
            )

    kinds = evidence.get("evidence_kinds")
    kinds = list(kinds) if isinstance(kinds, Sequence) and not isinstance(kinds, str) else []
    normalized_kinds = sorted({str(kind).strip().lower() for kind in kinds if str(kind).strip()})
    allowed_kinds = set(active_policy.allowed_direct_evidence_kinds)
    unknown_kinds = sorted(set(normalized_kinds) - allowed_kinds)
    if unknown_kinds:
        fail(
            "direct_evidence_kind_not_allowed",
            "Direct evidence kinds are not allowlisted: " + ", ".join(unknown_kinds) + ".",
        )
    direct_kinds = sorted(set(normalized_kinds) & allowed_kinds)
    if len(direct_kinds) < active_policy.minimum_direct_evidence_kinds:
        fail(
            "direct_evidence_kinds_insufficient",
            f"Found {len(direct_kinds)} direct evidence kinds; "
            f"need {active_policy.minimum_direct_evidence_kinds}.",
        )

    relationship_allowed = not failures
    return {
        "schema_version": "railway.support-relationship-claim-gate.v1",
        "claim_id": relation_id,
        "relation_type": relation_type,
        "subject_id": subject_id or None,
        "owner_id": owner_id or None,
        "relationship_allowed": relationship_allowed,
        "formal_acceptance": False,
        "status": (
            "candidate_support_relationship_pass_pending_review"
            if relationship_allowed
            else "blocked_support_relationship"
        ),
        "failures": failures,
        "metrics": {
            "contact_residual_m": residual_value,
            "absolute_contact_residual_m": (
                abs(residual_value) if residual_value is not None else None
            ),
            "direct_evidence_kinds": direct_kinds,
            "owner_geometry_present": _modeled_geometry_present(owner) if owner is not None else False,
        },
        "thresholds": {
            "maximum_absolute_contact_residual_m": (
                active_policy.maximum_absolute_contact_residual_m
            ),
            "minimum_direct_evidence_kinds": active_policy.minimum_direct_evidence_kinds,
            "require_modeled_owner": active_policy.require_modeled_owner,
            "allowed_direct_evidence_kinds": list(
                active_policy.allowed_direct_evidence_kinds
            ),
            "allowed_owner_identification_methods": list(
                active_policy.allowed_owner_identification_methods
            ),
        },
    }


def validate_support_relationship_patch(
    registry: Mapping[str, Any],
    patch: Mapping[str, Any],
    *,
    policy: SupportRelationshipPolicy | None = None,
) -> dict[str, Any]:
    """Validate support relations in a candidate patch and fail closed on release state."""

    active_policy = policy or SupportRelationshipPolicy()
    asset_rows = registry.get("assets")
    relation_rows = patch.get("relations")
    if not isinstance(asset_rows, Sequence) or isinstance(asset_rows, (str, bytes)):
        raise TypeError("registry.assets must be a sequence")
    if not isinstance(relation_rows, Sequence) or isinstance(relation_rows, (str, bytes)):
        raise TypeError("patch.relations must be a sequence")
    assets: dict[str, Mapping[str, Any]] = {}
    for row in asset_rows:
        if not isinstance(row, Mapping):
            raise TypeError("Every registry asset must be an object")
        asset_id = _identifier(row, "id", "asset_id")
        if not asset_id:
            raise ValueError("Every registry asset requires id or asset_id")
        if asset_id in assets:
            raise ValueError(f"Duplicate registry asset ID: {asset_id}")
        assets[asset_id] = row

    endpoint_map = dict(active_policy.owner_endpoint_by_relation)
    support_rows = [
        row
        for row in relation_rows
        if isinstance(row, Mapping) and str(row.get("type") or "") in endpoint_map
    ]
    malformed_rows = [row for row in relation_rows if not isinstance(row, Mapping)]
    results = [
        evaluate_support_relationship_claim(row, assets, policy=active_policy)
        for row in support_rows
    ]
    context_failures: list[dict[str, str]] = []
    if patch.get("formal_release") is not False:
        context_failures.append(
            {
                "code": "formal_release_not_false",
                "message": "Candidate support patch must declare formal_release=false.",
            }
        )
    if patch.get("delivery_allowed") is not False:
        context_failures.append(
            {
                "code": "delivery_allowed_not_false",
                "message": "Candidate support patch must declare delivery_allowed=false.",
            }
        )
    if _has_forbidden_status(patch.get("status"), active_policy):
        context_failures.append(
            {
                "code": "accepted_or_formal_patch_status",
                "message": "Candidate support patch status contains accepted/formal state.",
            }
        )
    if malformed_rows:
        context_failures.append(
            {
                "code": "malformed_relation",
                "message": "Every relation must be an object.",
            }
        )
    if not support_rows:
        context_failures.append(
            {
                "code": "no_support_relationships",
                "message": "Patch contains no support relationships to validate.",
            }
        )
    allowed = not context_failures and all(result["relationship_allowed"] for result in results)
    return {
        "schema_version": "railway.support-relationship-patch-gate.v1",
        "relationship_patch_allowed": allowed,
        "formal_acceptance": False,
        "status": "candidate_patch_pass_pending_review" if allowed else "blocked_support_patch",
        "summary": {
            "relation_count": len(relation_rows),
            "support_relation_count": len(support_rows),
            "passed_count": sum(result["relationship_allowed"] for result in results),
            "failed_count": sum(not result["relationship_allowed"] for result in results),
        },
        "context_failures": context_failures,
        "claims": results,
    }
