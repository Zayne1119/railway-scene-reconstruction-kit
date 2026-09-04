from __future__ import annotations

import copy

import pytest

from railway_recon.support_relationship_gate import (
    SupportRelationshipPolicy,
    evaluate_support_relationship_claim,
    validate_support_relationship_patch,
)


def _asset(asset_id: str, *, status: str = "review_candidate") -> dict:
    return {
        "id": asset_id,
        "status": status,
        "evidence_status": "direct_observation_candidate",
        "geometry": {"file": "candidate.obj", "node": f"NODE-{asset_id}"},
    }


def _claim() -> dict:
    return {
        "id": "REL-COLUMN-RESTS-ON-DECK",
        "type": "rests_on",
        "source": "COLUMN",
        "target": "DECK",
        "status": "review_candidate",
        "evidence_status": "direct_interface_candidate",
        "support_evidence": {
            "direct_contact_observed": True,
            "contact_residual_m": 0.008,
            "owner_semantics_resolved": True,
            "owner_identification_method": "surveyed_interface_boundary",
            "evidence_kinds": ["full_density_point_cloud", "registered_topology"],
            "proximity_only": False,
        },
    }


def _assets() -> dict[str, dict]:
    return {"COLUMN": _asset("COLUMN"), "DECK": _asset("DECK")}


def test_direct_candidate_support_claim_passes_without_formal_acceptance() -> None:
    result = evaluate_support_relationship_claim(_claim(), _assets())

    assert result["relationship_allowed"] is True
    assert result["formal_acceptance"] is False
    assert result["metrics"]["absolute_contact_residual_m"] == pytest.approx(0.008)


def test_missing_owner_fails_closed() -> None:
    claim = _claim()
    claim["target"] = None

    result = evaluate_support_relationship_claim(claim, _assets())

    assert result["relationship_allowed"] is False
    assert "missing_owner" in {failure["code"] for failure in result["failures"]}


def test_nearest_aabb_cannot_launder_support_ownership() -> None:
    claim = _claim()
    claim["support_evidence"].update(
        {
            "owner_identification_method": "nearest_aabb_overlap",
            "evidence_kinds": ["nearest_distance", "aabb_overlap"],
            "proximity_only": True,
            "contact_residual_m": 0.0,
        }
    )

    result = evaluate_support_relationship_claim(claim, _assets())
    failures = {failure["code"] for failure in result["failures"]}

    assert result["relationship_allowed"] is False
    assert "proximity_only_evidence" in failures
    assert "direct_evidence_kinds_insufficient" in failures
    assert "direct_evidence_kind_not_allowed" in failures


def test_unknown_evidence_kind_cannot_claim_direct_support() -> None:
    claim = _claim()
    claim["support_evidence"]["evidence_kinds"] = ["made_up"]

    result = evaluate_support_relationship_claim(claim, _assets())
    failures = {failure["code"] for failure in result["failures"]}

    assert result["relationship_allowed"] is False
    assert "direct_evidence_kind_not_allowed" in failures
    assert "direct_evidence_kinds_insufficient" in failures


def test_unknown_owner_identification_method_is_rejected() -> None:
    claim = _claim()
    claim["support_evidence"]["owner_identification_method"] = "made_up"

    result = evaluate_support_relationship_claim(claim, _assets())

    assert result["relationship_allowed"] is False
    assert "owner_identification_method_not_allowed" in {
        failure["code"] for failure in result["failures"]
    }


@pytest.mark.parametrize("field,value", [("status", "accepted"), ("evidence_status", "formal")])
def test_relation_accepted_or_formal_status_is_rejected(field: str, value: str) -> None:
    claim = _claim()
    claim[field] = value

    result = evaluate_support_relationship_claim(claim, _assets())

    assert "accepted_or_formal_status_pollution" in {
        failure["code"] for failure in result["failures"]
    }


def test_accepted_owner_state_is_rejected() -> None:
    assets = _assets()
    assets["DECK"]["status"] = "formally-accepted"

    result = evaluate_support_relationship_claim(_claim(), assets)

    assert result["relationship_allowed"] is False
    assert "accepted_or_formal_status_pollution" in {
        failure["code"] for failure in result["failures"]
    }


def test_patch_requires_explicit_nonformal_nondelivery_context() -> None:
    registry = {"assets": list(_assets().values())}
    patch = {"formal_release": False, "delivery_allowed": False, "relations": [_claim()]}

    passed = validate_support_relationship_patch(registry, patch)
    polluted = copy.deepcopy(patch)
    polluted["formal_release"] = True
    polluted["delivery_allowed"] = True
    blocked = validate_support_relationship_patch(registry, polluted)

    assert passed["relationship_patch_allowed"] is True
    assert blocked["relationship_patch_allowed"] is False
    assert {failure["code"] for failure in blocked["context_failures"]} == {
        "delivery_allowed_not_false",
        "formal_release_not_false",
    }


def test_patch_level_accepted_status_is_rejected() -> None:
    registry = {"assets": list(_assets().values())}
    patch = {
        "formal_release": False,
        "delivery_allowed": False,
        "status": "formally_accepted",
        "relations": [_claim()],
    }

    result = validate_support_relationship_patch(registry, patch)

    assert result["relationship_patch_allowed"] is False
    assert result["context_failures"][0]["code"] == "accepted_or_formal_patch_status"


def test_relation_requires_explicit_candidate_state() -> None:
    claim = _claim()
    claim.pop("status")
    claim.pop("evidence_status")

    result = evaluate_support_relationship_claim(claim, _assets())
    failures = {failure["code"] for failure in result["failures"]}

    assert "relation_status_missing" in failures
    assert "relation_evidence_status_missing" in failures


def test_unmodeled_or_unresolved_owner_and_large_gap_fail() -> None:
    assets = _assets()
    assets["DECK"].pop("geometry")
    claim = _claim()
    claim["support_evidence"]["owner_semantics_resolved"] = False
    claim["support_evidence"]["contact_residual_m"] = 0.08

    result = evaluate_support_relationship_claim(
        claim,
        assets,
        policy=SupportRelationshipPolicy(maximum_absolute_contact_residual_m=0.05),
    )
    failures = {failure["code"] for failure in result["failures"]}

    assert {
        "contact_residual_exceeds_tolerance",
        "owner_geometry_missing",
        "owner_semantics_unresolved",
    } <= failures


def test_patch_with_no_support_claim_is_not_a_pass() -> None:
    result = validate_support_relationship_patch(
        {"assets": list(_assets().values())},
        {"formal_release": False, "delivery_allowed": False, "relations": []},
    )

    assert result["relationship_patch_allowed"] is False
    assert result["context_failures"][0]["code"] == "no_support_relationships"
