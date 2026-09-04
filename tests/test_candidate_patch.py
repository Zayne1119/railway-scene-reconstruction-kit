from __future__ import annotations

import copy

import pytest

from railway_recon.candidate_patch import apply_candidate_registry_patch


def base_registry() -> dict:
    return {
        "release_id": "candidate-01",
        "assets": [
            {
                "id": "A",
                "status": "candidate_pending_review",
                "evidence_status": "direct_points",
                "geometry": {"node": "NODE-A"},
                "parameters": {},
                "limitations": [],
            }
        ],
        "relations": [],
    }


def candidate_patch() -> dict:
    return {
        "target_release_id": "candidate-01",
        "formal_release": False,
        "status": "candidate_pending_review",
        "asset_updates": [
            {
                "id": "A",
                "match_geometry_node": "NODE-A",
                "set": {"parameters.audit": "round-2"},
                "limitations_append_unique": ["Candidate only."],
            }
        ],
        "asset_additions": [
            {
                "id": "B",
                "status": "candidate_pending_review",
                "evidence_status": "direct_points",
                "geometry": {"node": "NODE-B"},
            }
        ],
        "relations": [
            {
                "id": "A-SUPPORTS-B",
                "source": "A",
                "target": "B",
                "from": "A",
                "to": "B",
                "status": "candidate_pending_review",
            }
        ],
    }


def test_candidate_patch_is_additive_and_idempotent() -> None:
    original = base_registry()
    patch = candidate_patch()

    first, report = apply_candidate_registry_patch(original, patch)
    second, second_report = apply_candidate_registry_patch(first, patch)

    assert original == base_registry()
    assert first == second
    assert report["added_asset_ids"] == ["B"]
    assert report["added_relation_ids"] == ["A-SUPPORTS-B"]
    assert second_report["existing_asset_noop_ids"] == ["B"]
    assert second_report["existing_relation_noop_ids"] == ["A-SUPPORTS-B"]
    assert first["assets"][0]["parameters"]["audit"] == "round-2"
    assert first["assets"][0]["limitations"] == ["Candidate only."]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (("status", "accepted_candidate"), "promotes status"),
        (("status", "accepted_relative_geometry"), "promotes status"),
        (("status", "reviewed-accepted"), "promotes status"),
        (("evidence_status", "supported"), "promotes evidence_status"),
    ],
)
def test_candidate_patch_rejects_acceptance_promotion(mutation, message) -> None:
    patch = candidate_patch()
    field, value = mutation
    patch["asset_additions"][0][field] = value

    with pytest.raises(ValueError, match=message):
        apply_candidate_registry_patch(base_registry(), patch)


def test_candidate_patch_rejects_destructive_or_divergent_reapplication() -> None:
    destructive = candidate_patch()
    destructive["asset_deletions"] = ["A"]
    with pytest.raises(ValueError, match="destructive"):
        apply_candidate_registry_patch(base_registry(), destructive)

    first, _ = apply_candidate_registry_patch(base_registry(), candidate_patch())
    divergent = copy.deepcopy(candidate_patch())
    divergent["asset_additions"][0]["geometry"]["node"] = "DIFFERENT"
    with pytest.raises(ValueError, match="collides"):
        apply_candidate_registry_patch(first, divergent)


@pytest.mark.parametrize(
    "nested",
    [
        {"parameters_merge": {"review": {"status": "accepted"}}},
        {"parameters_merge": {"review": {"evidence_status": "supported"}}},
        {"parameters_merge": {"review": {"formal_release": True}}},
        {"parameters_merge": {"review": {"asset_deletions": ["A"]}}},
    ],
)
def test_candidate_patch_rejects_nested_promotion_or_deletion(nested) -> None:
    patch = candidate_patch()
    patch["asset_updates"][0].update(nested)

    with pytest.raises(ValueError):
        apply_candidate_registry_patch(base_registry(), patch)
