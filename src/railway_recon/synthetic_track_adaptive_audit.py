"""Keep geometric inference available when observations cannot verify a cause.

No customer assets, truth, filenames, split labels or new numeric thresholds
are consumed. Confidence labels identify decision provenance, not calibrated
probabilities. `evidence_verified` means internal support criteria passed; it
does not certify the inferred cause, observation authenticity or survey quality.
"""
from __future__ import annotations

import copy
from typing import Any

from .synthetic_track_study_audit import STUDY_MODES, StudyAuditPolicy, audit_study_candidate

ADAPTIVE_MODES = (*STUDY_MODES, "adaptive_evidence")
DECISION_BASES = (
    "local_geometry", "topology_only", "geometry_fallback", "evidence_supported",
    "evidence_conflict", "evidence_abstention", "geometry_duplicate",
)


def _annotate(finding: dict, basis: str, reason: str) -> None:
    verified = basis == "evidence_supported"
    confidence = "observation_supported" if verified else "inferred_geometry"
    if basis == "evidence_conflict":
        confidence = "unresolved_conflict"
    elif basis == "evidence_abstention":
        confidence = "unresolved_no_evidence"
    finding.update(
        decision_basis=basis, evidence_verified=verified,
        confidence_level=confidence, decision_reason=reason,
    )


def _summarize(result: dict) -> dict:
    result["summary"].update(
        diagnosis_abstain_count=sum(item["cause_status"] == "abstain" for item in result["findings"]),
        geometry_fallback_count=sum(item["decision_basis"] == "geometry_fallback" for item in result["findings"]),
        evidence_verified_count=sum(item["evidence_verified"] for item in result["findings"]),
    )
    result["confidence_interpretation"] = (
        "Qualitative decision provenance only, not calibrated probability. Evidence verified "
        "means coverage/path-support criteria passed, not that the cause is ground-truth correct."
    )
    return result


def _annotated_baseline(value: dict, mode: str, policy: dict | None) -> dict:
    result = audit_study_candidate(value, mode, policy)
    active = StudyAuditPolicy(**(policy or {}))
    for finding in result["findings"]:
        if finding["kind"] == "duplicate":
            _annotate(finding, "geometry_duplicate", "overlap_geometry_only")
        elif mode != "topology_evidence":
            _annotate(finding, mode, "unchanged_v1_baseline")
        elif finding["cause_status"] == "abstain":
            _annotate(finding, "evidence_abstention", "strict_v1_missing_or_sparse_evidence")
        else:
            measurements = finding["measurements"]
            verified = (
                measurements.get("evidence_status") == "available"
                and (
                    measurements.get("bridge_observation_support", 0) >= active.minimum_bridge_support
                    or finding["kind"] == "wrong_connection"
                )
            )
            _annotate(finding, "evidence_supported" if verified else "geometry_fallback",
                      "unchanged_v1_assignment")
    return _summarize(result)


def audit_adaptive_candidate(
    value: dict[str, Any], mode: str = "adaptive_evidence", policy: dict | None = None
) -> dict:
    """Use observed support when sufficient; otherwise retain topology inference.

    All modes preserve the original geometric alarm entities/positions. Current
    or alternative neighborhood absence is not negative evidence. If current
    and all alternative paths lack support despite adequate local coverage,
    the conflict is exposed as an unresolved cause, without clearing the alarm.
    """
    if mode not in ADAPTIVE_MODES:
        raise ValueError(f"Unknown adaptive study mode: {mode}")
    if mode in STUDY_MODES:
        return _annotated_baseline(value, mode, policy)
    active = StudyAuditPolicy(**(policy or {}))
    result = audit_study_candidate(value, "topology_only", policy)
    measured = audit_study_candidate(value, "topology_evidence", policy)
    result.update(schema_version="railway.synthetic-track-adaptive-audit.v1", mode=mode)
    observations = {tuple(item["entity_ids"]): item for item in measured["checks"]}
    findings = {tuple(item["entity_ids"]): item for item in result["findings"]}
    for check in result["checks"]:
        key = tuple(check["entity_ids"])
        if len(key) != 1:
            if key in findings:
                _annotate(findings[key], "geometry_duplicate", "overlap_geometry_only")
            continue
        measurements = copy.deepcopy(observations[key]["measurements"])
        check["measurements"] = measurements
        if key not in findings:
            continue
        finding = findings[key]
        finding["measurements"] = measurements
        measurements["topology_inferred_kind"] = finding["kind"]
        finding["cause_status"] = "assigned"
        if measurements.get("evidence_status") != "available":
            _annotate(finding, "geometry_fallback", "current_neighborhood_missing_or_sparse")
            continue
        alternatives = measurements.get("alternative_candidates", [])
        if measurements["bridge_observation_support"] >= active.minimum_bridge_support:
            finding["kind"] = check["kind"] = "gap"
            _annotate(finding, "evidence_supported", "current_path_supported")
            continue
        fully_covered = [
            item for item in alternatives
            if item.get("neighborhood_support") is not None
            and item["neighborhood_support"] >= active.minimum_neighborhood_support
        ]
        supporting = [item for item in fully_covered
                      if item["bridge_support"] >= active.minimum_bridge_support]
        if supporting:
            finding["kind"] = check["kind"] = "wrong_connection"
            _annotate(finding, "evidence_supported", "alternative_path_supported")
        elif not alternatives:
            _annotate(finding, "geometry_fallback", "no_geometric_alternative")
        elif len(fully_covered) != len(alternatives):
            _annotate(finding, "geometry_fallback", "alternative_neighborhood_missing_or_sparse")
        else:
            finding["cause_status"] = "abstain"
            _annotate(finding, "evidence_conflict", "covered_current_and_alternative_paths_unsupported")
    result["summary"]["evidence_abstain_count"] = measured["summary"]["evidence_abstain_count"]
    result["limitations"] = [
        "All compared modes share geometric alarms; only cause assignment and provenance change.",
        "Fallback is topology/geometry inference, not observation verification or known truth.",
        "Confidence levels are qualitative provenance, not calibrated probabilities.",
        "Clutter can imitate path support; endpoint coverage does not prove bridge visibility.",
        "No repair, customer-model mutation, manual task creation or production gate change.",
        "Independent synthetic centerline proxies, not field LiDAR or full track meshes.",
    ]
    return _summarize(result)
