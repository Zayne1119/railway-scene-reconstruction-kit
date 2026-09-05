"""P2 connection-cause ablations; shared geometry alarms, no truth access.

This explicitly tests diagnosis, not a detection improvement: all three modes
start from exactly the same P1 geometric findings. Observations can veto a
topology-only wrong-target hypothesis, or leave its cause unresolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .synthetic_track_audit import (
    PilotAuditPolicy,
    _alternative_targets,
    _bridge_support,
    _validate_input,
    audit_track_candidate,
)

STUDY_MODES = ("local_geometry", "topology_only", "topology_evidence")


@dataclass(frozen=True)
class StudyAuditPolicy(PilotAuditPolicy):
    neighborhood_length_m: float = 2.0
    minimum_neighborhood_support: float = 0.60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minimum_neighborhood_support > 1:
            raise ValueError("minimum_neighborhood_support must not exceed one")


def _neighborhood_support(
    points: np.ndarray, tree: cKDTree | None, policy: StudyAuditPolicy, *, at_end: bool
) -> float | None:
    if tree is None:
        return None
    ordered = points[::-1] if at_end else points
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))]
    samples = np.linspace(0, min(length[-1], policy.neighborhood_length_m), 41)
    query = np.column_stack([np.interp(samples, length, ordered[:, axis]) for axis in range(3)])
    return float(np.mean(tree.query(query)[0] <= policy.observation_support_radius_m))


def audit_study_candidate(
    value: dict[str, Any], mode: str = "local_geometry", policy: dict | None = None
) -> dict:
    if mode not in STUDY_MODES:
        raise ValueError(f"Unknown study mode: {mode}")
    active = StudyAuditPolicy(**(policy or {}))
    base_policy = {field.name: getattr(active, field.name) for field in fields(PilotAuditPolicy)}
    result = audit_track_candidate(value, mode="local_geometry", policy=base_policy)
    result.update(schema_version="railway.synthetic-track-study-audit.v1", mode=mode,
                  policy=asdict(active))
    result["limitations"] = [
        "All three modes share geometry/entity alarms; only connection-cause diagnosis differs.",
        "No track identity labels, repair, support ownership, turnout inference or field accuracy.",
        "Observation absence/sparsity leaves geometric alarms active and may prevent diagnosis.",
        "Endpoint neighborhood coverage cannot exclude every unobserved bridge or false return.",
        "Synthetic noisy centerlines are not LiDAR or full rail meshes.",
    ]
    for finding in result["findings"]:
        finding["cause_status"] = "assigned"
    result["summary"]["diagnosis_abstain_count"] = 0
    if mode == "local_geometry":
        return result
    segments, observations = _validate_input(value)
    tree = cKDTree(observations) if mode == "topology_evidence" and len(observations) else None
    connections = {edge["id"]: edge for edge in value["connections"]}
    findings = {tuple(item["entity_ids"]): item for item in result["findings"]}
    for check in result["checks"]:
        if len(check["entity_ids"]) != 1:
            continue
        edge = connections[check["entity_ids"][0]]
        source, target = segments[edge["source"]], segments[edge["target"]]
        measurements = check["measurements"]
        gap = measurements["endpoint_separation_m"]
        supported_context = False
        if mode == "topology_evidence":
            source_support = _neighborhood_support(source, tree, active, at_end=True)
            target_support = _neighborhood_support(target, tree, active, at_end=False)
            supported_context = all(
                support is not None and support >= active.minimum_neighborhood_support
                for support in (source_support, target_support)
            )
            measurements.update(
                source_neighborhood_support=source_support,
                target_neighborhood_support=target_support,
                bridge_observation_support=_bridge_support(source[-1], target[0], tree, active),
                evidence_status="available" if supported_context else "abstain",
            )
        if check["status"] != "flag":
            continue
        alternatives = [
            {"target_id": key, "separation_m": distance}
            for key, distance in _alternative_targets(edge["source"], edge["target"], segments, active)
            if gap - distance >= active.minimum_alternative_improvement_m
        ]
        measurements["alternative_candidates"] = alternatives
        wrong_target = bool(alternatives)
        if mode == "topology_evidence":
            for candidate in alternatives:
                points = segments[candidate["target_id"]]
                candidate["neighborhood_support"] = _neighborhood_support(
                    points, tree, active, at_end=False
                )
                candidate["bridge_support"] = _bridge_support(source[-1], points[0], tree, active)
            supporting = [
                item for item in alternatives
                if item["neighborhood_support"] is not None
                and item["neighborhood_support"] >= active.minimum_neighborhood_support
                and item["bridge_support"] >= active.minimum_bridge_support
            ]
            measurements["supported_alternative_count"] = len(supporting)
            wrong_target = bool(
                supported_context and supporting
                and measurements["bridge_observation_support"] < active.minimum_bridge_support
            )
        check["kind"] = "wrong_connection" if wrong_target else "gap"
        finding = findings[tuple(check["entity_ids"])]
        finding["kind"] = check["kind"]
        if mode == "topology_evidence" and not supported_context:
            # Retain the coarse geometric gap finding, but do not count it as a
            # correct gap diagnosis just because observation evidence is absent.
            finding["cause_status"] = "abstain"
    result["summary"]["evidence_abstain_count"] = sum(
        check["measurements"].get("evidence_status") == "abstain" for check in result["checks"]
    )
    result["summary"]["diagnosis_abstain_count"] = sum(
        finding["cause_status"] == "abstain" for finding in result["findings"]
    )
    return result
