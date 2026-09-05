"""Development-only direction baseline and bounded observation-support guard.

The guard rejects one inconsistency, not arbitrary observation contamination.
Smooth but incorrect connections can pass it, and curved gaps can violate its
straight-chord assumption. No truth, seed, condition or customer data is read.
The v1/v2 auditors remain unchanged controls, including their known failures.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .synthetic_track_adaptive_audit import (
    ADAPTIVE_MODES,
    _annotate,
    _summarize,
    audit_adaptive_candidate,
)
from .synthetic_track_audit import _unit, _validate_input
from .synthetic_track_study_audit import StudyAuditPolicy

GUARDED_MODES = (*ADAPTIVE_MODES, "direction_geometry", "guarded_evidence")


def _direction_measurements(
    source: np.ndarray, target: np.ndarray, policy: StudyAuditPolicy,
) -> dict:
    """Compare a proposed straight bridge with both forward endpoint tangents.

    Threshold is the pre-existing minimum_direction_cosine, not a tuned score.
    No coordinate axis or opaque ID is used to choose a continuation.
    """
    chord = _unit(target[0] - source[-1])
    source_direction = _unit(source[-1] - source[-2])
    target_direction = _unit(target[1] - target[0])
    source_cosine = float(np.clip(np.dot(chord, source_direction), -1.0, 1.0))
    target_cosine = float(np.clip(np.dot(chord, target_direction), -1.0, 1.0))
    return {
        "source_chord_direction_cosine": source_cosine,
        "target_chord_direction_cosine": target_cosine,
        "endpoint_tangent_cosine": float(np.clip(
            np.dot(source_direction, target_direction), -1.0, 1.0,
        )),
        "current_chord_direction_compatible": bool(
            min(source_cosine, target_cosine) >= policy.minimum_direction_cosine
        ),
        "direction_threshold_source": "existing_minimum_direction_cosine",
    }


def audit_guarded_candidate(
    value: dict[str, Any], mode: str = "guarded_evidence", policy: dict | None = None,
) -> dict:
    """Refine cause assignments without changing geometric alarms or locations.

    Direction baseline: a chord compatible with both endpoint tangents remains
    a gap hypothesis, even if a different route has a closer fragment. Otherwise
    keep the old topology hypothesis. Guarded evidence: preserve v2 except use
    this stronger baseline for missing-context fallbacks, and reject supported
    current bridges when their chord is incompatible and a plausible alternative
    exists. Rejected support becomes a geometric inference, never verification.

    Evidence may still override a smooth geometric gap when the current bridge
    lacks support and an alternative is supported. All such judgments remain
    hypotheses; neither forward orientation nor route semantics is inferred.
    """
    if mode not in GUARDED_MODES:
        raise ValueError(f"Unknown guarded study mode: {mode}")
    if mode in ADAPTIVE_MODES:
        return audit_adaptive_candidate(value, mode, policy)
    active = StudyAuditPolicy(**(policy or {}))
    segments, _ = _validate_input(value)
    baseline_mode = "topology_only" if mode == "direction_geometry" else "adaptive_evidence"
    result = audit_adaptive_candidate(value, baseline_mode, policy)
    result.update(schema_version="railway.synthetic-track-guarded-audit.v1", mode=mode)
    connections = {edge["id"]: edge for edge in value["connections"]}
    findings = {tuple(item["entity_ids"]): item for item in result["findings"]}
    for check in result["checks"]:
        key = tuple(check["entity_ids"])
        if len(key) != 1 or key not in findings:
            continue
        finding = findings[key]
        edge = connections[key[0]]
        measurements = finding["measurements"]
        measurements.update(_direction_measurements(
            segments[edge["source"]], segments[edge["target"]], active,
        ))
        alternatives = measurements.get("alternative_candidates", [])
        compatible = measurements["current_chord_direction_compatible"]
        geometric_kind = "gap" if compatible or not alternatives else "wrong_connection"
        measurements["direction_geometry_inferred_kind"] = geometric_kind
        measurements["observation_support_rejected_by_direction"] = False
        if mode == "direction_geometry":
            finding["kind"] = check["kind"] = geometric_kind
            _annotate(finding, "topology_only", "direction_geometry_current_chord_compatible"
                      if compatible else "direction_geometry_retained_topology")
        elif finding["decision_basis"] == "geometry_fallback":
            finding["kind"] = check["kind"] = geometric_kind
            _annotate(finding, "geometry_fallback", finding["decision_reason"] + "_direction_geometry")
        elif (finding["decision_reason"] == "current_path_supported"
              and not compatible and alternatives):
            measurements["observation_support_rejected_by_direction"] = True
            finding["kind"] = check["kind"] = geometric_kind
            _annotate(finding, "geometry_fallback", "current_support_rejected_by_endpoint_direction")
        # v2 shares the measurements dict between check and finding. Preserve
        # that relationship explicitly and never clear a geometric alarm.
        check["measurements"] = measurements
    result["summary"]["direction_guard_rejection_count"] = sum(
        bool(item["measurements"].get("observation_support_rejected_by_direction"))
        for item in result["findings"]
    )
    result["limitations"] = [
        "All six modes share geometric alarm entities and locations; this is cause diagnosis only.",
        "Current chord alignment assumes locally straight, forward-oriented fragment endpoints.",
        "Curved true gaps may fail the direction check; smooth wrong connections may pass it.",
        "The guard checks a directional contradiction, not observation authenticity or route identity.",
        "Geometry fallback is an assigned hypothesis, not evidence verification or calibrated certainty.",
        "Development revision after v1/v2 results; synthetic proxies, not held-out or field accuracy.",
        "No customer data, repair, production gate change or additional manual signoff.",
    ]
    return _summarize(result)
