"""Geometry-derived diagnostics for the independent, development-only track pilot.

Inputs are centerline polylines and synthetic centerline observations, not rail
meshes or survey LiDAR. No generator truth, layout identity, or cached residual
is consumed. The two modes share geometric separation and overlap checks; the
enhanced mode attempts evidence-supported diagnosis of a wrong connection.
This is an experimental adapter, not a replacement for production TrackGraph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .relationships import contact_status


@dataclass(frozen=True)
class PilotAuditPolicy:
    maximum_connection_gap_m: float = 0.08
    alternative_connection_radius_m: float = 0.12
    minimum_alternative_improvement_m: float = 0.20
    minimum_direction_cosine: float = 0.98
    duplicate_candidate_radius_m: float = 0.25
    maximum_duplicate_distance_m: float = 0.06
    minimum_duplicate_coverage: float = 0.80
    minimum_duplicate_length_m: float = 2.0
    observation_support_radius_m: float = 0.12
    minimum_bridge_support: float = 0.70

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in (
            "minimum_direction_cosine", "minimum_duplicate_coverage", "minimum_bridge_support"
        ):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must not exceed one")
        if self.maximum_duplicate_distance_m > self.duplicate_candidate_radius_m:
            raise ValueError("Duplicate candidate radius must cover the detection distance")


def _points(value: object, name: str, minimum: int = 2) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if minimum == 0 and points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} three-dimensional points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return points


def _validate_input(value: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    allowed = {
        "schema_version", "coordinate_system", "gauge_m", "segments", "connections", "observations"
    }
    if set(value) - allowed:
        raise ValueError("Unexpected input fields; audit accepts geometry and observations only")
    if value.get("schema_version") != "railway.synthetic-track-input.v1":
        raise ValueError("Unsupported synthetic input schema")
    if value.get("coordinate_system") != "synthetic_local_m":
        raise ValueError("This pilot accepts synthetic local metres only")
    segments: dict[str, np.ndarray] = {}
    for segment in value["segments"]:
        if set(segment) != {"id", "points"}:
            raise ValueError("Segments must not carry identity labels or audit measurements")
        key = segment["id"]
        if not isinstance(key, str) or not key or key in segments:
            raise ValueError("Segment IDs must be nonempty and unique")
        points = _points(segment["points"], key)
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-9):
            raise ValueError("Polyline contains consecutive coincident points")
        segments[key] = points
    if not segments:
        raise ValueError("At least one segment is required")
    connection_ids: set[str] = set()
    for edge in value["connections"]:
        if set(edge) != {"id", "source", "target"}:
            raise ValueError("Connections must contain only ID and candidate endpoints")
        key = edge["id"]
        if not isinstance(key, str) or not key or key in connection_ids or key in segments:
            raise ValueError("Connection IDs must be unique and disjoint from segment IDs")
        if edge["source"] not in segments or edge["target"] not in segments:
            raise ValueError("Connection references an unknown segment")
        if edge["source"] == edge["target"]:
            raise ValueError("This pilot does not support self-connections")
        connection_ids.add(key)
    observations = _points(value["observations"], "observations", minimum=0)
    return segments, observations


def _sample_polyline(points: np.ndarray, count: int = 81) -> tuple[np.ndarray, float]:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.linspace(0.0, distance[-1], count)
    samples = np.column_stack([np.interp(query, distance, points[:, axis]) for axis in range(3)])
    return samples, float(distance[-1])


def _distances_to_polyline(query: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts = polyline[:-1]
    vectors = np.diff(polyline, axis=0)
    relative = query[:, None, :] - starts[None, :, :]
    factors = np.einsum("nmi,mi->nm", relative, vectors) / np.sum(vectors * vectors, axis=1)
    closest = starts[None, :, :] + np.clip(factors, 0.0, 1.0)[:, :, None] * vectors[None]
    return np.sqrt(np.min(np.sum((query[:, None] - closest) ** 2, axis=2), axis=1))


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-12 else np.zeros(3)


def _bridge_support(
    start: np.ndarray, end: np.ndarray, tree: cKDTree | None, policy: PilotAuditPolicy
) -> float | None:
    if tree is None:
        return None
    count = min(81, max(2, int(np.ceil(np.linalg.norm(end - start) / 0.10)) + 1))
    query = np.linspace(start, end, count)
    distances = tree.query(query)[0]
    return float(np.mean(distances <= policy.observation_support_radius_m))


def _alternative_targets(
    source_id: str, target_id: str, segments: dict[str, np.ndarray], policy: PilotAuditPolicy
) -> list[tuple[str, float]]:
    source = segments[source_id]
    direction = _unit(source[-1] - source[-2])
    candidates = []
    for key, points in segments.items():
        if key in {source_id, target_id}:
            continue
        gap = float(np.linalg.norm(source[-1] - points[0]))
        cosine = float(np.dot(direction, _unit(points[1] - points[0])))
        if gap <= policy.alternative_connection_radius_m and cosine >= policy.minimum_direction_cosine:
            candidates.append((key, gap))
    # Opaque IDs must never select the evidence used to classify an error.
    # Evaluate every plausible alternative; no unique correct target is claimed.
    return sorted(candidates, key=lambda item: (item[1], *segments[item[0]][0]))


def _connection_checks(
    value: dict[str, Any], segments: dict[str, np.ndarray], tree: cKDTree | None,
    mode: str, policy: PilotAuditPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings, checks = [], []
    for edge in sorted(value["connections"], key=lambda row: row["id"]):
        source, target = segments[edge["source"]], segments[edge["target"]]
        start, end = source[-1], target[0]
        gap = float(np.linalg.norm(end - start))
        status = contact_status(
            gap, maximum_gap_m=policy.maximum_connection_gap_m, maximum_penetration_m=0.0
        )
        measurements: dict[str, Any] = {"endpoint_separation_m": gap}
        kind = "gap"
        if mode == "topology_evidence":
            support = _bridge_support(start, end, tree, policy)
            measurements["bridge_observation_support"] = support
            measurements["evidence_status"] = "available" if support is not None else "abstain"
            if status != "pass":
                alternatives = []
                for alternative_id, alternative_gap in _alternative_targets(
                    edge["source"], edge["target"], segments, policy
                ):
                    alternatives.append({
                        "target_id": alternative_id, "separation_m": alternative_gap,
                        "observation_support": _bridge_support(
                            start, segments[alternative_id][0], tree, policy
                        ),
                    })
                supported = [
                    item for item in alternatives
                    if gap - item["separation_m"] >= policy.minimum_alternative_improvement_m
                    and item["observation_support"] is not None
                    and item["observation_support"] >= policy.minimum_bridge_support
                ]
                measurements["alternative_candidates"] = alternatives
                measurements["supported_alternative_count"] = len(supported)
                if supported and support is not None and support < policy.minimum_bridge_support:
                    kind = "wrong_connection"
        check = {
            "kind": kind, "entity_ids": [edge["id"]],
            "status": "pass" if status == "pass" else "flag", "measurements": measurements,
        }
        checks.append(check)
        if status != "pass":
            findings.append({
                "kind": kind, "entity_ids": [edge["id"]],
                "position": ((start + end) / 2).tolist(),
                "score": min(1.0, gap / policy.maximum_connection_gap_m),
                "measurements": measurements,
            })
    return findings, checks


def _overlap_checks(
    segments: dict[str, np.ndarray], policy: PilotAuditPolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings, checks = [], []
    samples = {key: _sample_polyline(points) for key, points in segments.items()}
    for first, second in combinations(sorted(segments), 2):
        a, length_a = samples[first]
        b, length_b = samples[second]
        # Broad phase excludes distant/non-interacting segment pairs from the
        # normal-check denominator. These are geometric candidates, not all n^2 pairs.
        low_a, high_a = a.min(axis=0), a.max(axis=0)
        low_b, high_b = b.min(axis=0), b.max(axis=0)
        separation = np.maximum(0, np.maximum(low_a - high_b, low_b - high_a))
        if np.linalg.norm(separation) > policy.duplicate_candidate_radius_m:
            continue
        da = _distances_to_polyline(a, segments[second])
        db = _distances_to_polyline(b, segments[first])
        if min(da.min(), db.min()) > policy.duplicate_candidate_radius_m:
            continue
        coverage_a = float(np.mean(da <= policy.maximum_duplicate_distance_m))
        coverage_b = float(np.mean(db <= policy.maximum_duplicate_distance_m))
        overlap_length = min(coverage_a * length_a, coverage_b * length_b)
        shorter_coverage = coverage_a if length_a <= length_b else coverage_b
        alignment = abs(float(np.dot(_unit(a[-1] - a[0]), _unit(b[-1] - b[0]))))
        flagged = (
            shorter_coverage >= policy.minimum_duplicate_coverage
            and overlap_length >= policy.minimum_duplicate_length_m
            and alignment >= policy.minimum_direction_cosine
        )
        measurements = {
            "first_coverage": coverage_a, "second_coverage": coverage_b,
            "estimated_overlap_m": overlap_length, "direction_cosine_abs": alignment,
            "first_distance_p50_m": float(np.median(da)),
            "second_distance_p50_m": float(np.median(db)),
        }
        checks.append({
            "kind": "duplicate", "entity_ids": [first, second],
            "status": "flag" if flagged else "pass", "measurements": measurements,
        })
        if flagged:
            near = np.concatenate((
                a[da <= policy.maximum_duplicate_distance_m],
                b[db <= policy.maximum_duplicate_distance_m],
            ))
            findings.append({
                "kind": "duplicate", "entity_ids": [first, second],
                "position": np.mean(near, axis=0).tolist(),
                "score": shorter_coverage, "measurements": measurements,
            })
    return findings, checks


def audit_track_candidate(
    value: dict[str, Any], mode: str = "local_geometry", policy: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Measure and diagnose a candidate without accessing truth or any files.

    Scores are uncalibrated priority scores. Empty observations leave geometric
    detection active but prevent evidence-supported wrong-connection diagnosis.
    A geometric separation finding can detect a bad connection even when its
    precise cause is unknown; evaluators must separate detection from diagnosis.
    """
    if mode not in {"local_geometry", "topology_evidence"}:
        raise ValueError(f"Unknown audit mode: {mode}")
    active = PilotAuditPolicy(**(policy or {}))
    segments, observations = _validate_input(value)
    tree = cKDTree(observations) if len(observations) and mode == "topology_evidence" else None
    findings, checks = _connection_checks(value, segments, tree, mode, active)
    duplicate_findings, duplicate_checks = _overlap_checks(segments, active)
    findings.extend(duplicate_findings)
    checks.extend(duplicate_checks)
    return {
        "schema_version": "railway.synthetic-track-audit.v1", "mode": mode,
        "policy": asdict(active), "findings": findings, "checks": checks,
        "summary": {
            "segment_count": len(segments), "connection_count": len(value["connections"]),
            "observation_count": len(observations), "finding_count": len(findings),
            "check_count": len(checks),
            "evidence_abstain_count": sum(
                item["measurements"].get("evidence_status") == "abstain" for item in checks
            ),
        },
        "limitations": [
            "Development pilot on centerline proxies, not measured LiDAR or complete rail meshes.",
            "Assumes forward-oriented fragments and known candidate directed connections.",
            "No independent track identity, turnout reasoning, repair or production acceptance.",
            "Evidence only refines diagnosis; identical detection across modes is possible.",
            "Missing observation support alone is not evidence of a geometry defect.",
        ],
    }
