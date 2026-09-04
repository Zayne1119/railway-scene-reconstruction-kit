from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, write_json

ALLOWED_ACTIONS = {
    "build_candidate",
    "refine_existing_group",
    "require_geometry_gate",
    "no_build",
}


def compile_gap_candidate_dispositions(
    gap_report: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Join explicit point/photo review decisions to gap-candidate geometry.

    This intentionally does not infer semantics from a candidate's dimensions.  The
    candidate detector proposes review locations; only an explicit evidence review
    can authorize new geometry.
    """

    candidates = {
        str(item["candidate_id"]): item
        for item in gap_report.get("vertical_candidates", [])
    }
    decisions = review.get("decisions", [])
    if not decisions:
        raise ValueError("At least one explicit semantic decision is required")

    seen: set[str] = set()
    compiled: list[dict[str, Any]] = []
    for decision in decisions:
        candidate_id = str(decision.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"Invalid or duplicate candidate decision: {candidate_id}")
        seen.add(candidate_id)
        if candidate_id not in candidates:
            raise ValueError(f"Reviewed candidate is absent from gap report: {candidate_id}")
        action = str(decision.get("model_action", ""))
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported model action for {candidate_id}: {action}")
        confidence = float(decision.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence is outside [0, 1] for {candidate_id}")
        evidence = list(decision.get("evidence", []))
        if not evidence:
            raise ValueError(f"Evidence references are required for {candidate_id}")

        source = candidates[candidate_id]
        compiled.append(
            {
                **decision,
                "candidate_id": candidate_id,
                "model_action": action,
                "confidence": confidence,
                "candidate_geometry": {
                    "station_m": float(source["station_m"]),
                    "cross_m": float(source["cross_m"]),
                    "minimum_xyz_m": source["minimum_xyz_m"],
                    "maximum_xyz_m": source["maximum_xyz_m"],
                    "sample_point_count": int(source["sample_point_count"]),
                    "corridor_scope_ownership": source["corridor_scope_ownership"],
                },
            }
        )

    action_counts = Counter(item["model_action"] for item in compiled)
    class_counts = Counter(str(item["reviewed_class"]) for item in compiled)
    build_ids = [
        item["candidate_id"]
        for item in compiled
        if item["model_action"] == "build_candidate"
    ]
    return {
        "schema_version": "railway.gap-candidate-disposition.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_gap_report": review.get("source_gap_report"),
        "reviewed_candidate_count": len(compiled),
        "model_action_counts": dict(sorted(action_counts.items())),
        "reviewed_class_counts": dict(sorted(class_counts.items())),
        "build_candidate_ids": build_ids,
        "decisions": compiled,
        "status": "explicit_evidence_review_compiled_geometry_unchanged",
        "limitations": [
            "A build_candidate decision authorizes candidate generation, not baseline promotion.",
            "Professional subtypes remain provisional where the panorama cannot resolve a product type.",
        ],
    }


def write_gap_candidate_dispositions(
    *,
    gap_report_path: str | Path,
    review_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    gap_path = Path(gap_report_path).resolve()
    input_review_path = Path(review_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite disposition report: {output}")
    review = load_json(input_review_path)
    review["source_gap_report"] = str(gap_path)
    result = compile_gap_candidate_dispositions(load_json(gap_path), review)
    result["source_review"] = str(input_review_path)
    write_json(output, result)
    return result
