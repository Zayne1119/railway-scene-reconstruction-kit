"""Two-tier audit for fast internal candidate iteration.

The existing :mod:`candidate_release_audit` remains deliberately strict and is
still the authority before an external/formal release.  This module reuses all
of its checks, but demotes review-process and legacy-contract debt to warnings
while keeping damaged, missing, mismatched, or promoted artifacts as hard
blockers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .candidate_release_audit import CandidateReleasePolicy, audit_candidate_release


@dataclass(frozen=True)
class RapidCandidatePolicy:
    """Policy for one-person, model-first internal iteration."""

    warning_codes: tuple[str, ...] = (
        "delivery_gate_blocking_reasons_missing",
        "gate_artifacts_not_list",
        "gate_required_artifact_roles_missing",
        "critical_stages_missing",
        "required_critical_stages_missing",
        "registry_parent_release_id_mismatch",
        "lineage_delivery_allowed_not_false",
    )


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _lineage_delivery_is_explicitly_true(
    lineage_path: str | Path | None,
) -> bool:
    if lineage_path is None:
        return False
    return _read_object(Path(lineage_path).resolve()).get("delivery_allowed") is True


def classify_candidate_failures(
    strict_failures: Sequence[Mapping[str, Any]],
    *,
    policy: RapidCandidatePolicy | None = None,
    lineage_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split strict findings into hard blockers and iteration warnings."""

    contract = policy or RapidCandidatePolicy()
    warning_codes = set(contract.warning_codes)
    explicit_delivery = _lineage_delivery_is_explicitly_true(lineage_path)
    hard_blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for finding in strict_failures:
        item = dict(finding)
        code = str(item.get("code", ""))
        # Missing legacy metadata is a warning.  An explicit delivery promotion
        # is never softened, even when it uses the same strict-audit code.
        soft = code in warning_codes and not (
            code == "lineage_delivery_allowed_not_false" and explicit_delivery
        )
        item["severity"] = "warning" if soft else "hard_blocker"
        (warnings if soft else hard_blockers).append(item)
    return hard_blockers, warnings


def _review_summary(
    manifest_path: Path,
    release_directory: Path,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path)
    records = manifest.get("artifacts")
    fixed_path: Path | None = None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict) or record.get("role") != "fixed_views":
                continue
            value = record.get("path")
            if isinstance(value, str) and value.strip():
                fixed_path = Path(value)
                if not fixed_path.is_absolute():
                    fixed_path = release_directory / fixed_path
            break
    fixed = _read_object(fixed_path.resolve() if fixed_path is not None else None)
    pending = 0
    decided = 0
    for value in fixed.values():
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, dict) or "review_status" not in row:
                continue
            if row.get("review_status") == "pending_human_review":
                pending += 1
            else:
                decided += 1
    return {
        "required_for_internal_iteration": False,
        "required_before_external_formal_release": True,
        "workflow": "single_owner_batch_confirmation_before_external_release",
        "pending_view_count": pending,
        "decided_view_count": decided,
    }


def audit_rapid_candidate(
    release_directory: str | Path,
    manifest_path: str | Path,
    delivery_gate_path: str | Path,
    *,
    lineage_path: str | Path | None = None,
    expected_release_id: str | None = None,
    expected_parent_release_id: str | None = None,
    policy: RapidCandidatePolicy | None = None,
) -> dict[str, Any]:
    """Audit a candidate for continued internal modeling, not for delivery."""

    root = Path(release_directory).resolve()
    manifest = Path(manifest_path).resolve()
    strict = audit_candidate_release(
        root,
        manifest,
        Path(delivery_gate_path).resolve(),
        lineage_path=lineage_path,
        expected_release_id=expected_release_id,
        expected_parent_release_id=expected_parent_release_id,
        policy=CandidateReleasePolicy(),
    )
    hard_blockers, warnings = classify_candidate_failures(
        strict.get("failures", []), policy=policy, lineage_path=lineage_path
    )
    ready = not hard_blockers
    return {
        "schema_version": "railway.rapid-candidate-audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "release_id": strict.get("release_id"),
        "profile": "solo_fast_model_iteration",
        "scope": "internal_iteration_only_not_formal_release_authorization",
        "passed": ready,
        "iteration_ready": ready,
        "delivery_allowed": False,
        "formal_release": False,
        "decision": (
            "continue_modeling_with_automatic_warning_queue"
            if ready
            else "stop_for_broken_or_unsafe_artifact"
        ),
        "hard_blocker_count": len(hard_blockers),
        "hard_blockers": hard_blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "manual_review": _review_summary(manifest, root),
        "strict_audit": {
            "passed": strict.get("passed"),
            "failure_count": strict.get("failure_count"),
            "decision": strict.get("decision"),
        },
        "policy": {
            "default_unrecognized_finding_severity": "hard_blocker",
            "warning_codes": list((policy or RapidCandidatePolicy()).warning_codes),
            "formal_release_requires_separate_strict_audit": True,
        },
    }


def write_rapid_candidate_audit(
    output_path: str | Path,
    release_directory: str | Path,
    manifest_path: str | Path,
    delivery_gate_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = audit_rapid_candidate(
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
        description="Two-tier audit for fast internal model iteration"
    )
    parser.add_argument("--release-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--delivery-gate", type=Path, required=True)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--expected-release-id")
    parser.add_argument("--expected-parent-release-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = write_rapid_candidate_audit(
            args.output,
            args.release_directory,
            args.manifest,
            args.delivery_gate,
            lineage_path=args.lineage,
            expected_release_id=args.expected_release_id,
            expected_parent_release_id=args.expected_parent_release_id,
        )
    except FileExistsError:
        print(f"Refusing to overwrite rapid audit: {args.output.resolve()}", file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
