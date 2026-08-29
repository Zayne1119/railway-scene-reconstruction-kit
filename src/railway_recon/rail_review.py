from __future__ import annotations

import csv
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json

REVIEW_COLUMNS = [
    "segment_id",
    "source_report",
    "source_sha256",
    "diagnostic_image",
    "rail_pair_count",
    "minimum_selected_support",
    "maximum_position_correction_m",
    "maximum_crosslevel_m",
    "maximum_cross_fit_p90_m",
    "maximum_z_fit_p90_m",
    "decision",
    "reviewer",
    "reviewed_at",
    "notes",
]


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("Review package name may contain only letters, digits, dot, dash and underscore")
    return value


def _finite_max(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def _selected_support(report: dict[str, Any]) -> float | None:
    selected = {
        int(index)
        for pair in report.get("rail_pairs", [])
        for index in pair.get("peak_indexes", [])
    }
    values = [
        float(peak["coverage"])
        for peak in report.get("peaks", [])
        if int(peak.get("grid_index", -1)) in selected
    ]
    return min(values) if values else None


def create_rail_review_package(
    project: ProjectConfig,
    segment_sources: list[tuple[str, str | Path]],
    output_name: str = "track_graph_rail_review_v1",
) -> Path:
    if not segment_sources:
        raise ValueError("At least one SEGMENT=REPORT source is required")
    root = project.workspace_path("reports") / _safe_name(output_name)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite review package: {root}")
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment_id, source_value in segment_sources:
        if segment_id in seen:
            raise ValueError(f"Duplicate review source segment: {segment_id}")
        seen.add(segment_id)
        source = Path(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        report = load_json(source)
        if report.get("schema_version") != "railway.rail-candidates.v1":
            raise ValueError(f"Unsupported rail candidate schema: {source}")
        if report.get("segment_id") != segment_id:
            raise ValueError(f"Review source segment mismatch: {source}")
        diagnostic = Path(str(report.get("diagnostic_image", ""))).resolve()
        source_hash = sha256_file(source)
        pairs = list(report.get("rail_pairs", []))
        lines = list(report.get("rail_lines", []))
        row = {
            "segment_id": segment_id,
            "source_report": str(source),
            "source_sha256": source_hash,
            "diagnostic_image": str(diagnostic),
            "rail_pair_count": len(pairs),
            "minimum_selected_support": _selected_support(report),
            "maximum_position_correction_m": _finite_max(
                [item.get("rail_position_correction_m", 0.0) for item in pairs]
            ),
            "maximum_crosslevel_m": _finite_max(
                [item.get("rail_top_crosslevel_m") for item in pairs]
            ),
            "maximum_cross_fit_p90_m": _finite_max(
                [item.get("cross_fit_residual_p90_m") for item in lines]
            ),
            "maximum_z_fit_p90_m": _finite_max(
                [item.get("z_fit_residual_p90_m") for item in lines]
            ),
            "decision": "",
            "reviewer": "",
            "reviewed_at": "",
            "notes": "",
        }
        rows.append(row)
        manifest_sources.append(
            {
                "segment_id": segment_id,
                "source_report": str(source),
                "source_sha256": source_hash,
                "diagnostic_image": str(diagnostic),
                "diagnostic_exists": diagnostic.is_file(),
            }
        )
    task_path = root / "rail-candidate-review.csv"
    with task_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        root / "manifest.json",
        {
            "schema_version": "railway.rail-source-review-package.v1",
            "project_id": project.project_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "pending_independent_review",
            "task_file": task_path.name,
            "decision_values": ["accepted", "rejected"],
            "instructions": (
                "Open every diagnostic image, verify all physical tracks and paired rails, "
                "then fill decision, reviewer, reviewed_at and notes."
            ),
            "sources": manifest_sources,
        },
    )
    return root


def apply_rail_review_package(project: ProjectConfig, package_value: str | Path) -> dict[str, Any]:
    root = Path(package_value).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "railway.rail-source-review-package.v1":
        raise ValueError("Unsupported rail review package schema")
    if manifest.get("project_id") != project.project_id:
        raise ValueError("Rail review package project_id does not match project")
    task_path = root / str(manifest["task_file"])
    with task_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    source_by_segment = {
        str(item["segment_id"]): item for item in manifest.get("sources", [])
    }
    failures: list[dict[str, Any]] = []
    reviewed_sources: list[dict[str, Any]] = []
    decisions: list[str] = []
    for row in rows:
        segment_id = str(row.get("segment_id", ""))
        source_record = source_by_segment.get(segment_id)
        if source_record is None:
            failures.append({"segment_id": segment_id, "reason": "unknown_segment"})
            continue
        source = Path(str(source_record["source_report"])).resolve()
        if not source.is_file() or sha256_file(source) != source_record["source_sha256"]:
            failures.append({"segment_id": segment_id, "reason": "source_hash_mismatch"})
            continue
        decision = str(row.get("decision", "")).strip().lower()
        decisions.append(decision)
        if decision not in {"accepted", "rejected"}:
            failures.append({"segment_id": segment_id, "reason": "decision_pending"})
            continue
        reviewer = str(row.get("reviewer", "")).strip()
        if not reviewer:
            failures.append({"segment_id": segment_id, "reason": "reviewer_missing"})
            continue
        if decision == "rejected":
            failures.append({"segment_id": segment_id, "reason": "candidate_rejected"})
            continue
        reviewed = load_json(source)
        reviewed["review_status"] = "reviewed_accepted"
        reviewed["review"] = {
            "decision": "accepted",
            "reviewer": reviewer,
            "reviewed_at": str(row.get("reviewed_at", "")).strip()
            or datetime.now(UTC).isoformat(),
            "notes": str(row.get("notes", "")).strip(),
            "source_candidate_path": str(source),
            "source_candidate_sha256": source_record["source_sha256"],
        }
        reviewed_dir = root / "reviewed_sources"
        reviewed_dir.mkdir(exist_ok=True)
        output = reviewed_dir / f"{segment_id}_rail_candidates.reviewed.json"
        write_json(output, reviewed)
        reviewed_sources.append(
            {"segment_id": segment_id, "path": str(output), "sha256": sha256_file(output)}
        )
    status = "accepted" if rows and not failures and len(reviewed_sources) == len(rows) else "blocked"
    result = {
        "schema_version": "railway.rail-source-review-result.v1",
        "project_id": project.project_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "passed": status == "accepted",
        "task_count": len(rows),
        "reviewed_source_count": len(reviewed_sources),
        "failures": failures,
        "reviewed_sources": reviewed_sources,
    }
    output_path = root / "review-result.json"
    write_json(output_path, result)
    return {**result, "output_report_path": str(output_path)}


def apply_rail_owner_override(
    project: ProjectConfig,
    package_value: str | Path,
    approved_by: str,
    reason: str,
) -> dict[str, Any]:
    """Accept hash-bound rail candidates through an explicit project-owner waiver."""
    approver = approved_by.strip()
    rationale = reason.strip()
    if not approver:
        raise ValueError("Owner override requires --approved-by")
    if not rationale:
        raise ValueError("Owner override requires --reason")

    root = Path(package_value).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "railway.rail-source-review-package.v1":
        raise ValueError("Unsupported rail review package schema")
    if manifest.get("project_id") != project.project_id:
        raise ValueError("Rail review package project_id does not match project")

    approved_at = datetime.now(UTC).isoformat()
    output_dir = root / "owner_override_sources"
    output_dir.mkdir(exist_ok=True)
    failures: list[dict[str, Any]] = []
    approved_sources: list[dict[str, Any]] = []
    for source_record in manifest.get("sources", []):
        segment_id = str(source_record.get("segment_id", ""))
        source = Path(str(source_record.get("source_report", ""))).resolve()
        expected_hash = str(source_record.get("source_sha256", ""))
        if not source.is_file() or sha256_file(source) != expected_hash:
            failures.append({"segment_id": segment_id, "reason": "source_hash_mismatch"})
            continue
        report = load_json(source)
        if report.get("segment_id") != segment_id:
            failures.append({"segment_id": segment_id, "reason": "source_segment_mismatch"})
            continue
        overridden = dict(report)
        overridden["review_status"] = "owner_override_accepted"
        overridden["review"] = {
            "decision": "accepted",
            "mode": "project_owner_override",
            "approved_by": approver,
            "approved_at": approved_at,
            "reason": rationale,
            "independent_review_completed": False,
            "source_candidate_path": str(source),
            "source_candidate_sha256": expected_hash,
        }
        output = output_dir / f"{segment_id}_rail_candidates.owner-override.json"
        write_json(output, overridden)
        approved_sources.append(
            {"segment_id": segment_id, "path": str(output), "sha256": sha256_file(output)}
        )

    status = (
        "accepted"
        if manifest.get("sources")
        and not failures
        and len(approved_sources) == len(manifest.get("sources", []))
        else "blocked"
    )
    result = {
        "schema_version": "railway.rail-source-owner-override-result.v1",
        "project_id": project.project_id,
        "created_at": approved_at,
        "status": status,
        "passed": status == "accepted",
        "review_mode": "project_owner_override",
        "independent_review_completed": False,
        "approved_by": approver,
        "reason": rationale,
        "source_count": len(manifest.get("sources", [])),
        "approved_source_count": len(approved_sources),
        "failures": failures,
        "approved_sources": approved_sources,
        "warning": "Accepted by project-owner waiver; this is not independent source review.",
    }
    output_path = root / "owner-override-result.json"
    write_json(output_path, result)
    return {**result, "output_report_path": str(output_path)}
