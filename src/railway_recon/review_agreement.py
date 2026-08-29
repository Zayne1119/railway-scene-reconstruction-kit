from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .annotation_tasks import FEATURE_COLUMNS, LABEL_COLUMNS
from .io import load_json, sha256_file, write_json

OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
DECISION_COLUMNS = ["existence", "asset_type", "geometry_evaluable", "evidence_state"]
ALLOWED_VALUES = {
    "existence": {"present", "absent", "uncertain"},
    "asset_type": {
        "catenary_mast",
        "canopy_column",
        "building_edge",
        "signal_or_lineside_pole",
        "other_asset",
        "false_positive",
        "uncertain",
    },
    "geometry_evaluable": {"yes", "no"},
    "evidence_state": {"observed", "weak", "inferred", "unsupported"},
}
ADJUDICATION_COLUMNS = [
    "adjudicated_existence",
    "adjudicated_asset_type",
    "adjudicated_geometry_evaluable",
    "adjudicated_evidence_state",
    "adjudicator_confidence",
    "adjudication_reason",
]


def _read_review(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        task_id = row.get("task_id", "").strip()
        if not task_id:
            raise ValueError(f"Missing task_id in {path} row {row_number}")
        if task_id in by_id:
            raise ValueError(f"Duplicate task_id {task_id} in {path}")
        for column, allowed in ALLOWED_VALUES.items():
            value = row.get(column, "").strip()
            if value not in allowed:
                raise ValueError(
                    f"Invalid or incomplete {column}={value!r} in {path} row {row_number}"
                )
        confidence = float(row.get("reviewer_confidence", ""))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"reviewer_confidence outside [0, 1] in {path} row {row_number}")
        by_id[task_id] = row
    return rows, by_id


def _cohen_kappa(values_a: list[str], values_b: list[str]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("Cohen's kappa requires two equally sized non-empty label lists")
    total = len(values_a)
    observed = sum(left == right for left, right in zip(values_a, values_b, strict=True)) / total
    counts_a = Counter(values_a)
    counts_b = Counter(values_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[item] / total) * (counts_b[item] / total) for item in categories)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _prefixed_labels(prefix: str, row: dict[str, str]) -> dict[str, str]:
    return {f"{prefix}_{column}": row.get(column, "") for column in LABEL_COLUMNS}


def compare_vertical_reviews(
    annotation_package: str | Path,
    output_name: str = "vertical_review_comparison_v1",
) -> Path:
    package_root = Path(annotation_package).resolve()
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("output_name must use lowercase letters, digits, '-' or '_'")
    manifest = load_json(package_root / "manifest.json")
    reviewer_files = manifest.get("reviewer_files", [])
    if len(reviewer_files) != 2:
        raise ValueError("Review comparison requires exactly two reviewer files")
    reviewer_paths = [package_root / name for name in reviewer_files]
    rows_a, by_id_a = _read_review(reviewer_paths[0])
    _, by_id_b = _read_review(reviewer_paths[1])
    if set(by_id_a) != set(by_id_b):
        missing_a = sorted(set(by_id_b) - set(by_id_a))
        missing_b = sorted(set(by_id_a) - set(by_id_b))
        raise ValueError(
            f"Reviewer task mismatch: missing_a={len(missing_a)}, missing_b={len(missing_b)}"
        )

    output_root = package_root / "review_comparisons" / output_name
    if output_root.exists():
        raise FileExistsError(f"Review comparison already exists: {output_root}")
    output_root.mkdir(parents=True)

    ordered_ids = sorted(by_id_a)
    column_metrics: dict[str, Any] = {}
    for column in DECISION_COLUMNS:
        values_a = [by_id_a[task_id][column] for task_id in ordered_ids]
        values_b = [by_id_b[task_id][column] for task_id in ordered_ids]
        agreement = sum(a == b for a, b in zip(values_a, values_b, strict=True)) / len(ordered_ids)
        column_metrics[column] = {
            "raw_agreement": agreement,
            "cohen_kappa": _cohen_kappa(values_a, values_b),
        }

    disagreements: list[dict[str, str]] = []
    consensus: list[dict[str, str]] = []
    for task_id in ordered_ids:
        row_a = by_id_a[task_id]
        row_b = by_id_b[task_id]
        decisions_match = all(row_a[column] == row_b[column] for column in DECISION_COLUMNS)
        base = {column: row_a.get(column, "") for column in FEATURE_COLUMNS}
        comparison = {
            **base,
            **_prefixed_labels("reviewer_a", row_a),
            **_prefixed_labels("reviewer_b", row_b),
        }
        if decisions_match:
            consensus.append(
                {
                    **base,
                    **{column: row_a[column] for column in DECISION_COLUMNS},
                    "consensus_confidence": str(
                        min(float(row_a["reviewer_confidence"]), float(row_b["reviewer_confidence"]))
                    ),
                    "consensus_source": "independent_reviewer_agreement",
                }
            )
        else:
            disagreements.append(
                {**comparison, **{column: "" for column in ADJUDICATION_COLUMNS}}
            )

    disagreement_columns = [
        *FEATURE_COLUMNS,
        *[f"reviewer_a_{column}" for column in LABEL_COLUMNS],
        *[f"reviewer_b_{column}" for column in LABEL_COLUMNS],
        *ADJUDICATION_COLUMNS,
    ]
    with (output_root / "disagreements.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=disagreement_columns)
        writer.writeheader()
        writer.writerows(disagreements)

    consensus_columns = [
        *FEATURE_COLUMNS,
        *DECISION_COLUMNS,
        "consensus_confidence",
        "consensus_source",
    ]
    with (output_root / "consensus_seed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=consensus_columns)
        writer.writeheader()
        writer.writerows(consensus)

    joint_agreement = len(consensus) / len(ordered_ids)
    report = {
        "schema_version": "railway.review-agreement.v1",
        "comparison_id": output_name,
        "created_at": datetime.now(UTC).isoformat(),
        "annotation_package_id": manifest["package_id"],
        "annotation_package_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "reviewers": [
            {"file": reviewer_files[index], "sha256": sha256_file(path)}
            for index, path in enumerate(reviewer_paths)
        ],
        "task_count": len(rows_a),
        "joint_decision_agreement": joint_agreement,
        "consensus_count": len(consensus),
        "disagreement_count": len(disagreements),
        "per_column": column_metrics,
        "outputs": {
            "consensus_seed": "consensus_seed.csv",
            "disagreements": "disagreements.csv",
        },
        "ground_truth_status": "requires_adjudication" if disagreements else "consensus_complete",
    }
    write_json(output_root / "agreement-report.json", report)
    return output_root
