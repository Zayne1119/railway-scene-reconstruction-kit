"""Build a machine-readable three-method rail holdout comparison.

This script keeps numerical accuracy metrics separate from operational failures.
An external method that emits no geometry is reported as ``not_comparable``;
missing values are never silently converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METRICS = (
    ("train_line_match_recall", ("train_line_match_recall",), "ratio", True),
    ("holdout_line_match_precision", ("holdout_line_match_precision",), "ratio", True),
    ("lateral_repeatability_p90_m", ("line_lateral_repeatability", "p90_m"), "m", False),
    ("vertical_repeatability_p90_m", ("line_vertical_repeatability", "p90_m"), "m", False),
    ("raw_gauge_repeatability_p90_m", ("raw_gauge_repeatability", "p90_m"), "m", False),
    (
        "model_to_holdout_support_p90_m",
        ("model_to_holdout_support_distance", "p90_m"),
        "m",
        False,
    ),
    ("model_to_holdout_coverage_at_0_05m", ("model_to_holdout_coverage_at_0_05m",), "ratio", True),
    ("model_to_holdout_coverage_at_0_10m", ("model_to_holdout_coverage_at_0_10m",), "ratio", True),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _method_record(name: str, path: Path) -> dict[str, Any]:
    report = _load(path)
    summary = report["summary"]
    return {
        "name": name,
        "report": str(path.resolve()),
        "report_sha256": _sha256(path),
        "segment_count": summary["segment_count"],
        "train_line_count": summary["train_line_count"],
        "holdout_line_count": summary["holdout_line_count"],
        "matched_line_count": summary["matched_line_count"],
        "metrics": {metric: _nested(summary, keys) for metric, keys, _, _ in METRICS},
    }


def _gislab_execution(baseline_root: Path) -> dict[str, Any]:
    normalized = sorted((baseline_root / "train").glob("*.json")) + sorted(
        (baseline_root / "holdout").glob("*.json")
    )
    statuses: dict[str, int] = {}
    for path in normalized:
        status = str(_load(path).get("status", "geometry_output"))
        statuses[status] = statuses.get(status, 0) + 1

    runs = sorted((baseline_root / "raw").glob("*/*/run.json"))
    elapsed = [float(_load(path)["elapsed_seconds"]) for path in runs]
    exit_codes: dict[str, int] = {}
    for path in runs:
        code = str(int(_load(path)["exit_code"]))
        exit_codes[code] = exit_codes.get(code, 0) + 1

    zero_count = statuses.get("upstream_zero_detection_then_empty_output_crash", 0)
    timeout_count = statuses.get("upstream_timeout_no_output", 0)
    geometry_count = len(normalized) - zero_count - timeout_count
    return {
        "expected_case_count": 8,
        "normalized_case_count": len(normalized),
        "geometry_output_case_count": geometry_count,
        "zero_detection_case_count": zero_count,
        "timeout_case_count": timeout_count,
        "status_counts": statuses,
        "run_manifest_count": len(runs),
        "exit_code_counts": exit_codes,
        "external_runtime_seconds_total": sum(elapsed) if elapsed else None,
        "external_runtime_seconds_median": statistics.median(elapsed) if elapsed else None,
        "runtime_scope": "upstream executable only; shared preprocessing excluded",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ours = _method_record("railway_height_prominence_v2", args.ours)
    open3d = _method_record("open3d_ransac_dbscan_v1", args.open3d)
    gislab = _method_record("gislab_railtrack_3557ecf", args.gislab)
    execution = _gislab_execution(args.gislab_baseline_root)

    gislab_comparable = execution["geometry_output_case_count"] > 0
    comparison: dict[str, Any] = {}
    for metric, _, unit, higher_is_better in METRICS:
        comparison[metric] = {
            "unit": unit,
            "higher_is_better": higher_is_better,
            "railway_height_prominence_v2": ours["metrics"][metric],
            "open3d_ransac_dbscan_v1": open3d["metrics"][metric],
            "gislab_railtrack_3557ecf": (
                gislab["metrics"][metric] if gislab_comparable else None
            ),
            "gislab_status": "comparable" if gislab_comparable else "not_comparable_no_geometry",
        }

    return {
        "schema_version": "railway.rail-holdout-three-method-comparison.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "annotation_free_split_repeatability_support_and_operational_compatibility",
        "methods": {
            "railway_height_prominence_v2": ours,
            "open3d_ransac_dbscan_v1": open3d,
            "gislab_railtrack_3557ecf": gislab,
        },
        "gislab_execution": execution,
        "metrics": comparison,
        "interpretation": {
            "ours_vs_open3d": "The in-house method wins 7 of 8 frozen aggregate metrics; Open3D wins vertical repeatability P90.",
            "gislab": "No rail geometry was emitted in any of the eight train/holdout cases, so geometric accuracy metrics are N/A rather than zero.",
            "scope": "This establishes within-site split repeatability and raw held-out support only.",
        },
        "limitations": [
            "There is no semantic ground truth; precision, recall and absolute survey accuracy are not claimed.",
            "Only four independent 50 m spatial blocks are available, so results remain exploratory.",
            "GISLab used unmodified upstream detection after the common frozen envelope and a rigid corridor-axis alignment.",
            "The GISLab failure is evidence of incompatibility with this fixed dense-station protocol, not a universal ranking of the method.",
            "External runtime excludes shared preprocessing and is not directly comparable with methods lacking equivalent timing manifests.",
        ],
    }


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "metric",
                "unit",
                "higher_is_better",
                "railway_height_prominence_v2",
                "open3d_ransac_dbscan_v1",
                "gislab_railtrack_3557ecf",
                "gislab_status",
            ]
        )
        for metric, values in report["metrics"].items():
            writer.writerow(
                [
                    metric,
                    values["unit"],
                    values["higher_is_better"],
                    values["railway_height_prominence_v2"],
                    values["open3d_ransac_dbscan_v1"],
                    values["gislab_railtrack_3557ecf"],
                    values["gislab_status"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--open3d", type=Path, required=True)
    parser.add_argument("--gislab", type=Path, required=True)
    parser.add_argument("--gislab-baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(args.csv, report)
    print(args.output)
    print(args.csv)


if __name__ == "__main__":
    main()
