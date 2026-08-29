from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


def _metric(
    baseline: dict[str, Any],
    method: dict[str, Any],
    key: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    baseline_value = baseline.get(key)
    method_value = method.get(key)
    if baseline_value is None or method_value is None:
        return {
            "baseline": baseline_value,
            "method": method_value,
            "higher_is_better": higher_is_better,
            "winner": "not_comparable",
            "absolute_delta_method_minus_baseline": None,
            "relative_change": None,
        }
    baseline_number = float(baseline_value)
    method_number = float(method_value)
    if abs(method_number - baseline_number) <= 1e-12:
        winner = "tie"
    elif (method_number > baseline_number) == higher_is_better:
        winner = "method"
    else:
        winner = "baseline"
    return {
        "baseline": baseline_number,
        "method": method_number,
        "higher_is_better": higher_is_better,
        "winner": winner,
        "absolute_delta_method_minus_baseline": method_number - baseline_number,
        "relative_change": (
            (method_number - baseline_number) / abs(baseline_number)
            if abs(baseline_number) > 1e-12
            else None
        ),
    }


def compare_rail_holdout_reports(
    baseline_path: str | Path,
    method_path: str | Path,
    output_path: str | Path,
    *,
    baseline_name: str = "baseline",
    method_name: str = "method",
) -> Path:
    baseline_source = Path(baseline_path).resolve()
    method_source = Path(method_path).resolve()
    baseline_report = load_json(baseline_source)
    method_report = load_json(method_source)
    expected_schema = "railway.rail-holdout-evaluation.v1"
    if baseline_report.get("schema_version") != expected_schema:
        raise ValueError("Unsupported baseline holdout report")
    if method_report.get("schema_version") != expected_schema:
        raise ValueError("Unsupported method holdout report")
    if baseline_report.get("holdout_manifest_sha256") != method_report.get(
        "holdout_manifest_sha256"
    ):
        raise ValueError("Methods were not evaluated on the same frozen holdout manifest")
    for key in ("core_length_m", "sample_step_m", "holdout_seed", "voxel_size_m"):
        if baseline_report["parameters"].get(key) != method_report["parameters"].get(key):
            raise ValueError(f"Holdout evaluation parameter differs: {key}")

    baseline = baseline_report["summary"]
    method = method_report["summary"]
    metrics = {
        "train_line_match_recall": _metric(
            baseline, method, "train_line_match_recall", higher_is_better=True
        ),
        "holdout_line_match_precision": _metric(
            baseline, method, "holdout_line_match_precision", higher_is_better=True
        ),
        "lateral_repeatability_p90_m": _metric(
            {"value": baseline["line_lateral_repeatability"]["p90_m"]},
            {"value": method["line_lateral_repeatability"]["p90_m"]},
            "value",
            higher_is_better=False,
        ),
        "vertical_repeatability_p90_m": _metric(
            {"value": baseline["line_vertical_repeatability"]["p90_m"]},
            {"value": method["line_vertical_repeatability"]["p90_m"]},
            "value",
            higher_is_better=False,
        ),
        "raw_gauge_repeatability_p90_m": _metric(
            {"value": baseline["raw_gauge_repeatability"]["p90_m"]},
            {"value": method["raw_gauge_repeatability"]["p90_m"]},
            "value",
            higher_is_better=False,
        ),
        "model_to_holdout_support_p90_m": _metric(
            {"value": baseline["model_to_holdout_support_distance"]["p90_m"]},
            {"value": method["model_to_holdout_support_distance"]["p90_m"]},
            "value",
            higher_is_better=False,
        ),
        "model_to_holdout_coverage_at_0_05m": _metric(
            baseline,
            method,
            "model_to_holdout_coverage_at_0_05m",
            higher_is_better=True,
        ),
        "model_to_holdout_coverage_at_0_10m": _metric(
            baseline,
            method,
            "model_to_holdout_coverage_at_0_10m",
            higher_is_better=True,
        ),
    }
    wins = {
        name: sum(item["winner"] == name for item in metrics.values())
        for name in ("method", "baseline", "tie", "not_comparable")
    }
    report = {
        "schema_version": "railway.rail-holdout-comparison.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "annotation_free_paired_holdout_method_comparison",
        "baseline": {
            "name": baseline_name,
            "report": str(baseline_source),
            "report_sha256": sha256_file(baseline_source),
        },
        "method": {
            "name": method_name,
            "report": str(method_source),
            "report_sha256": sha256_file(method_source),
        },
        "frozen_holdout_manifest_sha256": method_report["holdout_manifest_sha256"],
        "metrics": metrics,
        "win_counts": wins,
        "interpretation": {
            "method_strengths": [name for name, item in metrics.items() if item["winner"] == "method"],
            "baseline_strengths": [
                name for name, item in metrics.items() if item["winner"] == "baseline"
            ],
        },
        "limitations": [
            "This compares split repeatability and held-out raw-point support, not semantic ground truth.",
            "The Open3D baseline was tuned on the train partition before holdout evaluation.",
            "A method win is not an absolute survey-accuracy claim.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite holdout comparison: {output}")
    write_json(output, report)
    return output
