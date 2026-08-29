from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    summary = report["summary"]
    return {
        "train_line_match_recall": float(summary["train_line_match_recall"]),
        "holdout_line_match_precision": float(summary["holdout_line_match_precision"]),
        "lateral_repeatability_p90_m": float(
            summary["line_lateral_repeatability"]["p90_m"]
        ),
        "vertical_repeatability_p90_m": float(
            summary["line_vertical_repeatability"]["p90_m"]
        ),
        "raw_gauge_repeatability_p90_m": float(
            summary["raw_gauge_repeatability"]["p90_m"]
        ),
        "model_to_holdout_support_p90_m": float(
            summary["model_to_holdout_support_distance"]["p90_m"]
        ),
        "model_to_holdout_coverage_at_0_10m": float(
            summary["model_to_holdout_coverage_at_0_10m"]
        ),
    }


def select_rail_vertical_fit(
    variants: list[tuple[str, str | Path, str | Path]],
    baseline_name: str,
    output_path: str | Path,
) -> Path:
    """Select a vertical fit using only a nested outer-train validation split."""
    if len(variants) < 2:
        raise ValueError("At least two vertical-fit variants are required")
    if len({name for name, _, _ in variants}) != len(variants):
        raise ValueError("Vertical-fit variant names must be unique")

    loaded: dict[str, dict[str, Any]] = {}
    common_manifest_hash: str | None = None
    common_parameters: dict[str, Any] | None = None
    for name, evaluation_value, settings_value in variants:
        evaluation_path = Path(evaluation_value).resolve()
        settings_path = Path(settings_value).resolve()
        evaluation = load_json(evaluation_path)
        settings = load_json(settings_path)
        if evaluation.get("schema_version") != "railway.rail-holdout-evaluation.v1":
            raise ValueError(f"Unsupported holdout report for {name}")
        manifest_hash = str(evaluation["holdout_manifest_sha256"])
        comparison_parameters = {
            key: evaluation["parameters"].get(key)
            for key in (
                "maximum_match_distance_m",
                "core_length_m",
                "sample_step_m",
                "voxel_size_m",
                "holdout_fraction",
                "holdout_seed",
            )
        }
        if common_manifest_hash is None:
            common_manifest_hash = manifest_hash
            common_parameters = comparison_parameters
        elif manifest_hash != common_manifest_hash:
            raise ValueError("Vertical-fit variants do not share one nested holdout")
        elif comparison_parameters != common_parameters:
            raise ValueError("Vertical-fit evaluation parameters differ")
        loaded[name] = {
            "evaluation_path": evaluation_path,
            "settings_path": settings_path,
            "evaluation_sha256": sha256_file(evaluation_path),
            "settings_sha256": sha256_file(settings_path),
            "settings": settings,
            "metrics": _metrics(evaluation),
        }

    if baseline_name not in loaded:
        raise ValueError(f"Baseline variant is missing: {baseline_name}")
    baseline = loaded[baseline_name]["metrics"]
    tolerances = {
        "train_line_match_recall_minimum": baseline["train_line_match_recall"] - 0.05,
        "holdout_line_match_precision_minimum": (
            baseline["holdout_line_match_precision"] - 0.05
        ),
        "lateral_repeatability_p90_m_maximum": (
            baseline["lateral_repeatability_p90_m"] + 0.02
        ),
        "raw_gauge_repeatability_p90_m_maximum": (
            baseline["raw_gauge_repeatability_p90_m"] + 0.02
        ),
        "model_to_holdout_support_p90_m_maximum": (
            baseline["model_to_holdout_support_p90_m"] + 0.05
        ),
        "model_to_holdout_coverage_at_0_10m_minimum": (
            baseline["model_to_holdout_coverage_at_0_10m"] - 0.05
        ),
    }

    candidates: dict[str, Any] = {}
    for name, item in loaded.items():
        metrics = item["metrics"]
        guardrails = {
            "train_line_match_recall": (
                metrics["train_line_match_recall"]
                >= tolerances["train_line_match_recall_minimum"]
            ),
            "holdout_line_match_precision": (
                metrics["holdout_line_match_precision"]
                >= tolerances["holdout_line_match_precision_minimum"]
            ),
            "lateral_repeatability_p90_m": (
                metrics["lateral_repeatability_p90_m"]
                <= tolerances["lateral_repeatability_p90_m_maximum"]
            ),
            "raw_gauge_repeatability_p90_m": (
                metrics["raw_gauge_repeatability_p90_m"]
                <= tolerances["raw_gauge_repeatability_p90_m_maximum"]
            ),
            "model_to_holdout_support_p90_m": (
                metrics["model_to_holdout_support_p90_m"]
                <= tolerances["model_to_holdout_support_p90_m_maximum"]
            ),
            "model_to_holdout_coverage_at_0_10m": (
                metrics["model_to_holdout_coverage_at_0_10m"]
                >= tolerances["model_to_holdout_coverage_at_0_10m_minimum"]
            ),
        }
        candidates[name] = {
            "evaluation": str(item["evaluation_path"]),
            "evaluation_sha256": item["evaluation_sha256"],
            "settings": str(item["settings_path"]),
            "settings_sha256": item["settings_sha256"],
            "metrics": metrics,
            "guardrails": guardrails,
            "passes_all_guardrails": all(guardrails.values()),
        }

    eligible = [name for name, item in candidates.items() if item["passes_all_guardrails"]]
    if not eligible:
        raise ValueError("No vertical-fit variant passes the predefined guardrails")
    selected_name = min(
        eligible,
        key=lambda name: (
            candidates[name]["metrics"]["vertical_repeatability_p90_m"],
            name,
        ),
    )
    report = {
        "schema_version": "railway.rail-vertical-fit-selection.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_scope": "nested_validation_within_outer_train_only",
        "baseline_variant": baseline_name,
        "nested_holdout_manifest_sha256": common_manifest_hash,
        "evaluation_parameters": common_parameters,
        "selection_rule": {
            "primary_metric": "vertical_repeatability_p90_m",
            "direction": "minimize",
            "guardrail_thresholds": tolerances,
            "tie_break": "lexicographic_variant_name",
        },
        "variants": candidates,
        "selected_variant": selected_name,
        "selected_settings": candidates[selected_name]["settings"],
        "selected_settings_sha256": candidates[selected_name]["settings_sha256"],
        "limitations": [
            "Selection uses only a nested validation split made from the outer training clouds.",
            "No manual labels or outer held-out points are used for variant selection.",
            "A later outer-holdout rerun is exploratory because that holdout has been observed before.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite vertical-fit selection: {output}")
    write_json(output, report)
    return output
