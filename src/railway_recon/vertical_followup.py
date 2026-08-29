from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json


def _metric_values(report: dict[str, Any]) -> dict[str, float]:
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


def audit_rail_vertical_followup(
    baseline_path: str | Path,
    initial_selection_path: str | Path,
    trigger_followup_path: str | Path,
    nested_correction_path: str | Path,
    final_followup_path: str | Path,
    corrected_settings_path: str | Path,
    output_path: str | Path,
) -> Path:
    sources = {
        "baseline": Path(baseline_path).resolve(),
        "initial_selection": Path(initial_selection_path).resolve(),
        "trigger_followup": Path(trigger_followup_path).resolve(),
        "nested_correction": Path(nested_correction_path).resolve(),
        "final_followup": Path(final_followup_path).resolve(),
        "corrected_settings": Path(corrected_settings_path).resolve(),
    }
    values = {name: load_json(path) for name, path in sources.items()}
    evaluation_schema = "railway.rail-holdout-evaluation.v1"
    for name in ("baseline", "trigger_followup", "nested_correction", "final_followup"):
        if values[name].get("schema_version") != evaluation_schema:
            raise ValueError(f"Unsupported evaluation report: {name}")
    if (
        values["initial_selection"].get("schema_version")
        != "railway.rail-vertical-fit-selection.v1"
    ):
        raise ValueError("Unsupported initial vertical-fit selection")
    outer_hashes = {
        values[name]["holdout_manifest_sha256"]
        for name in ("baseline", "trigger_followup", "final_followup")
    }
    if len(outer_hashes) != 1:
        raise ValueError("Outer vertical follow-ups do not share the baseline holdout")
    if values["nested_correction"]["holdout_manifest_sha256"] in outer_hashes:
        raise ValueError("Nested correction validation unexpectedly uses the outer holdout")

    baseline = _metric_values(values["baseline"])
    final = _metric_values(values["final_followup"])
    trigger = _metric_values(values["trigger_followup"])
    metrics: dict[str, Any] = {}
    for metric_id, baseline_value in baseline.items():
        final_value = final[metric_id]
        metrics[metric_id] = {
            "baseline": baseline_value,
            "final": final_value,
            "delta_final_minus_baseline": final_value - baseline_value,
            "relative_change": (
                (final_value - baseline_value) / abs(baseline_value)
                if abs(baseline_value) > 1e-12
                else None
            ),
        }

    guardrails = {
        "train_recall_not_lower": (
            final["train_line_match_recall"] >= baseline["train_line_match_recall"]
        ),
        "holdout_precision_not_lower": (
            final["holdout_line_match_precision"]
            >= baseline["holdout_line_match_precision"]
        ),
        "lateral_p90_not_worse": (
            final["lateral_repeatability_p90_m"]
            <= baseline["lateral_repeatability_p90_m"] + 1e-12
        ),
        "gauge_p90_not_worse": (
            final["raw_gauge_repeatability_p90_m"]
            <= baseline["raw_gauge_repeatability_p90_m"] + 1e-12
        ),
        "vertical_p90_improved": (
            final["vertical_repeatability_p90_m"]
            < baseline["vertical_repeatability_p90_m"]
        ),
        "support_p90_improved": (
            final["model_to_holdout_support_p90_m"]
            < baseline["model_to_holdout_support_p90_m"]
        ),
        "coverage_10cm_not_lower": (
            final["model_to_holdout_coverage_at_0_10m"]
            >= baseline["model_to_holdout_coverage_at_0_10m"]
        ),
        "trigger_precision_regression_removed": (
            trigger["holdout_line_match_precision"]
            < final["holdout_line_match_precision"]
        ),
    }
    report = {
        "schema_version": "railway.rail-vertical-development-audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "passes_development_guardrails"
            if all(guardrails.values())
            else "fails_development_guardrails"
        ),
        "evaluation_scope": "single_site_post_holdout_development_exploratory",
        "chronology": [
            "A nested outer-train validation selected anchored q50 for rail-top height.",
            "The first outer follow-up improved height but changed pair eligibility and precision.",
            "Pair eligibility was decoupled from the selected height estimator.",
            "The correction retained nested height results and restored outer topology metrics.",
        ],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "outer_holdout_manifest_sha256": next(iter(outer_hashes)),
        "nested_holdout_manifest_sha256": values["nested_correction"][
            "holdout_manifest_sha256"
        ],
        "initial_selected_variant": values["initial_selection"]["selected_variant"],
        "correction": {
            "rail_top_fit_method": values["corrected_settings"].get(
                "rail_top_fit_method"
            ),
            "rail_pair_top_method": values["corrected_settings"].get(
                "rail_pair_top_method"
            ),
        },
        "metrics": metrics,
        "guardrails": guardrails,
        "limitations": [
            "The outer holdout was already observed when the decoupling correction was designed.",
            "This audit supports engineering adoption on this site, not a new confirmatory claim.",
            "Cross-site validation remains required before making the corrected estimator the global default.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite vertical development audit: {output}")
    write_json(output, report)
    return output
