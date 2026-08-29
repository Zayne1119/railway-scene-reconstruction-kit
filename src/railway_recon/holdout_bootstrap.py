from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, sha256_file, write_json

MetricExtractor = Callable[[dict[str, Any]], float | None]


def _nested(*keys: str) -> MetricExtractor:
    def extract(segment: dict[str, Any]) -> float | None:
        value: Any = segment
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return None
        return float(value)

    return extract


METRICS: dict[str, tuple[MetricExtractor, bool]] = {
    "train_line_match_recall": (
        _nested("line_repeatability", "train_match_recall"),
        True,
    ),
    "holdout_line_match_precision": (
        _nested("line_repeatability", "holdout_match_precision"),
        True,
    ),
    "lateral_repeatability_p90_m": (
        _nested("line_repeatability", "lateral_difference", "p90_m"),
        False,
    ),
    "vertical_repeatability_p90_m": (
        _nested("line_repeatability", "vertical_difference", "p90_m"),
        False,
    ),
    "raw_gauge_repeatability_p90_m": (
        _nested("track_pair_repeatability", "raw_gauge_difference", "p90_m"),
        False,
    ),
    "model_to_holdout_support_p90_m": (
        _nested("model_to_holdout_support", "distance", "p90_m"),
        False,
    ),
    "model_to_holdout_coverage_at_0_05m": (
        _nested("model_to_holdout_support", "coverage_at_0_05m"),
        True,
    ),
    "model_to_holdout_coverage_at_0_10m": (
        _nested("model_to_holdout_support", "coverage_at_0_10m"),
        True,
    ),
}


def bootstrap_rail_holdout_comparison(
    baseline_path: str | Path,
    method_path: str | Path,
    output_path: str | Path,
    *,
    iterations: int = 10_000,
    seed: int = 20260827,
    minimum_confirmatory_blocks: int = 10,
) -> Path:
    if iterations < 100 or minimum_confirmatory_blocks < 2:
        raise ValueError("Bootstrap settings are too small")
    baseline_source = Path(baseline_path).resolve()
    method_source = Path(method_path).resolve()
    baseline = load_json(baseline_source)
    method = load_json(method_source)
    schema = "railway.rail-holdout-evaluation.v1"
    if baseline.get("schema_version") != schema or method.get("schema_version") != schema:
        raise ValueError("Unsupported rail holdout report")
    if baseline.get("holdout_manifest_sha256") != method.get("holdout_manifest_sha256"):
        raise ValueError("Bootstrap inputs do not share the same frozen holdout")
    baseline_segments = {item["segment_id"]: item for item in baseline["segments"]}
    method_segments = {item["segment_id"]: item for item in method["segments"]}
    if set(baseline_segments) != set(method_segments):
        raise ValueError("Bootstrap inputs do not share the same spatial blocks")
    segment_ids = sorted(baseline_segments)
    rng = np.random.default_rng(seed)
    sampled_indexes = rng.integers(0, len(segment_ids), size=(iterations, len(segment_ids)))

    metrics: dict[str, Any] = {}
    for metric_id, (extract, higher_is_better) in METRICS.items():
        paired: list[tuple[str, float, float]] = []
        for segment_id in segment_ids:
            baseline_value = extract(baseline_segments[segment_id])
            method_value = extract(method_segments[segment_id])
            if baseline_value is not None and method_value is not None:
                paired.append((segment_id, baseline_value, method_value))
        if not paired:
            metrics[metric_id] = {
                "paired_block_count": 0,
                "status": "not_comparable",
            }
            continue
        differences = np.asarray(
            [method_value - baseline_value for _, baseline_value, method_value in paired],
            dtype=np.float64,
        )
        metric_indexes = sampled_indexes[:, : len(paired)] % len(paired)
        bootstrap_means = np.mean(differences[metric_indexes], axis=1)
        better = bootstrap_means > 0 if higher_is_better else bootstrap_means < 0
        metrics[metric_id] = {
            "higher_is_better": higher_is_better,
            "paired_block_count": len(paired),
            "paired_segment_ids": [item[0] for item in paired],
            "observed_macro_mean_delta_method_minus_baseline": float(np.mean(differences)),
            "bootstrap_mean_delta_ci95": [
                float(np.percentile(bootstrap_means, 2.5)),
                float(np.percentile(bootstrap_means, 97.5)),
            ],
            "bootstrap_probability_method_better": float(np.mean(better)),
            "status": (
                "confirmatory"
                if len(paired) >= minimum_confirmatory_blocks
                else "exploratory_insufficient_independent_blocks"
            ),
        }

    block_count = len(segment_ids)
    report = {
        "schema_version": "railway.rail-holdout-block-bootstrap.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "paired_spatial_block_bootstrap",
        "inputs": {
            "baseline": str(baseline_source),
            "baseline_sha256": sha256_file(baseline_source),
            "method": str(method_source),
            "method_sha256": sha256_file(method_source),
            "holdout_manifest_sha256": method["holdout_manifest_sha256"],
        },
        "parameters": {
            "iterations": iterations,
            "seed": seed,
            "minimum_confirmatory_blocks": minimum_confirmatory_blocks,
            "resampling_unit": "whole_50m_segment",
        },
        "independent_block_count": block_count,
        "overall_status": (
            "confirmatory"
            if block_count >= minimum_confirmatory_blocks
            else "exploratory_insufficient_independent_blocks"
        ),
        "metrics": metrics,
        "limitations": [
            "Only whole spatial segments are resampled; individual points are never treated as independent.",
            "Four blocks are insufficient for a confirmatory confidence claim.",
            "Intervals quantify within-site segment variation, not cross-site generalization.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bootstrap report: {output}")
    write_json(output, report)
    return output
