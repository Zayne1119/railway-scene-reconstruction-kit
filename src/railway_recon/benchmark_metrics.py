from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .benchmark import validate_benchmark_file, validate_benchmark_value
from .io import write_json


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _distance_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mae_m": None, "rmse_m": None, "p50_m": None, "p90_m": None, "p95_m": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mae_m": float(np.mean(np.abs(array))),
        "rmse_m": float(np.sqrt(np.mean(np.square(array)))),
        "p50_m": float(np.percentile(array, 50)),
        "p90_m": float(np.percentile(array, 90)),
        "p95_m": float(np.percentile(array, 95)),
    }


def _geometry_metrics(distance_sets: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in distance_sets:
        reference_to_model = [float(value) for value in item["reference_to_model_m"]]
        model_to_reference = [float(value) for value in item["model_to_reference_m"]]
        thresholds: dict[str, Any] = {}
        for threshold in item.get("thresholds_m", []):
            threshold = float(threshold)
            completeness = _safe_div(
                sum(value <= threshold for value in reference_to_model), len(reference_to_model)
            )
            precision = _safe_div(
                sum(value <= threshold for value in model_to_reference), len(model_to_reference)
            )
            fscore = None
            if completeness is not None and precision is not None and completeness + precision:
                fscore = 2.0 * completeness * precision / (completeness + precision)
            thresholds[f"{threshold:.6g}"] = {
                "threshold_m": threshold,
                "completeness": completeness,
                "precision": precision,
                "fscore": fscore,
            }
        result[item["name"]] = {
            "reference_to_model": _distance_statistics(reference_to_model),
            "model_to_reference": _distance_statistics(model_to_reference),
            "thresholds": thresholds,
        }
    return result


def _asset_metrics(assets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[tuple[float, int]]]:
    evaluable = [
        item
        for item in assets
        if item["evaluable"] and item["ground_truth"]["existence"] != "unknown"
    ]
    detection_tp = detection_fp = detection_fn = 0
    matched_present = 0
    matched_type_correct = 0
    confidence_pairs: list[tuple[float, int]] = []
    labels: set[str] = set()
    unsupported_present = 0
    predicted_present = 0
    for item in evaluable:
        truth = item["ground_truth"]
        prediction = item["prediction"]
        truth_present = truth["existence"] == "present"
        prediction_present = prediction["existence"] == "present"
        truth_type = truth.get("type") if truth_present else None
        prediction_type = prediction.get("type") if prediction_present else None
        labels.update(value for value in (truth_type, prediction_type) if value)
        if truth_present and prediction_present:
            detection_tp += 1
            matched_present += 1
            matched_type_correct += int(truth_type == prediction_type)
        elif not truth_present and prediction_present:
            detection_fp += 1
        elif truth_present and not prediction_present:
            detection_fn += 1
        if prediction_present:
            predicted_present += 1
            unsupported_present += int(prediction.get("evidence_level") == "unsupported")
        correct = int(
            (truth_present == prediction_present)
            and (not truth_present or truth_type == prediction_type)
        )
        confidence_pairs.append((float(prediction["confidence"]), correct))

    per_class: dict[str, Any] = {}
    for label in sorted(labels):
        true_positive = false_positive = false_negative = 0
        for item in evaluable:
            truth = item["ground_truth"]
            prediction = item["prediction"]
            truth_label = truth.get("type") if truth["existence"] == "present" else None
            prediction_label = (
                prediction.get("type") if prediction["existence"] == "present" else None
            )
            true_positive += int(truth_label == label and prediction_label == label)
            false_positive += int(truth_label != label and prediction_label == label)
            false_negative += int(truth_label == label and prediction_label != label)
        per_class[label] = _prf(true_positive, false_positive, false_negative)
    class_f1 = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    result = {
        "evaluable_count": len(evaluable),
        "detection": _prf(detection_tp, detection_fp, detection_fn),
        "classification_accuracy_on_matched": _safe_div(matched_type_correct, matched_present),
        "macro_f1": float(np.mean(class_f1)) if class_f1 else None,
        "per_class": per_class,
        "unsupported_hallucination_rate": _safe_div(unsupported_present, predicted_present),
    }
    return result, confidence_pairs


def _topology_metrics(
    ground_truth_relations: list[dict[str, Any]], predicted_relations: list[dict[str, Any]]
) -> dict[str, Any]:
    def key(item: dict[str, Any]) -> tuple[str, str, str]:
        return str(item["type"]), str(item["from"]), str(item["to"])

    truth = {key(item) for item in ground_truth_relations}
    prediction = {key(item) for item in predicted_relations}
    return _prf(len(truth & prediction), len(prediction - truth), len(truth - prediction))


def _confidence_metrics(pairs: list[tuple[float, int]], bin_count: int = 10) -> dict[str, Any]:
    if not pairs:
        return {"count": 0, "brier": None, "ece": None, "aurc": None, "bins": []}
    confidence = np.asarray([item[0] for item in pairs], dtype=np.float64)
    correctness = np.asarray([item[1] for item in pairs], dtype=np.float64)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        mask = (confidence >= lower) & (
            confidence <= upper if index == bin_count - 1 else confidence < upper
        )
        count = int(np.sum(mask))
        if not count:
            continue
        mean_confidence = float(np.mean(confidence[mask]))
        accuracy = float(np.mean(correctness[mask]))
        ece += count / len(pairs) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    order = np.argsort(-confidence)
    ordered_correctness = correctness[order]
    cumulative_risk = 1.0 - np.cumsum(ordered_correctness) / np.arange(1, len(pairs) + 1)
    return {
        "count": len(pairs),
        "brier": float(np.mean(np.square(confidence - correctness))),
        "ece": float(ece),
        "aurc": float(np.mean(cumulative_risk)),
        "risk_at_50pct_coverage": float(cumulative_risk[max(0, math.ceil(len(pairs) * 0.5) - 1)]),
        "bins": bins,
    }


def _human_metrics(value: dict[str, Any]) -> dict[str, Any]:
    events = value.get("events", [])
    total_seconds = float(sum(float(item["duration_seconds"]) for item in events))
    total_minutes = total_seconds / 60.0
    errors_found = sum(bool(item.get("error_found")) for item in events)
    corridor_length_m = float(value.get("corridor_length_m", 0.0))
    return {
        "event_count": len(events),
        "total_minutes": total_minutes,
        "minutes_per_100m": (
            total_minutes * 100.0 / corridor_length_m if corridor_length_m > 0 else None
        ),
        "errors_found": errors_found,
        "errors_per_minute": _safe_div(errors_found, total_minutes),
    }


def _evaluate_v1(value: dict[str, Any]) -> dict[str, Any]:
    errors = validate_benchmark_value(value)
    if errors:
        raise ValueError("Invalid benchmark evaluation input:\n- " + "\n- ".join(errors))
    asset_metrics, confidence_pairs = _asset_metrics(value["assets"])
    report = {
        "schema_version": "railway.benchmark.metric-report.v1",
        "report_id": value["evaluation_id"],
        "experiment_id": value["experiment_id"],
        "evaluator_version": "paper-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "instances": asset_metrics,
            "topology": _topology_metrics(
                value["ground_truth_relations"], value["predicted_relations"]
            ),
            "confidence": _confidence_metrics(confidence_pairs),
            "geometry": _geometry_metrics(value["distance_sets"]),
            "human": _human_metrics(value["human_review"]),
        },
        "sample_counts": {
            "assets": len(value["assets"]),
            "ground_truth_relations": len(value["ground_truth_relations"]),
            "predicted_relations": len(value["predicted_relations"]),
            "distance_sets": len(value["distance_sets"]),
        },
        "limitations": list(value.get("limitations", [])),
    }
    report_errors = validate_benchmark_value(report)
    if report_errors:
        raise ValueError("Generated metric report is invalid:\n- " + "\n- ".join(report_errors))
    return report


def _asset_matching_v2(
    ground_truth_assets: list[dict[str, Any]],
    predicted_assets: list[dict[str, Any]],
    matching: dict[str, Any],
    ignored_prediction_ids: set[str],
) -> dict[str, Any]:
    truth = [item for item in ground_truth_assets if bool(item["evaluable"])]
    predictions = [
        item for item in predicted_assets if str(item["id"]) not in ignored_prediction_ids
    ]
    maximum = float(matching["max_center_distance_m"])
    block_restricted = bool(matching["block_restricted"])
    invalid_cost = maximum + max(1.0, maximum) * 1_000_000.0
    costs = np.full((len(truth), len(predictions)), invalid_cost, dtype=np.float64)
    distances = np.full_like(costs, math.inf)
    for truth_index, truth_item in enumerate(truth):
        truth_center = np.asarray(truth_item["center_m"], dtype=np.float64)
        for prediction_index, prediction in enumerate(predictions):
            if block_restricted and truth_item["block_id"] != prediction["block_id"]:
                continue
            distance = float(
                np.linalg.norm(
                    truth_center - np.asarray(prediction["center_m"], dtype=np.float64)
                )
            )
            distances[truth_index, prediction_index] = distance
            if distance <= maximum:
                costs[truth_index, prediction_index] = distance

    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    pairs: list[dict[str, Any]] = []
    prediction_to_truth: dict[str, str] = {}
    if len(truth) and len(predictions):
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if costs[row, column] > maximum:
                continue
            truth_item = truth[row]
            prediction = predictions[column]
            matched_truth.add(row)
            matched_predictions.add(column)
            prediction_to_truth[str(prediction["id"])] = str(truth_item["id"])
            pairs.append(
                {
                    "ground_truth_id": truth_item["id"],
                    "prediction_id": prediction["id"],
                    "distance_m": float(distances[row, column]),
                    "ground_truth_type": truth_item["type"],
                    "prediction_type": prediction["type"],
                    "type_correct": truth_item["type"] == prediction["type"],
                }
            )

    unmatched_truth = [
        truth_item["id"]
        for index, truth_item in enumerate(truth)
        if index not in matched_truth
    ]
    unmatched_predictions = [
        prediction["id"]
        for index, prediction in enumerate(predictions)
        if index not in matched_predictions
    ]
    return {
        "ground_truth": truth,
        "predictions": predictions,
        "pairs": pairs,
        "prediction_to_truth": prediction_to_truth,
        "unmatched_ground_truth_ids": unmatched_truth,
        "unmatched_prediction_ids": unmatched_predictions,
        "ignored_prediction_ids": sorted(ignored_prediction_ids),
        "max_center_distance_m": maximum,
        "block_restricted": block_restricted,
    }


def _asset_metrics_v2(matching: dict[str, Any]) -> dict[str, Any]:
    truth = {str(item["id"]): item for item in matching["ground_truth"]}
    predictions = {str(item["id"]): item for item in matching["predictions"]}
    pairs = matching["pairs"]
    detection = _prf(
        len(pairs),
        len(matching["unmatched_prediction_ids"]),
        len(matching["unmatched_ground_truth_ids"]),
    )
    labels = sorted(
        {str(item["type"]) for item in truth.values()}
        | {str(item["type"]) for item in predictions.values()}
    )
    pair_by_truth = {str(item["ground_truth_id"]): item for item in pairs}
    pair_by_prediction = {str(item["prediction_id"]): item for item in pairs}
    per_class: dict[str, Any] = {}
    for label in labels:
        true_positive = sum(
            item["ground_truth_type"] == label and item["prediction_type"] == label
            for item in pairs
        )
        false_negative = sum(
            item["type"] == label
            and (
                str(item["id"]) not in pair_by_truth
                or pair_by_truth[str(item["id"])]["prediction_type"] != label
            )
            for item in truth.values()
        )
        false_positive = sum(
            item["type"] == label
            and (
                str(item["id"]) not in pair_by_prediction
                or pair_by_prediction[str(item["id"])]["ground_truth_type"] != label
            )
            for item in predictions.values()
        )
        per_class[label] = _prf(true_positive, false_positive, false_negative)
    class_f1 = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    return {
        "detection": detection,
        "classification_accuracy_on_matched": _safe_div(
            sum(bool(item["type_correct"]) for item in pairs), len(pairs)
        ),
        "macro_f1": float(np.mean(class_f1)) if class_f1 else None,
        "per_class": per_class,
    }


def _topology_metrics_v2(
    ground_truth_relations: list[dict[str, Any]],
    predicted_relations: list[dict[str, Any]],
    prediction_to_truth: dict[str, str],
) -> dict[str, Any]:
    def key(item: dict[str, Any]) -> tuple[str, str, str]:
        return str(item["type"]), str(item["from"]), str(item["to"])

    truth = {key(item) for item in ground_truth_relations}
    matched_truth_ids = set(prediction_to_truth.values())
    conditional_truth = {
        item for item in truth if item[1] in matched_truth_ids and item[2] in matched_truth_ids
    }
    conditional_prediction: set[tuple[str, str, str]] = set()
    end_to_end_prediction: set[tuple[str, str, str]] = set()
    for relation in predicted_relations:
        source = prediction_to_truth.get(
            str(relation["from"]), f"PRED_UNMATCHED:{relation['from']}"
        )
        target = prediction_to_truth.get(
            str(relation["to"]), f"PRED_UNMATCHED:{relation['to']}"
        )
        mapped = str(relation["type"]), source, target
        end_to_end_prediction.add(mapped)
        if str(relation["from"]) in prediction_to_truth and str(
            relation["to"]
        ) in prediction_to_truth:
            conditional_prediction.add(mapped)

    def compare(
        reference: set[tuple[str, str, str]],
        prediction: set[tuple[str, str, str]],
    ) -> dict[str, Any]:
        return _prf(
            len(reference & prediction),
            len(prediction - reference),
            len(reference - prediction),
        )

    return {
        "conditional_on_matched_instances": compare(
            conditional_truth, conditional_prediction
        ),
        "end_to_end": compare(truth, end_to_end_prediction),
    }


def _evidence_metrics_v2(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    released = [item for item in predictions if bool(item["released"])]
    statuses = ("supported", "confirmed_false", "unsupported", "unverifiable")
    counts = {
        status: sum(item["independent_evidence_status"] == status for item in released)
        for status in statuses
    }
    return {
        "prediction_count": len(predictions),
        "released_count": len(released),
        "coverage": _safe_div(len(released), len(predictions)),
        "independent_status_counts": counts,
        "supported_rate": _safe_div(counts["supported"], len(released)),
        "confirmed_false_rate": _safe_div(counts["confirmed_false"], len(released)),
        "unsupported_rate": _safe_div(counts["unsupported"], len(released)),
        "unverifiable_rate": _safe_div(counts["unverifiable"], len(released)),
    }


def _risk_metrics_v2(
    predictions: list[dict[str, Any]], prediction_to_truth: dict[str, str], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    pair_by_prediction = {str(item["prediction_id"]): item for item in pairs}
    scored: list[tuple[float, int, str]] = []
    for prediction in predictions:
        prediction_id = str(prediction["id"])
        pair = pair_by_prediction.get(prediction_id)
        correct = int(
            prediction_id in prediction_to_truth and pair is not None and pair["type_correct"]
        )
        scored.append((float(prediction["risk_score"]), correct, prediction_id))
    if not scored:
        return {"count": 0, "aurc": None, "risk_at_50pct_coverage": None, "curve": []}
    scored.sort(key=lambda item: (item[0], item[2]))
    correctness = np.asarray([item[1] for item in scored], dtype=np.float64)
    cumulative_risk = 1.0 - np.cumsum(correctness) / np.arange(1, len(scored) + 1)
    curve = [
        {
            "coverage": (index + 1) / len(scored),
            "empirical_risk": float(cumulative_risk[index]),
            "risk_threshold": scored[index][0],
        }
        for index in range(len(scored))
    ]
    return {
        "count": len(scored),
        "aurc": float(np.mean(cumulative_risk)),
        "risk_at_50pct_coverage": float(
            cumulative_risk[max(0, math.ceil(len(scored) * 0.5) - 1)]
        ),
        "curve": curve,
        "note": "risk_score is ranked only and is not interpreted as correctness probability",
    }


def _track_measurement_metrics_v2(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("raw_detection", "constrained_output"):
        stage_values = [item for item in measurements if item["stage"] == stage]
        errors = [
            abs(float(item["observed_gauge_m"]) - float(item["reference_gauge_m"]))
            for item in stage_values
        ]
        corrections = [abs(float(item["correction_m"])) for item in stage_values if "correction_m" in item]
        compliance = [
            error <= float(item["tolerance_m"])
            for item, error in zip(stage_values, errors, strict=True)
            if "tolerance_m" in item
        ]
        result[stage] = {
            "gauge_error": _distance_statistics(errors),
            "correction": _distance_statistics(corrections),
            "within_declared_tolerance_ratio": _safe_div(sum(compliance), len(compliance)),
        }
    return result


def _evaluate_v2(value: dict[str, Any]) -> dict[str, Any]:
    ignored = {str(item) for item in value.get("ignored_prediction_ids", [])}
    matching = _asset_matching_v2(
        value["ground_truth_assets"],
        value["predicted_assets"],
        value["matching"],
        ignored,
    )
    report = {
        "schema_version": "railway.benchmark.metric-report.v2",
        "report_id": value["evaluation_id"],
        "experiment_id": value["experiment_id"],
        "evaluator_version": "paper-v2-neutral-matching",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "matching": {
                "pairs": matching["pairs"],
                "unmatched_ground_truth_ids": matching["unmatched_ground_truth_ids"],
                "unmatched_prediction_ids": matching["unmatched_prediction_ids"],
                "ignored_prediction_ids": matching["ignored_prediction_ids"],
                "max_center_distance_m": matching["max_center_distance_m"],
                "block_restricted": matching["block_restricted"],
            },
            "instances": _asset_metrics_v2(matching),
            "topology": _topology_metrics_v2(
                value["ground_truth_relations"],
                value["predicted_relations"],
                matching["prediction_to_truth"],
            ),
            "evidence": _evidence_metrics_v2(matching["predictions"]),
            "risk": _risk_metrics_v2(
                matching["predictions"],
                matching["prediction_to_truth"],
                matching["pairs"],
            ),
            "geometry": _geometry_metrics(value["distance_sets"]),
            "track_gauge": _track_measurement_metrics_v2(value["track_measurements"]),
            "human": _human_metrics(value["human_review"]),
        },
        "sample_counts": {
            "ground_truth_assets": len(matching["ground_truth"]),
            "predicted_assets": len(matching["predictions"]),
            "matched_assets": len(matching["pairs"]),
            "ground_truth_relations": len(value["ground_truth_relations"]),
            "predicted_relations": len(value["predicted_relations"]),
            "distance_sets": len(value["distance_sets"]),
            "track_measurements": len(value["track_measurements"]),
        },
        "limitations": [
            *list(value.get("limitations", [])),
            "Instance matching uses independent centers and a declared radius; it does not use predicted class labels.",
            "Risk scores are ranking scores unless calibrated on an independent scene.",
        ],
    }
    report_errors = validate_benchmark_value(report)
    if report_errors:
        raise ValueError("Generated metric report is invalid:\n- " + "\n- ".join(report_errors))
    return report


def evaluate_benchmark_value(value: dict[str, Any]) -> dict[str, Any]:
    errors = validate_benchmark_value(value)
    if errors:
        raise ValueError("Invalid benchmark evaluation input:\n- " + "\n- ".join(errors))
    if value["schema_version"] == "railway.benchmark.evaluation-input.v1":
        return _evaluate_v1(value)
    if value["schema_version"] == "railway.benchmark.evaluation-input.v2":
        return _evaluate_v2(value)
    raise ValueError(f"Unsupported benchmark evaluation input: {value['schema_version']}")


def evaluate_benchmark_file(input_path: str | Path, output_path: str | Path) -> Path:
    value, errors = validate_benchmark_file(input_path)
    if errors:
        raise ValueError("Invalid benchmark evaluation input:\n- " + "\n- ".join(errors))
    report = evaluate_benchmark_value(value)
    output = Path(output_path).resolve()
    write_json(output, report)
    return output
