"""Pointwise reference-label agreement, not geometric or connection ground truth.

Classification is permitted only at this evaluation boundary. It must not be
passed to candidate extraction. Unknown numeric labels are retained, never
silently recoded or discarded. A target ID alone does not verify a rail meaning.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

import numpy as np

SCHEMA_VERSION = "railway.public-rail-reference-evaluation.v1"
_FACT_RESOURCE = "public-rail-reference-facts-v1.json"


def _reference_contract() -> dict[str, Any]:
    raw = files("railway_recon").joinpath("resources", _FACT_RESOURCE).read_bytes()
    facts = json.loads(raw)
    return {
        "schema_version": "railway.public-rail-reference-contract.v1",
        "dataset": "SemanticRail3D-V2",
        "dataset_doi": "10.5281/zenodo.15641832",
        "target_class_ids": [1],
        "ignored_class_ids": [0],
        "target_name": "numeric_class_1",
        "target_semantic_mapping_status": "rail_numeric_id_not_explicitly_verified",
        "zero_semantics": "intentionally_unclassified_background",
        "zero_semantics_status": "explicitly_stated_in_dataset_paper",
        "ignore_policy": "report_both_all_points_and_annotated_only_not_official_rule",
        "reference_role": "released_pointwise_semantic_labels_not_independent_survey_truth",
        "source_url": "https://www.nature.com/articles/s41597-025-06392-9",
        "source_facts_resource": _FACT_RESOURCE,
        "source_facts_sha256": hashlib.sha256(raw).hexdigest(),
        "source_facts": facts,
    }


PUBLIC_RAIL_REFERENCE_CONTRACT = _reference_contract()


def _class_ids(value: tuple[int, ...], name: str, *, required: bool) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an explicit tuple of integer IDs")
    if required and not value:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or item < 0
        for item in value
    ):
        raise ValueError(f"{name} must contain nonnegative integer IDs, not booleans")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return tuple(sorted(result))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    tn = int(np.count_nonzero(~reference & ~prediction))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "evaluated_points": tp + fp + fn + tn,
        "reference_positive_points": tp + fn,
        "predicted_positive_points": tp + fp,
    }


def evaluate_public_rail_mask(
    classification: np.ndarray,
    predicted_mask: np.ndarray,
    target_class_ids: tuple[int, ...],
    ignored_class_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Compare a boolean mask to caller-selected numeric reference classes.

    Both views are always returned: ``all_points`` includes every input point,
    treating other numeric labels as reference-negative; ``annotated_only``
    removes only explicitly supplied ignored IDs. The second name is a reporting
    scope, not a claim that this function can establish annotation completeness.
    Undefined ratios return JSON null (Python None), never NaN or a perfect score.
    """
    if not isinstance(classification, np.ndarray) or classification.ndim != 1:
        raise ValueError("classification must be a one-dimensional numpy array")
    if classification.dtype.kind not in "iu" or np.any(classification < 0):
        raise ValueError("classification must contain nonnegative integer labels")
    if not isinstance(predicted_mask, np.ndarray) or predicted_mask.ndim != 1:
        raise ValueError("predicted_mask must be a one-dimensional numpy array")
    if predicted_mask.dtype.kind != "b":
        raise ValueError("predicted_mask must have boolean dtype, not scores or integer masks")
    if classification.shape != predicted_mask.shape:
        raise ValueError("classification and predicted_mask must have the same point count")
    targets = _class_ids(target_class_ids, "target_class_ids", required=True)
    ignored = _class_ids(ignored_class_ids, "ignored_class_ids", required=False)
    if set(targets) & set(ignored):
        raise ValueError("target and ignored IDs must be disjoint")

    reference = np.isin(classification, targets)
    ignored_mask = np.isin(classification, ignored)
    labels, counts = np.unique(classification, return_counts=True)
    per_class: dict[str, Any] = {}
    for label, count in zip(labels, counts, strict=True):
        numeric_id = int(label)
        per_class[str(numeric_id)] = {
            "point_count": int(count),
            "predicted_positive_count": int(
                np.count_nonzero(predicted_mask & (classification == label))
            ),
            "reference_positive": numeric_id in targets,
            "ignored_in_annotated_only": numeric_id in ignored,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_contract": {
            "task": "pointwise_binary_agreement_with_selected_numeric_reference_labels",
            "semantic_name_inferred_from_numeric_id": False,
            "all_points": "all_reference_nontarget_ids_count_as_negative_including_ignored",
            "annotated_only": "exclude_only_explicit_ignored_ids",
            "all_points_false_positive_interpretation": "reference_label_disagreement",
            "ignored_background_is_verified_nonrail": False,
            "undefined_ratio": "null",
            "independent_point_samples_for_statistics": False,
            "independent_geometry_or_connection_truth": False,
        },
        "target_class_ids": list(targets),
        "ignored_class_ids": list(ignored),
        "input_point_count": int(classification.size),
        "ignored_point_count": int(np.count_nonzero(ignored_mask)),
        "ignored_predicted_point_count": int(np.count_nonzero(ignored_mask & predicted_mask)),
        "all_points": _metrics(reference, predicted_mask),
        "annotated_only": _metrics(reference[~ignored_mask], predicted_mask[~ignored_mask]),
        "per_numeric_class": per_class,
    }
