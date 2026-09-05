"""Numeric-label evaluation must preserve the unclassified/background boundary."""

from __future__ import annotations

import hashlib
import json
import unittest
from importlib.resources import files

import numpy as np

from railway_recon.public_rail_reference import (
    PUBLIC_RAIL_REFERENCE_CONTRACT,
    evaluate_public_rail_mask,
)


class PublicRailReferenceTests(unittest.TestCase):
    def test_both_views_preserve_background_disagreement(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
        predicted = np.array([True, False, True, False, True, False])
        result = evaluate_public_rail_mask(labels, predicted, (1,), (0,))
        self.assertEqual(result["input_point_count"], 6)
        self.assertEqual(result["ignored_point_count"], 2)
        self.assertEqual(result["ignored_predicted_point_count"], 1)
        for name, expected in (
            ("all_points", {"tp": 1, "fp": 2, "fn": 1, "tn": 2}),
            ("annotated_only", {"tp": 1, "fp": 1, "fn": 1, "tn": 1}),
        ):
            view = result[name]
            self.assertEqual({key: view[key] for key in expected}, expected)
            self.assertEqual(view["evaluated_points"], sum(expected.values()))
        self.assertAlmostEqual(result["all_points"]["precision"], 1 / 3)
        self.assertAlmostEqual(result["all_points"]["recall"], 0.5)
        self.assertAlmostEqual(result["all_points"]["f1"], 0.4)
        self.assertAlmostEqual(result["all_points"]["iou"], 0.25)
        self.assertEqual(result["annotated_only"]["f1"], 0.5)
        self.assertEqual(result["per_numeric_class"]["0"]["predicted_positive_count"], 1)
        self.assertFalse(result["metric_contract"]["ignored_background_is_verified_nonrail"])
        json.dumps(result, allow_nan=False)

    def test_default_does_not_silently_ignore_zero(self) -> None:
        result = evaluate_public_rail_mask(
            np.array([0, 1], dtype=np.uint8), np.array([True, True]), (1,)
        )
        self.assertEqual(result["all_points"], result["annotated_only"])
        self.assertEqual(result["all_points"]["fp"], 1)
        self.assertEqual(result["ignored_point_count"], 0)

    def test_empty_arrays_use_null_not_perfect_scores(self) -> None:
        result = evaluate_public_rail_mask(
            np.array([], dtype=np.int64), np.array([], dtype=bool), (1,), (0,)
        )
        for name in ("all_points", "annotated_only"):
            self.assertEqual(result[name]["evaluated_points"], 0)
            for metric in ("precision", "recall", "f1", "iou"):
                self.assertIsNone(result[name][metric])
        self.assertEqual(result["per_numeric_class"], {})
        json.dumps(result, allow_nan=False)

    def test_no_predictions_retains_false_negatives(self) -> None:
        result = evaluate_public_rail_mask(
            np.array([1, 1, 0], dtype=np.int32), np.zeros(3, dtype=bool), (1,), (0,)
        )
        self.assertIsNone(result["annotated_only"]["precision"])
        self.assertEqual(result["annotated_only"]["fn"], 2)
        self.assertEqual(result["annotated_only"]["recall"], 0)
        self.assertEqual(result["annotated_only"]["f1"], 0)
        self.assertEqual(result["annotated_only"]["iou"], 0)

    def test_absent_target_does_not_invent_recall(self) -> None:
        result = evaluate_public_rail_mask(
            np.array([2, 0], dtype=np.uint16), np.array([True, True]), (1,), (0,)
        )
        self.assertIsNone(result["all_points"]["recall"])
        self.assertEqual(result["all_points"]["f1"], 0)
        self.assertEqual(result["all_points"]["precision"], 0)

    def test_unknown_numeric_classes_and_multiple_targets_are_retained(self) -> None:
        labels = np.array([0, 1, 2, 99, 255], dtype=np.uint16)
        result = evaluate_public_rail_mask(
            labels, np.array([True, True, False, True, False]), (np.int64(99), 1), (0,)
        )
        self.assertEqual(result["target_class_ids"], [1, 99])
        self.assertEqual(list(result["per_numeric_class"]), ["0", "1", "2", "99", "255"])
        self.assertEqual(result["all_points"]["tp"], 2)
        self.assertFalse(result["metric_contract"]["semantic_name_inferred_from_numeric_id"])

    def test_inputs_unchanged_and_order_invariant(self) -> None:
        labels = np.array([0, 1, 2, 1, 99], dtype=np.int64)
        predicted = np.array([True, True, True, False, False])
        labels_before, predicted_before = labels.copy(), predicted.copy()
        original = evaluate_public_rail_mask(labels, predicted, (1,), (0,))
        permuted = evaluate_public_rail_mask(labels[::-1], predicted[::-1], (1,), (0,))
        self.assertEqual(original, permuted)
        np.testing.assert_array_equal(labels, labels_before)
        np.testing.assert_array_equal(predicted, predicted_before)

    def test_invalid_label_arrays_rejected(self) -> None:
        for labels in (
            [1, 2], np.array([[1, 2]]), np.array([True, False]),
            np.array([1.0, 2.0]), np.array([1, -1]), np.array([1, 2], dtype=object),
        ):
            with self.subTest(labels=repr(labels)), self.assertRaises(ValueError):
                evaluate_public_rail_mask(labels, np.array([True, False]), (1,))

    def test_invalid_mask_arrays_rejected(self) -> None:
        for predicted in (
            [True, False], np.array([[True, False]]), np.array([1, 0]),
            np.array([0.3, 0.6]), np.array([True]),
        ):
            with self.subTest(mask=repr(predicted)), self.assertRaises(ValueError):
                evaluate_public_rail_mask(np.array([1, 2]), predicted, (1,))

    def test_invalid_target_and_ignore_contract_rejected(self) -> None:
        for targets, ignored in (
            ((), ()), ([1], ()), ((True,), ()), ((-1,), ()), ((1.0,), ()),
            ((1, 1), ()), ((1,), [0]), ((1,), (True,)), ((1,), (1,)),
            ((1,), (0, 0)), ((1,), (-1,)),
        ):
            with self.subTest(targets=targets, ignored=ignored), self.assertRaises(
                (TypeError, ValueError)
            ):
                evaluate_public_rail_mask(np.array([1]), np.array([True]), targets, ignored)

    def test_contract_preserves_unverified_rail_name_and_hash_bound_facts(self) -> None:
        contract = PUBLIC_RAIL_REFERENCE_CONTRACT
        self.assertEqual(contract["target_name"], "numeric_class_1")
        self.assertEqual(contract["target_class_ids"], [1])
        self.assertEqual(contract["ignored_class_ids"], [0])
        self.assertEqual(
            contract["target_semantic_mapping_status"], "rail_numeric_id_not_explicitly_verified"
        )
        raw = files("railway_recon").joinpath(
            "resources", contract["source_facts_resource"]
        ).read_bytes()
        self.assertEqual(contract["source_facts_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(contract["source_facts"], json.loads(raw))
        json.dumps(contract, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
