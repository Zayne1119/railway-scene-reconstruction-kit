from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from railway_recon.public_rail_figure import render_public_rail_comparison


def _evaluation(labels: np.ndarray, mask: np.ndarray) -> dict:
    reference = labels == 2
    result = {}
    for scope, included in (
        ("all_points", np.ones(len(labels), bool)),
        ("annotated_only", labels != 0),
    ):
        tp = int(np.count_nonzero(included & reference & mask))
        fp = int(np.count_nonzero(included & ~reference & mask))
        fn = int(np.count_nonzero(included & reference & ~mask))
        tn = int(np.count_nonzero(included & ~reference & ~mask))
        result[scope] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "evaluated_points": int(included.sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        }
    return result


class PublicRailFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.xyz = np.array([[0, 0, 0], [1, 2, 0], [2, 4, 0], [3, 6, 0], [20, 5, 0]], float)
        self.labels = np.array([2, 0, 2, 3, 0], dtype=np.uint8)
        self.predictions = {
            "height_prominence": np.array([True, True, False, True, False]),
            "paired_geometry": np.array([True, False, True, False, False]),
        }
        self.evaluations = {
            mode: _evaluation(self.labels, mask) for mode, mask in self.predictions.items()
        }

    def _render(self, path: Path, **changes: object) -> dict:
        arguments = {
            "xyz": self.xyz,
            "classification": self.labels,
            "predictions": self.predictions,
            "evaluations": self.evaluations,
            "output": path,
            "target_class_ids": (2,),
            "ignored_class_ids": (0,),
        }
        arguments.update(changes)
        return render_public_rail_comparison(**arguments)

    def test_deterministic_subsample_full_bounds_full_metrics_and_no_png_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._render(root / "a.png", max_display_points=2)
            second = self._render(root / "b.png", max_display_points=2)
            self.assertEqual((root / "a.png").read_bytes(), (root / "b.png").read_bytes())
            self.assertEqual(first["display_indices_sha256"], second["display_indices_sha256"])
            self.assertEqual(first["display_point_count"], 2)
            self.assertTrue(first["display_is_subsample"])
            self.assertEqual(first["full_bounds_xy"]["maximum"], [20.0, 6.0])
            self.assertEqual(first["metrics"]["height_prominence"]["all_points"]["fp"], 2)
            self.assertEqual(first["metrics"]["height_prominence"]["annotated_only"]["fp"], 1)
            self.assertTrue(first["metrics_use_full_point_arrays"])
            self.assertNotIn(str(root), json.dumps(first))
            with Image.open(root / "a.png") as rendered:
                self.assertEqual(rendered.size, (1800, 916))
                self.assertEqual(rendered.info, {})
            self.assertEqual(first["target_label"], "numeric reference target")
            self.assertEqual(first["mapping_status"], "not_verified")

    def test_inputs_unchanged_and_refuses_overwrite(self) -> None:
        originals = copy.deepcopy((self.xyz, self.labels, self.predictions, self.evaluations))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "figure.png"
            self._render(path)
            saved = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self._render(path)
            self.assertEqual(path.read_bytes(), saved)
        np.testing.assert_array_equal(self.xyz, originals[0])
        np.testing.assert_array_equal(self.labels, originals[1])
        for mode in self.predictions:
            np.testing.assert_array_equal(self.predictions[mode], originals[2][mode])
        self.assertEqual(self.evaluations, originals[3])

    def test_class_zero_is_not_implicitly_ignored(self) -> None:
        evaluations = copy.deepcopy(self.evaluations)
        for evaluation in evaluations.values():
            evaluation["annotated_only"] = copy.deepcopy(evaluation["all_points"])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._render(
                Path(temporary) / "no-implicit-ignore.png",
                ignored_class_ids=(),
                evaluations=evaluations,
            )
        self.assertEqual(manifest["ignored_class_ids"], [])
        self.assertEqual(manifest["metrics"]["height_prominence"]["annotated_only"]["fp"], 2)

    def test_rejects_invalid_inputs_without_output(self) -> None:
        cases = [
            {"xyz": np.empty((0, 3))},
            {"xyz": np.zeros((5, 2))},
            {"xyz": np.full((5, 3), np.nan)},
            {"classification": np.zeros(5, float)},
            {"classification": np.zeros(4, int)},
            {"classification": np.full(5, -1)},
            {"target_class_ids": ()},
            {"target_class_ids": (True,)},
            {"target_class_ids": (2, 2)},
            {"ignored_class_ids": (2,)},
            {"max_display_points": 0},
            {"max_display_points": True},
            {"source_label": "C:\\Users\\private\\sample.laz"},
            {"source_label": "private /tmp/sample.laz"},
            {"source_label": "label\nline"},
            {"target_label": "private /tmp/sample.laz"},
            {"mapping_status": "guessed_from_numeric_order"},
            {"predictions": {"height_prominence": np.ones(5, bool)}},
            {"predictions": {mode: np.ones(5, int) for mode in self.predictions}},
            {"evaluations": {}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for index, changes in enumerate(cases):
                path = Path(temporary) / f"invalid-{index}.png"
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self._render(path, **changes)
                self.assertFalse(path.exists())

    def test_rejects_mismatched_or_undefined_metric_captions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for index, (scope, key, value) in enumerate(
                [
                    ("all_points", "tp", 200),
                    ("annotated_only", "f1", 0.0),
                    ("all_points", "recall", float("nan")),
                    ("annotated_only", "tp", True),
                ]
            ):
                evaluations = copy.deepcopy(self.evaluations)
                evaluations["height_prominence"][scope][key] = value
                with self.assertRaises(ValueError):
                    self._render(Path(temporary) / f"invalid-{index}.png", evaluations=evaluations)

    def test_degenerate_geometry_and_no_positive_predictions_render(self) -> None:
        xyz = np.zeros((5, 3))
        predictions = {mode: np.zeros(5, bool) for mode in self.predictions}
        evaluations = {mode: _evaluation(self.labels, mask) for mode, mask in predictions.items()}
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._render(
                Path(temporary) / "zero.png",
                xyz=xyz,
                predictions=predictions,
                evaluations=evaluations,
            )
            self.assertIsNone(manifest["metrics"]["height_prominence"]["all_points"]["precision"])
            self.assertFalse(manifest["display_is_subsample"])


if __name__ == "__main__":
    unittest.main()
