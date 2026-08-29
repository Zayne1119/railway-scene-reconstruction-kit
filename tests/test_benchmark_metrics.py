import unittest

from railway_recon.benchmark_metrics import evaluate_benchmark_value


class BenchmarkMetricTests(unittest.TestCase):
    def test_evaluator_reports_instances_topology_geometry_and_effort(self) -> None:
        value = {
            "schema_version": "railway.benchmark.evaluation-input.v1",
            "evaluation_id": "evaluation-001",
            "experiment_id": "experiment-001",
            "assets": [
                {
                    "id": "A",
                    "evaluable": True,
                    "ground_truth": {"existence": "present", "type": "track"},
                    "prediction": {
                        "existence": "present",
                        "type": "track",
                        "confidence": 0.9,
                        "evidence_level": "observed",
                    },
                },
                {
                    "id": "B",
                    "evaluable": True,
                    "ground_truth": {"existence": "present", "type": "mast"},
                    "prediction": {
                        "existence": "present",
                        "type": "column",
                        "confidence": 0.8,
                        "evidence_level": "photo_interpreted",
                    },
                },
                {
                    "id": "C",
                    "evaluable": True,
                    "ground_truth": {"existence": "absent", "type": None},
                    "prediction": {
                        "existence": "present",
                        "type": "mast",
                        "confidence": 0.7,
                        "evidence_level": "unsupported",
                    },
                },
            ],
            "ground_truth_relations": [{"type": "supports", "from": "A", "to": "B"}],
            "predicted_relations": [
                {"type": "supports", "from": "A", "to": "B"},
                {"type": "connects", "from": "A", "to": "C"},
            ],
            "distance_sets": [
                {
                    "name": "track",
                    "reference_to_model_m": [0.01, 0.02, 0.03],
                    "model_to_reference_m": [0.01, 0.02, 0.04],
                    "thresholds_m": [0.02, 0.05],
                }
            ],
            "human_review": {
                "corridor_length_m": 50.0,
                "events": [
                    {"duration_seconds": 60.0, "error_found": True},
                    {"duration_seconds": 60.0, "error_found": False},
                ],
            },
            "limitations": ["synthetic unit test"],
        }
        report = evaluate_benchmark_value(value)
        self.assertAlmostEqual(report["metrics"]["instances"]["detection"]["precision"], 2 / 3)
        self.assertEqual(
            report["metrics"]["instances"]["classification_accuracy_on_matched"], 0.5
        )
        self.assertEqual(report["metrics"]["topology"]["recall"], 1.0)
        self.assertEqual(report["metrics"]["topology"]["precision"], 0.5)
        self.assertEqual(
            report["metrics"]["geometry"]["track"]["reference_to_model"]["p90_m"],
            0.028,
        )
        self.assertEqual(report["metrics"]["human"]["minutes_per_100m"], 4.0)
        self.assertEqual(report["metrics"]["confidence"]["count"], 3)

    def test_v2_matches_spatially_without_using_class_labels(self) -> None:
        value = {
            "schema_version": "railway.benchmark.evaluation-input.v2",
            "evaluation_id": "evaluation-v2-001",
            "experiment_id": "experiment-v2-001",
            "matching": {
                "max_center_distance_m": 1.0,
                "block_restricted": True,
            },
            "ground_truth_assets": [
                {
                    "id": "GT-TRACK",
                    "type": "track",
                    "evaluable": True,
                    "center_m": [0.0, 0.0, 0.0],
                    "block_id": "block-a",
                    "evidence_status": "supported",
                },
                {
                    "id": "GT-MAST",
                    "type": "mast",
                    "evaluable": True,
                    "center_m": [10.0, 0.0, 0.0],
                    "block_id": "block-a",
                    "evidence_status": "supported",
                },
            ],
            "predicted_assets": [
                {
                    "id": "PRED-TRACK",
                    "type": "track",
                    "center_m": [0.1, 0.0, 0.0],
                    "block_id": "block-a",
                    "risk_score": 0.1,
                    "declared_evidence_level": "observed",
                    "independent_evidence_status": "supported",
                    "released": True,
                },
                {
                    "id": "PRED-WRONG-CLASS",
                    "type": "column",
                    "center_m": [10.1, 0.0, 0.0],
                    "block_id": "block-a",
                    "risk_score": 0.2,
                    "declared_evidence_level": "photo_interpreted",
                    "independent_evidence_status": "unsupported",
                    "released": True,
                },
                {
                    "id": "PRED-FALSE-POSITIVE",
                    "type": "mast",
                    "center_m": [50.0, 0.0, 0.0],
                    "block_id": "block-a",
                    "risk_score": 0.9,
                    "declared_evidence_level": "rule_inferred",
                    "independent_evidence_status": "confirmed_false",
                    "released": True,
                },
            ],
            "ground_truth_relations": [
                {"type": "supports", "from": "GT-TRACK", "to": "GT-MAST"}
            ],
            "predicted_relations": [
                {
                    "type": "supports",
                    "from": "PRED-TRACK",
                    "to": "PRED-WRONG-CLASS",
                },
                {
                    "type": "connects",
                    "from": "PRED-TRACK",
                    "to": "PRED-FALSE-POSITIVE",
                },
            ],
            "distance_sets": [],
            "track_measurements": [
                {
                    "name": "track-1-raw",
                    "stage": "raw_detection",
                    "observed_gauge_m": 1.5,
                    "reference_gauge_m": 1.435,
                    "correction_m": 0.0,
                    "tolerance_m": 0.01,
                },
                {
                    "name": "track-1-constrained",
                    "stage": "constrained_output",
                    "observed_gauge_m": 1.435,
                    "reference_gauge_m": 1.435,
                    "correction_m": -0.065,
                    "tolerance_m": 0.01,
                },
            ],
            "human_review": {"corridor_length_m": 50.0, "events": []},
            "limitations": ["synthetic neutral-matching test"],
        }

        report = evaluate_benchmark_value(value)
        metrics = report["metrics"]
        self.assertEqual(report["schema_version"], "railway.benchmark.metric-report.v2")
        self.assertEqual(report["sample_counts"]["matched_assets"], 2)
        self.assertEqual(metrics["instances"]["detection"]["true_positive"], 2)
        self.assertEqual(metrics["instances"]["detection"]["false_positive"], 1)
        self.assertEqual(metrics["instances"]["classification_accuracy_on_matched"], 0.5)
        self.assertEqual(
            metrics["topology"]["conditional_on_matched_instances"]["f1"], 1.0
        )
        self.assertEqual(metrics["topology"]["end_to_end"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["evidence"]["supported_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["evidence"]["unsupported_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["evidence"]["confirmed_false_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["track_gauge"]["raw_detection"]["gauge_error"]["mae_m"],
            0.065,
        )
        self.assertEqual(
            metrics["track_gauge"]["constrained_output"][
                "within_declared_tolerance_ratio"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
