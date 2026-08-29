import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.io import load_json, write_json
from railway_recon.prediction_lock import lock_vertical_predictions


class PredictionLockTests(unittest.TestCase):
    def test_locks_hitl_predictions_against_blind_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            package = benchmark / "annotations" / "vertical_candidates_blind_v1"
            package.mkdir(parents=True)
            write_json(
                package / "manifest.json",
                {
                    "package_id": "vertical_candidates_blind_v1",
                    "reviewer_files": ["reviewer_a.csv"],
                },
            )
            with (package / "reviewer_a.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["task_id"])
                writer.writeheader()
                writer.writerows(
                    [{"task_id": "VANN-000-001"}, {"task_id": "VANN-000-002"}]
                )

            classification = root / "classification.json"
            write_json(
                classification,
                {
                    "continuity_redetection": {
                        "candidate_count": 2,
                        "candidates": [
                            {
                                "id": 1,
                                "semantic_class": "catenary_mast",
                                "classification_confidence": 0.9,
                                "evidence_views": [{"camera_index": 1}],
                                "evidence_source": "geometry+photo",
                                "asset_id": "mast-001",
                            },
                            {
                                "id": 2,
                                "semantic_class": "false_positive",
                                "classification_confidence": 0.8,
                                "evidence_views": [],
                                "evidence_source": "geometry",
                                "asset_id": "",
                            },
                        ],
                    }
                },
            )

            locked = lock_vertical_predictions(
                benchmark, package, [("s000_50m", classification)]
            )
            manifest = load_json(locked / "manifest.json")
            self.assertEqual(manifest["prediction_count"], 2)
            self.assertEqual(manifest["execution_mode"], "HITL")
            self.assertFalse(manifest["eligible_as_fully_automatic_baseline"])
            self.assertTrue(manifest["locked_before_ground_truth"])
            self.assertEqual(len(manifest["prediction_file_sha256"]), 64)

            with (locked / "predictions.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["prediction_existence"], "present")
            self.assertEqual(rows[0]["prediction_type"], "catenary_mast")
            self.assertEqual(rows[0]["evidence_level"], "photo_interpreted")
            self.assertEqual(rows[1]["prediction_existence"], "absent")
            self.assertEqual(rows[1]["prediction_type"], "")

    def test_rejects_prediction_task_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            package = benchmark / "annotations" / "blind"
            package.mkdir(parents=True)
            write_json(
                package / "manifest.json",
                {"package_id": "blind", "reviewer_files": ["reviewer_a.csv"]},
            )
            with (package / "reviewer_a.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["task_id"])
                writer.writeheader()
                writer.writerow({"task_id": "VANN-000-999"})
            classification = root / "classification.json"
            write_json(
                classification,
                {
                    "continuity_redetection": {
                        "candidate_count": 1,
                        "candidates": [
                            {
                                "id": 1,
                                "semantic_class": "false_positive",
                                "classification_confidence": 0.5,
                            }
                        ],
                    }
                },
            )
            with self.assertRaisesRegex(ValueError, "task mismatch"):
                lock_vertical_predictions(
                    benchmark, package, [("s000_50m", classification)]
                )


if __name__ == "__main__":
    unittest.main()
