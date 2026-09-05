from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from railway_recon.synthetic_track_pilot import evaluate_track_prediction, run_synthetic_track_pilot


def _item(kind: str = "gap", entity_ids: list[str] | None = None) -> dict:
    return {"kind": kind, "entity_ids": entity_ids or ["entity-a"], "position": [1, 2, 3]}


def _prediction(findings: list[dict], checks: list[dict] | None = None) -> dict:
    return {"findings": findings, "checks": checks or []}


class SyntheticTrackPilotTests(unittest.TestCase):
    def test_repeated_alarms_never_inflate_true_positives(self) -> None:
        truth = {"defects": [_item()]}
        result = evaluate_track_prediction(truth, _prediction([_item(), _item()]))
        for metric in ("detection", "diagnosis"):
            with self.subTest(metric=metric):
                self.assertEqual(result[metric]["tp"], 1)
                self.assertEqual(result[metric]["fp"], 1)
                self.assertEqual(result[metric]["fn"], 0)
        self.assertEqual(result["extra_alerts_on_repeated_entities"], 1)

    def test_wrong_cause_still_detects_entity_but_fails_diagnosis(self) -> None:
        truth = {"defects": [_item("wrong_connection")]}
        checks = [{**_item("gap"), "status": "flag"}]
        result = evaluate_track_prediction(truth, _prediction([_item("gap")], checks))
        self.assertEqual(result["detection"]["tp"], 1)
        self.assertEqual(result["detection"]["fp"], 0)
        self.assertEqual(result["diagnosis"]["tp"], 0)
        self.assertEqual(result["diagnosis"]["fp"], 1)
        self.assertEqual(result["diagnosis"]["fn"], 1)
        self.assertEqual(result["per_kind"]["wrong_connection"]["detection_tp"], 1)
        self.assertEqual(result["checks"]["normal_assessed"], 0)
        self.assertIsNone(result["checks"]["normal_check_fpr"])

    def test_missing_checks_and_abstentions_cannot_hide_missed_faults(self) -> None:
        truth = {"defects": [_item()]}
        result = evaluate_track_prediction(truth, _prediction([], [{**_item(), "status": "abstain"}]))
        self.assertEqual(result["detection"]["fn"], 1)
        self.assertEqual(result["detection"]["recall"], 0)
        self.assertIsNone(result["detection"]["precision"])
        self.assertEqual(result["checks"]["abstain"], 1)
        self.assertEqual(result["checks"]["truth_defect_check_coverage"], 0)
        empty = evaluate_track_prediction({"defects": []}, _prediction([]))
        self.assertIsNone(empty["detection"]["recall"])
        self.assertIsNone(empty["detection"]["f1"])
        self.assertIsNone(empty["checks"]["normal_check_fpr"])

    def test_normal_false_positive_denominator_excludes_abstentions(self) -> None:
        checks = [
            {**_item(entity_ids=["a"]), "status": "pass"},
            {**_item(entity_ids=["b"]), "status": "flag"},
            {**_item(entity_ids=["c"]), "status": "abstain"},
        ]
        result = evaluate_track_prediction(
            {"defects": []}, _prediction([_item(entity_ids=["b"])], checks)
        )
        self.assertEqual(result["checks"]["normal_total"], 3)
        self.assertEqual(result["checks"]["normal_assessed"], 2)
        self.assertEqual(result["checks"]["normal_check_fpr"], 0.5)
        self.assertIs(result["normal_case_flagged"], True)

    def test_entity_pair_order_does_not_change_matching_and_position_error_is_measured(self) -> None:
        truth = {"defects": [_item("duplicate", ["a", "b"])]}
        finding = {**_item("duplicate", ["b", "a"]), "position": [4, 6, 3]}
        result = evaluate_track_prediction(truth, _prediction([finding]))
        self.assertEqual(result["detection"]["tp"], 1)
        self.assertEqual(result["diagnosis"]["tp"], 1)
        self.assertEqual(result["localization_error_m"], [5.0])

    def test_duplicate_checks_are_rejected_instead_of_diluting_false_positive_rate(self) -> None:
        check = {**_item(), "status": "pass"}
        with self.assertRaisesRegex(ValueError, "duplicate checks"):
            evaluate_track_prediction({"defects": []}, _prediction([], [check, check]))

    def test_runner_refuses_existing_output_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="railway-pilot-no-overwrite-") as temporary:
            output = Path(temporary) / "already-there"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_synthetic_track_pilot(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["keep.txt"])

    def test_full_pilot_is_reproducible_and_input_files_have_no_truth(self) -> None:
        with tempfile.TemporaryDirectory(prefix="railway-pilot-reproducible-") as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            report = run_synthetic_track_pilot(first)
            repeated = run_synthetic_track_pilot(second)
            self.assertEqual(report, repeated)  # Timing lives in a separate artifact.
            self.assertEqual(report["case_count"], 24)
            self.assertEqual(report["base_layout_count"], 6)
            self.assertIs(report["frozen_test"], False)
            self.assertIs(report["customer_data_used"], False)
            self.assertEqual(len(list((first / "figures").glob("*.png"))), 24)
            for path in (first / "inputs").glob("*.json"):
                candidate = json.loads(path.read_text(encoding="utf-8"))
                with self.subTest(input_file=path.name):
                    self.assertNotIn("truth", candidate)
                    self.assertNotIn("variant", candidate)
                    self.assertNotIn("case_id", candidate)
                    self.assertNotIn("layout_id", candidate)
            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            repeated_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["data_fingerprint_sha256"], repeated_manifest["data_fingerprint_sha256"]
            )
            self.assertEqual(
                manifest["prediction_fingerprint_sha256"],
                repeated_manifest["prediction_fingerprint_sha256"],
            )
            for record in manifest["files"]:
                payload = (first / record["path"]).read_bytes()
                with self.subTest(output_file=record["path"]):
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
                    self.assertEqual(len(payload), record["bytes"])
            self.assertTrue(manifest["source_sha256"])


if __name__ == "__main__":
    unittest.main()
