import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.annotation_tasks import FEATURE_COLUMNS, LABEL_COLUMNS
from railway_recon.io import load_json, write_json
from railway_recon.review_agreement import compare_vertical_reviews


def _review_row(task_id: str, asset_type: str) -> dict[str, str]:
    row = {column: "" for column in [*FEATURE_COLUMNS, *LABEL_COLUMNS]}
    row.update(
        {
            "task_id": task_id,
            "segment_id": "s000_50m",
            "existence": "absent" if asset_type == "false_positive" else "present",
            "asset_type": asset_type,
            "geometry_evaluable": "yes",
            "evidence_state": "observed",
            "reviewer_confidence": "0.9",
        }
    )
    return row


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*FEATURE_COLUMNS, *LABEL_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


class ReviewAgreementTests(unittest.TestCase):
    def test_builds_consensus_and_adjudication_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "annotations"
            package.mkdir()
            write_json(
                package / "manifest.json",
                {
                    "package_id": "blind-v1",
                    "reviewer_files": ["reviewer_a.csv", "reviewer_b.csv"],
                },
            )
            _write_review(
                package / "reviewer_a.csv",
                [
                    _review_row("VANN-000-001", "catenary_mast"),
                    _review_row("VANN-000-002", "canopy_column"),
                ],
            )
            _write_review(
                package / "reviewer_b.csv",
                [
                    _review_row("VANN-000-002", "building_edge"),
                    _review_row("VANN-000-001", "catenary_mast"),
                ],
            )

            output = compare_vertical_reviews(package)
            report = load_json(output / "agreement-report.json")
            self.assertEqual(report["task_count"], 2)
            self.assertEqual(report["consensus_count"], 1)
            self.assertEqual(report["disagreement_count"], 1)
            self.assertEqual(report["joint_decision_agreement"], 0.5)
            self.assertEqual(report["ground_truth_status"], "requires_adjudication")
            self.assertEqual(len(report["reviewers"][0]["sha256"]), 64)

            with (output / "disagreements.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                disagreements = list(csv.DictReader(stream))
            self.assertEqual(disagreements[0]["task_id"], "VANN-000-002")
            self.assertEqual(disagreements[0]["reviewer_a_asset_type"], "canopy_column")
            self.assertEqual(disagreements[0]["reviewer_b_asset_type"], "building_edge")
            self.assertEqual(disagreements[0]["adjudicated_asset_type"], "")

    def test_rejects_incomplete_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "annotations"
            package.mkdir()
            write_json(
                package / "manifest.json",
                {
                    "package_id": "blind-v1",
                    "reviewer_files": ["reviewer_a.csv", "reviewer_b.csv"],
                },
            )
            empty = _review_row("VANN-000-001", "catenary_mast")
            empty["asset_type"] = ""
            _write_review(package / "reviewer_a.csv", [empty])
            _write_review(
                package / "reviewer_b.csv",
                [_review_row("VANN-000-001", "catenary_mast")],
            )
            with self.assertRaisesRegex(ValueError, "Invalid or incomplete asset_type"):
                compare_vertical_reviews(package)


if __name__ == "__main__":
    unittest.main()
