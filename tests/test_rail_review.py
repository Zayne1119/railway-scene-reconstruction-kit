from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.rail_review import (
    apply_rail_owner_override,
    apply_rail_review_package,
    create_rail_review_package,
)


class RailReviewTests(unittest.TestCase):
    def _package(self, root: Path):
        project = load_project(initialize_project(root / "sample", "sample", "Sample"))
        report = project.workspace_path("reports") / "candidate.json"
        diagnostic = project.workspace_path("reports") / "candidate.png"
        diagnostic.write_bytes(b"png")
        write_json(
            report,
            {
                "schema_version": "railway.rail-candidates.v1",
                "project_id": project.project_id,
                "segment_id": "s0000_0050m",
                "diagnostic_image": str(diagnostic),
                "rail_pairs": [
                    {
                        "peak_indexes": [1, 2],
                        "rail_position_correction_m": 0.01,
                        "rail_top_crosslevel_m": 0.02,
                    }
                ],
                "peaks": [
                    {"grid_index": 1, "coverage": 0.4},
                    {"grid_index": 2, "coverage": 0.5},
                ],
                "rail_lines": [
                    {
                        "cross_fit_residual_p90_m": 0.03,
                        "z_fit_residual_p90_m": 0.04,
                    }
                ],
            },
        )
        package = create_rail_review_package(
            project, [("s0000_0050m", report)], "review"
        )
        return project, package, report

    def test_pending_review_does_not_mutate_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, package, source = self._package(Path(temporary))
            before = source.read_bytes()
            result = apply_rail_review_package(project, package)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(source.read_bytes(), before)

    def test_accepted_review_emits_hash_bound_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, package, source = self._package(Path(temporary))
            task = package / "rail-candidate-review.csv"
            with task.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0])
            rows[0]["decision"] = "accepted"
            rows[0]["reviewer"] = "reviewer-a"
            with task.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result = apply_rail_review_package(project, package)
            self.assertEqual(result["status"], "accepted")
            reviewed = load_json(Path(result["reviewed_sources"][0]["path"]))
            self.assertEqual(reviewed["review_status"], "reviewed_accepted")
            self.assertEqual(reviewed["review"]["reviewer"], "reviewer-a")
            self.assertNotIn("review_status", load_json(source))

    def test_owner_override_is_explicit_and_does_not_mutate_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, package, source = self._package(Path(temporary))
            before = source.read_bytes()
            result = apply_rail_owner_override(
                project,
                package,
                "project-owner",
                "Owner approved generation without independent signature",
            )
            self.assertEqual(result["status"], "accepted")
            self.assertFalse(result["independent_review_completed"])
            approved = load_json(Path(result["approved_sources"][0]["path"]))
            self.assertEqual(approved["review_status"], "owner_override_accepted")
            self.assertEqual(approved["review"]["mode"], "project_owner_override")
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
