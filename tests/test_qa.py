import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.io import write_json
from railway_recon.qa import quality_report
from railway_recon.registry import new_registry


class QualityReportTests(unittest.TestCase):
    def test_absolute_precision_requires_independent_check_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = initialize_project(
                Path(temporary) / "sample", "sample-project", "Sample"
            )
            project = load_project(config_path)
            project.value["project"]["crs"].update(
                {"epsg": 32650, "vertical_datum": "example-datum"}
            )
            report = quality_report(project)
            self.assertFalse(report["absolute_precision_ready"])
            self.assertTrue(any("Independent check points" in item for item in report["limitations"]))

            check_points = project.root / "input" / "independent-check-points.csv"
            check_points.write_text("id,x,y,z\nP1,1,2,3\n", encoding="utf-8")
            project.value["project"]["crs"]["independent_check_points"] = str(check_points)
            report = quality_report(project)
            self.assertTrue(report["absolute_precision_ready"])
            self.assertEqual(report["limitations"], [])

    def test_candidate_assets_block_release_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = initialize_project(
                Path(temporary) / "sample", "sample-project", "Sample"
            )
            project = load_project(config_path)
            registry = new_registry(project.project_id)
            registry["assets"] = [
                {
                    "id": "TRACK-001",
                    "type": "track",
                    "status": "candidate",
                    "evidence_level": "observed",
                    "confidence": 0.9,
                    "sources": [{"kind": "point_cloud", "reference": "pilot.laz"}],
                }
            ]
            write_json(project.workspace_path("asset_registry"), registry)

            report = quality_report(project)

            self.assertFalse(report["release_ready"])
            self.assertEqual(report["blocking_release_asset_count"], 1)
            self.assertEqual(report["blocking_release_asset_ids"], ["TRACK-001"])

    def test_failed_track_graph_blocks_an_accepted_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = initialize_project(
                Path(temporary) / "sample", "sample-project", "Sample"
            )
            project = load_project(config_path)
            registry = new_registry(project.project_id)
            registry["assets"] = [
                {
                    "id": "TRACK-001",
                    "type": "track",
                    "status": "accepted",
                    "evidence_level": "observed",
                    "confidence": 0.9,
                    "sources": [{"kind": "point_cloud", "reference": "pilot.laz"}],
                }
            ]
            write_json(project.workspace_path("asset_registry"), registry)
            write_json(
                project.workspace_path("reports") / "track_graph_audit.json",
                {"schema_version": "railway.track-graph-audit.v1", "status": "fail"},
            )

            report = quality_report(project)

            self.assertFalse(report["release_ready"])
            check = next(item for item in report["checks"] if item["id"] == "track_graph.pass")
            self.assertFalse(check["passed"])


if __name__ == "__main__":
    unittest.main()
