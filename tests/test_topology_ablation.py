from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.io import load_json, write_json
from railway_recon.topology_ablation import evaluate_topology_ablation


class TopologyAblationTests(unittest.TestCase):
    def test_reports_observability_delta_without_claiming_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_path = root / "graph.json"
            audit_path = root / "audit.json"
            output = root / "result.json"
            write_json(
                graph_path,
                {
                    "schema_version": "railway.track-graph.v1",
                    "project_id": "sample",
                    "sources": [{"segment_id": "a"}, {"segment_id": "b"}],
                    "observations": [
                        {"local_track_id": "TRACK-0001"},
                        {"local_track_id": "TRACK-0001"},
                    ],
                    "tracks": [{"id": "TRACK-0001"}],
                    "seams": [
                        {
                            "lateral_difference_m": 0.1,
                            "vertical_difference_m": 0.2,
                            "endpoint_3d_difference_m": 0.3,
                        }
                    ],
                    "identity_events": [
                        {"match_count": 1, "current_count": 1, "previous_count": 1}
                    ],
                },
            )
            write_json(
                audit_path,
                {
                    "schema_version": "railway.track-graph-audit.v1",
                    "project_id": "sample",
                    "status": "fail",
                    "checks": [
                        {"id": "segment_seams", "failure_count": 1},
                        {"id": "interior_track_termination", "failure_count": 0},
                    ],
                },
            )
            evaluate_topology_ablation(graph_path, audit_path, output)
            result = load_json(output)
            self.assertEqual(result["delta"]["fragment_identities_consolidated"], 1)
            self.assertEqual(result["track_graph_variant"]["seam_failure_count"], 1)
            self.assertTrue(result["delta"]["release_blocked_by_detected_defects"])
            self.assertIn("not geometric improvement", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
