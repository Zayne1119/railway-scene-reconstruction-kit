from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.config import ProjectConfig
from railway_recon.full_corridor import build_full_corridor_plan
from railway_recon.io import write_json


class FullCorridorPlanTests(unittest.TestCase):
    def test_plan_is_resumable_and_prioritizes_missing_core_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = {
                "project": {"id": "test-full-corridor"},
                "workspace": {
                    "segment_manifest": "workspace/manifests/segments.json",
                    "segments": "workspace/segments",
                    "reports": "workspace/reports",
                },
            }
            project = ProjectConfig(root / "project.json", value)
            write_json(
                project.workspace_path("segment_manifest"),
                {
                    "trajectory_length_m": 100.0,
                    "segments": [
                        {
                            "id": "s0000_0050m",
                            "chainage_start_m": 0.0,
                            "chainage_end_m": 50.0,
                            "primary_source_id": "a",
                            "context_source_ids": ["a"],
                        },
                        {
                            "id": "s0050_0100m",
                            "chainage_start_m": 50.0,
                            "chainage_end_m": 100.0,
                            "primary_source_id": "b",
                            "context_source_ids": ["a", "b"],
                        },
                    ],
                },
            )
            first_crop = project.workspace_path("segments") / "s0000_0050m.laz"
            first_crop.parent.mkdir(parents=True)
            first_crop.write_bytes(b"cached")

            result = build_full_corridor_plan(project)

            self.assertEqual(result["segment_count"], 2)
            self.assertEqual(result["stage_counts"]["crop"], {"complete": 1, "missing": 1})
            self.assertEqual(result["stage_counts"]["rails"], {"complete": 0, "missing": 2})
            self.assertEqual(result["priority_queue"][0]["priority"], "P0")
            self.assertTrue(Path(result["output"]).is_file())


if __name__ == "__main__":
    unittest.main()
