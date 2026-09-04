from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from railway_recon.config import initialize_project, load_project
from railway_recon.io import write_json
from railway_recon.seam_evidence import compare_seam_rail_reports


def _rail_report(segment_id: str, source: str, center: float) -> dict[str, object]:
    positions = [center - 0.754, center + 0.754]
    return {
        "segment_id": segment_id,
        "source": source,
        "rail_lines": [
            {"cross_position_m": positions[0], "median_z_m": 10.0},
            {"cross_position_m": positions[1], "median_z_m": 10.002},
        ],
        "rail_pairs": [
            {
                "track_id": "TRACK-0001",
                "cross_positions_m": positions,
                "gauge_m": 1.435,
                "observed_gauge_m": 1.437,
                "rail_top_crosslevel_m": 0.002,
                "joint_support_ratio": 0.8,
                "pair_continuity_status": "pass",
            }
        ],
    }


class SeamEvidenceTests(unittest.TestCase):
    def test_matching_shared_rail_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = load_project(initialize_project(root / "project", "test", "Test"))
            left = root / "left.json"
            right = root / "right.json"
            output = root / "comparison.json"
            write_json(left, _rail_report("s0000_0050m", "left.laz", 2.0))
            write_json(right, _rail_report("s0000_0050m", "right.laz", 2.01))
            report = compare_seam_rail_reports(project, left, right, output)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["matched_pair_count"], 1)
            self.assertTrue(output.is_file())

    def test_missing_side_is_insufficient_not_a_false_termination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = load_project(initialize_project(root / "project", "test", "Test"))
            left = root / "left.json"
            right = root / "right.json"
            output = root / "comparison.json"
            write_json(left, {"segment_id": "s", "source": "left", "rail_pairs": []})
            write_json(right, _rail_report("s", "right", 2.0))
            report = compare_seam_rail_reports(project, left, right, output)
            self.assertEqual(report["status"], "insufficient_shared_rail_support")
            self.assertEqual(report["matched_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
