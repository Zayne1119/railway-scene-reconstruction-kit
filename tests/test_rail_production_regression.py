from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.io import load_json, write_json
from railway_recon.rail_production_regression import audit_rail_production_regression


def _report(
    source: Path, z_value: float, crosses: tuple[float, ...] = (0.0, 1.5)
) -> dict:
    lines = []
    for index, cross in enumerate(crosses, start=1):
        lines.append(
            {
                "id": f"RAIL-{index:04d}",
                "cross_position_m": cross,
                "cross_fit_slope_m_per_m": 0.0,
                "cross_fit_intercept_m": cross,
                "z_fit_slope_m_per_m": 0.0,
                "z_fit_intercept_m": z_value,
            }
        )
    return {
        "schema_version": "railway.rail-candidates.v1",
        "segment_id": "s0000_0050m",
        "source": str(source),
        "frame": {
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
        "longitudinal_range_m": [0.0, 50.0],
        "rail_pair_count": len(crosses) // 2,
        "rail_lines": lines,
    }


class RailProductionRegressionTests(unittest.TestCase):
    def test_candidate_with_better_height_and_same_topology_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            longitudinal = np.repeat(np.linspace(0.0, 50.0, 101), 2)
            cloud.x = longitudinal
            cloud.y = np.tile([0.0, 1.5], 101)
            cloud.z = np.zeros(len(longitudinal))
            cloud.write(source)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "audit.json"
            write_json(baseline, _report(source, 0.08))
            write_json(candidate, _report(source, 0.0))

            audit_rail_production_regression(
                [("s0000_0050m", baseline)],
                [("s0000_0050m", candidate)],
                output,
            )

            report = load_json(output)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["guardrails"]["aggregate_support_p90_improved"])

    def test_expansion_mode_allows_one_complete_additional_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.las"
            header = laspy.LasHeader(point_format=3, version="1.2")
            cloud = laspy.LasData(header)
            crosses = (0.0, 1.5, 3.0, 4.5)
            longitudinal = np.repeat(np.linspace(0.0, 50.0, 201), len(crosses))
            cloud.x = longitudinal
            cloud.y = np.tile(crosses, 201)
            cloud.z = np.zeros(len(longitudinal))
            cloud.write(source)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "audit.json"
            write_json(baseline, _report(source, 0.08))
            write_json(candidate, _report(source, 0.0, crosses))

            audit_rail_production_regression(
                [("s0000_0050m", baseline)],
                [("s0000_0050m", candidate)],
                output,
                allow_additional_lines=True,
            )

            report = load_json(output)
            self.assertEqual(report["status"], "pass")
            segment = report["segments"][0]
            self.assertEqual(segment["additional_candidate_cross_positions_m"], [3.0, 4.5])


if __name__ == "__main__":
    unittest.main()
