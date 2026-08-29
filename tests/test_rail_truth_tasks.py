import csv
import tempfile
import unittest
from pathlib import Path

from railway_recon.io import load_json, write_json
from railway_recon.rail_truth_tasks import (
    create_rail_truth_annotation_package,
    validate_rail_truth_annotation_package,
)


class RailTruthTaskTests(unittest.TestCase):
    def test_package_is_double_blind_and_uses_neutral_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark"
            (benchmark / "annotations").mkdir(parents=True)
            stations = []
            for index in range(3):
                cross = root / f"cross-{index}.png"
                plan = root / f"plan-{index}.png"
                cross.write_bytes(b"cross-section")
                plan.write_bytes(b"plan-strip")
                stations.append(
                    {
                        "task_id": f"RAIL-TRUTH-{index:03d}",
                        "segment_id": "s000_50m",
                        "block_id": "scene-a-000-025",
                        "chainage_m": float(index),
                        "cross_section_image": str(cross),
                        "plan_strip_image": str(plan),
                        "point_count": 100,
                        "cross_min_m": -20.0,
                        "cross_max_m": 20.0,
                        "z_min_m": 20.0,
                        "z_max_m": 25.0,
                        "slice_half_width_m": 0.25,
                    }
                )
            evidence_manifest = root / "neutral-evidence.json"
            write_json(
                evidence_manifest,
                {
                    "schema_version": "railway.rail-neutral-evidence.v1",
                    "scene_id": "scene-a",
                    "evidence_source": "raw_point_cloud",
                    "model_overlay": False,
                    "candidate_overlay": False,
                    "stations": stations,
                },
            )

            package = create_rail_truth_annotation_package(
                benchmark, [evidence_manifest], seed=42
            )
            manifest = load_json(package / "manifest.json")
            self.assertTrue(manifest["blind_to_existing_model"])
            self.assertFalse(manifest["model_overlay"])
            orders = []
            for name in manifest["reviewer_files"]:
                with (package / name).open(
                    "r", encoding="utf-8-sig", newline=""
                ) as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(all(not row["track_pairs"] for row in rows))
                orders.append([row["task_id"] for row in rows])
            self.assertNotEqual(orders[0], orders[1])
            report = validate_rail_truth_annotation_package(package)
            self.assertTrue(report["passed"])
            self.assertEqual(report["evidence_file_count"], 6)

    def test_candidate_overlay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "benchmark" / "annotations").mkdir(parents=True)
            evidence_manifest = root / "leaking-evidence.json"
            write_json(
                evidence_manifest,
                {
                    "schema_version": "railway.rail-neutral-evidence.v1",
                    "scene_id": "scene-a",
                    "evidence_source": "raw_point_cloud",
                    "model_overlay": False,
                    "candidate_overlay": True,
                    "stations": [{}],
                },
            )
            with self.assertRaisesRegex(ValueError, "candidate_overlay=false"):
                create_rail_truth_annotation_package(
                    root / "benchmark", [evidence_manifest]
                )


if __name__ == "__main__":
    unittest.main()
