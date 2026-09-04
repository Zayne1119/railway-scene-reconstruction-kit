import tempfile
import unittest
from pathlib import Path

from railway_recon.benchmark import validate_benchmark_file, validate_benchmark_root
from railway_recon.io import load_json, sha256_file
from railway_recon.research_protocol import (
    freeze_research_baseline,
    initialize_prospective_benchmark,
    register_prospective_inputs,
    seal_prospective_intake,
    verify_research_baseline,
)


class ResearchProtocolTests(unittest.TestCase):
    def test_baseline_freeze_hashes_without_modifying_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            source = root / "src" / "railway_recon" / "method.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='sample'\n", encoding="utf-8")
            before_hash = sha256_file(source)
            before_mtime = source.stat().st_mtime_ns

            output = freeze_research_baseline(
                root,
                root / "research_freezes" / "baseline-v1.json",
                includes=["src/railway_recon", "pyproject.toml"],
            )

            manifest = load_json(output)
            self.assertTrue(manifest["non_mutating"])
            self.assertFalse(manifest["production_algorithms_modified_by_freeze"])
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(sha256_file(source), before_hash)
            self.assertEqual(source.stat().st_mtime_ns, before_mtime)
            verification = verify_research_baseline(root, output)
            self.assertEqual(verification["status"], "pass")

            source.write_text("VALUE = 2\n", encoding="utf-8")
            changed = verify_research_baseline(root, output)
            self.assertEqual(changed["status"], "fail")
            self.assertTrue(any("SHA-256 mismatch" in issue for issue in changed["issues"]))

    def test_prospective_init_is_isolated_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            production = temporary_path / "production-config.json"
            production.write_text('{"threshold": 0.05}\n', encoding="utf-8")
            production_hash = sha256_file(production)

            root = initialize_prospective_benchmark(
                temporary_path / "site-b",
                "railway-site-b",
                "station-b",
                chainage_end_m=200.0,
            )

            values, errors = validate_benchmark_root(root)
            self.assertEqual(errors, [])
            self.assertEqual(values["dataset"]["role"], "prospective_blind_test")
            self.assertEqual(len(values["split"]["blocks"]), 8)
            self.assertTrue(all(block["role"] == "test" for block in values["split"]["blocks"]))
            truth, truth_errors = validate_benchmark_file(root / "annotations" / "ground-truth.json")
            self.assertEqual(truth_errors, [])
            self.assertEqual(truth["truth_level"], "GT-L1S_observed_single_reviewer")
            self.assertEqual(truth["annotation"]["double_annotation_fraction"], 0.0)
            state = load_json(root / "blind-run-state.json")
            self.assertFalse(state["production_defaults_modified"])
            self.assertTrue(state["output_policy"]["never_overwrite_b0"])
            self.assertEqual(sha256_file(production), production_hash)

    def test_register_and_seal_full_hashes_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = initialize_prospective_benchmark(
                temporary_path / "site-b", "railway-site-b", "station-b"
            )
            point_cloud = temporary_path / "site.laz"
            point_cloud.write_bytes(b"point-cloud")
            camera = temporary_path / "camera.csv"
            camera.write_text("id,x,y,z\n1,0,0,0\n", encoding="utf-8")
            panoramas = temporary_path / "panoramas"
            panoramas.mkdir()
            (panoramas / "001.jpg").write_bytes(b"image-1")
            (panoramas / "002.jpg").write_bytes(b"image-2")

            register_prospective_inputs(
                root,
                point_cloud,
                camera_poses=camera,
                panoramas=panoramas,
                coordinate_mode="local",
            )
            freeze_path = seal_prospective_intake(root)

            freeze = load_json(freeze_path)
            self.assertEqual(freeze["status"], "completed")
            by_id = {item["id"]: item for item in freeze["inputs"]}
            self.assertEqual(by_id["point-cloud"]["sha256"], sha256_file(point_cloud))
            self.assertTrue(by_id["panoramas"]["exists"])
            self.assertEqual(by_id["panoramas"]["file_count"], 2)
            self.assertEqual(by_id["panoramas"]["hash_source"], "computed")
            state = load_json(root / "blind-run-state.json")
            self.assertEqual(state["status"], "intake_frozen_ready_for_experiment_lock")

    def test_prospective_blocks_require_atomic_length(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "positive multiple"),
        ):
            initialize_prospective_benchmark(
                Path(temporary) / "site-b",
                "railway-site-b",
                "station-b",
                chainage_end_m=210.0,
            )


if __name__ == "__main__":
    unittest.main()
