import tempfile
import unittest
from pathlib import Path

from railway_recon.benchmark import (
    freeze_benchmark,
    initialize_benchmark,
    validate_benchmark_file,
    validate_benchmark_root,
    validate_benchmark_value,
)
from railway_recon.io import load_json, sha256_file, write_json


class BenchmarkTests(unittest.TestCase):
    def test_initialize_creates_valid_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = initialize_benchmark(
                Path(temporary) / "benchmark", "sample-dataset", "scene-a"
            )
            documents = [
                root / "protocols" / "paper_v1.json",
                root / "manifests" / "dataset-manifest.json",
                root / "manifests" / "split-manifest.json",
                root / "annotations" / "ground-truth.json",
                root / "experiments" / "full-auto.json",
                root / "experiments" / "evaluation-input.json",
                root / "failures" / "failure-case.example.json",
            ]
            for document in documents:
                _, errors = validate_benchmark_file(document)
                self.assertEqual(errors, [], document)

    def test_ground_truth_rejects_unknown_relation_endpoint(self) -> None:
        value = {
            "schema_version": "railway.benchmark.ground-truth.v1",
            "ground_truth_id": "sample-gt",
            "dataset_id": "sample-dataset",
            "split_id": "sample-split",
            "truth_level": "GT-L1_observed_consensus",
            "status": "draft",
            "instances": [
                {
                    "id": "TRACK-001",
                    "type": "track",
                    "scene_id": "scene-a",
                    "block_id": "scene-a-000-025",
                    "existence": "present",
                    "geometry_evaluable": True,
                }
            ],
            "relations": [
                {"id": "REL-001", "type": "supports", "from": "TRACK-001", "to": "MISSING"}
            ],
            "exclusion_regions": [],
            "annotation": {
                "blind_to_final_model": True,
                "double_annotation_fraction": 0.2,
                "adjudication_required": True,
            },
        }
        errors = validate_benchmark_value(value)
        self.assertTrue(any("unknown instance" in error for error in errors))

    def test_freeze_computes_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = initialize_benchmark(
                Path(temporary) / "benchmark", "sample-dataset", "scene-a"
            )
            sample = root / "private-input.bin"
            sample.write_bytes(b"synthetic benchmark input")
            dataset_path = root / "manifests" / "dataset-manifest.json"
            dataset = load_json(dataset_path)
            dataset["scenes"][0]["inputs"] = [
                {
                    "id": "point-cloud",
                    "kind": "point_cloud",
                    "path": "private-input.bin",
                    "required": True,
                    "bytes": sample.stat().st_size,
                }
            ]
            write_json(dataset_path, dataset)

            manifest_path = freeze_benchmark(root, full_hash=True)
            manifest = load_json(manifest_path)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["inputs"][0]["sha256"], sha256_file(sample))
            self.assertEqual(manifest["inputs"][0]["hash_source"], "computed")

    def test_cross_file_validation_rejects_overlap_and_unknown_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = initialize_benchmark(
                Path(temporary) / "benchmark", "sample-dataset", "scene-a"
            )
            split_path = root / "manifests" / "split-manifest.json"
            split = load_json(split_path)
            split["blocks"].append(
                {
                    "block_id": "scene-a-overlap",
                    "evaluation_group_id": "scene-a-eval-000-050",
                    "scene_id": "scene-a",
                    "role": "development",
                    "core_start_m": 12.5,
                    "core_end_m": 37.5,
                    "context_before_m": 10.0,
                    "context_after_m": 10.0,
                }
            )
            split["blocks"].append(
                {
                    "block_id": "scene-missing-000-050",
                    "evaluation_group_id": "scene-missing-eval-000-050",
                    "scene_id": "scene-missing",
                    "role": "development",
                    "core_start_m": 0.0,
                    "core_end_m": 50.0,
                    "context_before_m": 0.0,
                    "context_after_m": 0.0,
                }
            )
            write_json(split_path, split)
            _, errors = validate_benchmark_root(root)
            self.assertTrue(any("overlapping score cores" in error for error in errors))
            self.assertTrue(any("unknown scene" in error for error in errors))

    def test_v2_rejects_duplicate_ids_and_unknown_relation_endpoints(self) -> None:
        ground_truth_asset = {
            "id": "GT-001",
            "type": "track",
            "evaluable": True,
            "center_m": [0.0, 0.0, 0.0],
            "block_id": "block-a",
            "evidence_status": "supported",
        }
        prediction = {
            "id": "PRED-001",
            "type": "track",
            "center_m": [0.0, 0.0, 0.0],
            "block_id": "block-a",
            "risk_score": 0.1,
            "declared_evidence_level": "observed",
            "independent_evidence_status": "supported",
            "released": True,
        }
        value = {
            "schema_version": "railway.benchmark.evaluation-input.v2",
            "evaluation_id": "evaluation-v2-invalid",
            "experiment_id": "experiment-v2-invalid",
            "matching": {
                "max_center_distance_m": 1.0,
                "block_restricted": True,
            },
            "ground_truth_assets": [ground_truth_asset, ground_truth_asset],
            "predicted_assets": [prediction],
            "ground_truth_relations": [
                {"type": "supports", "from": "GT-001", "to": "GT-MISSING"}
            ],
            "predicted_relations": [
                {"type": "supports", "from": "PRED-001", "to": "PRED-MISSING"}
            ],
            "distance_sets": [],
            "track_measurements": [],
            "human_review": {"corridor_length_m": 0.0, "events": []},
        }

        errors = validate_benchmark_value(value)
        self.assertTrue(any("duplicate ground-truth asset id" in error for error in errors))
        self.assertTrue(any("unknown ground-truth asset" in error for error in errors))
        self.assertTrue(any("unknown predicted asset" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
