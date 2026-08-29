import tempfile
import unittest
from pathlib import Path

from railway_recon.benchmark import freeze_benchmark, initialize_benchmark
from railway_recon.experiment_lock import lock_benchmark_experiment
from railway_recon.io import load_json, sha256_file, write_json


class ExperimentLockTests(unittest.TestCase):
    def test_lock_binds_executable_variant_and_frozen_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = initialize_benchmark(
                Path(temporary) / "benchmark", "sample-dataset", "scene-a"
            )
            input_path = root / "input.bin"
            input_path.write_bytes(b"frozen input")
            dataset_path = root / "manifests" / "dataset-manifest.json"
            dataset = load_json(dataset_path)
            dataset["scenes"][0]["inputs"] = [
                {
                    "id": "point-cloud",
                    "kind": "point_cloud",
                    "path": "input.bin",
                    "required": True,
                }
            ]
            write_json(dataset_path, dataset)
            freeze_path = freeze_benchmark(root, full_hash=True)

            experiment_path = root / "experiments" / "full-auto.json"
            experiment = load_json(experiment_path)
            experiment["command"] = ["railway-recon", "detect-rails", "--segment", "s000_50m"]
            write_json(experiment_path, experiment)
            config_path = root / "method-config.json"
            config_path.write_text('{"threshold": 0.1}\n', encoding="utf-8")

            output = lock_benchmark_experiment(
                root,
                experiment_path,
                [("method_config", config_path)],
                freeze_manifest_path=freeze_path,
            )
            locked = load_json(output)
            self.assertEqual(locked["status"], "locked")
            self.assertTrue(locked["locked_before_test"])
            self.assertTrue(locked["lock"]["ground_truth_not_used_by_lock"])
            self.assertEqual(
                locked["lock"]["bindings"][0]["sha256"], sha256_file(config_path)
            )

    def test_empty_command_cannot_be_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = initialize_benchmark(
                Path(temporary) / "benchmark", "sample-dataset", "scene-a"
            )
            freeze_path = freeze_benchmark(root, full_hash=True)
            binding = root / "config.json"
            binding.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "command is empty"):
                lock_benchmark_experiment(
                    root,
                    root / "experiments" / "full-auto.json",
                    [("method_config", binding)],
                    freeze_manifest_path=freeze_path,
                )


if __name__ == "__main__":
    unittest.main()
