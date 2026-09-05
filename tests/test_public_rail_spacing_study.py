"""Synthetic mathematical fixtures only; never real public/customer inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from railway_recon import public_rail_spacing_study as study
from railway_recon.public_rail_candidates import (
    DEFAULT_PUBLIC_RAIL_POLICY,
    extract_public_rail_candidates,
)
from railway_recon.public_rail_reference import evaluate_public_rail_mask
from railway_recon.synthetic_track_pilot import _write_json
from tests.test_public_rail_candidates import mathematical_corridor


@contextmanager
def fixture(root: Path, count: int = 2):
    samples = []
    for segment in range(1, count + 1):
        source = root / f"cloud900_Seg{segment}.laz"
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.01, 0.01, 0.01]
        cloud = laspy.LasData(header)
        cloud.x = np.linspace(0, 8 + segment, 30)
        cloud.y = np.zeros(30)
        cloud.z = np.zeros(30)
        cloud.classification = np.array([0, 1, 2] * 10, dtype=np.uint8)
        cloud.write(source)
        samples.append({**study.PUBLIC_SAMPLE, "filename": source.name, "parent_cloud": "cloud900",
                        "segment": segment, "point_count": 30,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    receipt_path = root / "receipt.json"
    _write_json(receipt_path, {"samples": [{"filename": sample["filename"], "sha256": sample["sha256"]}
                                          for sample in samples]})
    receipt = {"filename": receipt_path.name, "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    original_sample = copy.deepcopy(samples[0])
    samples = [{**sample, "acquisition_receipt": receipt} for sample in samples]
    metadata = root / "metadata"
    metadata.mkdir()
    records = {
        "zenodo_15641832.json": json.dumps({"id": 15641832, "doi": study.DATASET["doi"],
            "metadata": {"license": {"id": "cc-by-4.0"}, "access_right": "open"}}),
        "upstream_README.md": "Independent temporary mathematical fixture, not real metadata.",
        "train_clouds.txt": "cloud900_Seg 1,2,3\n", "val_clouds.txt": "cloud901_Seg 1\n",
        "test_clouds.txt": "cloud902_Seg 1\n", "removed_clouds.txt": "cloud903_Seg 1\n",
    }
    hashes = {}
    for name, content in records.items():
        data = content.encode("utf-8")
        (metadata / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    with patch.object(study, "PUBLIC_SAMPLE", original_sample), \
         patch.object(study, "DEFAULT_RECEIPT", receipt), \
         patch.object(study, "PUBLIC_METADATA_SHA256", hashes):
        protocol = root / "protocol.json"
        _write_json(protocol, study.create_public_spacing_protocol(samples))
        yield samples, protocol


class PublicRailSpacingStudyTests(unittest.TestCase):
    def test_protocol_changes_only_maximum_spacing_and_discloses_training_adaptation(self):
        protocol = study.create_public_spacing_protocol()
        study.validate_public_spacing_protocol(protocol)
        self.assertTrue(protocol["development_exposure"]["original_pilot_already_viewed"])
        self.assertEqual(protocol["reference_contract"]["target_name"], "numeric_class_1")
        self.assertEqual(protocol["policies"]["height_prominence"], DEFAULT_PUBLIC_RAIL_POLICY)
        self.assertEqual(protocol["policies"]["paired_default"], DEFAULT_PUBLIC_RAIL_POLICY)
        expected = {**DEFAULT_PUBLIC_RAIL_POLICY, "rail_pair_maximum_m": 1.90}
        self.assertEqual(protocol["policies"]["paired_broad_spacing"], expected)
        self.assertEqual(protocol["candidate_input"], ["xyz"])
        self.assertFalse(protocol["connection_truth_available"])
        self.assertFalse(protocol["absolute_accuracy_available"])

    def test_protocol_rejects_undeclared_tuning_or_semantic_claims(self):
        original = study.create_public_spacing_protocol()
        for mutate in (
            lambda value: value["policies"]["paired_broad_spacing"].update(rail_pair_maximum_m=2.0),
            lambda value: value["reference_contract"].update(target_name="rail"),
            lambda value: value.update(modes=["paired_broad_spacing"]),
            lambda value: value.update(role="held_out"),
        ):
            value = copy.deepcopy(original)
            mutate(value)
            with self.assertRaises(ValueError):
                study.validate_public_spacing_protocol(value)

    def test_samples_cannot_drop_original_or_add_test_or_unsafe_paths(self):
        original = study.create_public_spacing_protocol()["samples"]
        mutations = (
            {"filename": "../cloud901_Seg1.laz"}, {"split": "test"},
            {"point_count": True}, {"point_count": 2_000_001}, {"segment": True},
            {"sha256": "g" * 64}, {"selection": "../private"},
            {"acquisition_receipt": {"filename": "../receipt.json", "sha256": "a" * 64}},
        )
        for mutation in mutations:
            added = {**original[0], "filename": "cloud901_Seg1.laz", "parent_cloud": "cloud901",
                     "sha256": "a" * 64, **mutation}
            with self.assertRaises(ValueError):
                study.create_public_spacing_protocol([*original, added])
        with self.assertRaises(ValueError):
            study.create_public_spacing_protocol([])
        with self.assertRaises(ValueError):
            study.create_public_spacing_protocol([original[0], original[0]])
        with self.assertRaises(ValueError):
            study.create_public_spacing_protocol([{**original[0], "split": "test"}])

    def test_broad_spacing_accepts_mathematical_175_without_changing_pool_or_geometry(self):
        xyz, _ = mathematical_corridor(spacing=1.75)
        protocol = study.create_public_spacing_protocol()
        results = {mode: extract_public_rail_candidates(xyz, protocol["policies"][mode],
                                                       protocol["extractor_modes"][mode])
                   for mode in study.SPACING_MODES}
        self.assertFalse(results["paired_default"]["point_mask"].any())
        broad = results["paired_broad_spacing"]
        self.assertTrue(broad["point_mask"].any())
        self.assertEqual(len(broad["report"]["rail_pairs"]), 1)
        self.assertAlmostEqual(broad["report"]["rail_pairs"][0]["observed_center_spacing_m"], 1.75, delta=0.06)
        self.assertFalse(broad["report"]["rail_pairs"][0]["nominal_gauge_correction"])
        for mode in study.SPACING_MODES:
            self.assertEqual(results[mode]["report"]["candidate_pool"], results["height_prominence"]["report"]["candidate_pool"])
            self.assertFalse(np.any(results[mode]["point_mask"] & ~results["height_prominence"]["point_mask"]))

    def test_broad_spacing_does_not_accept_outside_bound_or_disjoint_support(self):
        policy = study.create_public_spacing_protocol()["policies"]["paired_broad_spacing"]
        for kwargs in ({"spacing": 2.1}, {"spacing": 1.75, "disjoint_support": True}):
            xyz, _ = mathematical_corridor(**kwargs)
            self.assertFalse(extract_public_rail_candidates(xyz, policy)["point_mask"].any())

    def test_receipt_requires_hash_and_same_record_filename_digest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fixture(root) as (samples, _):
                self.assertEqual(study.verify_spacing_receipts(root, samples)["status"], "hash_and_sample_binding_verified")
                changed = copy.deepcopy(samples)
                changed[1]["sha256"] = "1" * 64
                with self.assertRaisesRegex(ValueError, "does not bind"):
                    study.verify_spacing_receipts(root, changed)
                (root / "receipt.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    study.verify_spacing_receipts(root, samples)
        self.assertTrue(study._receipt_binds({"sample_file": "one.laz", "sample_sha256": "a"}, "one.laz", "a"))
        self.assertFalse(study._receipt_binds({"a": {"filename": "one.laz"}, "b": {"sha256": "a"}}, "one.laz", "a"))

    def test_complete_run_locks_all_sample_predictions_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "output"
            seen = []

            def extract(xyz, policy, mode):
                self.assertEqual(xyz.shape, (30, 3))
                self.assertFalse(xyz.flags.writeable)
                seen.append(mode)
                return extract_public_rail_candidates(xyz, policy, mode)

            def evaluate(*args):
                self.assertEqual(len(seen), 6)
                self.assertTrue((destination / "prediction_lock.json").is_file())
                return evaluate_public_rail_mask(*args)

            with fixture(root) as (_, protocol), \
                 patch("railway_recon.public_rail_candidates.extract_public_rail_candidates", side_effect=extract), \
                 patch("railway_recon.public_rail_reference.evaluate_public_rail_mask", side_effect=evaluate):
                report = study.run_public_spacing_study(root, root / "metadata", root, protocol, destination)
                self.assertEqual(report["sample_count"], 2)
                self.assertEqual(report["parent_cloud_count"], 1)
                self.assertEqual(report["by_parent_cloud"]["cloud900"]["segment_count"], 2)
                self.assertEqual(report["total_point_count"], 60)
                self.assertFalse(report["held_out_test"])
                self.assertIsNone(report["inference"]["confidence_intervals"])
                pooled = report["pooled_descriptive_evaluation"]["paired_broad_spacing"]
                self.assertEqual(pooled["all_points"]["fn"], 20)
                self.assertEqual(pooled["annotated_only"]["evaluated_points"], 40)
                with self.assertRaises(FileExistsError):
                    study.run_public_spacing_study(root, root / "metadata", root, protocol, destination)
            for item in report["samples"]:
                for mode in study.SPACING_MODES:
                    sample_root = destination / "samples" / Path(item["source"]["filename"]).stem
                    with np.load(sample_root / "evaluation" / mode / "error_indices.npz") as indices:
                        for view in ("all_points", "annotated_only"):
                            for metric in ("tp", "fp", "fn", "tn"):
                                self.assertEqual(len(indices[f"{view}_{metric}"]), item["evaluation_by_mode"][mode][view][metric])
            manifest = json.loads((destination / "manifest.json").read_text())
            for record in manifest["files"]:
                data = (destination / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(len(data), record["bytes"])
            lock = json.loads((destination / "prediction_lock.json").read_text())
            self.assertEqual(len(lock["files"]), 12)
            for relative, digest in lock["files"].items():
                self.assertEqual(hashlib.sha256((destination / relative).read_bytes()).hexdigest(), digest)

    def test_changed_source_fails_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fixture(root, count=1) as (_, protocol), \
                 patch.object(study, "_source_bindings", side_effect=[{"source": "a"}, {"source": "b"}]), \
                 self.assertRaisesRegex(ValueError, "Source code"):
                study.run_public_spacing_study(root, root / "metadata", root, protocol, root / "output")
            self.assertTrue((root / "output" / "run_failure.json").is_file())
            self.assertFalse((root / "output" / "report.json").exists())
            self.assertFalse((root / "output" / "manifest.json").exists())

    def test_changed_prediction_after_lock_fails_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "output"
            changed = False

            def evaluate(*args):
                nonlocal changed
                result = evaluate_public_rail_mask(*args)
                if not changed:
                    path = destination / "samples" / "cloud900_Seg1" / "predictions" / "height_prominence" / "candidates.json"
                    path.write_text("{}", encoding="utf-8")
                    changed = True
                return result

            with fixture(root, count=1) as (_, protocol), \
                 patch("railway_recon.public_rail_reference.evaluate_public_rail_mask", side_effect=evaluate), \
                 self.assertRaisesRegex(ValueError, "Prediction bytes changed"):
                study.run_public_spacing_study(root, root / "metadata", root, protocol, destination)
            self.assertFalse((destination / "manifest.json").exists())
            self.assertFalse((destination / "report.json").exists())

    def test_empty_prediction_metrics_are_not_perfect(self):
        metrics = study._pooled_counts([{"tp": 0, "fp": 0, "fn": 10, "tn": 20}])
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)


if __name__ == "__main__":
    unittest.main()
