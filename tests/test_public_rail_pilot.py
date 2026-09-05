"""All inputs below are temporary mathematical LAS files, never site data."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from railway_recon import public_rail_pilot as pilot
from railway_recon.synthetic_track_pilot import _write_json


def _fixture(root: Path) -> tuple[Path, dict, dict]:
    source = root / "cloud900_Seg1.laz"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    cloud = laspy.LasData(header)
    cloud.x = np.linspace(0, 8, 30)
    cloud.y = np.zeros(30)
    cloud.z = np.zeros(30)
    cloud.classification = np.array([0, 1, 2] * 10, dtype=np.uint8)
    cloud.write(source)
    metadata = root / "metadata"
    metadata.mkdir()
    records = {
        "zenodo_15641832.json": json.dumps({"id": 15641832, "doi": pilot.DATASET["doi"],
            "metadata": {"license": {"id": "cc-by-4.0"}, "access_right": "open"}}),
        "upstream_README.md": "Independent unit-test fixture, not real dataset metadata.",
        "train_clouds.txt": "cloud900_Seg 1,2\n", "val_clouds.txt": "cloud901_Seg 1\n",
        "test_clouds.txt": "cloud902_Seg 1\n", "removed_clouds.txt": "cloud903_Seg 1\n",
    }
    hashes = {}
    for name, content in records.items():
        data = content.encode("utf-8")
        (metadata / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    sample = {**pilot.PUBLIC_SAMPLE, "filename": source.name, "parent_cloud": "cloud900",
              "point_count": 30, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return source, sample, hashes


class PublicRailPilotTests(unittest.TestCase):
    def test_protocol_is_fixed_and_separates_reference_from_candidate_input(self):
        value = pilot.create_public_rail_protocol()
        pilot.validate_public_rail_protocol(value)
        self.assertEqual(value["candidate_input"], ["xyz"])
        self.assertEqual(value["sample"]["split"], "train")
        self.assertFalse(value["connection_truth_available"])
        for key, replacement in (("sample", {**value["sample"], "split": "test"}),
                                 ("policy", {**value["policy"], "cross_bin_m": 1.0}),
                                 ("reference_contract", {})):
            changed = copy.deepcopy(value)
            changed[key] = replacement
            with self.assertRaises(ValueError):
                pilot.validate_public_rail_protocol(changed)

    def test_split_parser_rejects_invalid_and_duplicate_entries(self):
        self.assertEqual(pilot._parse_split("cloud1_Seg 1,2\n"), {("cloud1", 1), ("cloud1", 2)})
        for text in ("../private", "cloud1_Seg 6", "cloud1_Seg 1,1"):
            with self.assertRaises(ValueError):
                pilot._parse_split(text)

    def test_metadata_hash_and_retained_train_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, sample, hashes = _fixture(root)
            protocol = {"sample": sample, "metadata_sha256": hashes}
            self.assertTrue(pilot.verify_public_metadata(root / "metadata", protocol)["retained_train_member"])
            protocol["sample"] = {**sample, "segment": 3}
            with self.assertRaises(ValueError):
                pilot.verify_public_metadata(root / "metadata", protocol)
            protocol["sample"] = sample
            (root / "metadata" / "train_clouds.txt").write_text("cloud901_Seg 1\n")
            with self.assertRaises(ValueError):
                pilot.verify_public_metadata(root / "metadata", protocol)

    def test_pinned_arrays_decode_all_points_and_reject_wrong_hash_or_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, sample, _ = _fixture(Path(tmp))
            xyz, labels = pilot.load_pinned_public_arrays(source, sample["sha256"], 30)
            self.assertEqual(xyz.shape, (30, 3))
            np.testing.assert_array_equal(labels, np.array([0, 1, 2] * 10))
            for digest, count in (("0" * 64, 30), (sample["sha256"], 29), (sample["sha256"], True)):
                with self.assertRaises(ValueError):
                    pilot.load_pinned_public_arrays(source, digest, count)

    def test_complete_run_locks_predictions_before_evaluation_and_verifies_outputs(self):
        from railway_recon.public_rail_candidates import extract_public_rail_candidates
        from railway_recon.public_rail_reference import evaluate_public_rail_mask

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, sample, hashes = _fixture(root)
            destination = root / "output"
            seen = []

            def extract(xyz, policy, mode):
                self.assertEqual(xyz.shape, (30, 3))
                self.assertFalse(xyz.flags.writeable)
                seen.append(mode)
                return extract_public_rail_candidates(xyz, policy, mode)

            def evaluate(*args):
                self.assertEqual(len(seen), 2)
                self.assertTrue((destination / "prediction_lock.json").is_file())
                return evaluate_public_rail_mask(*args)

            with patch.object(pilot, "PUBLIC_SAMPLE", sample), patch.object(pilot, "PUBLIC_METADATA_SHA256", hashes):
                protocol = root / "protocol.json"
                _write_json(protocol, pilot.create_public_rail_protocol())
                with patch("railway_recon.public_rail_candidates.extract_public_rail_candidates", side_effect=extract), \
                     patch("railway_recon.public_rail_reference.evaluate_public_rail_mask", side_effect=evaluate):
                    report = pilot.run_public_rail_pilot(source, root / "metadata", protocol, destination)
                self.assertEqual(report["point_count"], 30)
                self.assertFalse(report["held_out_test"])
                self.assertFalse(report["connection_accuracy"]["available"])
                with self.assertRaises(FileExistsError):
                    pilot.run_public_rail_pilot(source, root / "metadata", protocol, destination)
            with np.load(destination / "inputs" / "xyz.npz") as arrays:
                self.assertEqual(arrays.files, ["xyz"])
            manifest = json.loads((destination / "manifest.json").read_text())
            for record in manifest["files"]:
                data = (destination / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                self.assertEqual(len(data), record["bytes"])
            lock = json.loads((destination / "prediction_lock.json").read_text())
            for relative, digest in lock["files"].items():
                self.assertEqual(hashlib.sha256((destination / relative).read_bytes()).hexdigest(), digest)

    def test_changed_source_fails_without_success_report_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, sample, hashes = _fixture(root)
            with patch.object(pilot, "PUBLIC_SAMPLE", sample), patch.object(pilot, "PUBLIC_METADATA_SHA256", hashes):
                protocol = root / "protocol.json"
                _write_json(protocol, pilot.create_public_rail_protocol())
                with patch.object(pilot, "_source_bindings", side_effect=[{"code": "a"}, {"code": "b"}]), \
                     self.assertRaisesRegex(ValueError, "Source code"):
                    pilot.run_public_rail_pilot(source, root / "metadata", protocol, root / "output")
            self.assertTrue((root / "output" / "run_failure.json").is_file())
            self.assertFalse((root / "output" / "report.json").exists())
            self.assertFalse((root / "output" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
