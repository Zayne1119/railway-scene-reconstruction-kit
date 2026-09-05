from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from railway_recon.synthetic_track_formal_analysis import (
    ANALYSIS_CONTRACT,
    _safe_file,
    analyze_guarded_run,
    create_formal_analysis_plan,
    paired_layout_interval,
    validate_formal_analysis_plan,
    verify_completed_run,
)
from railway_recon.synthetic_track_guarded_protocol import (
    REQUIRED_EXECUTION_FILES as GUARDED_EXECUTION_FILES,
)
from railway_recon.synthetic_track_guarded_protocol import create_guarded_protocol
from railway_recon.synthetic_track_guarded_study import run_guarded_split
from railway_recon.synthetic_track_pilot import _write_json
from railway_recon.synthetic_track_study_protocol import create_study_protocol


class IntervalTests(unittest.TestCase):
    def test_constant_zero_is_not_equivalence_or_population_certainty(self):
        result = paired_layout_interval([0.0] * 60)
        self.assertEqual(result["percentile_95_interval"], [0.0, 0.0])
        self.assertTrue(result["degenerate_empirical_distribution"])
        self.assertFalse(result["equivalence_established"])
        self.assertFalse(result["superiority_established"])
        self.assertIsNone(result["p_value"])
        self.assertIn("not_population_certainty", result["interval_interpretation"])

    def test_constant_gain_is_not_automatic_superiority(self):
        result = paired_layout_interval([0.25] * 4)
        self.assertEqual(result["mean_difference"], 0.25)
        self.assertEqual(result["percentile_95_interval"], [0.25, 0.25])
        self.assertTrue(result["degenerate_empirical_distribution"])
        self.assertFalse(result["superiority_established"])

    def test_bootstrap_matches_direct_fixed_paired_numpy_calculation(self):
        values = np.array([0.0, 0.25, -0.5, 0.125])
        generator = np.random.Generator(np.random.PCG64(947320260909))
        replicates = values[generator.integers(0, 4, size=(10000, 4))].mean(axis=1)
        expected = np.quantile(replicates, [0.025, 0.975], method="linear").tolist()
        result = paired_layout_interval(values.tolist())
        self.assertEqual(result["percentile_95_interval"], expected)
        self.assertEqual(result["mean_difference"], float(values.mean()))
        self.assertEqual(result, paired_layout_interval(values.tolist()))
        self.assertFalse(result["degenerate_empirical_distribution"])

    def test_empty_nonfinite_multidimensional_out_of_range_rejected(self):
        for values in ([], [float("nan")], [float("inf")], [[0.0]], [1.01]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                paired_layout_interval(values)


class FormalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="railway-formal-fixture-")
        cls.directory = Path(cls.temporary.name)
        base = create_study_protocol(seed=53417, counts={"development": 2, "validation": 1, "test": 1})
        cls.protocol = cls.directory / "protocol.json"
        _write_json(cls.protocol, create_guarded_protocol(base))
        cls.run_directory = cls.directory / "development"
        run_guarded_split(cls.protocol, "development", cls.run_directory)
        cls.plan = cls.directory / "plan.json"
        _write_json(cls.plan, create_formal_analysis_plan(cls.protocol))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway-formal-test-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def altered_run(self):
        destination = self.work / "run"
        shutil.copytree(self.run_directory, destination)
        return destination

    def rewrite_bound_json(self, directory, relative, value):
        path = directory / relative
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        path.write_bytes(payload)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        record = next(item for item in manifest["files"] if item["path"] == relative)
        record.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_plan_creation_does_not_execute_a_case_generator(self):
        with patch("railway_recon.synthetic_track_study_scene.generate_study_layout") as generate:
            plan = create_formal_analysis_plan(self.protocol)
            validate_formal_analysis_plan(plan)
            generate.assert_not_called()
        self.assertEqual(plan["contract"]["primary"]["comparison"], "direction_geometry")
        self.assertEqual(plan["contract"]["bootstrap"]["resamples"], 10000)
        self.assertIn("+00:00", plan["created_at_utc"])
        self.assertEqual(plan["guarded_protocol_sha256"], hashlib.sha256(self.protocol.read_bytes()).hexdigest())

    def test_contract_mutation_and_source_changes_rejected(self):
        plan = create_formal_analysis_plan(self.protocol)
        plan["contract"]["bootstrap"]["resamples"] = 100
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_formal_analysis_plan(plan)
        plan = create_formal_analysis_plan(self.protocol)
        with (
            patch("railway_recon.synthetic_track_formal_analysis._bindings", return_value={}),
            self.assertRaisesRegex(ValueError, "source changed"),
        ):
            validate_formal_analysis_plan(plan)

    def test_plan_binds_full_guarded_execution_closure(self):
        plan = create_formal_analysis_plan(self.protocol)
        self.assertTrue(set(GUARDED_EXECUTION_FILES).issubset(plan["source_sha256"]))

    def test_changed_relationship_helper_or_guarded_cli_invalidates_plan(self):
        plan = create_formal_analysis_plan(self.protocol)
        repository = Path(__file__).resolve().parents[1]
        original_read = Path.read_bytes
        for relative in ("src/railway_recon/relationships.py",
                         "scripts/run_synthetic_track_guarded_study.py"):
            target = (repository / relative).resolve()

            def changed_read(path, target=target):
                payload = original_read(path)
                return payload + b"\n# simulated change\n" if path.resolve() == target else payload

            with (self.subTest(relative=relative), patch.object(Path, "read_bytes", changed_read),
                  self.assertRaisesRegex(ValueError, "source changed")):
                validate_formal_analysis_plan(plan)

    def test_entire_development_all_predictions_rescored_and_outputs_bound(self):
        result = analyze_guarded_run(self.run_directory, self.plan, self.work / "analysis")
        self.assertEqual(result["layout_count"], 2)
        self.assertEqual(result["case_count"], 24)
        self.assertTrue(result["all_allocated_groups_retained"])
        self.assertEqual(result["comparisons"][0]["role"], "primary")
        self.assertEqual(result["comparisons"][0]["comparison"],
                         "guarded_evidence_minus_direction_geometry")
        self.assertEqual(len(result["comparisons"]), 5)
        self.assertEqual(len(result["by_condition"]), 12)
        self.assertEqual(result["summary_by_mode"]["guarded_evidence"]["normal_case_count"], 10)
        output_manifest = json.loads((self.work / "analysis" / "manifest.json").read_bytes())
        for record in output_manifest["files"]:
            payload = (self.work / "analysis" / record["path"]).read_bytes()
            self.assertEqual(len(payload), record["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])

    def test_prediction_tamper_rejected_before_output(self):
        altered = self.altered_run()
        first = next((altered / "predictions" / "guarded_evidence").glob("*.json"))
        first.write_bytes(first.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            analyze_guarded_run(altered, self.plan, self.work / "analysis")
        self.assertFalse((self.work / "analysis").exists())

    def test_unbound_extra_file_rejected(self):
        altered = self.altered_run()
        (altered / "unbound.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "every artifact"):
            verify_completed_run(altered)

    def test_missing_method_source_binding_rejected(self):
        altered = self.altered_run()
        path = altered / "manifest.json"
        manifest = json.loads(path.read_bytes())
        del manifest["source_sha256"]["src/railway_recon/synthetic_track_guarded_audit.py"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "method/evaluator"):
            analyze_guarded_run(altered, self.plan, self.work / "analysis")

    def test_failed_run_not_formally_analyzed(self):
        altered = self.altered_run()
        (altered / "run_failure.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Failed runs"):
            analyze_guarded_run(altered, self.plan, self.work / "analysis")

    def test_modified_score_even_with_updated_manifest_is_rejected(self):
        altered = self.altered_run()
        report = json.loads((altered / "report.json").read_bytes())
        report["summary_by_mode"]["guarded_evidence"]["diagnosis"]["tp"] += 1
        self.rewrite_bound_json(altered, "report.json", report)
        with self.assertRaisesRegex(ValueError, "Recomputed saved predictions"):
            analyze_guarded_run(altered, self.plan, self.work / "analysis")

    def test_missing_layout_cannot_be_silently_filtered(self):
        altered = self.altered_run()
        index = json.loads((altered / "case_index.json").read_bytes())
        first_group = index[0]["group_id"]
        self.rewrite_bound_json(altered, "case_index.json",
                                [item for item in index if item["group_id"] == first_group])
        with self.assertRaisesRegex(ValueError, "all allocated groups"):
            analyze_guarded_run(altered, self.plan, self.work / "analysis")

    def test_unsafe_manifest_paths_rejected(self):
        for relative in ("../report.json", "/report.json", "x/../report.json", "x\\report.json", "C:/x"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                _safe_file(self.work, relative)

    def test_existing_or_nested_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            analyze_guarded_run(self.run_directory, self.plan, self.work)
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            analyze_guarded_run(self.run_directory, self.plan, self.run_directory / "nested")

    def test_modified_protocol_binding_rejected(self):
        plan = json.loads(self.plan.read_bytes())
        plan["guarded_protocol_sha256"] = "0" * 64
        path = self.work / "bad-plan.json"
        _write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "plan-bound protocol"):
            analyze_guarded_run(self.run_directory, path, self.work / "analysis")

    def test_disclosures_cannot_be_changed_to_blind_development(self):
        plan = create_formal_analysis_plan(self.protocol)
        plan["prior_data_access"]["validation_v1_v2_v3_observed"] = False
        with self.assertRaisesRegex(ValueError, "disclosures"):
            validate_formal_analysis_plan(plan)
        self.assertEqual(ANALYSIS_CONTRACT, copy.deepcopy(ANALYSIS_CONTRACT))
