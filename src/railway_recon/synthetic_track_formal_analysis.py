"""Locked, layout-paired analysis of completed six-mode synthetic runs.

This module never imports a case generator or executes an auditor. It verifies
saved artifacts, re-scores saved predictions, and retains every allocated group.
Bootstrap intervals describe the observed synthetic design, not field accuracy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

from .synthetic_track_adaptive_study import (
    aggregate_adaptive_evaluations,
    evaluate_adaptive_prediction,
)
from .synthetic_track_guarded_protocol import GUARDED_MODES, validate_guarded_protocol
from .synthetic_track_guarded_protocol import REQUIRED_EXECUTION_FILES as GUARDED_EXECUTION_FILES
from .synthetic_track_pilot import _canonical_bytes, _write_json

ANALYSIS_CONTRACT = {
    "modes": list(GUARDED_MODES),
    "primary": {"method": "guarded_evidence", "comparison": "direction_geometry"},
    "endpoint": "mean_layout_correct_diagnosis_rate_over_truth_faults",
    "fault_denominator": "all_truth_faults_in_all_variants_within_each_layout",
    "abstention": "diagnosis_false_negative_not_removed_from_denominator",
    "weighting": "equal_layout_weights_not_independent_variant_weights",
    "secondary": "guarded_minus_each_other_mode_exploratory_no_multiplicity_claim",
    "bootstrap": {
        "unit": "whole_layout_with_all_conditions_and_modes_paired",
        "resamples": 10000,
        "seed": 947320260909,
        "rng": "numpy_PCG64",
        "interval": "percentile",
        "confidence_level": 0.95,
        "quantile_method": "linear",
        "stratification": "none_design_mixture_not_population_sampling",
    },
    "all_normal_false_alarms_reported": True,
    "all_abstentions_reported": True,
    "p_values": False,
    "automatic_superiority_claim": False,
    "automatic_equivalence_claim": False,
    "degenerate_interval": "observed_invariance_not_population_certainty_or_equivalence",
    "external_baseline_status": "six_internal_control_policies_not_external_SOTA",
    "scope": "synthetic_centerline_and_observation_proxies_not_field_or_LiDAR_accuracy",
}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bindings() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = (
        "synthetic_track_formal_analysis.py", "synthetic_track_adaptive_study.py",
        "synthetic_track_study.py", "synthetic_track_pilot.py",
        "synthetic_track_study_audit.py", "synthetic_track_adaptive_audit.py",
        "synthetic_track_guarded_audit.py", "synthetic_track_guarded_protocol.py",
        "synthetic_track_guarded_study.py", "synthetic_track_study_protocol.py",
        "synthetic_track_adaptive_protocol.py", "synthetic_track_scene.py",
        "synthetic_track_study_scene.py",
    )
    # Bind the declared full execution closure, including relationship/scoring
    # helpers and protocol CLIs, not merely the directly imported audit modules.
    relative = {*GUARDED_EXECUTION_FILES, *(f"src/railway_recon/{name}" for name in names),
                "scripts/analyze_synthetic_track_formal.py", "pyproject.toml", "uv.lock"}
    return {name: _digest((root / name).read_bytes()) for name in sorted(relative)}


def create_formal_analysis_plan(protocol_path: str | Path) -> dict:
    """Record decisions without constructing or reading any reserved geometry."""
    payload = Path(protocol_path).read_bytes()
    protocol = json.loads(payload)
    validate_guarded_protocol(protocol)
    return {
        "schema_version": "railway.synthetic-track-formal-analysis-plan.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "contract": json.loads(json.dumps(ANALYSIS_CONTRACT)),
        "guarded_protocol_sha256": _digest(payload),
        "guarded_protocol_id": protocol["protocol_id"],
        "source_sha256": _bindings(),
        "prior_data_access": {
            "development_v1_v2_v3_observed": True,
            "validation_v1_v2_v3_observed": True,
            "legacy_stress_observed": True,
            "challenge_stress_observed": True,
            "public_training_pilot_observed": True,
            "reserved_test_generated_for_plan_creation": False,
            "creation_does_not_verify_historical_test_non_access": True,
        },
        "registration": "local_timestamp_and_hash_binding_not_external_preregistration",
        "test_result_policy": "retain_first_complete_run_no_post_test_method_tuning",
    }


def validate_formal_analysis_plan(plan: dict) -> None:
    required = {"schema_version", "created_at_utc", "contract", "guarded_protocol_sha256",
                "guarded_protocol_id", "source_sha256", "prior_data_access", "registration",
                "test_result_policy"}
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("Analysis plan has missing or unexpected fields")
    if plan["schema_version"] != "railway.synthetic-track-formal-analysis-plan.v1":
        raise ValueError("Unsupported analysis plan schema")
    if plan["contract"] != ANALYSIS_CONTRACT:
        raise ValueError("Analysis contract must remain exactly fixed")
    timestamp = datetime.fromisoformat(plan["created_at_utc"])
    if timestamp.utcoffset() is None:
        raise ValueError("Analysis plan timestamp must be timezone-aware")
    if re.fullmatch(r"[0-9a-f]{64}", plan["guarded_protocol_sha256"]) is None:
        raise ValueError("Invalid protocol SHA-256")
    if plan["source_sha256"] != _bindings():
        raise ValueError("Analysis source changed since plan creation; preserve the old plan")
    if plan["prior_data_access"] != {
        "development_v1_v2_v3_observed": True, "validation_v1_v2_v3_observed": True,
        "legacy_stress_observed": True, "challenge_stress_observed": True,
        "public_training_pilot_observed": True,
        "reserved_test_generated_for_plan_creation": False,
        "creation_does_not_verify_historical_test_non_access": True,
    }:
        raise ValueError("Analysis plan must retain all development access disclosures")
    if plan["registration"] != "local_timestamp_and_hash_binding_not_external_preregistration":
        raise ValueError("Local analysis plan is not external preregistration")
    if plan["test_result_policy"] != "retain_first_complete_run_no_post_test_method_tuning":
        raise ValueError("First complete test result retention policy is required")


def _safe_file(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Artifact path must be canonical relative POSIX text")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(item in {"", ".", ".."} for item in relative.split("/")):
        raise ValueError("Artifact path escapes or is not canonical")
    path = directory.joinpath(*parsed.parts)
    if not path.resolve().is_relative_to(directory.resolve()) or ":" in relative:
        raise ValueError("Artifact path escapes run directory")
    return path


def verify_completed_run(directory: str | Path) -> tuple[dict, dict, set[str]]:
    """Verify the exact complete artifact inventory, including every prediction."""
    root = Path(directory).resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "railway.synthetic-track-guarded-run-manifest.v1":
        raise ValueError("Expected a completed guarded v3 manifest")
    if manifest.get("source_unchanged_during_run") is not True:
        raise ValueError("Run source stability is not established")
    if (root / "run_failure.json").exists():
        raise ValueError("Failed runs cannot be formally analyzed")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires nonempty artifact bindings")
    bound = set()
    for entry in records:
        relative = entry["path"]
        path = _safe_file(root, relative)
        if relative in bound or relative == "manifest.json":
            raise ValueError("Duplicate or self-referential manifest artifact")
        bound.add(relative)
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or _digest(payload) != entry["sha256"]:
            raise ValueError(f"Artifact hash/size mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != bound | {"manifest.json"}:
        raise ValueError("Manifest must bind every artifact without extras or omissions")
    if (root / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed during verification")
    report = json.loads((root / "report.json").read_bytes())
    if report.get("schema_version") != "railway.synthetic-track-guarded-study-report.v1":
        raise ValueError("Expected guarded v3 report")
    return manifest, report, bound


def paired_layout_interval(differences: list[float]) -> dict:
    """Equal-weight paired layout mean and fixed percentile bootstrap interval."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Require a nonempty finite one-dimensional layout difference vector")
    if np.any(np.abs(values) > 1):
        raise ValueError("Correct-diagnosis rate differences must lie in [-1, 1]")
    settings = ANALYSIS_CONTRACT["bootstrap"]
    generator = np.random.Generator(np.random.PCG64(settings["seed"]))
    # Batches bound temporary memory without changing the PCG64 traversal.
    means = []
    for start in range(0, settings["resamples"], 250):
        size = min(250, settings["resamples"] - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        means.extend(np.mean(values[indices], axis=1).tolist())
    low, high = np.quantile(means, [0.025, 0.975], method="linear").tolist()
    degenerate = bool(np.all(values == values[0]))
    return {
        "layout_count": len(values), "mean_difference": float(np.mean(values)),
        "percentile_95_interval": [low, high], "degenerate_empirical_distribution": degenerate,
        "interval_interpretation": (
            "observed_invariance_not_population_certainty_or_equivalence" if degenerate
            else "paired_layout_bootstrap_for_observed_synthetic_design_not_field_population"
        ),
        "resamples": settings["resamples"], "seed": settings["seed"],
        "p_value": None, "superiority_established": False, "equivalence_established": False,
    }


def _recompute(root: Path, report: dict, manifest: dict, bound: set[str], protocol: dict) -> dict:
    modes = tuple(GUARDED_MODES)
    split = report.get("split")
    if split not in {"development", "validation", "test"} or manifest.get("split") != split:
        raise ValueError("Only complete registered splits are eligible; stress is separately reported")
    if report.get("modes") != list(modes):
        raise ValueError("Run must contain all six fixed modes")
    if report.get("held_out_test") != (split == "test"):
        raise ValueError("Report split and held-out status disagree")
    if report.get("registered_test_generated") != (split == "test"):
        raise ValueError("Report split and reserved-generation status disagree")
    if report.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Report protocol identity disagrees with snapshot")
    policy = json.loads((root / "policy_snapshot.json").read_bytes())
    if policy != protocol["policy"] or report.get("policy") != policy:
        raise ValueError("Report and snapshot policy differ from planned protocol")
    if _digest(_canonical_bytes(policy)) != manifest.get("policy_sha256"):
        raise ValueError("Policy fingerprint differs from manifest")
    expected = [item for item in protocol["base_protocol"]["groups"] if item["split"] == split]
    expected_ids = [item["group_id"] for item in expected]
    index = json.loads((root / "case_index.json").read_bytes())
    if not isinstance(index, list) or not index:
        raise ValueError("Case index must be nonempty")
    conditions = protocol["base_protocol"]["conditions"]
    cases_by_group = {}
    seen = set()
    expected_predictions = set()
    fingerprint = hashlib.sha256()
    data_fingerprint = hashlib.sha256()
    scores = {mode: [] for mode in modes}
    group_scores = {}
    condition_scores = {}
    for entry in index:
        case_id, group_id = entry["case_id"], entry["group_id"]
        if not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", case_id) is None:
            raise ValueError("Unsafe case identity")
        if case_id in seen or group_id not in expected_ids:
            raise ValueError("Duplicate case or unallocated layout")
        seen.add(case_id)
        cases_by_group.setdefault(group_id, []).append(entry)
        truth_payload = json.loads((root / "ground_truth" / f"{case_id}.json").read_bytes())
        if any(truth_payload[key] != entry[key] for key in ("group_id", "layout_id", "condition")):
            raise ValueError("Ground truth and case-index metadata disagree")
        truth = {k: v for k, v in truth_payload.items() if k not in {"group_id", "layout_id"}}
        candidate = json.loads((root / "inputs" / f"{case_id}.json").read_bytes())
        data_fingerprint.update(_canonical_bytes({
            "case_id": case_id, "layout_id": entry["layout_id"], "input": candidate, "truth": truth,
        }))
        group_scores.setdefault(group_id, {mode: [] for mode in modes})
        condition_scores.setdefault(entry["condition"], {mode: [] for mode in modes})
        for mode in modes:
            relative = f"predictions/{mode}/{case_id}.json"
            expected_predictions.add(relative)
            prediction = json.loads((root / relative).read_bytes())
            fingerprint.update(_canonical_bytes({"case_id": case_id, "mode": mode,
                                                  "prediction": prediction}))
            evaluation = evaluate_adaptive_prediction(truth, prediction)
            scores[mode].append(evaluation)
            group_scores[group_id][mode].append(evaluation)
            condition_scores[entry["condition"]][mode].append(evaluation)
    if list(cases_by_group) != expected_ids:
        raise ValueError("Must retain all allocated groups in the registered traversal")
    flattened = []
    for group_id in expected_ids:
        entries = cases_by_group[group_id]
        if Counter(item["condition"] for item in entries) != Counter(conditions):
            raise ValueError("Each layout must retain every registered condition exactly once")
        if len({item["layout_id"] for item in entries}) != 1:
            raise ValueError("Registered group must have exactly one underlying layout")
        flattened.extend(sorted(entries, key=lambda item: item["case_id"]))
    if index != flattened:
        raise ValueError("Case traversal differs from complete registered group/variant order")
    if len({entries[0]["layout_id"] for entries in cases_by_group.values()}) != len(expected_ids):
        raise ValueError("Underlying layout must not be reused across independent groups")
    if expected_predictions != {name for name in bound if name.startswith("predictions/")}:
        raise ValueError("Prediction inventory differs from all indexed case-mode pairs")
    for prefix in ("inputs", "ground_truth"):
        if {f"{prefix}/{case_id}.json" for case_id in seen} != {
            name for name in bound if name.startswith(prefix + "/")
        }:
            raise ValueError("Input/truth inventory differs from complete case index")
    if report["base_layout_count"] != len(expected_ids) or report["case_count"] != len(seen):
        raise ValueError("Report counts disagree with allocation")
    if report["detector_run_count"] != len(seen) * len(modes):
        raise ValueError("Report detector count disagrees with all case-mode pairs")
    for name, value in (("prediction_fingerprint_sha256", fingerprint.hexdigest()),
                        ("data_fingerprint_sha256", data_fingerprint.hexdigest())):
        if value != report[name] or value != manifest[name]:
            raise ValueError(f"Recomputed {name} differs from report/manifest")
    summary = {mode: aggregate_adaptive_evaluations(values) for mode, values in scores.items()}
    grouped = {group: {mode: aggregate_adaptive_evaluations(values)
                       for mode, values in methods.items()} for group, methods in group_scores.items()}
    by_condition = {condition: {mode: aggregate_adaptive_evaluations(values)
                                for mode, values in methods.items()}
                    for condition, methods in condition_scores.items()}
    if summary != report["summary_by_mode"] or grouped != report["by_group"]:
        raise ValueError("Recomputed saved predictions disagree with report summary or groups")
    if by_condition != report["by_condition"]:
        raise ValueError("Recomputed saved predictions disagree with condition report")
    return {"summary_by_mode": summary, "by_group": grouped, "by_condition": by_condition}


def analyze_guarded_run(run_directory: str | Path, plan_path: str | Path,
                        output: str | Path) -> dict:
    """Verify and analyze one entire registered split without changing its files."""
    root, destination = Path(run_directory).resolve(), Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Analysis output already exists; previous results must be preserved")
    if destination.is_relative_to(root):
        raise ValueError("Analysis output must be outside immutable run directory")
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    validate_formal_analysis_plan(plan)
    manifest, report, bound = verify_completed_run(root)
    initial_manifest = (root / "manifest.json").read_bytes()
    protocol_bytes = (root / "protocol_snapshot.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_guarded_protocol(protocol)
    if (manifest["protocol_sha256"] != plan["guarded_protocol_sha256"]
            or _digest(protocol_bytes) != plan["guarded_protocol_sha256"]
            or protocol["protocol_id"] != plan["guarded_protocol_id"]):
        raise ValueError("Run does not belong to the plan-bound protocol")
    analysis_module = "src/railway_recon/synthetic_track_formal_analysis.py"
    nonexecuting_clis = {
        "scripts/analyze_synthetic_track_formal.py",
        "scripts/run_synthetic_track_study.py",
        "scripts/run_synthetic_track_adaptive_study.py",
    }
    for relative, digest in plan["source_sha256"].items():
        if relative in nonexecuting_clis:
            # These CLIs are bound/checked by the plan but never execute in the
            # guarded run, whose own source inventory includes only its CLI.
            continue
        if relative == analysis_module and relative not in manifest["source_sha256"]:
            if report["split"] == "test":
                raise ValueError("Test execution did not bind the planned analysis implementation")
            continue  # Earlier development runs predate this additive analysis module.
        if manifest["source_sha256"].get(relative) != digest:
            raise ValueError("Planned method/evaluator differs from executed run sources")
    if report["split"] == "test":
        verification = manifest.get("freeze_verification") or {}
        if not report.get("source_freeze_verified") or verification.get("status") != "pass":
            raise ValueError("Formal test requires successful execution source-freeze verification")
    recomputed = _recompute(root, report, manifest, bound, protocol)
    comparisons = []
    layout_rows = []
    for group_id, methods in recomputed["by_group"].items():
        denominators = {value["truth_defect_count"] for value in methods.values()}
        if len(denominators) != 1 or next(iter(denominators)) <= 0:
            raise ValueError("Every paired layout requires the same nonzero truth-fault denominator")
        denominator = next(iter(denominators))
        layout_rows.append({"group_id": group_id, "truth_fault_count": denominator,
                            "correct_diagnosis_rate": {
                                mode: value["diagnosis"]["tp"] / denominator
                                for mode, value in methods.items()}})
    primary = ANALYSIS_CONTRACT["primary"]["comparison"]
    for mode in (primary, *(item for item in GUARDED_MODES[:-1] if item != primary)):
        differences = [row["correct_diagnosis_rate"]["guarded_evidence"]
                       - row["correct_diagnosis_rate"][mode] for row in layout_rows]
        comparisons.append({"comparison": f"guarded_evidence_minus_{mode}",
                            "role": "primary" if mode == primary else "exploratory_secondary",
                            **paired_layout_interval(differences)})
    analysis = {
        "schema_version": "railway.synthetic-track-formal-analysis-report.v1",
        "split": report["split"], "analysis_plan_sha256": _digest(plan_bytes),
        "run_manifest_sha256": _digest(initial_manifest),
        "guarded_protocol_sha256": plan["guarded_protocol_sha256"],
        "data_fingerprint_sha256": report["data_fingerprint_sha256"],
        "prediction_fingerprint_sha256": report["prediction_fingerprint_sha256"],
        "artifact_count_verified": len(bound), "case_count": report["case_count"],
        "layout_count": len(layout_rows), "all_allocated_groups_retained": True,
        "contract": ANALYSIS_CONTRACT, "comparisons": comparisons, "layout_rates": layout_rows,
        **recomputed,
        "limitations": [
            "Layouts, not their related variants, are the resampling units.",
            "Equal layout weights target the registered synthetic design mixture, not real railways.",
            "Degenerate intervals do not establish population certainty, equivalence or zero risk.",
            "No p-values, multiplicity-adjusted claims, or automatic superiority conclusion are made.",
            "Development and validation informed the method; only reserved test is a new allocation.",
            "Local hashes/timestamps are not external preregistration or proof of historical blindness.",
            "Six internal control policies do not constitute an external state-of-the-art comparison.",
            "Observation proxies and centerlines do not establish LiDAR, survey or field accuracy.",
        ],
    }
    # Detect source/input changes before a completed analysis can be written.
    validate_formal_analysis_plan(plan)
    verify_completed_run(root)
    if Path(plan_path).read_bytes() != plan_bytes or (root / "manifest.json").read_bytes() != initial_manifest:
        raise ValueError("Plan or run changed during analysis")
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "analysis_plan_snapshot.json", plan)
    _write_json(destination / "report.json", analysis)
    lines = ["# Fixed layout-paired diagnosis analysis", "",
             f"Split: {analysis['split']}; {len(layout_rows)} layouts; {report['case_count']} cases.",
             "", "| Comparison | Role | Mean difference | Percentile 95% interval | Degenerate |",
             "| --- | --- | ---: | --- | --- |"]
    for item in comparisons:
        low, high = item["percentile_95_interval"]
        lines.append(f"| {item['comparison']} | {item['role']} | {item['mean_difference']:.6f} | "
                     f"[{low:.6f}, {high:.6f}] | {item['degenerate_empirical_distribution']} |")
    lines.extend(["", "## Every policy, including normal alarms and abstentions", "",
                  "| Mode | Diagnosis correct / truth | Normal flagged / normal | Abstentions |",
                  "| --- | --- | --- | --- |"])
    for mode, result in recomputed["summary_by_mode"].items():
        lines.append(f"| {mode} | {result['diagnosis']['tp']}/{result['truth_defect_count']} | "
                     f"{result['normal_flagged_case_count']}/{result['normal_case_count']} | "
                     f"{result['cause_evaluation']['abstained_findings']} |")
    lines.extend(["", *[f"- {item}" for item in analysis["limitations"]], ""])
    with (destination / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    _write_json(destination / "manifest.json", {
        "schema_version": "railway.synthetic-track-formal-analysis-manifest.v1",
        "status": "completed", "analysis_plan_sha256": _digest(plan_bytes),
        "input_run_manifest_sha256": _digest(initial_manifest),
        "source_sha256": plan["source_sha256"],
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow")},
        "files": [{"path": path.name, "bytes": path.stat().st_size,
                   "sha256": _digest(path.read_bytes())} for path in sorted(destination.iterdir())],
        "manifest_self_hash_excluded": True,
    })
    return analysis
