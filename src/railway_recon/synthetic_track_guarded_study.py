"""Six-way guarded diagnosis replay and separately reported development probes.

Only the auditor receives isolated candidate inputs. Scoring and route accounting
reuse the unchanged v2 evaluator; no old runner is patched or monkeypatched.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from .synthetic_track_adaptive_study import (
    aggregate_adaptive_evaluations,
    evaluate_adaptive_prediction,
)
from .synthetic_track_pilot import _canonical_bytes, _draw_panel, _font, _write_json
from .synthetic_track_study import _percent
from .synthetic_track_study_audit import StudyAuditPolicy


def render_guarded_case(
    case: dict, predictions: dict, evaluations: dict, path: Path, modes: tuple
) -> None:
    """Render exactly six policies in a readable three-column, two-row layout."""
    if len(modes) != 6:
        raise ValueError("Guarded comparison figure requires exactly six modes")
    picture = Image.new("RGB", (1840, 1100), "white")
    draw = ImageDraw.Draw(picture)
    draw.text((24, 18), f"P2 guarded v3: {case['truth']['condition']}",
              fill="#173047", font=_font(23))
    draw.text((24, 54),
              "Synthetic centerline candidates and observation proxies; XY plan. Checks use 3D coordinates.",
              fill="#536b7e", font=_font(16))
    for index, mode in enumerate(modes):
        left, top = 24 + (index % 3) * 605, 105 + (index // 3) * 445
        draw.text((left, top), mode, fill="#173047", font=_font(20))
        _draw_panel(draw, (left, top + 36, left + 570, top + 298),
                    case["input"], case["truth"], predictions[mode]["findings"])
        diag, cause = evaluations[mode]["diagnosis"], evaluations[mode]["cause_evaluation"]
        draw.text((left, top + 311),
                  f"Diagnosis TP {diag['tp']} / FP {diag['fp']} / FN {diag['fn']}",
                  fill="#344d60", font=_font(17))
        draw.text((left, top + 340),
                  f"Assigned {cause['assigned_findings']}; abstained {cause['abstained_findings']}",
                  fill="#344d60", font=_font(16))
        labels = sorted({item["decision_basis"] for item in predictions[mode]["findings"]})
        draw.text((left, top + 370), ", ".join(labels) or "No geometric alarms",
                  fill="#344d60", font=_font(14))
    draw.text((24, 1000),
              "Green circle: evaluator truth. Red cross: shared geometric alarm, including unresolved causes.",
              fill="#344d60", font=_font(16))
    draw.text((24, 1032),
              "Geometry hypotheses are not observation verification. Support checks do not certify truth or observation authenticity.",
              fill="#536b7e", font=_font(16))
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)


def _source_bindings() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    sources = [*sorted((root / "src" / "railway_recon").rglob("*.py")),
               root / "scripts" / "run_synthetic_track_guarded_study.py",
               root / "pyproject.toml", root / "uv.lock"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}


def _report_text(report: dict) -> str:
    role = (f"Development pressure suite: **{report['suite']}**" if report["split"] == "stress"
            else f"Registered split: **{report['split']}**")
    lines = [
        "# P2 guarded diagnosis study (v3)", "",
        f"{role}; {report['base_layout_count']} layouts, {report['case_count']} cases, {report['detector_run_count']} detector runs.", "",
        "No customer data is used. Four unchanged v1/v2 controls, a direction-geometry baseline and guarded evidence are evaluated on the same cases with the unchanged abstention-aware evaluator.", "",
        "## Overall results", "",
        "| Mode | Detection TP/FP/FN | Diagnosis TP/FP/FN | Diagnosis recall | Assigned coverage | Correct among assigned | Normal flagged / normal | Abstentions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, value in report["summary_by_mode"].items():
        detection, diagnosis, cause = value["detection"], value["diagnosis"], value["cause_evaluation"]
        lines.append(
            f"| {mode} | {detection['tp']}/{detection['fp']}/{detection['fn']} | "
            f"{diagnosis['tp']}/{diagnosis['fp']}/{diagnosis['fn']} | {_percent(diagnosis['recall'])} | "
            f"{_percent(cause['diagnosis_coverage'])} | {_percent(cause['selective_diagnosis_accuracy'])} | "
            f"{value['normal_flagged_case_count']}/{value['normal_case_count']} | {cause['abstained_findings']} |"
        )
    lines.extend([
        "", "All six modes must preserve the same geometric alarm entities and positions. Differences concern cause assignment, abstention and provenance, not detection improvement. An abstention is a diagnosis FN; a wrong assigned cause contributes diagnosis FP and FN.", "",
        "## Decision-route accounting", "",
        "| Mode | Decision basis | Assigned / alerts | Diagnosis TP/FP/FN | Attributed truth | Wrong assigned / assigned truth | Evidence-verified alerts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for mode, value in report["summary_by_mode"].items():
        for basis, route in value["decision_routes"].items():
            lines.append(
                f"| {mode} | {basis} | {route['assigned_findings']}/{route['findings']} | "
                f"{route['diagnosis_tp']}/{route['diagnosis_fp']}/{route['diagnosis_fn']} | "
                f"{route['truth_count']} | {route['wrong_assigned_cause_count']}/{route['truth_with_assigned_diagnosis']} | "
                f"{route['evidence_verified_findings']} |"
            )
    lines.extend([
        "", "Route accounting uses the unchanged v2 one-to-one attribution rules and reconciles to total diagnosis TP/FP/FN. Route denominators are decision-selected, not independent trial groups. Qualitative confidence labels are not calibrated probabilities. `evidence_verified` means only internal coverage/support criteria passed; it does not certify a correct cause, authentic observations or survey accuracy.", "",
        "## Every condition", "",
        "| Condition | Mode | Correct diagnosis / truth | Assigned / truth | Wrong assigned | Normal flagged / normal |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for condition, methods in report["by_condition"].items():
        for mode, value in methods.items():
            wrong = sum(route["wrong_assigned_cause_count"] for route in value["decision_routes"].values())
            lines.append(
                f"| {condition} | {mode} | {value['diagnosis']['tp']}/{value['truth_defect_count']} | "
                f"{value['cause_evaluation']['truth_with_assigned_diagnosis']}/{value['truth_defect_count']} | "
                f"{wrong} | {value['normal_flagged_case_count']}/{value['normal_case_count']} |"
            )
    lines.extend([
        "", "## Reproducibility and scope", "",
        "- All inputs, evaluator truth and predictions are stored separately. The auditor receives only the candidate input, mode and policy; never split, condition or truth.",
        "- Every case/mode is retained in `case_results.csv`; all false positives and false negatives remain in `failure_cases.json`. No outcome filtering or automatic repair is performed.",
        "- Figures select the first case per condition in the registered traversal (or opaque-ID sorted stress traversal), independently of outcomes.",
        "- `manifest.json` binds every output, all Python source, dependency lock, CLI, actual policy and any supplied protocol/freeze. Source changes during a run cause failure instead of silently mixing code versions.",
        "- Development, validation and constructed pressure probes were used for method development. They are not independent field benchmarks, evidence of survey accuracy or human-review time measurements.",
        "- Old results are not overwritten. Separate pressure suites are never pooled into one headline score, so the original adverse examples remain directly comparable.",
    ])
    if report["split"] == "stress":
        lines.append(
            "- Legacy pressure uses the original 18-case generator unchanged; challenge uses a separate analytic construction. Neither suite generates registered formal-test groups. Challenge is a development probe suite, not a newly independent held-out test."
        )
    else:
        lines.extend([
            "- Original v1 grouped seeds, case generator, conditions, traversal and canonical data fingerprint are unchanged. The four old modes remain controls; previous development/validation results informed this revision.",
            "- Reserved test generation requires a verified new guarded protocol/source freeze belonging to the executing repository. A freeze is source binding, not external preregistration or proof of independence.",
        ])
    lines.extend(["", f"Data fingerprint: `{report['data_fingerprint_sha256']}`.", ""])
    return "\n".join(lines)


def _execute(
    batches: Iterable[tuple[str | None, list[dict]]],
    destination: Path,
    metadata: dict,
    policy: dict,
    *,
    expected_conditions: list[str] | None = None,
    protocol: dict | None = None,
    protocol_bytes: bytes | None = None,
    freeze_verification: dict | None = None,
    freeze_sha256: str | None = None,
) -> dict:
    from .synthetic_track_guarded_audit import GUARDED_MODES, audit_guarded_candidate

    modes = tuple(GUARDED_MODES)
    sources = _source_bindings()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Guarded output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evaluations = {mode: [] for mode in modes}
    conditions: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    rows, failures, case_index, timings = [], [], [], []
    hashes = {key: hashlib.sha256() for key in ("data", "predictions")}
    seen_ids, rendered = set(), set()
    completed_ids = []
    try:
        _write_json(destination / "policy_snapshot.json", policy)
        if protocol is not None:
            _write_json(destination / "protocol_snapshot.json", protocol)
        for batch_index, (registered_group, cases) in enumerate(batches):
            if not cases:
                raise ValueError("Every generated batch must contain cases")
            if expected_conditions is not None:
                if Counter(case["truth"]["condition"] for case in cases) != Counter(expected_conditions):
                    raise ValueError("Generated conditions differ from the registered base protocol")
                if len({case["layout_id"] for case in cases}) != 1:
                    raise ValueError("A registered group must have exactly one generated base layout")
            for case in sorted(cases, key=lambda item: item["case_id"]):
                case_id, condition = case["case_id"], case["truth"]["condition"]
                group_id = registered_group if registered_group is not None else case["layout_id"]
                if case_id in seen_ids or re.fullmatch(r"[A-Za-z0-9_-]+", case_id) is None:
                    raise ValueError("Case IDs must be unique safe opaque path components")
                if re.fullmatch(r"[A-Za-z0-9_-]+", condition) is None:
                    raise ValueError("Condition names must be safe path components")
                seen_ids.add(case_id)
                hashes["data"].update(_canonical_bytes(case))
                _write_json(destination / "inputs" / f"{case_id}.json", case["input"])
                _write_json(destination / "ground_truth" / f"{case_id}.json",
                            {**case["truth"], "layout_id": case["layout_id"], "group_id": group_id})
                case_index.append({"case_id": case_id, "group_id": group_id,
                                   "layout_id": case["layout_id"], "condition": condition})
                conditions.setdefault(condition, {mode: [] for mode in modes})
                groups.setdefault(group_id, {mode: [] for mode in modes})
                predictions, scores = {}, {}
                for mode in modes:
                    before = time.perf_counter()
                    prediction = audit_guarded_candidate(copy.deepcopy(case["input"]), mode, policy)
                    seconds = time.perf_counter() - before
                    hashes["predictions"].update(_canonical_bytes(
                        {"case_id": case_id, "mode": mode, "prediction": prediction}))
                    _write_json(destination / "predictions" / mode / f"{case_id}.json", prediction)
                    score = evaluate_adaptive_prediction(case["truth"], prediction)
                    predictions[mode], scores[mode] = prediction, score
                    evaluations[mode].append(score)
                    conditions[condition][mode].append(score)
                    groups[group_id][mode].append(score)
                    timings.append({"case_id": case_id, "mode": mode, "detector_seconds": seconds})
                    routes = score["decision_routes"].values()
                    rows.append({
                        "case_id": case_id, "group_id": group_id, "condition": condition, "mode": mode,
                        **{f"{metric}_{key}": score[metric][key]
                           for metric in ("detection", "diagnosis") for key in ("tp", "fp", "fn")},
                        "assigned_truth_count": score["cause_evaluation"]["truth_with_assigned_diagnosis"],
                        "truth_count": score["truth_defect_count"],
                        "cause_abstentions": score["cause_evaluation"]["abstained_findings"],
                        "wrong_assigned_cause_count": sum(route["wrong_assigned_cause_count"] for route in routes),
                        "evidence_verified_findings": sum(route["evidence_verified_findings"] for route in routes),
                        "normal_case_flagged": score["normal_case_flagged"],
                        "decision_bases": ";".join(sorted(score["decision_routes"])),
                    })
                    if any(score[metric][key] for metric in ("detection", "diagnosis")
                           for key in ("fp", "fn")):
                        failures.append({"case_id": case_id, "condition": condition,
                                         "mode": mode, "evaluation": score})
                reference = [(item["entity_ids"], item["position"])
                             for item in predictions[modes[0]]["findings"]]
                if any([(item["entity_ids"], item["position"])
                        for item in predictions[mode]["findings"]] != reference for mode in modes[1:]):
                    raise ValueError("A guarded diagnosis mode changed the geometric detection set")
                if condition not in rendered:
                    render_guarded_case(case, predictions, scores,
                                        destination / "figures" / f"{condition}.png", modes)
                    rendered.add(condition)
                completed_ids.append(case_id)
            print(f"guarded {metadata['split']}: {batch_index + 1} batches completed", flush=True)
        if not rows:
            raise ValueError("Selected run contains no cases")
        group_summary = {key: {mode: aggregate_adaptive_evaluations(values)
                               for mode, values in methods.items()}
                         for key, methods in groups.items()}
        paired = []
        for group_id, entries in group_summary.items():
            denominator = entries[modes[0]]["truth_defect_count"]
            if denominator:
                guarded = entries["guarded_evidence"]
                paired.append({
                    "group_id": group_id, "truth_count": denominator,
                    **{f"guarded_minus_{comparison}_diagnosis_recall":
                       (guarded["diagnosis"]["tp"] - entries[comparison]["diagnosis"]["tp"]) / denominator
                       for comparison in modes[:-1]},
                    **{f"guarded_minus_{comparison}_assigned_coverage":
                       (guarded["cause_evaluation"]["truth_with_assigned_diagnosis"]
                        - entries[comparison]["cause_evaluation"]["truth_with_assigned_diagnosis"]) / denominator
                       for comparison in modes[:-1]},
                })
        report = {
            "schema_version": "railway.synthetic-track-guarded-study-report.v1",
            **metadata,
            "modes": list(modes), "policy": policy,
            "case_count": len(seen_ids), "detector_run_count": len(rows),
            "base_layout_count": len(groups), "customer_data_used": False,
            "held_out_test": metadata["split"] == "test",
            "registered_test_generated": metadata["split"] == "test",
            "source_freeze_verified": freeze_verification is not None,
            "confidence_labels_are_probabilities": False,
            "summary_by_mode": {mode: aggregate_adaptive_evaluations(values)
                                for mode, values in evaluations.items()},
            "by_condition": {key: {mode: aggregate_adaptive_evaluations(values)
                                    for mode, values in methods.items()}
                             for key, methods in sorted(conditions.items())},
            "by_group": group_summary,
            "paired_layout_differences": paired,
            "mean_paired_differences": {
                key: sum(item[key] for item in paired) / len(paired)
                for key in paired[0] if key not in {"group_id", "truth_count"}
            } if paired else {},
            "data_fingerprint_sha256": hashes["data"].hexdigest(),
            "prediction_fingerprint_sha256": hashes["predictions"].hexdigest(),
            "data_fingerprint_definition": (
                "Canonical generated cases, opaque case_id sorted across this single stress suite"
                if metadata["split"] == "stress" else
                "Identical to v1/v2: canonical generated cases, base protocol group order, case_id-sorted variants"
            ),
        }
        _write_json(destination / "report.json", report)
        _write_json(destination / "case_index.json", case_index)
        _write_json(destination / "failure_cases.json", failures)
        _write_json(destination / "timings.json", {
            "elapsed_seconds": time.perf_counter() - started,
            "detector_runs": timings,
            "measurement": "Single sequential run; detector timing excludes scoring, serialization and rendering",
        })
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(_report_text(report))
        with (destination / "case_results.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if sources != _source_bindings():
            raise ValueError("Guarded execution source changed during run; results are not frozen")
        _write_json(destination / "manifest.json", {
            "schema_version": "railway.synthetic-track-guarded-run-manifest.v1",
            "split": metadata["split"], "suite": metadata.get("suite"),
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest() if protocol_bytes is not None else None,
            "policy_sha256": hashlib.sha256(_canonical_bytes(policy)).hexdigest(),
            "data_fingerprint_sha256": report["data_fingerprint_sha256"],
            "prediction_fingerprint_sha256": report["prediction_fingerprint_sha256"],
            "source_sha256": sources,
            "source_unchanged_during_run": True,
            "environment": {
                "python": platform.python_version(), "system": platform.system(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow")},
            },
            "freeze_verification": freeze_verification, "freeze_sha256": freeze_sha256,
            "files": [{"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(destination.rglob("*")) if path.is_file()],
            "manifest_self_hash_excluded": True,
        })
        return report
    except Exception as error:
        _write_json(destination / "run_failure.json", {
            "split": metadata["split"], "suite": metadata.get("suite"), "status": "failed",
            "exception_type": type(error).__name__, "started_case_ids": sorted(seen_ids),
            "completed_case_ids": completed_ids,
        })
        raise


def run_guarded_split(
    protocol_path: str | Path, split: str, output: str | Path,
    freeze_path: str | Path | None = None,
) -> dict:
    """Run a complete explicit allocation; test requires a new verified freeze."""
    from .synthetic_track_guarded_audit import GUARDED_MODES
    from .synthetic_track_guarded_protocol import validate_guarded_protocol, verify_guarded_freeze
    from .synthetic_track_study_protocol import _repository_root
    from .synthetic_track_study_scene import generate_study_layout

    source = Path(protocol_path).resolve()
    protocol_bytes = source.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    validate_guarded_protocol(protocol)
    if tuple(protocol["modes"]) != tuple(GUARDED_MODES):
        raise ValueError("Registered guarded modes differ from the executing auditor")
    if split not in {"development", "validation", "test"}:
        raise ValueError("Split must be development, validation or test")
    if split == "test" and freeze_path is None:
        raise ValueError("Test is reserved: a verified guarded protocol/source freeze is required")
    freeze_verification, freeze_sha256 = None, None
    if freeze_path is not None:
        if _repository_root(source) != Path(__file__).resolve().parents[2]:
            raise ValueError("Freeze verification repository differs from the executing toolkit")
        freeze = Path(freeze_path).resolve()
        freeze_bytes = freeze.read_bytes()
        freeze_verification = verify_guarded_freeze(source, freeze)
        if freeze_verification.get("status") != "pass":
            raise ValueError("Frozen guarded source/protocol verification failed")
        verified_root = freeze_verification.get("repository_root")
        if verified_root is None or Path(verified_root).resolve() != Path(__file__).resolve().parents[2]:
            raise ValueError("Verified freeze root differs from the executing toolkit")
        if source.read_bytes() != protocol_bytes or freeze.read_bytes() != freeze_bytes:
            raise ValueError("Guarded protocol or freeze changed during verification")
        freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    base = protocol["base_protocol"]
    groups = [group for group in base["groups"] if group["split"] == split]
    if not groups:
        raise ValueError("Selected split has no registered groups")
    batches = ((group["group_id"], generate_study_layout(group["seed"], group["family_index"]))
               for group in groups)
    return _execute(
        batches, Path(output).resolve(),
        {"split": split, "protocol_id": protocol["protocol_id"], "base_protocol_id": base["protocol_id"],
         "role": "reserved_test_with_verified_source_freeze" if split == "test" else "reused_method_development_split"},
        protocol["policy"], expected_conditions=base["conditions"], protocol=protocol,
        protocol_bytes=protocol_bytes, freeze_verification=freeze_verification, freeze_sha256=freeze_sha256,
    )


def run_guarded_stress(
    output: str | Path, suite: str = "legacy", seed: int | None = None, policy: dict | None = None,
) -> dict:
    """Run exactly one pressure suite; never pool old and new probe scores."""
    if suite not in {"legacy", "challenge"}:
        raise ValueError("Stress suite must be exactly legacy or challenge; mixed suites are not supported")
    seed = (20260907 if suite == "legacy" else 20260908) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if policy is not None and not isinstance(policy, dict):
        raise TypeError("policy must be a dictionary or None")
    active_policy = asdict(StudyAuditPolicy(**({} if policy is None else policy)))
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Guarded output already exists: {destination}")
    if suite == "legacy":
        from .synthetic_track_adaptive_stress import generate_adaptive_stress_cases

        generate = generate_adaptive_stress_cases
    else:
        from .synthetic_track_guarded_stress import generate_guarded_stress_cases

        generate = generate_guarded_stress_cases
    return _execute(
        ((None, generate(seed)) for _ in range(1)), destination,
        {"split": "stress", "suite": suite, "seed": seed,
         "role": "constructed_adversarial_development_probes_not_held_out_test",
         "suite_results_pooled": False},
        active_policy,
    )
