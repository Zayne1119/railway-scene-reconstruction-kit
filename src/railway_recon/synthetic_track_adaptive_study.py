"""Four-way adaptive diagnosis study using unchanged v1 cases and evaluation.

Development and validation are explicit runs. Reserved test generation requires
an adaptive protocol/source freeze verified against this executing repository.
Confidence labels describe decision routes; they are never probabilities.
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
from pathlib import Path

from PIL import Image, ImageDraw

from .synthetic_track_adaptive_audit import ADAPTIVE_MODES, DECISION_BASES
from .synthetic_track_pilot import _canonical_bytes, _draw_panel, _font, _write_json
from .synthetic_track_study import _percent, _study_aggregate, evaluate_study_prediction

CONFIDENCE_LABELS = (
    "inferred_geometry",
    "observation_supported",
    "unresolved_no_evidence",
    "unresolved_conflict",
)
_ROUTE_COUNTS = (
    "findings",
    "assigned_findings",
    "abstained_findings",
    "evidence_verified_findings",
    "truth_count",
    "truth_with_assigned_diagnosis",
    "evidence_verified_truth_count",
    "diagnosis_tp",
    "diagnosis_fp",
    "diagnosis_fn",
    "wrong_assigned_cause_count",
)


def _entity_key(item: dict) -> tuple:
    return tuple(sorted(item["entity_ids"]))


def _empty_route() -> dict:
    return {**{key: 0 for key in _ROUTE_COUNTS}, "confidence_labels": {}}


def _route_rates(route: dict) -> dict:
    truth_count = route["truth_count"]
    assigned = route["truth_with_assigned_diagnosis"]
    return {
        **route,
        "diagnosis_coverage": assigned / truth_count if truth_count else None,
        "diagnosis_recall": route["diagnosis_tp"] / truth_count if truth_count else None,
        "wrong_cause_among_assigned": route["wrong_assigned_cause_count"] / assigned
        if assigned
        else None,
    }


def evaluate_adaptive_prediction(truth: dict, prediction: dict) -> dict:
    """Reuse v1 metrics and attach additive, one-to-one decision-route counts.

    Each truth fault is attributed to a single route: a correctly assigned
    matching alert first, otherwise an assigned entity match, otherwise the
    detected/abstained alert, or ``undetected``. Extra alerts retain their own
    route and diagnosis FP without creating additional truth or coverage.
    """
    result = evaluate_study_prediction(truth, prediction)
    findings = prediction.get("findings", [])
    defects = truth.get("defects", [])
    routes: dict[str, dict] = {}
    for finding in findings:
        basis = finding.get("decision_basis")
        verified = finding.get("evidence_verified")
        confidence = finding.get("confidence_level")
        if basis not in DECISION_BASES:
            raise ValueError("Every finding needs an explicit known decision_basis")
        if not isinstance(verified, bool):
            raise TypeError("Every finding needs an explicit boolean evidence_verified")
        if confidence not in CONFIDENCE_LABELS:
            raise ValueError(
                "confidence_level must be a registered qualitative label, not a probability"
            )
        if verified != (basis == "evidence_supported"):
            raise ValueError("Only evidence_supported findings can claim evidence_verified")
        expected_confidence = {
            "evidence_supported": "observation_supported",
            "evidence_conflict": "unresolved_conflict",
            "evidence_abstention": "unresolved_no_evidence",
        }.get(basis, "inferred_geometry")
        if confidence != expected_confidence:
            raise ValueError("confidence_level contradicts the declared decision provenance")
        expected_status = "abstain" if basis in {"evidence_conflict", "evidence_abstention"} else "assigned"
        if finding["cause_status"] != expected_status:
            raise ValueError("cause_status contradicts the declared decision provenance")
        route = routes.setdefault(basis, _empty_route())
        route["findings"] += 1
        route["assigned_findings"] += finding["cause_status"] == "assigned"
        route["abstained_findings"] += finding["cause_status"] == "abstain"
        route["evidence_verified_findings"] += verified
        route["confidence_labels"][confidence] = route["confidence_labels"].get(confidence, 0) + 1

    correct = {
        match["truth_index"]: match["prediction_index"] for match in result["diagnosis"]["matches"]
    }
    detected = {
        match["truth_index"]: match["prediction_index"] for match in result["detection"]["matches"]
    }
    assigned_entities: dict[tuple, int] = {}
    for index, finding in enumerate(findings):
        if finding["cause_status"] == "assigned":
            assigned_entities.setdefault(_entity_key(finding), index)
    for truth_index, defect in enumerate(defects):
        index = correct.get(truth_index)
        if index is None:
            index = assigned_entities.get(_entity_key(defect), detected.get(truth_index))
        finding = findings[index] if index is not None else None
        basis = finding["decision_basis"] if finding is not None else "undetected"
        route = routes.setdefault(basis, _empty_route())
        route["truth_count"] += 1
        assigned = finding is not None and finding["cause_status"] == "assigned"
        route["truth_with_assigned_diagnosis"] += assigned
        route["evidence_verified_truth_count"] += bool(finding and finding["evidence_verified"])
        route["diagnosis_tp"] += truth_index in correct
        route["diagnosis_fn"] += truth_index not in correct
        route["wrong_assigned_cause_count"] += bool(assigned and finding["kind"] != defect["kind"])
    for index in result["diagnosis"]["unmatched_prediction_indices"]:
        routes[findings[index]["decision_basis"]]["diagnosis_fp"] += 1
    for key in ("tp", "fp", "fn"):
        if sum(route[f"diagnosis_{key}"] for route in routes.values()) != result["diagnosis"][key]:
            raise ValueError("Decision-route counts do not reconcile to the v1 evaluator")
    result["decision_routes"] = {
        basis: _route_rates(route) for basis, route in sorted(routes.items())
    }
    return result


def aggregate_adaptive_evaluations(evaluations: list[dict]) -> dict:
    """Aggregate scored cases without turning route labels into probabilities."""
    result = _study_aggregate(evaluations)
    routes: dict[str, dict] = {}
    for evaluation in evaluations:
        for basis, values in evaluation["decision_routes"].items():
            route = routes.setdefault(basis, _empty_route())
            for key in _ROUTE_COUNTS:
                route[key] += values[key]
            for confidence, count in values["confidence_labels"].items():
                route["confidence_labels"][confidence] = (
                    route["confidence_labels"].get(confidence, 0) + count
                )
    result["decision_routes"] = {
        basis: _route_rates(route) for basis, route in sorted(routes.items())
    }
    result["confidence_interpretation"] = (
        "Qualitative route labels only; evidence_verified means internal coverage/path support passed, not ground-truth correctness, survey precision or calibrated probability"
    )
    return result


def render_adaptive_case(
    case: dict,
    predictions: dict,
    evaluations: dict,
    path: Path,
    modes: tuple = ADAPTIVE_MODES,
) -> None:
    """Render four policies on one supplied synthetic case after evaluation."""
    picture = Image.new("RGB", (1840, 590), "white")
    draw = ImageDraw.Draw(picture)
    draw.text(
        (24, 18), f"P2 adaptive v2: {case['truth']['condition']}", fill="#173047", font=_font(23)
    )
    draw.text(
        (24, 51),
        "Synthetic centerline candidates and observation proxies; XY plan. Checks use 3D coordinates.",
        fill="#536b7e",
        font=_font(16),
    )
    for column, mode in enumerate(modes):
        left = 24 + column * 455
        draw.text((left, 86), mode, fill="#173047", font=_font(18))
        _draw_panel(
            draw,
            (left, 119, left + 430, 380),
            case["input"],
            case["truth"],
            predictions[mode]["findings"],
        )
        diag = evaluations[mode]["diagnosis"]
        cause = evaluations[mode]["cause_evaluation"]
        draw.text(
            (left, 395),
            f"Diagnosis TP {diag['tp']} / FP {diag['fp']} / FN {diag['fn']}",
            fill="#344d60",
            font=_font(16),
        )
        draw.text(
            (left, 422),
            f"Assigned {cause['assigned_findings']}; abstained {cause['abstained_findings']}",
            fill="#344d60",
            font=_font(15),
        )
        labels = sorted({item["decision_basis"] for item in predictions[mode]["findings"]})
        draw.text(
            (left, 449), ", ".join(labels) or "No geometric alarms", fill="#344d60", font=_font(13)
        )
    draw.text(
        (24, 500),
        "Green circle: evaluator truth. Red cross: geometric alarm, including unresolved causes.",
        fill="#344d60",
        font=_font(16),
    )
    draw.text(
        (24, 531),
        "Geometry fallback is an assigned hypothesis, not evidence verification. Confidence labels are not probabilities.",
        fill="#536b7e",
        font=_font(16),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)


def _report_text(report: dict) -> str:
    lines = [
        "# P2 adaptive diagnosis study (v2)",
        "",
        f"Split: **{report['split']}**; {report['base_layout_count']} base layouts, {report['case_count']} derived cases, {report['detector_run_count']} detector runs.",
        "",
        "This run reuses the v1 case generator, seeds, conditions and scoring. Geometry and truth are unchanged. Customer data is not used. Development and validation are not independent held-out tests.",
        "",
        "## Overall results",
        "",
        "| Mode | Detection TP/FP/FN | Assigned diagnosis TP/FP/FN | Diagnosis recall | Assigned coverage | Correct among assigned | Normal cases flagged | Cause abstentions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, result in report["summary_by_mode"].items():
        detection, diagnosis, cause = (
            result["detection"],
            result["diagnosis"],
            result["cause_evaluation"],
        )
        lines.append(
            f"| {mode} | {detection['tp']}/{detection['fp']}/{detection['fn']} | "
            f"{diagnosis['tp']}/{diagnosis['fp']}/{diagnosis['fn']} | {_percent(diagnosis['recall'])} | "
            f"{_percent(cause['diagnosis_coverage'])} | {_percent(cause['selective_diagnosis_accuracy'])} | "
            f"{result['normal_flagged_case_count']}/{result['normal_case_count']} | {cause['abstained_findings']} |"
        )
    lines.extend(
        [
            "",
            "All four modes share entity detection. The three original policies remain unchanged controls. Only cause assignment/abstention and decision provenance differ. An abstained cause remains a diagnosis FN; wrong assignment contributes diagnosis FP and FN. Increasing assigned coverage does not itself prove better correctness.",
            "",
            "## Decision route accounting",
            "",
            "| Mode | Decision basis | Assigned / alerts | Diagnosis TP/FP/FN | Attributed truth | Assigned coverage | Wrong assigned cause / assigned truth | Evidence-verified alerts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for mode, result in report["summary_by_mode"].items():
        for basis, route in result["decision_routes"].items():
            lines.append(
                f"| {mode} | {basis} | {route['assigned_findings']}/{route['findings']} | "
                f"{route['diagnosis_tp']}/{route['diagnosis_fp']}/{route['diagnosis_fn']} | "
                f"{route['truth_count']} | {_percent(route['diagnosis_coverage'])} | "
                f"{route['wrong_assigned_cause_count']}/{route['truth_with_assigned_diagnosis']} | "
                f"{route['evidence_verified_findings']} |"
            )
    lines.extend(
        [
            "",
            "Each truth defect is attributed once: correct assigned match first, otherwise an assigned entity match, then a detected/abstained alert, or undetected. Extra alerts retain their own false-positive route. Route TP/FP/FN sums reconcile to the unchanged v1 evaluator. Route denominators are selected after the method decision and are not independent trial groups.",
            "",
            "Confidence levels are qualitative provenance labels, never probabilities. Evidence verified only means the internal neighborhood coverage and path-support rules passed; it does not certify ground-truth correctness, observation authenticity or measurement accuracy. Geometry fallback, evidence conflict and strict evidence abstention must not be described as evidence-verified success. Conditioned route rates are descriptive, not probability calibration or human-review time measurements.",
            "",
            "## Condition results (all cases)",
            "",
            "| Condition | Mode | Correct diagnosis / truth | Assigned / truth | Wrong assigned cause | Normal cases flagged / normal |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for condition, modes in report["by_condition"].items():
        for mode, result in modes.items():
            wrong = sum(
                route["wrong_assigned_cause_count"] for route in result["decision_routes"].values()
            )
            lines.append(
                f"| {condition} | {mode} | {result['diagnosis']['tp']}/{result['truth_defect_count']} | "
                f"{result['cause_evaluation']['truth_with_assigned_diagnosis']}/{result['truth_defect_count']} | "
                f"{wrong} | {result['normal_flagged_case_count']}/{result['normal_case_count']} |"
            )
    lines.extend(
        [
            "",
            "## Reproducibility and boundaries",
            "",
            "- `case_results.csv`, `case_index.json` and `failure_cases.json` retain every case/mode, including wrong assignments and abstentions. No outcome-based case filtering is performed.",
            "- `inputs/`, `ground_truth/` and `predictions/` remain separate. The auditor sees only the isolated candidate input and policy, never case IDs, split, condition or truth.",
            "- Figures show only the first registered layout per condition; selection is independent of outcomes. Existing P1/v1 outputs are not modified or repackaged.",
            "- `data_fingerprint_sha256` uses the exact v1 traversal and canonical case encoding, allowing direct comparison with the matching v1 split. Prediction fingerprints intentionally differ by method metadata and the fourth policy.",
            "- `manifest.json` binds source, dependency lock, protocol, outputs and any verified test freeze. Wall-clock timing is separate from deterministic metrics/fingerprints.",
            "- The adaptive policy was developed after observing v1 development/validation limitations. These reused splits are not fresh blind tests. The reserved test requires a verified adaptive source/protocol freeze.",
            "- These within-generator cases do not establish cross-site generalization, measured LiDAR performance, survey accuracy, support-asset reasoning or actual time saved by a human reviewer.",
            "",
            f"Data fingerprint: `{report['data_fingerprint_sha256']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_adaptive_split(
    protocol_path: str | Path,
    split: str,
    output: str | Path,
    freeze_path: str | Path | None = None,
) -> dict:
    """Run one complete explicit split; no implicit test generation or case limit."""
    from .synthetic_track_adaptive_audit import ADAPTIVE_MODES, audit_adaptive_candidate
    from .synthetic_track_adaptive_protocol import (
        validate_adaptive_protocol,
        verify_adaptive_freeze,
    )
    from .synthetic_track_study_protocol import _repository_root
    from .synthetic_track_study_scene import generate_study_layout

    protocol_source = Path(protocol_path).resolve()
    protocol_bytes = protocol_source.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    validate_adaptive_protocol(protocol)
    base = protocol["base_protocol"]
    modes = tuple(ADAPTIVE_MODES)
    if tuple(protocol["modes"]) != modes:
        raise ValueError("Registered adaptive modes differ from the executing auditor")
    if split not in {"development", "validation", "test"}:
        raise ValueError("Split must be development, validation or test")
    if split == "test" and freeze_path is None:
        raise ValueError("Test is reserved: a verified adaptive protocol/source freeze is required")
    freeze_verification = None
    if freeze_path is not None:
        if _repository_root(protocol_source) != Path(__file__).resolve().parents[2]:
            raise ValueError("Freeze verification repository differs from the executing toolkit")
        freeze_verification = verify_adaptive_freeze(protocol_source, Path(freeze_path))
        if freeze_verification.get("status") != "pass":
            raise ValueError("Frozen adaptive source/protocol verification failed")
        verified_root = freeze_verification.get("repository_root")
        if (
            verified_root is None
            or Path(verified_root).resolve() != Path(__file__).resolve().parents[2]
        ):
            raise ValueError("Verified freeze root differs from the executing toolkit")
        if protocol_source.read_bytes() != protocol_bytes:
            raise ValueError("Adaptive protocol changed during freeze verification")
    groups = [group for group in base["groups"] if group["split"] == split]
    if not groups:
        raise ValueError("Selected split has no registered groups")
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Adaptive output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evaluations = {mode: [] for mode in modes}
    by_condition: dict[str, dict[str, list]] = {}
    by_group: dict[str, dict[str, list]] = {}
    timings, rows, failures, case_index = [], [], [], []
    fingerprints = {key: hashlib.sha256() for key in ("data", "predictions")}
    seen_ids: set[str] = set()
    completed_ids: list[str] = []
    try:
        _write_json(destination / "protocol_snapshot.json", protocol)
        for group_index, group in enumerate(groups):
            cases = generate_study_layout(group["seed"], group["family_index"])
            if Counter(case["truth"]["condition"] for case in cases) != Counter(base["conditions"]):
                raise ValueError("Generated conditions differ from the registered base protocol")
            if len({case["layout_id"] for case in cases}) != 1:
                raise ValueError("A generated group must have exactly one base layout")
            by_group[group["group_id"]] = {mode: [] for mode in modes}
            for case in sorted(cases, key=lambda item: item["case_id"]):
                case_id, condition = case["case_id"], case["truth"]["condition"]
                if case_id in seen_ids or not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
                    raise ValueError("Case IDs must be unique safe opaque path components")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", condition):
                    raise ValueError("Condition names must be safe registered path components")
                seen_ids.add(case_id)
                # Keep byte encoding and update order identical to the v1 runner.
                fingerprints["data"].update(_canonical_bytes(case))
                _write_json(destination / "inputs" / f"{case_id}.json", case["input"])
                _write_json(
                    destination / "ground_truth" / f"{case_id}.json",
                    {
                        **case["truth"],
                        "layout_id": case["layout_id"],
                        "group_id": group["group_id"],
                    },
                )
                case_index.append(
                    {
                        "case_id": case_id,
                        "group_id": group["group_id"],
                        "layout_id": case["layout_id"],
                        "condition": condition,
                    }
                )
                by_condition.setdefault(condition, {mode: [] for mode in modes})
                predictions, scores = {}, {}
                for mode in modes:
                    before = time.perf_counter()
                    prediction = audit_adaptive_candidate(
                        copy.deepcopy(case["input"]), mode=mode, policy=protocol["policy"]
                    )
                    seconds = time.perf_counter() - before
                    fingerprints["predictions"].update(
                        _canonical_bytes(
                            {
                                "case_id": case_id,
                                "mode": mode,
                                "prediction": prediction,
                            }
                        )
                    )
                    _write_json(destination / "predictions" / mode / f"{case_id}.json", prediction)
                    score = evaluate_adaptive_prediction(case["truth"], prediction)
                    predictions[mode], scores[mode] = prediction, score
                    evaluations[mode].append(score)
                    by_condition[condition][mode].append(score)
                    by_group[group["group_id"]][mode].append(score)
                    timings.append({"case_id": case_id, "mode": mode, "detector_seconds": seconds})
                    routes = score["decision_routes"].values()
                    rows.append(
                        {
                            "case_id": case_id,
                            "group_id": group["group_id"],
                            "condition": condition,
                            "mode": mode,
                            **{
                                f"{metric}_{key}": score[metric][key]
                                for metric in ("detection", "diagnosis")
                                for key in ("tp", "fp", "fn")
                            },
                            "assigned_truth_count": score["cause_evaluation"][
                                "truth_with_assigned_diagnosis"
                            ],
                            "truth_count": score["truth_defect_count"],
                            "cause_abstentions": score["cause_evaluation"]["abstained_findings"],
                            "wrong_assigned_cause_count": sum(
                                route["wrong_assigned_cause_count"] for route in routes
                            ),
                            "evidence_verified_findings": sum(
                                route["evidence_verified_findings"] for route in routes
                            ),
                            "normal_case_flagged": score["normal_case_flagged"],
                            "decision_bases": ";".join(sorted(score["decision_routes"])),
                        }
                    )
                    if any(
                        score[metric][key]
                        for metric in ("detection", "diagnosis")
                        for key in ("fp", "fn")
                    ):
                        failures.append(
                            {
                                "case_id": case_id,
                                "condition": condition,
                                "mode": mode,
                                "detection": score["detection"],
                                "diagnosis": score["diagnosis"],
                                "cause_evaluation": score["cause_evaluation"],
                                "decision_routes": score["decision_routes"],
                            }
                        )
                reference = [
                    (item["entity_ids"], item["position"])
                    for item in predictions[modes[0]]["findings"]
                ]
                if any(
                    [
                        (item["entity_ids"], item["position"])
                        for item in predictions[mode]["findings"]
                    ]
                    != reference
                    for mode in modes[1:]
                ):
                    raise ValueError(
                        "An adaptive diagnosis policy changed the geometric detection set"
                    )
                if group_index == 0:
                    render_adaptive_case(
                        case,
                        predictions,
                        scores,
                        destination / "figures" / f"{condition}.png",
                        modes,
                    )
                completed_ids.append(case_id)
            print(f"adaptive {split}: {group_index + 1}/{len(groups)} groups completed", flush=True)
        group_summary = {
            key: {mode: aggregate_adaptive_evaluations(items) for mode, items in entries.items()}
            for key, entries in by_group.items()
        }
        paired = []
        for group_id, entries in group_summary.items():
            denominator = entries[modes[0]]["truth_defect_count"]
            if denominator:
                adaptive = entries["adaptive_evidence"]
                paired.append(
                    {
                        "group_id": group_id,
                        "truth_count": denominator,
                        **{
                            f"adaptive_minus_{comparison}_diagnosis_recall": (
                                adaptive["diagnosis"]["tp"] - entries[comparison]["diagnosis"]["tp"]
                            )
                            / denominator
                            for comparison in ("topology_only", "topology_evidence")
                        },
                        **{
                            f"adaptive_minus_{comparison}_assigned_coverage": (
                                adaptive["cause_evaluation"]["truth_with_assigned_diagnosis"]
                                - entries[comparison]["cause_evaluation"][
                                    "truth_with_assigned_diagnosis"
                                ]
                            )
                            / denominator
                            for comparison in ("topology_only", "topology_evidence")
                        },
                    }
                )
        report = {
            "schema_version": "railway.synthetic-track-adaptive-study-report.v1",
            "protocol_id": protocol["protocol_id"],
            "base_protocol_id": base["protocol_id"],
            "split": split,
            "modes": list(modes),
            "base_layout_count": len(groups),
            "case_count": len(seen_ids),
            "detector_run_count": len(rows),
            "customer_data_used": False,
            "held_out_test": split == "test",
            "source_freeze_verified": freeze_verification is not None,
            "confidence_labels_are_probabilities": False,
            "summary_by_mode": {
                mode: aggregate_adaptive_evaluations(items) for mode, items in evaluations.items()
            },
            "by_condition": {
                key: {
                    mode: aggregate_adaptive_evaluations(items) for mode, items in entries.items()
                }
                for key, entries in sorted(by_condition.items())
            },
            "by_group": group_summary,
            "paired_layout_differences": paired,
            "mean_paired_differences": {
                key: sum(item[key] for item in paired) / len(paired)
                for key in paired[0]
                if key not in {"group_id", "truth_count"}
            }
            if paired
            else {},
            "data_fingerprint_sha256": fingerprints["data"].hexdigest(),
            "prediction_fingerprint_sha256": fingerprints["predictions"].hexdigest(),
            "data_fingerprint_definition": "Identical to v1: canonical generated cases, base protocol group order, case_id-sorted variants; no policy or timing metadata",
        }
        _write_json(destination / "report.json", report)
        _write_json(destination / "failure_cases.json", failures)
        _write_json(destination / "case_index.json", case_index)
        _write_json(
            destination / "timings.json",
            {
                "elapsed_seconds": time.perf_counter() - started,
                "detector_runs": timings,
                "measurement": "Single sequential run; per-detector timing excludes serialization and rendering",
            },
        )
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(_report_text(report))
        with (destination / "case_results.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        root = Path(__file__).resolve().parents[2]
        sources = sorted((root / "src" / "railway_recon").rglob("*.py"))
        sources += [
            root / "pyproject.toml",
            root / "uv.lock",
            root / "scripts" / "run_synthetic_track_adaptive_study.py",
        ]
        _write_json(
            destination / "manifest.json",
            {
                "schema_version": "railway.synthetic-track-adaptive-run-manifest.v1",
                "split": split,
                "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
                "data_fingerprint_sha256": report["data_fingerprint_sha256"],
                "prediction_fingerprint_sha256": report["prediction_fingerprint_sha256"],
                "source_sha256": {
                    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources
                },
                "environment": {
                    "python": platform.python_version(),
                    "system": platform.system(),
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in ("numpy", "scipy", "Pillow")
                    },
                },
                "freeze_verification": freeze_verification,
                "files": [
                    {
                        "path": path.relative_to(destination).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in sorted(destination.rglob("*"))
                    if path.is_file()
                ],
                "manifest_self_hash_excluded": True,
            },
        )
        return report
    except Exception as error:
        _write_json(
            destination / "run_failure.json",
            {
                "split": split,
                "status": "failed",
                "exception_type": type(error).__name__,
                "started_case_ids": sorted(seen_ids),
                "completed_case_ids": completed_ids,
            },
        )
        raise
