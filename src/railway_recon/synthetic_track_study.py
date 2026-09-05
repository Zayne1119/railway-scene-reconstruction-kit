"""P2 three-way development/validation study with explicit diagnosis abstention.

Split and truth metadata stay outside the auditor. A test split requires a
verified source/protocol freeze and is never implicitly run by this module.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from .synthetic_track_pilot import (
    _aggregate,
    _canonical_bytes,
    _draw_panel,
    _font,
    _write_json,
    evaluate_track_prediction,
)
from .synthetic_track_study_audit import STUDY_MODES, audit_study_candidate


def evaluate_study_prediction(truth: dict, prediction: dict) -> dict:
    """Never score an abstained cause as a correct gap; keep its entity alarm."""
    findings = prediction.get("findings", [])
    if any(item.get("cause_status") not in {"assigned", "abstain"} for item in findings):
        raise ValueError("Every study finding must explicitly assign or abstain on its cause")
    result = evaluate_track_prediction(truth, prediction)
    assigned = copy.deepcopy(prediction)
    assigned_original_indices = [index for index, item in enumerate(findings)
                                 if item["cause_status"] == "assigned"]
    assigned["findings"] = [item for item in findings if item["cause_status"] == "assigned"]
    diagnosis = evaluate_track_prediction(truth, assigned)
    result["diagnosis"] = diagnosis["diagnosis"]
    for kind in result["per_kind"]:
        result["per_kind"][kind]["diagnosis"] = diagnosis["per_kind"][kind]["diagnosis"]
    matched = {item["truth_index"]: item["prediction_index"]
               for item in result["detection"]["matches"]}
    assigned_matches = {item["truth_index"]: item["prediction_index"]
                        for item in diagnosis["detection"]["matches"]}
    correct_matches = {item["truth_index"]: item["prediction_index"]
                       for item in diagnosis["diagnosis"]["matches"]}
    confusion: Counter[str] = Counter()
    covered = 0
    for index, defect in enumerate(truth.get("defects", [])):
        if index in assigned_matches:
            prediction_index = correct_matches.get(index, assigned_matches[index])
            label = assigned["findings"][prediction_index]["kind"]
            covered += 1
        elif index not in matched:
            label = "missed"
        else:
            finding = findings[matched[index]]
            label = finding["kind"] if finding["cause_status"] == "assigned" else "abstain"
        confusion[f"{defect['kind']}->{label}"] += 1
    result["cause_evaluation"] = {
        "assigned_findings": len(assigned["findings"]),
        "abstained_findings": len(findings) - len(assigned["findings"]),
        "truth_with_assigned_diagnosis": covered,
        "confusion": dict(confusion),
    }
    # Diagnosis was evaluated on the assigned subset, but persisted prediction
    # references must always point into the original full findings array.
    for match in result["diagnosis"]["matches"]:
        match["prediction_index"] = assigned_original_indices[match["prediction_index"]]
    result["diagnosis"]["unmatched_prediction_indices"] = [
        assigned_original_indices[index]
        for index in result["diagnosis"]["unmatched_prediction_indices"]
    ]
    return result


def _study_aggregate(evaluations: list[dict]) -> dict:
    result = _aggregate(evaluations)
    confusion: Counter[str] = Counter()
    for item in evaluations:
        confusion.update(item["cause_evaluation"]["confusion"])
    totals = {key: sum(item["cause_evaluation"][key] for item in evaluations) for key in (
        "assigned_findings", "abstained_findings", "truth_with_assigned_diagnosis"
    )}
    count = result["truth_defect_count"]
    covered = totals["truth_with_assigned_diagnosis"]
    result["cause_evaluation"] = {
        **totals, "confusion": dict(sorted(confusion.items())),
        "diagnosis_coverage": covered / count if count else None,
        "selective_diagnosis_accuracy": result["diagnosis"]["tp"] / covered if covered else None,
        "interpretation": "Coverage uses all truth defects; abstentions remain diagnosis FN. Selective accuracy must not be reported without coverage and overall recall.",
    }
    return result


def _render(case: dict, predictions: dict, evaluations: dict, path: Path) -> None:
    picture = Image.new("RGB", (1440, 550), "white")
    draw = ImageDraw.Draw(picture)
    draw.text((24, 18), f"P2: {case['truth']['condition']}", fill="#163047", font=_font(22))
    draw.text((24, 49), "Independent synthetic centerlines; XY plan. Heights are used in all 3D checks.",
              fill="#536b7e", font=_font(15))
    for column, mode in enumerate(STUDY_MODES):
        left = 24 + column * 475
        draw.text((left, 83), mode, fill="#163047", font=_font(18))
        _draw_panel(draw, (left, 115, left + 445, 380), case["input"], case["truth"],
                    predictions[mode]["findings"])
        diagnosis = evaluations[mode]["diagnosis"]
        draw.text((left, 397), f"Diagnosis TP {diagnosis['tp']} / FP {diagnosis['fp']} / FN {diagnosis['fn']}",
                  fill="#344d60", font=_font(16))
        n = evaluations[mode]["cause_evaluation"]["abstained_findings"]
        draw.text((left, 424), f"Cause abstentions: {n}", fill="#344d60", font=_font(15))
    draw.text((24, 476), "Green circle: evaluator truth. Red cross: geometric alarm, including unresolved causes.",
              fill="#344d60", font=_font(16))
    draw.text((24, 506), "Shared detection rules; compare diagnosis and its coverage. This is not a field accuracy result.",
              fill="#536b7e", font=_font(15))
    path.parent.mkdir(parents=True, exist_ok=True)
    picture.save(path)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _report_text(report: dict) -> str:
    lines = [
        "# P2 three-way synthetic track study", "",
        f"Split: **{report['split']}**. {report['base_layout_count']} base layouts; {report['case_count']} derived cases.",
        "",
        "These are independently parameterized centerline proxies, not physical field sites or LiDAR. Customer data is not used. Development/validation results are not held-out test results.",
        "",
        "## Overall results", "",
        "| Mode | Detection TP/FP/FN | Diagnosis TP/FP/FN | Diagnosis recall | Diagnosis coverage | Correct among diagnosed | Normal cases flagged | Cause abstentions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, item in report["summary_by_mode"].items():
        det, diag, cause = item["detection"], item["diagnosis"], item["cause_evaluation"]
        lines.append(
            f"| {mode} | {det['tp']}/{det['fp']}/{det['fn']} | {diag['tp']}/{diag['fp']}/{diag['fn']} | "
            f"{_percent(diag['recall'])} | {_percent(cause['diagnosis_coverage'])} | "
            f"{_percent(cause['selective_diagnosis_accuracy'])} | "
            f"{item['normal_flagged_case_count']}/{item['normal_case_count']} | {cause['abstained_findings']} |"
        )
    lines.extend([
        "", "All modes share geometric flags and entity matching; detection equality is expected by design. Only diagnosis differs. Wrong cause yields a diagnosis FP and FN, not an extra geometric false alarm. An abstained cause remains a detected entity but is a diagnosis FN, never a correct default gap.",
        "", "## Conditions (all cases, no outcome-based selection)", "",
        "| Condition | Mode | Correct diagnosis / truth | Normal cases flagged / normal cases | Cause abstentions |",
        "| --- | --- | --- | --- | --- |",
    ])
    for condition, modes in report["by_condition"].items():
        for mode, item in modes.items():
            lines.append(
                f"| {condition} | {mode} | {item['diagnosis']['tp']}/{item['truth_defect_count']} | "
                f"{item['normal_flagged_case_count']}/{item['normal_case_count']} | "
                f"{item['cause_evaluation']['abstained_findings']} |"
            )
    lines.extend([
        "", "## Interpretation and reproducibility", "",
        "- `report.json` includes cause confusion, per-layout results, normal-check denominators and paired layout-level diagnosis differences. No confidence or significance claim is made from development/validation results.",
        "- `case_results.csv` contains every case/mode. `inputs`, `ground_truth` and `predictions` are separate. No case ID, split label, truth or condition is passed to the auditor.",
        "- Representative figures use the first preordered group for each condition, independent of outcomes. Failure case IDs are all recorded in `failure_cases.json`; figures do not replace the complete result table.",
        "- Empty/sparse observation coverage may prevent cause assignment without clearing a geometric alarm. Endpoint support does not rule out all bridge occlusions or incorrect observations.",
        "- A fixed pilot policy is used without performance-based threshold optimization. Evaluation at other operating points and real-data method validation remain separate work.",
        "- `manifest.json` binds inputs, truth, predictions, sources and outputs; wall-clock timings are excluded from deterministic fingerprints. P1 files and outputs are not overwritten.",
        "- These within-generator results do not establish cross-site generalization, support-asset reasoning, survey precision or saved human review time.",
    ])
    return "\n".join(lines) + "\n"


def run_study_split(
    protocol_path: str | Path, split: str, output: str | Path,
    freeze_path: str | Path | None = None,
) -> dict:
    from .synthetic_track_study_protocol import (
        _repository_root,
        validate_study_protocol,
        verify_study_freeze,
    )
    from .synthetic_track_study_scene import generate_study_layout

    protocol_source = Path(protocol_path).resolve()
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    validate_study_protocol(protocol)
    if split not in {"development", "validation", "test"}:
        raise ValueError("Split must be development, validation or test")
    if split == "test" and freeze_path is None:
        raise ValueError("Test is reserved: a verified protocol/source freeze is required")
    freeze_verification = None
    if freeze_path is not None:
        if _repository_root(protocol_source) != Path(__file__).resolve().parents[2]:
            raise ValueError("Freeze verification repository differs from the executing toolkit")
        freeze_verification = verify_study_freeze(protocol_source, Path(freeze_path))
        if freeze_verification.get("status") != "pass":
            raise ValueError("Frozen study source/protocol verification failed")
    groups = [item for item in protocol["groups"] if item["split"] == split]
    if not groups:
        raise ValueError("Selected split has no registered groups")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evaluations: dict[str, list] = {mode: [] for mode in STUDY_MODES}
    by_condition: dict[str, dict[str, list]] = {}
    by_group: dict[str, dict[str, list]] = {}
    timings, rows, failures = [], [], []
    fingerprints = {key: hashlib.sha256() for key in ("data", "predictions")}
    seen_ids: set[str] = set()
    case_index = []
    try:
        _write_json(destination / "protocol_snapshot.json", protocol)
        for group_index, group in enumerate(groups):
            cases = generate_study_layout(group["seed"], group["family_index"])
            if Counter(case["truth"]["condition"] for case in cases) != Counter(protocol["conditions"]):
                raise ValueError("Generated conditions differ from the registered protocol")
            if len({case["layout_id"] for case in cases}) != 1:
                raise ValueError("A generated group must have exactly one base layout")
            by_group[group["group_id"]] = {mode: [] for mode in STUDY_MODES}
            for case in sorted(cases, key=lambda item: item["case_id"]):
                case_id, condition = case["case_id"], case["truth"]["condition"]
                if case_id in seen_ids:
                    raise ValueError("Case IDs collide across groups")
                seen_ids.add(case_id)
                fingerprints["data"].update(_canonical_bytes(case))
                _write_json(destination / "inputs" / f"{case_id}.json", case["input"])
                _write_json(destination / "ground_truth" / f"{case_id}.json", {
                    **case["truth"], "layout_id": case["layout_id"], "group_id": group["group_id"]
                })
                case_index.append({"case_id": case_id, "group_id": group["group_id"],
                                   "layout_id": case["layout_id"], "condition": condition})
                by_condition.setdefault(condition, {mode: [] for mode in STUDY_MODES})
                predictions, scored = {}, {}
                for mode in STUDY_MODES:
                    before = time.perf_counter()
                    prediction = audit_study_candidate(copy.deepcopy(case["input"]), mode,
                                                       policy=protocol["policy"])
                    seconds = time.perf_counter() - before
                    fingerprints["predictions"].update(_canonical_bytes({"case_id": case_id,
                                                                         "mode": mode, "prediction": prediction}))
                    _write_json(destination / "predictions" / mode / f"{case_id}.json", prediction)
                    score = evaluate_study_prediction(case["truth"], prediction)
                    predictions[mode], scored[mode] = prediction, score
                    evaluations[mode].append(score)
                    by_condition[condition][mode].append(score)
                    by_group[group["group_id"]][mode].append(score)
                    timings.append({"case_id": case_id, "mode": mode, "detector_seconds": seconds})
                    rows.append({"case_id": case_id, "group_id": group["group_id"],
                                 "condition": condition, "mode": mode,
                                 **{f"{metric}_{key}": score[metric][key] for metric in ("detection", "diagnosis")
                                    for key in ("tp", "fp", "fn")},
                                 "cause_abstentions": score["cause_evaluation"]["abstained_findings"],
                                 "normal_case_flagged": score["normal_case_flagged"]})
                    if any(score[metric][key] for metric in ("detection", "diagnosis") for key in ("fp", "fn")):
                        failures.append({"case_id": case_id, "condition": condition, "mode": mode,
                                         "detection": score["detection"], "diagnosis": score["diagnosis"],
                                         "cause_evaluation": score["cause_evaluation"]})
                # Failures must remain visible; shared entity detection is a contract,
                # not an empirical result to be portrayed as an evidence advantage.
                reference = [(item["entity_ids"], item["position"]) for item in predictions[STUDY_MODES[0]]["findings"]]
                if any([(item["entity_ids"], item["position"]) for item in predictions[mode]["findings"]]
                       != reference for mode in STUDY_MODES[1:]):
                    raise ValueError("A diagnosis ablation changed the geometric detection set")
                if group_index == 0:
                    _render(case, predictions, scored, destination / "figures" / f"{condition}.png")
            print(f"{split}: {group_index + 1}/{len(groups)} groups completed", flush=True)
        group_summary = {key: {mode: _study_aggregate(items) for mode, items in modes.items()}
                         for key, modes in by_group.items()}
        paired = []
        for key, modes in group_summary.items():
            denominator = modes["local_geometry"]["truth_defect_count"]
            if denominator:
                paired.append({"group_id": key, "truth_count": denominator,
                               "evidence_minus_topology_diagnosis_recall":
                               (modes["topology_evidence"]["diagnosis"]["tp"]
                                - modes["topology_only"]["diagnosis"]["tp"]) / denominator})
        report = {
            "schema_version": "railway.synthetic-track-study-report.v1",
            "protocol_id": protocol["protocol_id"], "split": split,
            "base_layout_count": len(groups), "case_count": len(seen_ids),
            "customer_data_used": False, "held_out_test": split == "test",
            "source_freeze_verified": freeze_verification is not None,
            "summary_by_mode": {mode: _study_aggregate(items) for mode, items in evaluations.items()},
            "by_condition": {key: {mode: _study_aggregate(items) for mode, items in modes.items()}
                             for key, modes in sorted(by_condition.items())},
            "by_group": group_summary, "paired_layout_differences": paired,
            "mean_paired_evidence_minus_topology_diagnosis_recall":
                sum(item["evidence_minus_topology_diagnosis_recall"] for item in paired) / len(paired) if paired else None,
            "data_fingerprint_sha256": fingerprints["data"].hexdigest(),
            "prediction_fingerprint_sha256": fingerprints["predictions"].hexdigest(),
        }
        _write_json(destination / "report.json", report)
        _write_json(destination / "failure_cases.json", failures)
        _write_json(destination / "case_index.json", case_index)
        _write_json(destination / "timings.json", {"elapsed_seconds": time.perf_counter() - started,
                                                  "detector_runs": timings})
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(_report_text(report))
        with (destination / "case_results.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        root = Path(__file__).resolve().parents[2]
        # Bind all toolkit source dependencies, not just a hand-picked top module.
        sources = sorted((root / "src" / "railway_recon").rglob("*.py"))
        sources += [root / "pyproject.toml", root / "uv.lock",
                    root / "scripts" / "run_synthetic_track_study.py"]
        _write_json(destination / "manifest.json", {
            "schema_version": "railway.synthetic-track-study-run-manifest.v1",
            "split": split, "protocol_sha256": hashlib.sha256(protocol_source.read_bytes()).hexdigest(),
            "data_fingerprint_sha256": report["data_fingerprint_sha256"],
            "prediction_fingerprint_sha256": report["prediction_fingerprint_sha256"],
            "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow")}},
            "freeze_verification": freeze_verification,
            "files": [{"path": path.relative_to(destination).as_posix(),
                       "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(destination.rglob("*")) if path.is_file()],
            "manifest_self_hash_excluded": True,
        })
        return report
    except Exception as error:
        _write_json(destination / "run_failure.json", {"split": split, "status": "failed",
                    "exception_type": type(error).__name__, "completed_case_ids": sorted(seen_ids)})
        raise
