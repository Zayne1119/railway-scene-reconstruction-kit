"""Run and evaluate the independent synthetic track development pilot.

Ground truth belongs exclusively to this evaluator. The detector receives a
JSON-isolated candidate input, never a case envelope or injection metadata.
This small pilot is not a frozen benchmark or evidence of field performance.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

KINDS = ("gap", "wrong_connection", "duplicate")
MODES = ("local_geometry", "topology_evidence")
_COUNT_KEYS = ("tp", "fp", "fn")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _entities(item: dict[str, Any]) -> tuple[str, ...]:
    identifiers = item.get("entity_ids")
    if not isinstance(identifiers, list) or not identifiers:
        raise ValueError("Every defect, finding and check must identify at least one entity")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("Entity identifiers must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Repeated entity identifiers within one item are ambiguous")
    if item.get("kind") not in KINDS:
        raise ValueError(f"Unsupported defect kind: {item.get('kind')!r}")
    return tuple(sorted(identifiers))


def _check_key(item: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return item["kind"], _entities(item)


def _score_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _distance(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    if len(first) != 3 or len(second) != 3:
        raise ValueError("Localization positions must have three coordinates")
    result = math.dist(first, second)
    if not math.isfinite(result):
        raise ValueError("Localization positions must be finite")
    return result


def _match(
    defects: list[dict[str, Any]], findings: list[dict[str, Any]], *, strict_kind: bool
) -> dict[str, Any]:
    key = _check_key if strict_kind else _entities
    remaining = {key(defect): index for index, defect in enumerate(defects)}
    if len(remaining) != len(defects):
        raise ValueError("Multiple truth defects have the same matching key")
    matches: list[dict[str, Any]] = []
    unmatched_predictions = []
    for index, finding in enumerate(findings):
        truth_index = remaining.pop(key(finding), None)
        if truth_index is None:
            unmatched_predictions.append(index)
        else:
            matches.append(
                {
                    "prediction_index": index,
                    "truth_index": truth_index,
                    "truth_kind": defects[truth_index]["kind"],
                    "predicted_kind": finding["kind"],
                    "localization_error_m": _distance(
                        finding.get("position"), defects[truth_index].get("position")
                    ),
                }
            )
    result = _score_counts(len(matches), len(unmatched_predictions), len(remaining))
    result.update(
        {
            "matches": matches,
            "unmatched_prediction_indices": unmatched_predictions,
            "unmatched_truth_indices": sorted(remaining.values()),
        }
    )
    return result


def evaluate_track_prediction(truth: dict[str, Any], prediction: dict[str, Any]) -> dict:
    """Score detection and diagnosis separately with strict one-to-one matching.

    Detection matches the affected entity set regardless of the named cause;
    diagnosis additionally requires the correct defect kind. Extra alarms for
    a previously matched fault are false positives, not extra true positives.
    FPR counts only assessed checks on entities with no injected fault; normal
    abstentions remain visible in a separate denominator and coverage measure.
    """
    defects = truth.get("defects", [])
    findings = prediction.get("findings", [])
    checks = prediction.get("checks", [])
    for item in [*defects, *findings, *checks]:
        _entities(item)
    detection = _match(defects, findings, strict_kind=False)
    diagnosis = _match(defects, findings, strict_kind=True)
    per_kind: dict[str, dict] = {}
    for kind in KINDS:
        truth_count = sum(defect["kind"] == kind for defect in defects)
        detected = sum(match["truth_kind"] == kind for match in detection["matches"])
        correct_kind = sum(match["truth_kind"] == kind for match in diagnosis["matches"])
        diagnosed_fp = sum(
            findings[index]["kind"] == kind for index in diagnosis["unmatched_prediction_indices"]
        )
        per_kind[kind] = {
            "truth_count": truth_count,
            "detection_tp": detected,
            "detection_fn": truth_count - detected,
            "detection_recall": _ratio(detected, truth_count),
            "detection_fp_reported_as_kind": sum(
                findings[index]["kind"] == kind
                for index in detection["unmatched_prediction_indices"]
            ),
            "diagnosis": _score_counts(correct_kind, diagnosed_fp, truth_count - correct_kind),
        }

    faulty_entities = {_entities(defect) for defect in defects}
    unique_checks: dict[tuple, dict] = {}
    for check in checks:
        if check.get("status") not in {"pass", "flag", "abstain"}:
            raise ValueError("Check status must be pass, flag or abstain")
        identifier = _check_key(check)
        if identifier in unique_checks:
            raise ValueError("Detector emitted duplicate checks with the same kind and entity set")
        unique_checks[identifier] = check
    normal_checks = [
        check for check in unique_checks.values() if _entities(check) not in faulty_entities
    ]
    normal_assessed = sum(check["status"] != "abstain" for check in normal_checks)
    normal_flags = sum(check["status"] == "flag" for check in normal_checks)
    abstain_count = sum(check["status"] == "abstain" for check in unique_checks.values())
    tested_entities = {
        _entities(check) for check in unique_checks.values() if check["status"] != "abstain"
    }
    covered_defects = sum(_entities(defect) in tested_entities for defect in defects)
    repeated_entities = Counter(_entities(finding) for finding in findings)
    errors = [
        match["localization_error_m"]
        for match in detection["matches"]
        if match["localization_error_m"] is not None
    ]
    return {
        "detection": detection,
        "diagnosis": diagnosis,
        "per_kind": per_kind,
        "truth_defect_count": len(defects),
        "finding_count": len(findings),
        "normal_case": not defects,
        "normal_case_flagged": not defects and bool(findings),
        "extra_alerts_on_repeated_entities": sum(count - 1 for count in repeated_entities.values()),
        "localization_error_m": errors,
        "checks": {
            "total": len(unique_checks),
            "assessed": len(unique_checks) - abstain_count,
            "abstain": abstain_count,
            "evidence_abstain": sum(
                check.get("measurements", {}).get("evidence_status") == "abstain"
                for check in unique_checks.values()
            ),
            "assessment_coverage": _ratio(len(unique_checks) - abstain_count, len(unique_checks)),
            "normal_total": len(normal_checks),
            "normal_assessed": normal_assessed,
            "normal_abstain": len(normal_checks) - normal_assessed,
            "normal_flagged": normal_flags,
            "normal_check_fpr": _ratio(normal_flags, normal_assessed),
            "truth_defects_with_assessed_entity_check": covered_defects,
            "truth_defect_check_coverage": _ratio(covered_defects, len(defects)),
        },
    }


def _aggregate(evaluations: list[dict]) -> dict:
    result: dict[str, Any] = {}
    for metric in ("detection", "diagnosis"):
        result[metric] = _score_counts(
            *(sum(value[metric][key] for value in evaluations) for key in _COUNT_KEYS)
        )
    for key in ("truth_defect_count", "finding_count", "extra_alerts_on_repeated_entities"):
        result[key] = sum(value[key] for value in evaluations)
    result["normal_case_count"] = sum(value["normal_case"] for value in evaluations)
    result["normal_flagged_case_count"] = sum(value["normal_case_flagged"] for value in evaluations)
    result["normal_case_flag_rate"] = _ratio(
        result["normal_flagged_case_count"], result["normal_case_count"]
    )
    integer_checks = (
        "total",
        "assessed",
        "abstain",
        "normal_total",
        "normal_assessed",
        "normal_abstain",
        "normal_flagged",
        "evidence_abstain",
        "truth_defects_with_assessed_entity_check",
    )
    counts = {key: sum(value["checks"][key] for value in evaluations) for key in integer_checks}
    counts.update(
        {
            "assessment_coverage": _ratio(counts["assessed"], counts["total"]),
            "normal_check_fpr": _ratio(counts["normal_flagged"], counts["normal_assessed"]),
            "truth_defect_check_coverage": _ratio(
                counts["truth_defects_with_assessed_entity_check"], result["truth_defect_count"]
            ),
        }
    )
    result["checks"] = counts
    result["per_kind"] = {}
    for kind in KINDS:
        entries = [value["per_kind"][kind] for value in evaluations]
        sums = {
            key: sum(entry[key] for entry in entries)
            for key in (
                "truth_count",
                "detection_tp",
                "detection_fn",
                "detection_fp_reported_as_kind",
            )
        }
        sums["detection_recall"] = _ratio(sums["detection_tp"], sums["truth_count"])
        sums["diagnosis"] = _score_counts(
            *(sum(entry["diagnosis"][key] for entry in entries) for key in _COUNT_KEYS)
        )
        result["per_kind"][kind] = sums
    errors = [error for value in evaluations for error in value["localization_error_m"]]
    result["localization"] = {
        "matched_with_positions": len(errors),
        "mean_error_m": sum(errors) / len(errors) if errors else None,
        "median_error_m": statistics.median(errors) if errors else None,
        "max_error_m": max(errors) if errors else None,
        "interpretation": "3D centerline marker difference on entity-matched alerts; not survey or reconstruction accuracy",
    }
    return result


def _font(size: int) -> Any:
    # Pillow's bundled default font avoids machine-specific font dependencies.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 does not yet expose the bundled scalable font.
        return ImageFont.load_default()


def _dashed_line(draw: Any, first: tuple, last: tuple, color: str, width: int = 2) -> None:
    length = math.dist(first, last)
    if length < 1:
        return
    for start in range(0, math.ceil(length), 9):
        end = min(start + 5, length)
        draw.line(
            [
                tuple(first[axis] + (last[axis] - first[axis]) * start / length for axis in (0, 1)),
                tuple(first[axis] + (last[axis] - first[axis]) * end / length for axis in (0, 1)),
            ],
            fill=color,
            width=width,
        )


def _draw_panel(draw: Any, bounds: tuple, candidate: dict, truth: dict, findings: list) -> None:
    left, top, right, bottom = bounds
    segments = candidate["segments"]
    points = [point for segment in segments for point in segment["points"]]
    positions = [item["position"] for item in [*truth["defects"], *findings]]
    extent = [*points, *positions]
    xmin, xmax = min(point[0] for point in extent), max(point[0] for point in extent)
    ymin, ymax = min(point[1] for point in extent), max(point[1] for point in extent)
    scale = min(
        (right - left - 28) / max(xmax - xmin, 1), (bottom - top - 32) / max(ymax - ymin, 1)
    )
    xmiddle, ymiddle = (xmin + xmax) / 2, (ymin + ymax) / 2

    def xy(point: list) -> tuple[float, float]:
        return (
            (left + right) / 2 + (point[0] - xmiddle) * scale,
            (top + bottom) / 2 - (point[1] - ymiddle) * scale,
        )

    draw.rectangle(bounds, outline="#d8e1e9", width=1)
    observations = candidate.get("observations", [])
    stride = max(1, len(observations) // 3000)
    for point in observations[::stride]:
        pixel = xy(point)
        if left < pixel[0] < right and top < pixel[1] < bottom:
            draw.point(pixel, fill="#cbd5df")
    alerted = {identifier for finding in findings for identifier in finding["entity_ids"]}
    for segment in segments:
        color = "#b33939" if segment["id"] in alerted else "#32688c"
        draw.line([xy(point) for point in segment["points"]], fill=color, width=3)
    segment_map = {segment["id"]: segment for segment in segments}
    for connection in candidate["connections"]:
        source = segment_map[connection["source"]]["points"][-1]
        target = segment_map[connection["target"]]["points"][0]
        color = "#b33939" if connection["id"] in alerted else "#b18b55"
        _dashed_line(draw, xy(source), xy(target), color)
    for defect in truth["defects"]:
        x, y = xy(defect["position"])
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline="#20774f", width=3)
    for finding in findings:
        x, y = xy(finding["position"])
        draw.line((x - 6, y - 6, x + 6, y + 6), fill="#c52b3c", width=3)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill="#c52b3c", width=3)
    # Equal XY scale and an explicit length reference avoid distorted geometry.
    scale_length = 10 ** math.floor(math.log10(max((right - left) / scale / 5, 0.01)))
    bar_x, bar_y = left + 14, bottom - 14
    draw.line((bar_x, bar_y, bar_x + scale_length * scale, bar_y), fill="#2f4559", width=2)
    draw.text((bar_x, bar_y - 15), f"{scale_length:g} m", fill="#2f4559", font=_font(12))


def _render_case(case: dict, predictions: dict[str, dict], output: Path) -> Image.Image:
    picture = Image.new("RGB", (1120, 500), "#ffffff")
    draw = ImageDraw.Draw(picture)
    draw.text(
        (24, 18),
        f"{case['case_id']}  |  {case['truth']['variant']}",
        fill="#172c40",
        font=_font(20),
    )
    draw.text(
        (24, 48),
        "Independent synthetic centerline proxy; XY plan, equal scale",
        fill="#546779",
        font=_font(15),
    )
    for column, mode in enumerate(MODES):
        left = 24 + column * 558
        draw.text((left, 83), mode, fill="#172c40", font=_font(17))
        _draw_panel(
            draw,
            (left, 112, left + 532, 362),
            case["input"],
            case["truth"],
            predictions[mode]["findings"],
        )
        evaluation = evaluate_track_prediction(case["truth"], predictions[mode])
        for offset, metric in enumerate(("detection", "diagnosis")):
            counts = evaluation[metric]
            draw.text(
                (left, 376 + offset * 23),
                f"{metric}: TP {counts['tp']}   FP {counts['fp']}   FN {counts['fn']}",
                fill="#344d60",
                font=_font(15),
            )
    draw.ellipse((25, 443, 43, 461), outline="#20774f", width=3)
    draw.text((51, 443), "Truth (evaluator only)", fill="#344d60", font=_font(15))
    draw.line((279, 445, 291, 457), fill="#c52b3c", width=3)
    draw.line((279, 457, 291, 445), fill="#c52b3c", width=3)
    draw.text((302, 443), "Alert (detector)", fill="#344d60", font=_font(15))
    draw.text(
        (489, 443),
        "Blue: candidate; gray: observations; dashed: connection",
        fill="#344d60",
        font=_font(14),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    picture.save(output)
    return picture


def _render_overview(cases: list[dict], predictions: dict, output: Path) -> None:
    groups = sorted({case["layout_id"] for case in cases})
    variants = ("normal", "gap", "wrong_connection", "duplicate")
    tile_width, tile_height = 420, 235
    picture = Image.new("RGB", (tile_width * 4, tile_height * len(groups) + 95), "white")
    draw = ImageDraw.Draw(picture)
    draw.text(
        (20, 12),
        "Synthetic development pilot: independent layouts, paired variants",
        fill="#172c40",
        font=_font(23),
    )
    draw.text(
        (20, 43),
        "Topology/evidence alerts only here; both modes shown in individual figures. Green circle: Truth; red cross: Alert.",
        fill="#4c6274",
        font=_font(17),
    )
    for row, group in enumerate(groups):
        for column, variant in enumerate(variants):
            case = next(
                case
                for case in cases
                if case["layout_id"] == group and case["truth"]["variant"] == variant
            )
            left, top = column * tile_width + 12, row * tile_height + 84
            draw.text((left, top), f"Layout {row + 1} / {variant}", fill="#172c40", font=_font(17))
            draw.text((left, top + 24), case["case_id"], fill="#4c6274", font=_font(12))
            _draw_panel(
                draw,
                (left, top + 48, left + 395, top + 206),
                case["input"],
                case["truth"],
                predictions[case["case_id"]]["topology_evidence"]["findings"],
            )
    picture.save(output)


def _rate_text(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _write_report(output: Path, report: dict) -> None:
    lines = [
        "# Independent synthetic track pilot (P1)",
        "",
        (
            f"Seed: `{report['seed']}`. {report['case_count']} paired cases from "
            f"{report['base_layout_count']} base layouts. Development pilot, not a held-out test."
        ),
        "",
        (
            "All geometry and observations are generated from generic parameters. No customer data, "
            "production models, site layouts, coordinates or imagery are inputs. Curves represent "
            "track centerlines, not complete track meshes or measured LiDAR scans."
        ),
        "",
        "## Results",
        "",
        (
            "| Mode | Detection TP / FP / FN | Diagnosis TP / FP / FN | Detection recall | "
            "Normal cases flagged | Normal check flags / assessed | Abstained checks |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        result = report["summary_by_mode"][mode]
        detection, diagnosis, checks = result["detection"], result["diagnosis"], result["checks"]
        lines.append(
            f"| {mode} | {detection['tp']} / {detection['fp']} / {detection['fn']} | "
            f"{diagnosis['tp']} / {diagnosis['fp']} / {diagnosis['fn']} | "
            f"{_rate_text(detection['recall'])} | {result['normal_flagged_case_count']} / "
            f"{result['normal_case_count']} | {checks['normal_flagged']} / "
            f"{checks['normal_assessed']} | {checks['abstain']} / {checks['total']} |"
        )
    lines.extend(
        [
            "",
            (
                "Detection matches the exact affected entity set, regardless of the predicted cause. "
                "Diagnosis also requires the correct cause. For example, identifying a wrong connection "
                "as a gap can correctly detect the faulty connection while failing diagnosis. Matching "
                "is one-to-one: repeated alerts cannot increase true positives."
            ),
            "",
            (
                "Both modes share every fault-triggering/entity-detection rule in this implementation. "
                "Equal detection is therefore expected by design; the enhanced mode only refines the "
                "cause assigned to an already flagged connection. This pilot cannot establish improved "
                "detection or isolate the contribution of observation evidence. P2 needs separate "
                "local_geometry, topology_only and topology_evidence ablations on frozen data."
            ),
            "",
            (
                "Normal-check FPR excludes any check on an injected-fault entity set, even when the "
                "check names another cause. Its denominator contains assessed normal checks only; "
                "abstentions, all normal candidates, and fault-check coverage are reported separately. "
                "Check counts depend on each mode's explicit check universe and are not independent "
                "samples. No confidence interval or significance claim is made from these six layouts."
            ),
            "",
            "## Detection by injected defect",
            "",
            "| Mode | Ground-truth kind | Detected / total | Correct diagnosis / total |",
            "| --- | --- | --- | --- |",
        ]
    )
    for mode in MODES:
        for kind, values in report["summary_by_mode"][mode]["per_kind"].items():
            lines.append(
                f"| {mode} | {kind} | {values['detection_tp']} / {values['truth_count']} | "
                f"{values['diagnosis']['tp']} / {values['truth_count']} |"
            )
    lines.extend(
        [
            "",
            "## Matched centerline marker localization",
            "",
            "| Mode | Matched markers | Median difference (m) | Maximum difference (m) | Evidence abstentions |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for mode in MODES:
        summary = report["summary_by_mode"][mode]
        localization = summary["localization"]
        median = localization["median_error_m"]
        maximum = localization["max_error_m"]
        median_text = "N/A" if median is None else f"{median:.6f}"
        maximum_text = "N/A" if maximum is None else f"{maximum:.6f}"
        lines.append(
            f"| {mode} | {localization['matched_with_positions']} | {median_text} | "
            f"{maximum_text} | {summary['checks']['evidence_abstain']} |"
        )
    lines.extend(
        [
            "",
            (
                "These are 3D differences between detector and evaluator centerline marker positions, "
                "conditional on entity matching. They are not survey accuracy or reconstruction "
                "accuracy. Evidence abstention is separate from geometric-check abstention: absent "
                "observations may prevent cause refinement while leaving geometry checking active."
            ),
            "",
            "## Artifacts and interpretation",
            "",
            "- [Overview](overview.png): all paired cases, with truth and alerts distinguished.",
            "- [Case table](case_results.csv) and [full metrics](report.json): machine-recomputable results.",
            "- `figures/`: one comparison figure per case, including failures and normal controls.",
            "- `inputs/`, `ground_truth/`, `predictions/`: separate candidate, evaluator, and detector records.",
            (
                "- [Timings](timings.json): measured generator, detector, render/write and total elapsed time; "
                "one cold sequential run, not a speed benchmark."
            ),
            (
                "- [Manifest](manifest.json): input/truth fingerprint, source hashes, dependency versions, "
                "and hashes of every output file except the manifest itself."
            ),
            "",
            (
                "The detector is called with candidate geometry, proposed connections and observation "
                "points only. Case IDs, group IDs, variant names and truth records are unavailable to it. "
                "Figures may display truth because visualization takes place after prediction."
            ),
            "",
            (
                "All four variants of one base layout form one group. These development cases share "
                "the generator and are not independent field sites. P2 must freeze definitions, tuning "
                "and splits before a larger test; real observation validation remains separate. The "
                "pilot does not demonstrate field reconstruction accuracy or reduced human review time."
            ),
            "",
            f"Deterministic data fingerprint (excludes timings): `{report['data_fingerprint_sha256']}`.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_track_pilot(output: str | Path, seed: int = 20260905) -> dict:
    """Create a new, self-contained run directory; existing paths are never overwritten."""
    from . import synthetic_track_audit, synthetic_track_scene

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Pilot output already exists: {destination}")
    started = time.perf_counter()
    cases = synthetic_track_scene.generate_pilot_cases(seed=seed)
    generation_seconds = time.perf_counter() - started
    if not cases:
        raise ValueError("Generator returned no cases")
    identifiers = [case["case_id"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case identifiers must be unique")
    if any(
        not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
        for identifier in identifiers
    ):
        raise ValueError("Case identifiers must be safe opaque path components")
    groups = sorted({case["layout_id"] for case in cases})
    for group in groups:
        variants = Counter(case["truth"]["variant"] for case in cases if case["layout_id"] == group)
        if variants != Counter({kind: 1 for kind in ("normal", *KINDS)}):
            raise ValueError("Every layout must have one normal and three single-fault cases")
    data_fingerprint = _digest({"seed": seed, "cases": cases})
    destination.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "railway.synthetic-track-pilot-report.v1",
        "phase": "P1_development_pilot",
        "customer_data_used": False,
        "frozen_test": False,
        "seed": seed,
        "case_count": len(cases),
        "base_layout_count": len(groups),
        "statistical_unit": "base_layout_not_variant_or_point",
        "method_comparison_scope": "shared detection rules; enhanced cause refinement only",
        "data_fingerprint_sha256": data_fingerprint,
        "matching": {
            "detection": "exact sorted entity set, one-to-one, cause ignored",
            "diagnosis": "exact sorted entity set plus cause, one-to-one",
            "extra_alarms": "unmatched alarms are FP, including duplicate alarms",
            "normal_checks": "exclude faulty entity sets; assessed denominator excludes abstentions",
            "undefined_rates": "null when denominator is zero",
        },
        "cases": [],
    }
    predictions: dict[str, dict] = {}
    runtime_rows = []
    csv_rows = []
    for case in cases:
        case_id = case["case_id"]
        _write_json(destination / "inputs" / f"{case_id}.json", case["input"])
        _write_json(
            destination / "ground_truth" / f"{case_id}.json",
            {
                **case["truth"],
                "case_id": case_id,
                "layout_id": case["layout_id"],
            },
        )
        case_result = {
            "case_id": case_id,
            "layout_id": case["layout_id"],
            "variant": case["truth"]["variant"],
            "by_mode": {},
        }
        predictions[case_id] = {}
        for mode in MODES:
            # Each detector gets the same isolated input; modifications cannot carry across modes.
            candidate = copy.deepcopy(case["input"])
            detector_started = time.perf_counter()
            prediction = synthetic_track_audit.audit_track_candidate(candidate, mode=mode)
            elapsed = time.perf_counter() - detector_started
            predictions[case_id][mode] = prediction
            _write_json(destination / "predictions" / mode / f"{case_id}.json", prediction)
            evaluation = evaluate_track_prediction(case["truth"], prediction)
            case_result["by_mode"][mode] = evaluation
            runtime_rows.append({"case_id": case_id, "mode": mode, "detector_seconds": elapsed})
            csv_rows.append(
                {
                    "case_id": case_id,
                    "layout_id": case["layout_id"],
                    "variant": case["truth"]["variant"],
                    "mode": mode,
                    **{
                        f"{metric}_{key}": evaluation[metric][key]
                        for metric in ("detection", "diagnosis")
                        for key in _COUNT_KEYS
                    },
                    "normal_case_flagged": evaluation["normal_case_flagged"],
                    "normal_assessed_checks": evaluation["checks"]["normal_assessed"],
                    "normal_flagged_checks": evaluation["checks"]["normal_flagged"],
                    "abstain_checks": evaluation["checks"]["abstain"],
                    "total_checks": evaluation["checks"]["total"],
                }
            )
        report["cases"].append(case_result)
        _render_case(case, predictions[case_id], destination / "figures" / f"{case_id}.png")
    report["summary_by_mode"] = {
        mode: _aggregate([case["by_mode"][mode] for case in report["cases"]]) for mode in MODES
    }
    report["summary_by_layout"] = {
        group: {
            mode: _aggregate(
                [case["by_mode"][mode] for case in report["cases"] if case["layout_id"] == group]
            )
            for mode in MODES
        }
        for group in groups
    }
    _write_json(destination / "report.json", report)
    with (destination / "case_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _render_overview(cases, predictions, destination / "overview.png")
    _write_report(destination, report)
    detection_seconds = sum(row["detector_seconds"] for row in runtime_rows)
    elapsed = time.perf_counter() - started
    _write_json(
        destination / "timings.json",
        {
            "measurement": "perf_counter; one sequential run; detector excludes serialization/render",
            "generation_seconds": generation_seconds,
            "detector_seconds": detection_seconds,
            "other_processing_seconds": elapsed - generation_seconds - detection_seconds,
            "elapsed_before_manifest_seconds": elapsed,
            "by_case_and_mode": runtime_rows,
            "detector_seconds_by_mode": {
                mode: sum(row["detector_seconds"] for row in runtime_rows if row["mode"] == mode)
                for mode in MODES
            },
        },
    )
    toolkit = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__),
        Path(synthetic_track_scene.__file__),
        Path(synthetic_track_audit.__file__),
        toolkit / "src" / "railway_recon" / "relationships.py",
        toolkit / "scripts" / "run_synthetic_track_pilot.py",
        toolkit / "pyproject.toml",
        toolkit / "uv.lock",
    ]
    source_hashes = {
        path.resolve().relative_to(toolkit).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
        if path.is_file()
    }
    output_files = [
        {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": "railway.synthetic-track-pilot-manifest.v1",
            "seed": seed,
            "customer_data_used": False,
            "data_fingerprint_sha256": data_fingerprint,
            "fingerprint_definition": "Canonical sorted compact UTF-8 JSON of seed and generated cases; no runtime metadata",
            "prediction_fingerprint_sha256": _digest(predictions),
            "source_sha256": source_hashes,
            "environment": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "machine": platform.machine(),
                "packages": {
                    name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow")
                },
            },
            "case_groups": [
                {"case_id": case["case_id"], "layout_id": case["layout_id"]} for case in cases
            ],
            "files": output_files,
            "manifest_self_hash_excluded": True,
        },
    )
    return report
