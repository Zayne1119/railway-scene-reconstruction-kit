"""Execute the separate, non-held-out adaptive observation stress probes."""
from __future__ import annotations

import copy
import csv
import hashlib
from dataclasses import asdict
from pathlib import Path

from .synthetic_track_adaptive_audit import ADAPTIVE_MODES, audit_adaptive_candidate
from .synthetic_track_adaptive_stress import generate_adaptive_stress_cases
from .synthetic_track_adaptive_study import (
    aggregate_adaptive_evaluations,
    evaluate_adaptive_prediction,
    render_adaptive_case,
)
from .synthetic_track_pilot import _canonical_bytes, _write_json
from .synthetic_track_study_audit import StudyAuditPolicy


def run_adaptive_stress(output: str | Path, seed: int = 20260907) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cases = generate_adaptive_stress_cases(seed)
    policy = asdict(StudyAuditPolicy())
    evaluations = {mode: [] for mode in ADAPTIVE_MODES}
    conditions: dict[str, dict] = {}
    rows, failures = [], []
    data_hash = hashlib.sha256()
    prediction_hash = hashlib.sha256()
    rendered = set()
    try:
        for case in sorted(cases, key=lambda item: item["case_id"]):
            identifier, condition = case["case_id"], case["truth"]["condition"]
            data_hash.update(_canonical_bytes(case))
            _write_json(destination / "inputs" / f"{identifier}.json", case["input"])
            _write_json(destination / "ground_truth" / f"{identifier}.json", {
                **case["truth"], "case_id": identifier, "layout_id": case["layout_id"]
            })
            conditions.setdefault(condition, {mode: [] for mode in ADAPTIVE_MODES})
            predictions, scored = {}, {}
            for mode in ADAPTIVE_MODES:
                prediction = audit_adaptive_candidate(copy.deepcopy(case["input"]), mode, policy)
                score = evaluate_adaptive_prediction(case["truth"], prediction)
                predictions[mode], scored[mode] = prediction, score
                prediction_hash.update(_canonical_bytes({"case_id": identifier, "mode": mode,
                                                         "prediction": prediction}))
                _write_json(destination / "predictions" / mode / f"{identifier}.json", prediction)
                evaluations[mode].append(score)
                conditions[condition][mode].append(score)
                rows.append({"case_id": identifier, "layout_id": case["layout_id"],
                             "condition": condition, "mode": mode,
                             **{f"{metric}_{key}": score[metric][key]
                                for metric in ("detection", "diagnosis") for key in ("tp", "fp", "fn")}})
                if score["diagnosis"]["fn"] or score["diagnosis"]["fp"]:
                    failures.append({"case_id": identifier, "condition": condition,
                                     "mode": mode, "evaluation": score})
            if condition not in rendered:
                render_adaptive_case(case, predictions, scored,
                                     destination / "figures" / f"{condition}.png")
                rendered.add(condition)
        summary = {mode: aggregate_adaptive_evaluations(values) for mode, values in evaluations.items()}
        by_condition = {condition: {mode: aggregate_adaptive_evaluations(values)
                                    for mode, values in methods.items()}
                        for condition, methods in sorted(conditions.items())}
        report = {
            "schema_version": "railway.synthetic-track-adaptive-stress-report.v1",
            "seed": seed, "policy": policy, "case_count": len(cases),
            "base_layout_count": len({case["layout_id"] for case in cases}),
            "role": "constructed_adversarial_development_probes_not_held_out_test",
            "customer_data_used": False, "registered_test_generated": False,
            "summary_by_mode": summary, "by_condition": by_condition,
            "data_fingerprint_sha256": data_hash.hexdigest(),
            "prediction_fingerprint_sha256": prediction_hash.hexdigest(),
        }
        _write_json(destination / "report.json", report)
        _write_json(destination / "failure_cases.json", failures)
        with (destination / "case_results.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# Adaptive diagnosis: adversarial development probes", "",
            f"{len(cases)} constructed cases in {report['base_layout_count']} layout groups. No customer data; not a held-out test or field benchmark.",
            "",
            "| Condition | Method | Diagnosis TP / FP / FN |",
            "| --- | --- | --- |",
        ]
        for condition, methods in by_condition.items():
            for mode in ADAPTIVE_MODES:
                counts = methods[mode]["diagnosis"]
                lines.append(f"| {condition} | {mode} | {counts['tp']} / {counts['fp']} / {counts['fn']} |")
        lines.extend([
            "", "All 18 known defects remain in the denominator. Missing observations do not change the injected truth. Clutter is deliberately false support along a wrong proposed connection.",
            "", "An `evidence_verified` finding only passed internal support rules. The cause can still be wrong, especially with bridge clutter. Fallback findings are not promoted to observation verification.",
            "", "These cases use negative derived seeds outside the registered nonnegative split domain. Their results are separate from the original development/validation study; no registered test case is generated.",
            "", "Figures show the first case by opaque case ID in each condition, not outcome-selected cases. Full predictions and all failures are retained.",
        ])
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        root = Path(__file__).resolve().parents[2]
        sources = [*sorted((root / "src" / "railway_recon").rglob("*.py")),
                   root / "scripts" / "run_synthetic_track_adaptive_stress.py",
                   root / "pyproject.toml", root / "uv.lock"]
        _write_json(destination / "manifest.json", {
            "schema_version": "railway.synthetic-track-adaptive-stress-manifest.v1",
            "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "files": [{"path": path.relative_to(destination).as_posix(),
                       "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(destination.rglob("*")) if path.is_file()],
            "manifest_self_hash_excluded": True,
        })
        return report
    except Exception as error:
        _write_json(destination / "run_failure.json", {"status": "failed", "exception_type": type(error).__name__})
        raise
