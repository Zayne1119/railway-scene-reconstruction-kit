from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .algorithms.rail_candidates import detect_rail_candidates
from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .track_graph import build_track_graph

PROTECTED_CHECKS = {
    "track_graph_structure",
    "topology_degree",
    "canonical_direction",
    "track_identity_order",
    "duplicate_tracks",
}

RECOVERY_CHECKS = {
    "source_rail_pair_presence",
    "rail_gauge",
    "rail_top_crosslevel",
    "rail_top_crosslevel_consistency",
    "observation_support",
    "segment_seams",
    "track_observation_coverage",
    "interior_track_termination",
}


def _audit_failure_counts(audit: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["id"]): int(item.get("failure_count", 0))
        for item in audit.get("checks", [])
    }


def _trial_is_pareto_improvement(
    baseline_graph: dict[str, Any],
    baseline_audit: dict[str, Any],
    trial_graph: dict[str, Any],
    trial_audit: dict[str, Any],
) -> tuple[bool, list[str], dict[str, int]]:
    """Accept a recovery only when topology is stable and no tracked QA regresses."""

    reasons: list[str] = []
    baseline_track_count = len(baseline_graph.get("tracks", []))
    trial_track_count = len(trial_graph.get("tracks", []))
    if trial_track_count != baseline_track_count:
        reasons.append(
            f"global_track_count_changed:{baseline_track_count}->{trial_track_count}"
        )

    baseline_counts = _audit_failure_counts(baseline_audit)
    trial_counts = _audit_failure_counts(trial_audit)
    deltas = {
        check: trial_counts.get(check, 0) - baseline_counts.get(check, 0)
        for check in sorted(PROTECTED_CHECKS | RECOVERY_CHECKS)
    }
    for check in sorted(PROTECTED_CHECKS):
        if deltas[check] > 0:
            reasons.append(f"protected_check_regressed:{check}:{deltas[check]:+d}")
    for check in sorted(RECOVERY_CHECKS):
        if deltas[check] > 0:
            reasons.append(f"quality_check_regressed:{check}:{deltas[check]:+d}")
    if not any(deltas[check] < 0 for check in RECOVERY_CHECKS):
        reasons.append("no_recovery_check_improved")
    return not reasons, reasons, deltas


def _candidate_segment_ids(
    graph: dict[str, Any], segment_manifest: dict[str, Any]
) -> list[str]:
    """Find interior no-pair and graph-gap segments for a conservative second pass."""

    segments = list(segment_manifest.get("segments", []))
    by_id = {str(item["id"]): item for item in segments}
    if not segments or not graph.get("tracks"):
        return []
    authoritative_start = min(
        float(item["chainage_start_m"]) for item in graph["tracks"]
    )
    authoritative_end = max(float(item["chainage_end_m"]) for item in graph["tracks"])
    selected: set[str] = set()
    for source in graph.get("sources", []):
        segment = by_id.get(str(source["segment_id"]))
        if segment is None or int(source.get("rail_pair_count", 0)) > 0:
            continue
        start = float(segment["chainage_start_m"])
        end = float(segment["chainage_end_m"])
        if start >= authoritative_start and end <= authoritative_end:
            selected.add(str(segment["id"]))

    observations = {
        str(item["id"]): item for item in graph.get("observations", [])
    }
    for seam in graph.get("seams", []):
        left = observations.get(str(seam.get("from_observation_id")))
        right = observations.get(str(seam.get("to_observation_id")))
        gap_start = float(
            seam.get(
                "from_chainage_end_m",
                0.0 if left is None else left["chainage_end_m"],
            )
        )
        gap_end = float(
            seam.get(
                "to_chainage_start_m",
                0.0 if right is None else right["chainage_start_m"],
            )
        )
        if gap_end - gap_start <= 1e-6:
            continue
        for segment in segments:
            start = float(segment["chainage_start_m"])
            end = float(segment["chainage_end_m"])
            if start >= gap_start - 1e-6 and end <= gap_end + 1e-6:
                selected.add(str(segment["id"]))
    return sorted(selected, key=lambda value: float(by_id[value]["chainage_start_m"]))


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def select_corridor_rail_recoveries(
    project: ProjectConfig,
    baseline_graph_value: str | Path,
    baseline_audit_value: str | Path,
    recovery_settings_value: str | Path,
    *,
    output_name: str = "rail_recovery_selection_v1",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a fail-closed, greedy A/B recovery over all interior rail gaps.

    Recovery reports and trial graphs are versioned separately.  A trial becomes the
    new reference only if global track count is unchanged, protected topology checks
    do not regress, every monitored quality check is non-worse, and at least one
    monitored check improves.
    """

    baseline_graph_path = Path(baseline_graph_value).resolve()
    baseline_audit_path = Path(baseline_audit_value).resolve()
    recovery_settings_path = Path(recovery_settings_value).resolve()
    for path in (baseline_graph_path, baseline_audit_path, recovery_settings_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_graph = load_json(baseline_graph_path)
    baseline_audit = load_json(baseline_audit_path)
    if baseline_graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError("Rail recovery selection requires TrackGraph v1")
    if baseline_audit.get("graph_sha256") != sha256_file(baseline_graph_path):
        raise ValueError("Baseline TrackGraph and audit hashes do not match")

    output_root = project.workspace_path("reports") / _safe_tag(output_name)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite rail recovery output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = output_root / "candidates"
    trial_root = output_root / "trials"
    candidate_root.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json(project.workspace_path("segment_manifest"))
    candidate_ids = _candidate_segment_ids(baseline_graph, manifest)
    current_graph = baseline_graph
    current_audit = baseline_audit
    current_sources = {
        str(item["segment_id"]): str(item["path"])
        for item in baseline_graph.get("sources", [])
    }
    decisions: list[dict[str, Any]] = []
    for segment_id in candidate_ids:
        candidate_report = candidate_root / f"{segment_id}_rail_candidates.json"
        detect_rail_candidates(
            project,
            segment_id,
            overwrite=overwrite,
            settings_value=recovery_settings_path,
            report_value=candidate_report,
        )
        baseline_source = load_json(Path(current_sources[segment_id]))
        candidate = load_json(candidate_report)
        if int(candidate.get("rail_pair_count", 0)) <= int(
            baseline_source.get("rail_pair_count", 0)
        ):
            decisions.append(
                {
                    "segment_id": segment_id,
                    "decision": "rejected",
                    "reasons": ["candidate_did_not_add_a_rail_pair"],
                    "baseline_pair_count": int(baseline_source.get("rail_pair_count", 0)),
                    "candidate_pair_count": int(candidate.get("rail_pair_count", 0)),
                    "candidate_report": str(candidate_report),
                }
            )
            continue

        trial_sources = dict(current_sources)
        trial_sources[segment_id] = str(candidate_report)
        tag = _safe_tag(segment_id)
        trial_graph_path = trial_root / f"{tag}.track_graph.json"
        trial_audit_path = trial_root / f"{tag}.track_graph_audit.json"
        build_track_graph(
            project,
            sorted(trial_sources.items()),
            overwrite=True,
            output_value=trial_graph_path,
            audit_value=trial_audit_path,
        )
        trial_graph = load_json(trial_graph_path)
        trial_audit = load_json(trial_audit_path)
        accepted, reasons, deltas = _trial_is_pareto_improvement(
            current_graph, current_audit, trial_graph, trial_audit
        )
        decisions.append(
            {
                "segment_id": segment_id,
                "decision": "accepted" if accepted else "rejected",
                "reasons": reasons,
                "baseline_pair_count": int(baseline_source.get("rail_pair_count", 0)),
                "candidate_pair_count": int(candidate.get("rail_pair_count", 0)),
                "qa_failure_deltas": deltas,
                "candidate_report": str(candidate_report),
                "trial_graph": str(trial_graph_path),
                "trial_audit": str(trial_audit_path),
            }
        )
        if accepted:
            current_sources = trial_sources
            current_graph = trial_graph
            current_audit = trial_audit

    final_graph_path = project.workspace_path("derived") / f"{_safe_tag(output_name)}.json"
    final_audit_path = output_root / "final_track_graph_audit.json"
    write_json(final_graph_path, current_graph)
    final_audit = dict(current_audit)
    final_audit["graph_path"] = str(final_graph_path)
    final_audit["graph_sha256"] = sha256_file(final_graph_path)
    write_json(final_audit_path, final_audit)

    accepted_ids = [
        item["segment_id"] for item in decisions if item["decision"] == "accepted"
    ]
    report = {
        "schema_version": "railway.rail-recovery-selection.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "status": "candidate",
        "baseline_graph": str(baseline_graph_path),
        "baseline_graph_sha256": sha256_file(baseline_graph_path),
        "baseline_audit": str(baseline_audit_path),
        "recovery_settings": str(recovery_settings_path),
        "recovery_settings_sha256": sha256_file(recovery_settings_path),
        "candidate_segment_count": len(candidate_ids),
        "accepted_segment_count": len(accepted_ids),
        "accepted_segment_ids": accepted_ids,
        "rejected_segment_count": len(decisions) - len(accepted_ids),
        "decisions": decisions,
        "selected_sources": [
            {"segment_id": segment_id, "path": path}
            for segment_id, path in sorted(current_sources.items())
        ],
        "final_graph": str(final_graph_path),
        "final_graph_sha256": sha256_file(final_graph_path),
        "final_audit": str(final_audit_path),
        "policy": (
            "Greedy fail-closed A/B selection: stable global track count, no protected "
            "or monitored QA regression, and at least one monitored QA improvement."
        ),
    }
    report_path = output_root / "selection_report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path)}
