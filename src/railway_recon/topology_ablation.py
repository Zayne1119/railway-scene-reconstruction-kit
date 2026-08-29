from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, sha256_file, write_json


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50_m": None, "p90_m": None, "maximum_m": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50_m": float(np.percentile(array, 50)),
        "p90_m": float(np.percentile(array, 90)),
        "maximum_m": float(np.max(array)),
    }


def evaluate_topology_ablation(
    graph_path: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Quantify what a segment-only assembly cannot check without TrackGraph.

    This is deliberately an observability ablation.  It does not pretend that a
    failing TrackGraph has repaired the geometry; it records identities and seam
    defects that the segment-only variant leaves unmeasured.
    """

    graph_source = Path(graph_path).resolve()
    audit_source = Path(audit_path).resolve()
    graph = load_json(graph_source)
    audit = load_json(audit_source)
    if graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError("Unsupported TrackGraph")
    if audit.get("schema_version") != "railway.track-graph-audit.v1":
        raise ValueError("Unsupported TrackGraph audit")
    if graph.get("project_id") != audit.get("project_id"):
        raise ValueError("TrackGraph and audit project ids differ")

    observations = list(graph.get("observations", []))
    tracks = list(graph.get("tracks", []))
    seams = list(graph.get("seams", []))
    events = list(graph.get("identity_events", []))
    local_ids = [str(item.get("local_track_id")) for item in observations]
    unique_local_ids = set(local_ids)
    local_id_collision_count = len(local_ids) - len(unique_local_ids)
    identity_links = sum(int(item.get("match_count", 0)) for item in events)
    unmatched_births = sum(
        max(0, int(item.get("current_count", 0)) - int(item.get("match_count", 0)))
        for item in events
    )
    unmatched_deaths = sum(
        max(0, int(item.get("previous_count", 0)) - int(item.get("match_count", 0)))
        for item in events
    )
    check_by_id = {str(item["id"]): item for item in audit.get("checks", [])}
    seam_check = check_by_id.get("segment_seams", {})
    termination_check = check_by_id.get("interior_track_termination", {})

    report = {
        "schema_version": "railway.track-topology-ablation.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "annotation_free_topology_observability_ablation",
        "inputs": {
            "track_graph": str(graph_source),
            "track_graph_sha256": sha256_file(graph_source),
            "track_graph_audit": str(audit_source),
            "track_graph_audit_sha256": sha256_file(audit_source),
        },
        "segment_only_variant": {
            "local_fragment_count": len(observations),
            "segment_count": len(graph.get("sources", [])),
            "local_track_id_collision_count_if_not_namespaced": local_id_collision_count,
            "cross_segment_identity_links_available": 0,
            "cross_segment_seams_measured": 0,
            "interior_births_or_deaths_checked": 0,
            "status": "not_evaluated_across_segments",
        },
        "track_graph_variant": {
            "global_track_count": len(tracks),
            "observation_fragment_count": len(observations),
            "cross_segment_identity_link_count": identity_links,
            "seam_count": len(seams),
            "unmatched_birth_count": unmatched_births,
            "unmatched_death_count": unmatched_deaths,
            "seam_failure_count": int(seam_check.get("failure_count", 0)),
            "interior_termination_failure_count": int(
                termination_check.get("failure_count", 0)
            ),
            "lateral_seam_difference": _stats(
                [float(item["lateral_difference_m"]) for item in seams]
            ),
            "vertical_seam_difference": _stats(
                [float(item["vertical_difference_m"]) for item in seams]
            ),
            "endpoint_3d_seam_difference": _stats(
                [float(item["endpoint_3d_difference_m"]) for item in seams]
            ),
            "audit_status": audit.get("status"),
        },
        "delta": {
            "fragment_identities_consolidated": len(observations) - len(tracks),
            "new_cross_segment_seams_observed": len(seams),
            "new_topology_events_observed": unmatched_births + unmatched_deaths,
            "release_blocked_by_detected_defects": audit.get("status") != "pass",
        },
        "interpretation": (
            "TrackGraph exposes cross-segment identity and seam defects that the local-only "
            "variant cannot measure. The current graph does not pass, so this report proves "
            "defect observability and fail-closed behavior, not geometric improvement."
        ),
        "limitations": [
            "No semantic labels or survey control are used.",
            "Fragment consolidation is an identity count, not a claim that every match is true.",
            "A geometric improvement claim requires a passing corrected graph and held-out re-evaluation.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite topology ablation report: {output}")
    write_json(output, report)
    return output
