from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json


def _nearest_baseline_id(
    baseline: list[dict[str, Any]],
    *,
    station_m: float,
    cross_m: float,
    tolerance_m: float,
) -> str | None:
    if not baseline:
        return None
    distances = np.asarray(
        [
            np.hypot(
                float(item["station_m"]) - station_m,
                float(item["cross_m"]) - cross_m,
            )
            for item in baseline
        ],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    return str(baseline[index]["candidate_id"]) if distances[index] <= tolerance_m else None


def build_vertical_gap_disposition_ledger(
    *,
    baseline_gap_report_path: str | Path,
    current_gap_report_path: str | Path,
    candidate_gate_paths: tuple[str | Path, ...],
    closure_report_paths: tuple[str | Path, ...],
    output_path: str | Path,
    position_tolerance_m: float = 0.40,
) -> Path:
    """Reconcile changing gap IDs into an evidence-backed P1 disposition ledger."""
    baseline_report = load_json(Path(baseline_gap_report_path))
    current_report = load_json(Path(current_gap_report_path))
    baseline = [
        item for item in baseline_report["vertical_candidates"] if item["priority"] == "P1"
    ]
    baseline_by_id = {str(item["candidate_id"]): item for item in baseline}
    dispositions: dict[str, dict[str, Any]] = {}

    def assign(
        candidate_id: str,
        *,
        status: str,
        source: Path,
        rationale: str,
        precedence: int,
    ) -> None:
        if candidate_id not in baseline_by_id:
            return
        previous = dispositions.get(candidate_id)
        if previous is not None and int(previous["precedence"]) >= precedence:
            return
        dispositions[candidate_id] = {
            "status": status,
            "source": str(source.resolve()),
            "rationale": rationale,
            "precedence": precedence,
        }

    for raw_path in candidate_gate_paths:
        path = Path(raw_path)
        gate = load_json(path)
        if not bool(gate.get("passed")):
            continue
        target = gate.get("target", {})
        for candidate_id in target.get("baseline_candidate_ids", []):
            assign(
                str(candidate_id),
                status="built_candidate_gate_passed",
                source=path,
                rationale="Target disappeared after an accepted candidate build without regression.",
                precedence=30,
            )

    for raw_path in closure_report_paths:
        path = Path(raw_path)
        closure = load_json(path)
        records = closure.get("records", closure.get("decisions", []))
        for record in records:
            candidate_id: str | None = None
            if "station_m" in record and "cross_m" in record:
                candidate_id = _nearest_baseline_id(
                    baseline,
                    station_m=float(record["station_m"]),
                    cross_m=float(record["cross_m"]),
                    tolerance_m=position_tolerance_m,
                )
            if candidate_id is None:
                direct = str(record.get("candidate_id", ""))
                candidate_id = direct if direct in baseline_by_id else None
            if candidate_id is None:
                continue
            semantic = str(
                record.get(
                    "semantic_class",
                    record.get(
                        "reviewed_class",
                        closure.get("semantic_decision", {}).get("class", "reviewed_residual"),
                    ),
                )
            )
            assign(
                candidate_id,
                status="closed_no_geometry",
                source=path,
                rationale=semantic,
                precedence=20,
            )

    for current in current_report["vertical_candidates"]:
        if current.get("semantic_review_hint") != (
            "existing_station_entry_surface_edge_or_residual"
        ):
            continue
        candidate_id = _nearest_baseline_id(
            baseline,
            station_m=float(current["station_m"]),
            cross_m=float(current["cross_m"]),
            tolerance_m=position_tolerance_m,
        )
        if candidate_id is not None:
            assign(
                candidate_id,
                status="existing_group_refinement_not_new_asset",
                source=Path(current_gap_report_path),
                rationale=str(current.get("automatic_geometry_action", "refine_existing_group")),
                precedence=10,
            )

    records: list[dict[str, Any]] = []
    for item in baseline:
        candidate_id = str(item["candidate_id"])
        disposition = dispositions.get(
            candidate_id,
            {
                "status": "unresolved",
                "source": None,
                "rationale": "No accepted build, closure or existing-group relation was found.",
                "precedence": 0,
            },
        )
        records.append(
            {
                "baseline_candidate_id": candidate_id,
                "station_m": float(item["station_m"]),
                "cross_m": float(item["cross_m"]),
                "height_m": float(item["extent_xyz_m"][2]),
                "disposition": disposition["status"],
                "rationale": disposition["rationale"],
                "evidence": disposition["source"],
            }
        )

    counts: dict[str, int] = {}
    for record in records:
        key = str(record["disposition"])
        counts[key] = counts.get(key, 0) + 1
    unresolved = counts.get("unresolved", 0)
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.vertical-gap-disposition-ledger.v1",
            "inputs": {
                "baseline_gap_report": str(Path(baseline_gap_report_path).resolve()),
                "current_gap_report": str(Path(current_gap_report_path).resolve()),
                "candidate_gates": [str(Path(path).resolve()) for path in candidate_gate_paths],
                "closure_reports": [str(Path(path).resolve()) for path in closure_report_paths],
            },
            "matching": {
                "basis": "station_cross_position",
                "position_tolerance_m": position_tolerance_m,
                "warning": "Gap candidate IDs can change when accepted geometry removes earlier candidates.",
            },
            "summary": {
                "baseline_p1_count": len(baseline),
                "disposition_counts": counts,
                "unresolved_p1_count": unresolved,
                "all_baseline_p1_disposed": unresolved == 0,
                "formal_baseline_changed": False,
            },
            "records": records,
            "status": (
                "all_baseline_p1_candidates_disposed_candidate_chain_not_promoted"
                if unresolved == 0
                else "unresolved_baseline_p1_candidates_remain"
            ),
        },
    )
    return destination
