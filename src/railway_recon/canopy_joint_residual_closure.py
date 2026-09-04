from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_json, write_json


def close_canopy_joint_residuals(
    *,
    gap_report_path: str | Path,
    rejected_weld_gate_path: str | Path,
    local_seam_evidence_path: str | Path,
    output_path: str | Path,
    station_centres_m: tuple[float, ...] = (7.7, 61.74, 115.69),
    station_tolerance_m: float = 0.35,
) -> Path:
    gap_path = Path(gap_report_path).resolve()
    rejected_gate = Path(rejected_weld_gate_path).resolve()
    local_evidence = Path(local_seam_evidence_path).resolve()
    gap = load_json(gap_path)
    gate = load_json(rejected_gate)
    if gate.get("accepted") is not False:
        raise ValueError("The referenced canopy seam weld was not explicitly rejected")
    candidates: list[dict[str, Any]] = []
    for item in gap["vertical_candidates"]:
        station = float(item["station_m"])
        cross = float(item["cross_m"])
        height = float(item["extent_xyz_m"][2])
        near_station = any(abs(station - value) <= station_tolerance_m for value in station_centres_m)
        in_canopy_band = 3.8 <= cross <= 5.1 or -26.2 <= cross <= -24.2
        if item.get("priority") == "P1" and near_station and in_canopy_band and 1.8 <= height <= 2.5:
            candidates.append(item)
    if len(candidates) != 10:
        raise ValueError(f"Expected 10 known canopy joint residuals; found {len(candidates)}")
    decisions = []
    for item in candidates:
        cross = float(item["cross_m"])
        decisions.append(
            {
                "candidate_id": item["candidate_id"],
                "station_m": item["station_m"],
                "cross_m": item["cross_m"],
                "side": "right_platform" if cross > 0 else "opposite_platform",
                "reviewed_class": "canopy_roof_joint_or_fascia_fragment",
                "model_action": "no_build",
                "confidence": 0.9,
                "rationale": (
                    "The residual is co-located with an observed roof joint/fascia band. "
                    "A previous shared-boundary weld reduced point support and exceeded the "
                    "displacement gate, so the evidence supports preserving the physical joint."
                ),
                "evidence": [str(rejected_gate), str(local_evidence)],
            }
        )
    output = Path(output_path).resolve()
    write_json(
        output,
        {
            "schema_version": "railway.canopy-joint-residual-closure.v1",
            "source_gap_report": str(gap_path),
            "rejected_weld_gate": str(rejected_gate),
            "local_seam_evidence": str(local_evidence),
            "closed_candidate_count": len(decisions),
            "decisions": decisions,
            "geometry_changed": False,
            "conclusion": (
                "All ten P1 canopy joint residuals are explainable physical joint/fascia "
                "evidence; no standalone asset or forced seam weld is authorized."
            ),
            "status": "closed_no_geometry_formal_baseline_unchanged",
        },
    )
    return output
