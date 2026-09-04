from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .camera import load_camera_rows
from .io import load_json, write_json
from .vertical_conflict_photo_evidence import (
    _render_candidate_sheet,
    _render_summary,
    _resource_settings,
    analyze_vertical_conflict_photo_evidence_data,
    build_vertical_conflict_photo_graph,
)


def gap_candidate_to_vertical_hypothesis(candidate: dict[str, Any]) -> dict[str, Any]:
    extent = [float(value) for value in candidate["extent_xyz_m"]]
    minimum = [float(value) for value in candidate["minimum_xyz_m"]]
    maximum = [float(value) for value in candidate["maximum_xyz_m"]]
    return {
        "id": str(candidate["candidate_id"]),
        "longitudinal_position_m": float(candidate["station_m"]),
        "cross_position_m": float(candidate["cross_m"]),
        "minimum_z": minimum[2],
        "maximum_z": maximum[2],
        "footprint_m": max(extent[0], extent[1]),
        "predicted_class": str(candidate["classification"]),
        "confidence": {"P0": 0.9, "P1": 0.65, "P2": 0.4, "P3": 0.2}.get(
            str(candidate.get("priority")), 0.2
        ),
    }


def render_gap_candidate_photo_evidence(
    *,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    projection_consensus_path: str | Path,
    candidate_ids: list[str],
    output_directory: str | Path,
    settings_path: str | Path | None = None,
) -> dict[str, Path]:
    if not candidate_ids:
        raise ValueError("At least one gap candidate ID is required")
    gap = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    consensus = load_json(Path(projection_consensus_path))
    candidates_by_id = {
        str(item["candidate_id"]): item for item in gap["vertical_candidates"]
    }
    selected_ids = list(dict.fromkeys(str(value) for value in candidate_ids))
    missing = sorted(set(selected_ids) - set(candidates_by_id))
    if missing:
        raise ValueError(f"Gap candidates are absent: {missing}")
    vertical = {
        "frame": frame,
        "candidates": [
            gap_candidate_to_vertical_hypothesis(candidates_by_id[candidate_id])
            for candidate_id in selected_ids
        ],
    }
    interface = {
        "project_id": gap.get("project_id", "gap-audit"),
        "segment_id": "cloud-model-gap-review",
        "vertical_interfaces": [
            {"candidate_id": candidate_id, "semantic_conflict": True}
            for candidate_id in selected_ids
        ],
    }
    settings = load_json(Path(settings_path)) if settings_path else _resource_settings()
    cache: dict[Path, Image.Image] = {}

    def load_image(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path]

    report, crops = analyze_vertical_conflict_photo_evidence_data(
        interface,
        vertical,
        consensus,
        load_camera_rows(Path(consensus["camera_csv"])),
        settings,
        load_image,
    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for candidate in report["candidates"]:
        sheet = output / f"{candidate['candidate_id']}.jpg"
        _render_candidate_sheet(candidate, crops[candidate["candidate_id"]], sheet, settings)
        candidate["evidence_sheet"] = str(sheet)
    report_path = output / "gap_candidate_photo_evidence.json"
    graph_path = output / "gap_candidate_photo_evidence_graph.json"
    summary_path = output / "gap_candidate_photo_evidence.png"
    report.update(
        {
            "selection_mode": "explicit_cloud_model_gap_candidate_ids",
            "selected_candidate_ids": selected_ids,
            "gap_report": str(Path(gap_report_path).resolve()),
            "projection_consensus_report": str(
                Path(projection_consensus_path).resolve()
            ),
            "limitations": [
                *report.get("limitations", []),
                "Photo windows are semantic evidence only and never trigger automatic geometry changes.",
            ],
        }
    )
    write_json(report_path, report)
    write_json(graph_path, build_vertical_conflict_photo_graph(report))
    _render_summary(report, summary_path)
    for image in cache.values():
        image.close()
    return {"report": report_path, "graph": graph_path, "summary": summary_path}
