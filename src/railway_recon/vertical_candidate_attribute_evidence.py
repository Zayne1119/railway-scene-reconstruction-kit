from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .io import load_json, write_json

_CLASS_NAMES = {
    0: "created_never_classified",
    1: "unclassified",
    2: "ground",
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    7: "low_point_noise",
    9: "water",
    17: "bridge_deck",
}


def _rgb8(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.column_stack((red, green, blue)).astype(np.float64)
    if not np.any(rgb):
        return np.zeros((len(rgb), 3), dtype=np.float64)
    scale = 257.0 if float(np.percentile(rgb, 99)) > 255.0 else 1.0
    return np.clip(rgb / scale, 0.0, 255.0)


def analyze_vertical_candidate_attributes(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    candidate_ids: tuple[str, ...],
    output_path: str | Path,
    chunk_size: int = 2_000_000,
) -> Path:
    if not candidate_ids:
        raise ValueError("At least one candidate id is required")
    gap = load_json(Path(gap_report_path))
    requested = set(candidate_ids)
    candidates = {
        str(item["candidate_id"]): item
        for item in gap.get("vertical_candidates", [])
        if str(item.get("candidate_id")) in requested
    }
    missing = sorted(requested - set(candidates))
    if missing:
        raise ValueError(f"Unknown vertical candidate ids: {missing}")
    samples: dict[str, dict[str, list[np.ndarray]]] = {
        candidate_id: {
            "rgb": [],
            "classification": [],
            "intensity": [],
            "return_number": [],
            "number_of_returns": [],
        }
        for candidate_id in candidate_ids
    }
    with laspy.open(Path(cloud_path).resolve()) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            for candidate_id, candidate in candidates.items():
                minimum = np.asarray(candidate["minimum_xyz_m"], dtype=np.float64)
                maximum = np.asarray(candidate["maximum_xyz_m"], dtype=np.float64)
                selected = (
                    (x >= minimum[0])
                    & (x <= maximum[0])
                    & (y >= minimum[1])
                    & (y <= maximum[1])
                    & (z >= minimum[2])
                    & (z <= maximum[2])
                )
                if not np.any(selected):
                    continue
                target = samples[candidate_id]
                target["rgb"].append(
                    _rgb8(
                        np.asarray(chunk.red)[selected],
                        np.asarray(chunk.green)[selected],
                        np.asarray(chunk.blue)[selected],
                    )
                )
                target["classification"].append(
                    np.asarray(chunk.classification)[selected]
                )
                target["intensity"].append(np.asarray(chunk.intensity)[selected])
                target["return_number"].append(
                    np.asarray(chunk.return_number)[selected]
                )
                target["number_of_returns"].append(
                    np.asarray(chunk.number_of_returns)[selected]
                )
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        data = samples[candidate_id]
        if not data["rgb"]:
            raise ValueError(f"No cloud points found for {candidate_id}")
        rgb = np.vstack(data["rgb"])
        classification = np.concatenate(data["classification"]).astype(np.int64)
        intensity = np.concatenate(data["intensity"]).astype(np.float64)
        return_number = np.concatenate(data["return_number"]).astype(np.int64)
        number_of_returns = np.concatenate(data["number_of_returns"]).astype(np.int64)
        total = np.sum(rgb, axis=1)
        excess_green = (2.0 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(
            total, 1.0
        )
        maximum_channel = np.max(rgb, axis=1)
        saturation = (maximum_channel - np.min(rgb, axis=1)) / np.maximum(
            maximum_channel, 1.0
        )
        class_counts = Counter(int(value) for value in classification)
        vegetation_fraction = sum(class_counts.get(code, 0) for code in (3, 4, 5)) / len(
            classification
        )
        green_dominant_fraction = float(
            np.mean(
                (rgb[:, 1] > rgb[:, 0] * 1.05)
                & (rgb[:, 1] > rgb[:, 2] * 1.05)
            )
        )
        if vegetation_fraction >= 0.50:
            interpretation = "classified_vegetation_no_build"
        elif float(np.median(excess_green)) >= 0.08 and green_dominant_fraction >= 0.45:
            interpretation = "rgb_vegetation_like_review_no_build"
        else:
            interpretation = "non_vegetation_or_unclassified_semantic_review"
        records.append(
            {
                "candidate_id": candidate_id,
                "full_resolution_point_count": len(rgb),
                "classification_counts": {
                    f"{code}:{_CLASS_NAMES.get(code, 'other')}": count
                    for code, count in sorted(class_counts.items())
                },
                "vegetation_class_fraction": vegetation_fraction,
                "median_rgb_8bit": [float(value) for value in np.median(rgb, axis=0)],
                "green_dominant_fraction": green_dominant_fraction,
                "median_normalized_excess_green": float(np.median(excess_green)),
                "median_rgb_saturation": float(np.median(saturation)),
                "intensity_p50": float(np.percentile(intensity, 50)),
                "intensity_p90": float(np.percentile(intensity, 90)),
                "single_return_fraction": float(
                    np.mean((return_number == 1) & (number_of_returns == 1))
                ),
                "interpretation": interpretation,
            }
        )
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.vertical-candidate-attribute-evidence.v1",
            "cloud": str(Path(cloud_path).resolve()),
            "gap_report": str(Path(gap_report_path).resolve()),
            "candidate_count": len(records),
            "records": records,
            "warning": (
                "LAS classes and RGB are supporting evidence only; fixed assets still require "
                "geometric and photo review."
            ),
        },
    )
    return destination
