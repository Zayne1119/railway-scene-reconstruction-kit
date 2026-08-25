from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = ("index", "timestamp", "file", "x", "y", "z")


def load_camera_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fields]
        if missing:
            raise ValueError(f"Camera CSV is missing columns {missing}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No camera records found in {path}")
    return rows


def camera_trajectory(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cumulative = 0.0
    previous: tuple[float, float, float] | None = None
    for row_number, row in enumerate(rows, start=2):
        try:
            xyz = (float(row["x"]), float(row["y"]), float(row["z"]))
            index = int(row["index"])
            timestamp = float(row["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric camera value at CSV row {row_number}") from exc
        if previous is not None:
            cumulative += math.dist(previous, xyz)
        result.append(
            {
                "index": index,
                "timestamp": timestamp,
                "file": row["file"],
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "rot_x": _optional_float(row.get("rot_x")),
                "rot_y": _optional_float(row.get("rot_y")),
                "rot_z": _optional_float(row.get("rot_z")),
                "distance_m": cumulative,
            }
        )
        previous = xyz
    return result


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def audit_camera_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    trajectory = camera_trajectory(rows)
    indexes = [item["index"] for item in trajectory]
    timestamps = [item["timestamp"] for item in trajectory]
    files = [item["file"] for item in trajectory]
    return {
        "row_count": len(rows),
        "trajectory_length_m": trajectory[-1]["distance_m"],
        "indexes_strictly_increasing": all(b > a for a, b in zip(indexes, indexes[1:])),
        "timestamps_strictly_increasing": all(
            b > a for a, b in zip(timestamps, timestamps[1:])
        ),
        "duplicate_index_count": len(indexes) - len(set(indexes)),
        "duplicate_file_count": len(files) - len(set(files)),
        "rotation_complete_count": sum(
            all(item[key] is not None for key in ("rot_x", "rot_y", "rot_z"))
            for item in trajectory
        ),
        "first_index": indexes[0],
        "last_index": indexes[-1],
    }

