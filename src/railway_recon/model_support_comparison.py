from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json


def compare_support_values(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_objects = {str(item["object_name"]): item for item in before["objects"]}
    after_objects = {str(item["object_name"]): item for item in after["objects"]}
    common = sorted(set(before_objects) & set(after_objects))
    removed = sorted(set(before_objects) - set(after_objects))
    added = sorted(set(after_objects) - set(before_objects))
    changes = []
    for name in common:
        old = before_objects[name]
        new = after_objects[name]
        changes.append(
            {
                "object_name": name,
                "asset_type": new.get("asset_type"),
                "before_p90_m": float(old["p90_m"]),
                "after_p90_m": float(new["p90_m"]),
                "p90_change_m": float(new["p90_m"]) - float(old["p90_m"]),
                "before_coverage_at_0_10m": float(old["coverage_at_0_10m"]),
                "after_coverage_at_0_10m": float(new["coverage_at_0_10m"]),
                "coverage_at_0_10m_change": float(new["coverage_at_0_10m"])
                - float(old["coverage_at_0_10m"]),
            }
        )
    platform = [item for item in changes if item["asset_type"] == "platform_surface"]
    common_before_p90 = np.asarray(
        [float(before_objects[name]["p90_m"]) for name in common], dtype=np.float64
    )
    common_after_p90 = np.asarray(
        [float(after_objects[name]["p90_m"]) for name in common], dtype=np.float64
    )
    return {
        "common_object_count": len(common),
        "removed_objects": removed,
        "added_objects": added,
        "common_object_p90_median": {
            "before": float(np.median(common_before_p90)),
            "after": float(np.median(common_after_p90)),
        },
        "opposite_platform_surface": platform,
        "changes": changes,
    }


def compare_support_reports(
    before_path: str | Path, after_path: str | Path, output_path: str | Path
) -> Path:
    before_source = Path(before_path).resolve()
    after_source = Path(after_path).resolve()
    comparison = compare_support_values(load_json(before_source), load_json(after_source))
    report = {
        "schema_version": "railway.model-support-comparison.v1",
        "before": str(before_source),
        "after": str(after_source),
        **comparison,
    }
    destination = Path(output_path).resolve()
    write_json(destination, report)
    return destination
