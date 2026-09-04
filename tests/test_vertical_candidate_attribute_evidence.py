from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from railway_recon.vertical_candidate_attribute_evidence import (
    analyze_vertical_candidate_attributes,
)


def test_analyze_vertical_candidate_attributes_flags_classified_vegetation(
    tmp_path: Path,
) -> None:
    cloud = tmp_path / "cloud.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.asarray([0.0, 0.1, 0.2])
    las.y = np.asarray([0.0, 0.1, 0.2])
    las.z = np.asarray([0.0, 1.0, 2.0])
    las.red = np.asarray([10000, 12000, 11000], dtype=np.uint16)
    las.green = np.asarray([30000, 32000, 31000], dtype=np.uint16)
    las.blue = np.asarray([9000, 8000, 10000], dtype=np.uint16)
    las.classification = np.asarray([5, 5, 2], dtype=np.uint8)
    las.intensity = np.asarray([20, 30, 40], dtype=np.uint16)
    las.return_number = np.asarray([1, 1, 1], dtype=np.uint8)
    las.number_of_returns = np.asarray([1, 1, 1], dtype=np.uint8)
    las.write(cloud)
    gap = tmp_path / "gap.json"
    gap.write_text(
        json.dumps(
            {
                "vertical_candidates": [
                    {
                        "candidate_id": "GAP-VERTICAL-0001",
                        "minimum_xyz_m": [-0.1, -0.1, -0.1],
                        "maximum_xyz_m": [0.3, 0.3, 2.1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "attributes.json"
    analyze_vertical_candidate_attributes(
        cloud_path=cloud,
        gap_report_path=gap,
        candidate_ids=("GAP-VERTICAL-0001",),
        output_path=output,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    record = result["records"][0]
    assert record["vegetation_class_fraction"] == 2 / 3
    assert record["interpretation"] == "classified_vegetation_no_build"
