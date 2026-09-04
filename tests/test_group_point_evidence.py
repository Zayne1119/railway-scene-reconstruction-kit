from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from railway_recon.group_point_evidence import render_group_point_evidence


def test_render_group_point_evidence(tmp_path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.asarray((0.0, 1.0, 2.0, 3.0))
    cloud.y = np.asarray((0.0, 0.5, 1.0, 1.5))
    cloud.z = np.asarray((10.0, 11.0, 12.0, 13.0))
    cloud.red = np.asarray((65535, 0, 0, 65535), dtype=np.uint16)
    cloud.green = np.asarray((0, 65535, 0, 65535), dtype=np.uint16)
    cloud.blue = np.asarray((0, 0, 65535, 65535), dtype=np.uint16)
    cloud_path = tmp_path / "cloud.las"
    cloud.write(cloud_path)

    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        json.dumps({"frame": {"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0], "cross_xy": [0.0, 1.0]}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "gap.json"
    report_path.write_text(
        json.dumps(
            {
                "vertical_candidates": [
                    {
                        "candidate_id": "GAP-VERTICAL-0001",
                        "station_m": 1.0,
                        "cross_m": 0.5,
                        "minimum_xyz_m": [0.8, 0.3, 10.5],
                        "maximum_xyz_m": [1.2, 0.7, 11.5],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    outputs = render_group_point_evidence(
        cloud_path=cloud_path,
        gap_report_path=report_path,
        frame_report_path=frame_path,
        output_directory=tmp_path / "out",
        candidate_ids=("GAP-VERTICAL-0001",),
        station_margin_m=2.0,
        cross_margin_m=2.0,
        z_margin_m=2.0,
    )
    assert outputs["figure"].is_file()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_crop_point_count"] == 4
    assert manifest["candidate_ids"] == ["GAP-VERTICAL-0001"]
