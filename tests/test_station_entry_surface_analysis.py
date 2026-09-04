from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from railway_recon.station_entry_surface_analysis import analyse_station_entry_surfaces


def test_analyse_station_entry_surfaces(tmp_path: Path) -> None:
    count = 200
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.linspace(100.0, 120.0, count)
    cloud.y = np.where(np.arange(count) % 2 == 0, 15.6, 20.0)
    cloud.z = np.linspace(21.0, 25.0, count)
    cloud_path = tmp_path / "cloud.las"
    cloud.write(cloud_path)
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        json.dumps({"frame": {"origin_xy": [0.0, 0.0], "along_xy": [1.0, 0.0], "cross_xy": [0.0, 1.0]}}),
        encoding="utf-8",
    )
    output = analyse_station_entry_surfaces(
        cloud_path=cloud_path,
        frame_report_path=frame_path,
        output_path=tmp_path / "analysis.json",
        station_bounds_m=(98.0, 122.0),
        cross_bounds_m=(14.0, 21.0),
        z_bounds_m=(20.0, 26.0),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["crop_point_count"] == count
    assert len(report["plane_bands"]) == 2
    assert all(item["point_count"] == count // 2 for item in report["plane_bands"])
