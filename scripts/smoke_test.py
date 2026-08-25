"""Offline end-to-end smoke test using a temporary synthetic railway segment."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from railway_recon.algorithms.rail_candidates import detect_rail_candidates  # noqa: E402
from railway_recon.algorithms.track import build_parametric_track  # noqa: E402
from railway_recon.audit import audit_project  # noqa: E402
from railway_recon.config import initialize_project, load_project  # noqa: E402
from railway_recon.io import load_json, write_json  # noqa: E402
from railway_recon.mesh_audit import audit_obj  # noqa: E402
from railway_recon.qa import quality_report  # noqa: E402
from railway_recon.registry import initialize_registry, validate_registry_file  # noqa: E402
from railway_recon.segments import crop_segments, plan_segments  # noqa: E402


def build_synthetic_cloud(path: Path) -> None:
    random = np.random.default_rng(7)
    longitudinal = np.arange(0.0, 60.0, 0.25)
    rail_x = np.repeat(longitudinal, 2)
    rail_y = np.tile(np.asarray([-0.7175, 0.7175]), len(longitudinal))
    rail_z = np.full_like(rail_x, 0.30)

    ground_x = random.uniform(0.0, 60.0, 300)
    ground_y = random.uniform(-4.0, 4.0, 300)
    ground_z = random.normal(0.0, 0.01, 300)
    x = np.concatenate((rail_x, ground_x))
    y = np.concatenate((rail_y, ground_y))
    z = np.concatenate((rail_z, ground_z))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x = x
    cloud.y = y
    cloud.z = z
    cloud.red = np.full(len(x), 30000, dtype=np.uint16)
    cloud.green = np.full(len(x), 30000, dtype=np.uint16)
    cloud.blue = np.full(len(x), 30000, dtype=np.uint16)
    cloud.write(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="railway-recon-smoke-") as temporary:
        root = Path(temporary) / "synthetic"
        config_path = initialize_project(root, "synthetic-smoke", "Synthetic Smoke Test")
        config = load_json(config_path)
        config["segmentation"]["corridor_half_width_m"] = 5.0
        config["segmentation"]["z_below_camera_m"] = 5.0
        config["segmentation"]["z_above_camera_m"] = 5.0
        write_json(config_path, config)

        rail_config_path = root / "rail_detection.json"
        rail_config = load_json(rail_config_path)
        rail_config.update(
            {
                "z_mode": "absolute",
                "minimum_z": 0.20,
                "maximum_z": 0.40,
                "minimum_peak_prominence": 0.01,
                "median_filter_bins": 1,
                "gaussian_sigma_bins": 0.6,
            }
        )
        write_json(rail_config_path, rail_config)

        camera_path = root / "input/cameras.csv"
        with camera_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "timestamp", "file", "x", "y", "z"])
            for index, x_value in enumerate((0.0, 20.0, 40.0, 60.0)):
                writer.writerow([index, index, f"camera/{index:04d}.jpg", x_value, 0.0, 2.0])
        build_synthetic_cloud(root / "input/pointcloud/site.laz")

        project = load_project(config_path)
        audit_project(project)
        segment_plan = plan_segments(project)
        segment_id = segment_plan["segments"][0]["id"]
        crop_segments(project, {segment_id})
        candidates = detect_rail_candidates(project, segment_id)
        if candidates["rail_pair_count"] < 1:
            raise RuntimeError(
                "Synthetic rail pair was not detected: "
                + json.dumps(candidates["peaks"], ensure_ascii=False)
            )
        initialize_registry(project)
        track = build_parametric_track(project, segment_id)
        mesh_report = audit_obj(Path(track["output_obj"]))
        if not mesh_report["passed"]:
            raise RuntimeError("Synthetic OBJ mesh audit failed")
        _, registry_errors = validate_registry_file(project.workspace_path("asset_registry"))
        if registry_errors:
            raise RuntimeError("Registry validation failed: " + "; ".join(registry_errors))
        qa = quality_report(project)
        if not qa["passed"]:
            raise RuntimeError("Synthetic project QA did not pass")
        print(
            json.dumps(
                {
                    "passed": True,
                    "segment_id": segment_id,
                    "rail_pairs": candidates["rail_pair_count"],
                    "track_faces": track["face_count"],
                    "mesh_triangles": mesh_report["triangle_count_after_fan_triangulation"],
                    "asset_count": qa["asset_summary"]["asset_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
