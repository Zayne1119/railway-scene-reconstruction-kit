import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from railway_recon.benchmark import initialize_benchmark
from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.rail_truth_evidence import render_neutral_rail_evidence
from railway_recon.rail_truth_tasks import (
    create_rail_truth_annotation_package,
    validate_rail_truth_annotation_package,
)


class RailTruthEvidenceTests(unittest.TestCase):
    def test_raw_cloud_renderer_feeds_blind_task_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = initialize_benchmark(
                root / "benchmark", "sample-dataset", "scene-a"
            )
            project_path = initialize_project(root / "project", "sample-project", "Sample")

            camera_path = root / "camera.csv"
            camera_path.write_text(
                "index,time(hh:mm:ss.ss),timestamp,file,x,y,z,rot_x,rot_y,rot_z,\n"
                "0,,0,camera/0.jpg,0,0,4,0,0,0,\n"
                "1,,1,camera/1.jpg,0,10,4,0,0,0,\n",
                encoding="utf-8",
            )
            point_path = root / "cloud.las"
            y = np.linspace(0.0, 10.0, 501)
            x = np.concatenate(
                (
                    np.full_like(y, -0.75),
                    np.full_like(y, 0.75),
                    np.full_like(y, 4.0),
                )
            )
            cloud = laspy.create(point_format=3, file_version="1.2")
            cloud.x = x
            cloud.y = np.tile(y, 3)
            cloud.z = np.concatenate(
                (np.full_like(y, 1.0), np.full_like(y, 1.0), np.full_like(y, 0.5))
            )
            cloud.write(point_path)

            project_value = load_json(project_path)
            project_value["inputs"]["point_cloud"] = str(point_path)
            project_value["inputs"]["camera_csv"] = str(camera_path)
            write_json(project_path, project_value)
            write_json(
                root
                / "project"
                / "workspace"
                / "manifests"
                / "segments.generated.json",
                {
                    "schema_version": "railway.segments.v1",
                    "project_id": "sample-project",
                    "segments": [
                        {
                            "id": "s000_010m",
                            "chainage_start_m": 0.0,
                            "chainage_end_m": 10.0,
                            "camera_index_from": 0,
                            "camera_index_to": 1,
                        }
                    ],
                },
            )

            manifest_path = render_neutral_rail_evidence(
                load_project(project_path),
                benchmark,
                "scene-a",
                0.0,
                4.0,
                spacing_m=2.0,
                output_name="neutral_test_v1",
                chunk_size_points=100,
            )
            evidence = load_json(manifest_path)
            self.assertEqual(len(evidence["stations"]), 2)
            self.assertFalse(evidence["model_overlay"])
            self.assertFalse(evidence["candidate_overlay"])
            self.assertGreater(evidence["selected_point_count"], 0)
            for station in evidence["stations"]:
                self.assertTrue(Path(station["cross_section_image"]).is_file())
                self.assertTrue(Path(station["plan_strip_image"]).is_file())

            package = create_rail_truth_annotation_package(
                benchmark,
                [manifest_path],
                output_name="rail_truth_test_v1",
                seed=42,
            )
            self.assertTrue(validate_rail_truth_annotation_package(package)["passed"])


if __name__ == "__main__":
    unittest.main()
