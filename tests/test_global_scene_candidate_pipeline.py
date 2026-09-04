from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from railway_recon.config import initialize_project, load_project
from railway_recon.global_scene_candidate_pipeline import (
    run_global_scene_candidate_pipeline,
)
from railway_recon.io import write_json


class GlobalSceneCandidatePipelineTests(unittest.TestCase):
    def test_orchestrates_candidate_components_without_registry_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = load_project(
                initialize_project(Path(temporary) / "sample", "sample-project", "Sample")
            )
            graph = project.root / "graph.json"
            audit = project.root / "audit.json"
            station = project.root / "station.obj"
            write_json(graph, {"schema_version": "railway.track-graph.v1"})
            write_json(audit, {"status": "fail"})
            station.write_text("o station\n", encoding="utf-8")
            track_obj = project.root / "track.obj"
            catenary_obj = project.root / "catenary.obj"
            conductor_obj = project.root / "conductor.obj"
            for path in (track_obj, catenary_obj, conductor_obj):
                path.write_text("o candidate\n", encoding="utf-8")
            assembly = {
                "status": "candidate_mesh_assembly_written_not_formal_release",
                "output_obj": str(project.root / "scene.obj"),
                "output_mtl": str(project.root / "scene.mtl"),
                "output_origin": str(project.root / "model_origin.json"),
                "output_mesh_audit": str(project.root / "mesh_audit.json"),
                "component_count": 4,
            }
            with (
                patch(
                    "railway_recon.global_scene_candidate_pipeline.build_track_graph_mesh",
                    return_value={
                        "status": "candidate_review_required_nonpassing_source_audit",
                        "output_obj": str(track_obj),
                        "tracks": [{"track_id": "TRACK-1"}],
                        "face_count": 100,
                        "include_inferred_gap_hypotheses": False,
                    },
                ) as track_mock,
                patch(
                    "railway_recon.global_scene_candidate_pipeline.run_corridor_catenary_pipeline",
                    return_value={
                        "status": "candidate_mesh_written_review_required_not_formal_release",
                        "output_obj": str(catenary_obj),
                        "mast_count": 3,
                    },
                ),
                patch(
                    "railway_recon.global_scene_candidate_pipeline.run_corridor_conductor_pipeline",
                    return_value={
                        "status": "candidate_fragments_written_review_required_not_formal_release",
                        "output_obj": str(conductor_obj),
                        "span_count": 4,
                        "passing_seam_count": 2,
                        "failing_adjacent_seam_count": 1,
                    },
                ),
                patch(
                    "railway_recon.global_scene_candidate_pipeline.assemble_candidate_meshes",
                    return_value=assembly,
                ) as assembly_mock,
            ):
                report = run_global_scene_candidate_pipeline(
                    project,
                    graph,
                    audit,
                    project.root / "global",
                    "scene_candidate",
                    station_assets_obj=station,
                )

            self.assertEqual(
                report["status"],
                "global_candidate_assembled_optimization_and_review_required",
            )
            self.assertFalse(report["formal_release"])
            self.assertFalse(report["canonical_registry_updated"])
            self.assertTrue(report["candidate_nonpassing_track_audit_mode"])
            self.assertFalse(track_mock.call_args.kwargs["update_registry"])
            self.assertTrue(
                track_mock.call_args.kwargs["candidate_only_nonpassing_audit"]
            )
            self.assertFalse(
                track_mock.call_args.kwargs["include_inferred_gap_hypotheses"]
            )
            self.assertEqual(len(assembly_mock.call_args.args[3]), 4)


if __name__ == "__main__":
    unittest.main()
