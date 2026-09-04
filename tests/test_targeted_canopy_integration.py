import json
import tempfile
import unittest
from pathlib import Path

from railway_recon.targeted_canopy_integration import (
    _candidate_assets,
    audit_column_platform_contacts,
    integrate_targeted_canopy_candidate,
    merge_candidate_objs,
)


class TargetedCanopyIntegrationTests(unittest.TestCase):
    def _source(
        self,
        root: Path,
        name: str,
        origin: list[float],
        material: str,
    ) -> tuple[Path, Path]:
        obj = root / f"{name}.obj"
        mtl = root / f"{name}.mtl"
        origin_path = root / f"{name}.origin.json"
        obj.write_text(
            "\n".join(
                [
                    f"mtllib {mtl.name}",
                    f"o {name.upper()}",
                    f"usemtl {material}",
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "f 1 2 3",
                ]
            ),
            encoding="utf-8",
        )
        mtl.write_text(f"newmtl {material}\nKd 0.5 0.5 0.5\n", encoding="utf-8")
        origin_path.write_text(
            json.dumps({"origin_xyz": origin, "units": "metre", "axis": "Z-up"}),
            encoding="utf-8",
        )
        return obj, origin_path

    def test_merge_translates_origins_and_offsets_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, "first", [100.0, 200.0, 0.0], "First")
            second = self._source(root, "second", [102.0, 203.0, 0.0], "Second")
            output_obj = root / "merged.obj"
            result = merge_candidate_objs(
                [first, second],
                output_obj,
                root / "merged.mtl",
                root / "origin.json",
            )
            lines = output_obj.read_text(encoding="utf-8").splitlines()
            vertices = [line for line in lines if line.startswith("v ")]
            faces = [line for line in lines if line.startswith("f ")]
            self.assertEqual(vertices[3], "v 2.000000 3.000000 0.000000")
            self.assertEqual(faces, ["f 1 2 3", "f 4 5 6"])
            self.assertEqual(result["vertex_count"], 6)

    def test_namespaced_merge_allows_repeated_component_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, "first", [0.0, 0.0, 0.0], "Shared")
            second = self._source(root, "second", [1.0, 0.0, 0.0], "Shared")
            output_obj = root / "merged.obj"
            output_mtl = root / "merged.mtl"

            merge_candidate_objs(
                [first, second],
                output_obj,
                output_mtl,
                root / "origin.json",
                source_namespaces=["SEG-A", "SEG-B"],
            )

            obj_text = output_obj.read_text(encoding="utf-8")
            mtl_text = output_mtl.read_text(encoding="utf-8")
            self.assertIn("o SEG-A--FIRST", obj_text)
            self.assertIn("o SEG-B--SECOND", obj_text)
            self.assertIn("usemtl SEG-A--Shared", obj_text)
            self.assertIn("usemtl SEG-B--Shared", obj_text)
            self.assertIn("newmtl SEG-A--Shared", mtl_text)
            self.assertIn("newmtl SEG-B--Shared", mtl_text)

    def test_candidate_assets_include_only_three_reviewed_support_chains(self) -> None:
        surfaces = []
        for index, residual in enumerate((0.18, 0.26, 0.03), start=1):
            surfaces.append(
                {
                    "id": f"ROOF-{index}",
                    "surface_type": "planar",
                    "longitudinal_range_m": [0.0, 10.0],
                    "cross_range_m": [float(index), float(index + 1)],
                    "plane_diagnostic": {"absolute_residual_p90_m": residual},
                    "surface_sampling": "cell_90th_percentile_top_envelope",
                }
            )
        columns = []
        nodes = []
        decisions = []
        for index in (1, 4, 10):
            column_id = f"RIGHT-CANOPY-GRID-{index:03d}"
            seed_id = f"V{index}"
            columns.append(
                {
                    "id": column_id,
                    "reviewed_seed_id": seed_id,
                    "longitudinal_position_m": float(index),
                    "cross_position_m": 7.0,
                    "minimum_z": 1.0,
                }
            )
            nodes.append(
                {
                    "column_grid_id": column_id,
                    "reviewed_seed_id": seed_id,
                    "nearest_observed_roof_surface_id": f"ROOF-{1 if index == 1 else 3}",
                    "capital_bottom_z_m": 4.0,
                    "capital_top_z_m": 4.32,
                    "capital_along_size_m": 1.0,
                    "capital_cross_size_m": 1.0,
                    "cross_recovery_distance_m": 1.0,
                    "confidence": 0.72,
                }
            )
            decisions.append({"candidate_id": seed_id, "confidence": 0.9})
        recovery = {
            "segment_id": "s0000_0050m",
            "output_mesh_audit": "workspace/reports/mesh.json",
            "output_obj": "candidate.obj",
            "semantic_review": "semantic.json",
            "grid_review": "grid-review.json",
            "roof_surfaces": surfaces,
            "column_grid": columns,
            "local_column_roof_nodes": nodes,
        }
        assets, relations = _candidate_assets(
            recovery,
            {"decisions": decisions},
            Path("visual.json"),
            Path("scene.obj"),
        )
        self.assertEqual(len(assets), 12)
        self.assertEqual(len(relations), 9)
        ids = {item["id"] for item in assets}
        self.assertNotIn("RIGHT-CANOPY-GRID-002", ids)
        self.assertIn("RIGHT-CANOPY-GRID-010-LOCAL-UNDERROOF", ids)

    def test_merge_requires_explicit_approval(self) -> None:
        with self.assertRaisesRegex(PermissionError, "approve-candidate-merge"):
            integrate_targeted_canopy_candidate(  # type: ignore[arg-type]
                None,
                "s0000_0050m",
                "recovery.json",
                "visual.json",
                "candidate-v1",
                approve_candidate_merge=False,
            )

    def test_column_base_matches_segmented_platform_fit(self) -> None:
        recovery = {
            "platform_component_id": "P1",
            "column_grid": [
                {
                    "id": "C1",
                    "reviewed_seed_id": "V1",
                    "status": "confirmed_photo_seed",
                    "longitudinal_position_m": 4.0,
                    "cross_position_m": 3.0,
                    "minimum_z": 10.05,
                }
            ],
        }
        platform = {
            "platform_components": [
                {
                    "id": "P1",
                    "fit_segments": [
                        {
                            "longitudinal_range_m": [0.0, 5.0],
                            "observed_cross_range_m": [2.0, 4.0],
                            "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 10.0],
                        }
                    ],
                }
            ]
        }
        audit = audit_column_platform_contacts(recovery, platform, tolerance_m=0.08)
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["contacts"][0]["vertical_gap_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
