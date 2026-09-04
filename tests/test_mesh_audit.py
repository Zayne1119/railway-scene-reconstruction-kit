import tempfile
import unittest
from pathlib import Path

from railway_recon.mesh_audit import audit_obj


class MeshAuditTests(unittest.TestCase):
    def test_cross_object_coordinate_duplicate_triangles_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cross-object-duplicate.obj"
            path.write_text(
                """o LEFT
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o RIGHT
v 0 1 0
v 1 0 0
v 0 0 0
f 4 5 6
""",
                encoding="utf-8",
            )

            report = audit_obj(path)

            self.assertFalse(report["passed"])
            self.assertEqual(report["duplicate_face_count"], 0)
            self.assertEqual(report["coordinate_duplicate_triangle_count"], 1)
            self.assertEqual(
                report["cross_object_coordinate_duplicate_triangle_count"], 1
            )
            self.assertEqual(
                report["coordinate_duplicate_examples"],
                [{"first_object": "LEFT", "duplicate_object": "RIGHT"}],
            )

    def test_duplicate_object_names_fail_asset_identity_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate-nodes.obj"
            path.write_text(
                """o TRACK-0001-RAIL-LEFT
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o TRACK-0001-RAIL-LEFT
v 0 0 1
v 1 0 1
v 0 1 1
f 4 5 6
""",
                encoding="utf-8",
            )

            report = audit_obj(path)

            self.assertFalse(report["passed"])
            self.assertEqual(report["object_count"], 1)
            self.assertEqual(report["object_declaration_count"], 2)
            self.assertEqual(report["duplicate_object_name_count"], 1)

    def test_unreferenced_vertices_are_reported_and_optionally_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "loose-vertex.obj"
            path.write_text(
                """o TRACK
v 0 0 0
v 1 0 0
v 0 1 0
v 9 9 9
f 1 2 3
""",
                encoding="utf-8",
            )

            advisory = audit_obj(path)
            strict = audit_obj(path, fail_on_unreferenced_vertices=True)

            self.assertTrue(advisory["passed"])
            self.assertEqual(advisory["referenced_vertex_count"], 3)
            self.assertEqual(advisory["unreferenced_vertex_count"], 1)
            self.assertEqual(advisory["unreferenced_vertex_ratio"], 0.25)
            self.assertFalse(strict["passed"])


if __name__ == "__main__":
    unittest.main()
