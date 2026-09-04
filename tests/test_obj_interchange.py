import tempfile
import unittest
from pathlib import Path

from railway_recon.mesh_audit import audit_obj
from railway_recon.obj_interchange import prepare_obj_interchange


class ObjInterchangeTests(unittest.TestCase):
    def test_triangulates_repairs_winding_and_writes_explicit_normals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "prepared.obj"
            source.write_text(
                """o PANEL
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 2 1 0
f 1 2 3 4
f 2 3 6 5
""",
                encoding="utf-8",
            )

            report = prepare_obj_interchange(source, output)
            text = output.read_text(encoding="utf-8")

            self.assertTrue(report["passed"])
            self.assertEqual(report["object_count"], 1)
            self.assertEqual(report["triangle_count"], 4)
            self.assertGreater(report["explicit_normal_count"], 0)
            self.assertIn("vn ", text)
            self.assertIn("//", text)
            self.assertEqual(audit_obj(output)["degenerate_triangle_count"], 0)

    def test_rejects_same_input_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.obj"
            source.write_text("o A\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_obj_interchange(source, source)

    def test_removes_coordinate_duplicate_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.obj"
            output = root / "prepared.obj"
            source.write_text(
                """o A
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
f 4 5 6
""",
                encoding="utf-8",
            )

            report = prepare_obj_interchange(source, output)

            self.assertTrue(report["passed"])
            self.assertEqual(report["source_coordinate_duplicate_triangle_count"], 1)
            self.assertEqual(report["removed_coordinate_duplicate_triangle_count"], 1)
            self.assertEqual(report["coordinate_duplicate_triangle_count"], 0)
            self.assertEqual(report["triangle_count"], 1)


if __name__ == "__main__":
    unittest.main()
