import tempfile
import unittest
from pathlib import Path

from railway_recon.mesh_audit import audit_obj


class MeshAuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
