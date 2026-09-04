import unittest

import numpy as np

from railway_recon.geometry import CorridorFrame
from railway_recon.platform_mesh import (
    build_platform_component_mesh,
    platform_sections,
    validate_platform_mesh_gate,
)


def _component() -> dict:
    return {
        "id": "PLATFORM-SURFACE-CANDIDATE-001",
        "side": "right",
        "longitudinal_range_m": [0.0, 10.0],
        "fit_segments": [
            {
                "longitudinal_range_m": [0.0, 5.0],
                "observed_cross_range_m": [1.0, 5.0],
                "rail_side_edge_cross_m": 1.0,
                "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.0],
            },
            {
                "longitudinal_range_m": [5.0, 10.0],
                "observed_cross_range_m": [1.2, 5.2],
                "rail_side_edge_cross_m": 1.2,
                "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.2],
            },
        ],
    }


class PlatformMeshTests(unittest.TestCase):
    def test_adjacent_fits_share_one_averaged_boundary_section(self) -> None:
        sections = platform_sections(_component())
        self.assertEqual(len(sections), 3)
        self.assertAlmostEqual(sections[1]["longitudinal_m"], 5.0)
        self.assertAlmostEqual(sections[1]["low_cross_m"], 1.1)
        self.assertAlmostEqual(sections[1]["high_cross_m"], 5.1)
        self.assertAlmostEqual(sections[1]["low_z_m"], 1.1)
        self.assertEqual(sections[1]["contributing_fit_count"], 2.0)

    def test_component_mesh_has_no_independent_segment_seams(self) -> None:
        frame = CorridorFrame(
            np.asarray([100.0, 200.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        )
        top, top_faces, shell, shell_faces, sections = build_platform_component_mesh(
            _component(), frame, 0.35
        )
        self.assertEqual(top.shape, (6, 3))
        self.assertEqual(len(top_faces), 2)
        self.assertEqual(shell.shape, (12, 3))
        self.assertEqual(len(shell_faces), 8)
        self.assertEqual(len(sections), 3)
        self.assertTrue(np.allclose(shell[6:, 2], shell[:6, 2] - 0.35))
        top_normal = np.cross(
            top[top_faces[0][1]] - top[top_faces[0][0]],
            top[top_faces[0][2]] - top[top_faces[0][0]],
        )
        bottom_normal = np.cross(
            shell[shell_faces[0][1]] - shell[shell_faces[0][0]],
            shell[shell_faces[0][2]] - shell[shell_faces[0][0]],
        )
        self.assertGreater(top_normal[2], 0.0)
        self.assertLess(bottom_normal[2], 0.0)

    def test_gate_fails_closed_when_any_gap_is_unresolved(self) -> None:
        gate = {
            "schema_version": "railway.platform-mesh-gate.v1",
            "segment_id": "s0000_0050m",
            "platform_component_id": "PLATFORM-SURFACE-CANDIDATE-001",
            "unresolved_gap_count": 1,
            "mesh_gate": "passed_candidate",
        }
        with self.assertRaisesRegex(ValueError, "unresolved"):
            validate_platform_mesh_gate(gate, "s0000_0050m", "PLATFORM-SURFACE-CANDIDATE-001")


if __name__ == "__main__":
    unittest.main()
