from __future__ import annotations

from pathlib import Path

import numpy as np

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.mesh_seam_reconciliation import reconcile_mesh_seams


def _box_vertices(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    return [
        (x0, y0, z0),
        (x0, y1, z0),
        (x0, y1, z1),
        (x0, y0, z1),
        (x1, y0, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x1, y0, z1),
    ]


BOX_FACES = [
    (1, 2, 3, 4),
    (5, 8, 7, 6),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 8, 4),
    (4, 8, 5, 1),
]


def _write_source(path: Path) -> None:
    first = _box_vertices(0.0, 5.0, 0.0, 2.0, 0.0, 1.0)
    second = _box_vertices(5.02, 10.0, 0.02, 2.02, 0.02, 1.02)
    lines = ["mtllib source.mtl"]
    for point in [*first, *second]:
        lines.append(f"v {point[0]} {point[1]} {point[2]}")
    lines.append("o A-VOLUME")
    lines.append("usemtl Platform")
    lines.extend("f " + " ".join(str(value) for value in face) for face in BOX_FACES)
    lines.append("o B-VOLUME")
    lines.append("usemtl Platform")
    lines.extend("f " + " ".join(str(value + 8) for value in face) for face in BOX_FACES)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_name("source.mtl").write_text("newmtl Platform\nKd 0.5 0.5 0.5\n", encoding="utf-8")


def test_reconciliation_welds_endpoints_and_removes_one_duplicate_cap(tmp_path) -> None:
    project = load_project(initialize_project(tmp_path / "project", "sample", "Sample"))
    source = project.root / "source.obj"
    _write_source(source)
    write_json(project.root / "origin.json", {"origin_xyz": [0.0, 0.0, 0.0]})
    write_json(
        project.root / "frame.json",
        {
            "frame": {
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            }
        },
    )
    write_json(
        project.root / "settings.json",
        {
            "schema_version": "railway.mesh-seam-reconciliation-settings.v1",
            "endpoint_search_tolerance_m": 0.05,
            "endpoint_cluster_tolerance_m": 0.00001,
            "cap_station_tolerance_m": 0.0001,
            "maximum_cross_delta_m": 0.05,
            "maximum_z_delta_m": 0.05,
            "seams": [
                {
                    "id": "A-B",
                    "station_m": 5.01,
                    "pairs": [
                        {
                            "left_object": "A-VOLUME",
                            "right_object": "B-VOLUME",
                            "internal_cap_policy": "drop_right_duplicate",
                            "expected_removed_cap_polygon_count": 1,
                        }
                    ],
                }
            ],
        },
    )

    result = reconcile_mesh_seams(
        project,
        source,
        project.root / "origin.json",
        project.root / "frame.json",
        project.root / "settings.json",
        project.root / "output",
    )

    assert result["passed"] is True
    assert result["removed_internal_cap_polygon_count"] == 1
    cleanup = result["seams"][0]["pairs"][0]["internal_cap_cleanup"]
    assert cleanup["duplicate_triangle_count_before"] == 2
    assert cleanup["duplicate_triangle_count_after"] == 0
    output_lines = (
        (project.root / "output" / "reconciled.obj").read_text(encoding="utf-8").splitlines()
    )
    assert sum(line.startswith("f ") for line in output_lines) == 11
    vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in output_lines
            if line.startswith("v ")
        ]
    )
    seam_x = vertices[np.isclose(vertices[:, 0], 5.01)]
    assert len(seam_x) == 8
    assert load_json(project.root / "output" / "mesh_audit.json")["passed"] is True
