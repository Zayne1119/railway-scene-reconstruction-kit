import numpy as np

from railway_recon.catenary_candidate_mesh import analyze_catenary_section_data


def test_section_audit_ignores_wide_foundation_and_arm_bins() -> None:
    rng = np.random.default_rng(7)
    candidate = {
        "id": "V1",
        "longitudinal_position_m": 5.0,
        "cross_position_m": -2.0,
        "minimum_z": 0.0,
        "maximum_z": 5.0,
    }
    frame = {
        "origin_xy": [0.0, 0.0],
        "along_xy": [1.0, 0.0],
        "cross_xy": [0.0, 1.0],
    }
    points = []
    for index in range(10):
        z_low = index * 0.5
        count = 120
        width_s = 1.0 if index == 0 else (0.9 if index == 7 else 0.24)
        width_c = 1.0 if index == 0 else (1.1 if index == 7 else 0.30)
        points.append(
            np.column_stack(
                (
                    5.0 + rng.uniform(-width_s / 2.0, width_s / 2.0, count),
                    -2.0 + rng.uniform(-width_c / 2.0, width_c / 2.0, count),
                    rng.uniform(z_low, z_low + 0.5, count),
                )
            )
        )
    xyz = np.vstack(points)
    report = analyze_catenary_section_data(
        candidate, frame, xyz[:, 0], xyz[:, 1], xyz[:, 2]
    )
    assert report["section_gate"] == "passed"
    assert report["estimated_shaft"]["supporting_height_bin_count"] == 8
    assert 0.18 <= report["estimated_shaft"]["along_width_p90_m"] <= 0.26
    assert 0.22 <= report["estimated_shaft"]["cross_width_p90_m"] <= 0.32
