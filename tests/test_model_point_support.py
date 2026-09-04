from __future__ import annotations

import numpy as np

from railway_recon.model_point_support import (
    evaluate_model_support,
    parse_obj_model,
    sample_object_surfaces,
)


def test_parse_and_sample_obj(tmp_path) -> None:
    source = tmp_path / "small.obj"
    source.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\no TRIANGLE\nf 1 2 3",
        encoding="utf-8",
    )
    model = parse_obj_model(source)
    samples = sample_object_surfaces(model, origin_xyz=np.asarray([10.0, 20.0, 30.0]))
    assert model.vertices.shape == (3, 3)
    assert set(samples) == {"TRIANGLE"}
    assert len(samples["TRIANGLE"]) == 4
    assert np.min(samples["TRIANGLE"], axis=0).tolist() == [10.0, 20.0, 30.0]


def test_evaluate_model_support_supported() -> None:
    samples = {"A": np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])}
    cloud = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    report, distances = evaluate_model_support(samples, cloud)
    assert report[0]["disposition"] == "supported_keep"
    assert report[0]["p90_m"] == 0.0
    assert np.allclose(distances["A"], 0.0)
