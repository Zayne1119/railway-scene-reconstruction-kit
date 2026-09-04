from __future__ import annotations

from railway_recon.adjacent_candidate_closure import evaluate_adjacent_candidate_closure


def test_adjacent_candidate_closure_passes_complete_inputs() -> None:
    registry = {
        "schema_version": "railway.asset-registry.v1",
        "project_id": "sample",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "assets": [
            {
                "id": "ASSET-001",
                "type": "rail",
                "status": "candidate",
                "evidence_level": "observed",
                "confidence": 0.9,
                "sources": [{"kind": "point_cloud", "reference": "sample.laz"}],
            }
        ],
        "relations": [],
        "summary": {
            "asset_count": 1,
            "by_type": {"rail": 1},
            "by_evidence_level": {"observed": 1},
        },
    }
    result = evaluate_adjacent_candidate_closure(
        mesh={
            "passed": True,
            "object_names": ["TRACK-001"],
            "degenerate_triangle_count": 0,
            "duplicate_face_count": 0,
        },
        seams={"passed": True},
        columns={
            "fit_records": [
                {"point_to_mesh_support": {"passed": True}},
                {"point_to_mesh_support": {"passed": True}},
                {"point_to_mesh_support": {"passed": True}},
            ]
        },
        conductors={"passed": True},
        surfaces={"built_count": 1, "target_count": 1, "records": [{"passed": True}]},
        transition={"passed": True, "gates": {"point_support": True}},
        registry=registry,
        fixed_views=[{"view_count": 6}, {"view_count": 6}],
    )
    assert result["passed"] is True
    assert result["status"].endswith("not_promoted")
