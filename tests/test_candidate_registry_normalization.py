from __future__ import annotations

from railway_recon.candidate_registry_normalization import normalize_candidate_registry_value


def test_normalization_preserves_legacy_values_and_drops_dangling_relation() -> None:
    source = {
        "schema_version": "railway.asset-registry.v1",
        "project_id": "sample",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "assets": [
            {
                "id": "AAA",
                "type": "canopy",
                "status": "candidate_custom",
                "evidence_level": "supplemental_point_cloud",
                "confidence": 0.8,
                "sources": [],
            }
        ],
        "relations": [
            {"id": "REL-AAA-MISSING", "type": "supports", "from": "AAA", "to": "MISSING"}
        ],
        "summary": {},
    }
    normalized, report = normalize_candidate_registry_value(source)
    asset = normalized["assets"][0]
    assert asset["status"] == "candidate"
    assert asset["evidence_level"] == "observed"
    assert asset["parameters"]["legacy_status_before_normalization"] == "candidate_custom"
    assert normalized["relations"] == []
    assert report["dropped_dangling_relation_count"] == 1
    assert report["passed"] is True
