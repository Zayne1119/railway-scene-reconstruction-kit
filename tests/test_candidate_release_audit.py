from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from railway_recon.candidate_release_audit import (
    CandidateReleasePolicy,
    audit_candidate_release,
    main,
)
from railway_recon.io import sha256_file
from railway_recon.rapid_candidate_audit import (
    audit_rapid_candidate,
    classify_candidate_failures,
)

RELEASE_ID = "corridor_candidate_rc2"
PARENT_ID = "corridor_candidate_rc1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_glb(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total_length = 20 + len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _artifact(root: Path, role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "release_id": RELEASE_ID,
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _candidate_fixture(tmp_path: Path) -> dict[str, Path]:
    parent_root = tmp_path / PARENT_ID
    parent_model = parent_root / "model" / "parent.obj"
    parent_model.parent.mkdir(parents=True)
    parent_model.write_text("o parent\nv 0 0 0\n", encoding="utf-8")
    parent_registry = parent_root / "registry.json"
    _write_json(
        parent_registry,
        {
            "release_id": PARENT_ID,
            "assets": [
                {
                    "id": "asset-1",
                    "type": "platform",
                    "status": "review_candidate",
                    "evidence_status": "candidate_pending_review",
                }
            ],
            "relations": [
                {
                    "id": "rel-1",
                    "type": "rests_on",
                    "from": "asset-1",
                    "to": "external-support-1",
                    "status": "candidate_pending_review",
                }
            ],
        },
    )
    parent_artifacts = [
        {
            "role": "compact_model",
            "release_id": PARENT_ID,
            "path": str(parent_model.relative_to(parent_root)),
            "sha256": sha256_file(parent_model),
            "size_bytes": parent_model.stat().st_size,
        },
        {
            "role": "registry",
            "release_id": PARENT_ID,
            "path": str(parent_registry.relative_to(parent_root)),
            "sha256": sha256_file(parent_registry),
            "size_bytes": parent_registry.stat().st_size,
        },
    ]
    _write_json(
        parent_root / "candidate_release_manifest.json",
        {
            "release_id": PARENT_ID,
            "artifacts": parent_artifacts,
            "artifact_count": len(parent_artifacts),
        },
    )
    root = tmp_path / RELEASE_ID
    model = root / "model" / "candidate.obj"
    model.parent.mkdir(parents=True)
    model.write_text("o asset-1\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    registry = root / "registry.json"
    registry_value = {
        "release_id": RELEASE_ID,
        "formal_release": False,
        "candidate_state_policy": {
            "source_release_id": PARENT_ID,
            "formal_acceptance": False,
            "delivery_allowed": False,
        },
        "model_sha256": sha256_file(model),
        "assets": [
            {
                "id": "asset-1",
                "type": "platform",
                "status": "review_candidate",
                "evidence_status": "candidate_pending_review",
            }
        ],
        "relations": [
            {
                "id": "rel-1",
                "type": "rests_on",
                "from": "asset-1",
                "to": "external-support-1",
                "status": "candidate_pending_review",
            }
        ],
    }
    _write_json(registry, registry_value)
    fixed_dir = root / "evidence" / "fixed"
    image = fixed_dir / "view.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"candidate-view")
    fixed = fixed_dir / "manifest.json"
    _write_json(
        fixed,
        {
            "release_id": RELEASE_ID,
            "formal_release": False,
            "source_obj": str(model),
            "source_obj_sha256": sha256_file(model),
            "source_registry": str(registry),
            "source_registry_sha256": sha256_file(registry),
            "review_status": "pending_human_review",
            "views": [
                {
                    "id": "view-1",
                    "image": image.name,
                    "image_sha256": sha256_file(image),
                    "review_status": "pending_human_review",
                }
            ],
        },
    )
    embedded_asset = {
        "id": "asset-1",
        "type": "platform",
        "parameters": {
            "status": "review_candidate",
            "evidence_status": "candidate_pending_review",
        },
    }
    glb_document = {
        "asset": {"version": "2.0"},
        "meshes": [{"name": "asset-1"}],
        "extras": {
            "sourceRelease": RELEASE_ID,
            "releaseStatus": "candidate_not_formal_release",
            "assetCount": 1,
            "assetRegistry": [embedded_asset],
        },
    }
    hq_glb = root / "engine" / "hq.glb"
    web_glb = root / "engine" / "web.glb"
    _write_glb(hq_glb, glb_document)
    _write_glb(web_glb, glb_document)
    lineage = root / "integration.json"
    _write_json(
        lineage,
        {
            "release_id": RELEASE_ID,
            "source_release_id": PARENT_ID,
            "parent_model_sha256": sha256_file(parent_model),
            "parent_registry_sha256": sha256_file(parent_registry),
            "formal_release": False,
            "delivery_allowed": False,
            "status": "candidate_integrated",
        },
    )
    stage = root / "stages" / "registry.json"
    _write_json(
        stage,
        {
            "release_id": RELEASE_ID,
            "stage_id": "registry",
            "critical": True,
            "status": "candidate_checked",
            "formal_release": False,
            "source_report": str(registry),
            "source_report_sha256": sha256_file(registry),
        },
    )
    gate = root / "candidate_delivery_gate.json"
    gate_artifacts = [
        _artifact(root, "model", model),
        _artifact(root, "registry", registry),
        _artifact(root, "fixed_views", fixed),
    ]
    _write_json(
        gate,
        {
            "release_id": RELEASE_ID,
            "release_class": "internal_review_candidate",
            "status": "blocked_not_formal_release",
            "formal_release": False,
            "delivery_allowed": False,
            "blocking_reasons": ["human_review_pending"],
            "critical_stages": [
                {
                    "id": "registry",
                    "critical": True,
                    "release_id": RELEASE_ID,
                    "status": "candidate_checked",
                    "report": str(stage),
                    "sha256": sha256_file(stage),
                }
            ],
            "artifacts": gate_artifacts,
        },
    )
    manifest = root / "candidate_release_manifest.json"
    manifest_artifacts = [
        _artifact(root, "compact_model", model),
        _artifact(root, "registry", registry),
        _artifact(root, "integration", lineage),
        _artifact(root, "hq_assetized_glb", hq_glb),
        _artifact(root, "web_assetized_glb", web_glb),
        _artifact(root, "fixed_views", fixed),
        _artifact(root, "delivery_gate", gate),
    ]
    _write_json(
        manifest,
        {
            "release_id": RELEASE_ID,
            "release_class": "internal_review_candidate",
            "formal_release": False,
            "delivery_allowed": False,
            "package_generated": False,
            "published": False,
            "artifacts": manifest_artifacts,
            "artifact_count": len(manifest_artifacts),
        },
    )
    return {
        "root": root,
        "manifest": manifest,
        "gate": gate,
        "lineage": lineage,
        "registry": registry,
        "fixed": fixed,
        "hq_glb": hq_glb,
        "model": model,
    }


def _audit(paths: dict[str, Path]) -> dict[str, Any]:
    return audit_candidate_release(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
        policy=CandidateReleasePolicy(required_stage_ids=("registry",)),
    )


def test_candidate_closure_passes_without_promoting_delivery(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)

    report = _audit(paths)

    assert report["passed"] is True
    assert report["formal_release"] is False
    assert report["delivery_allowed"] is False
    assert report["registry"]["asset_count"] == 1
    assert report["registry"]["unbound_relation_endpoint_count"] == 1


def test_rapid_candidate_keeps_strict_success_and_defers_human_review(
    tmp_path: Path,
) -> None:
    paths = _candidate_fixture(tmp_path)

    report = audit_rapid_candidate(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        lineage_path=paths["lineage"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
    )

    assert report["passed"] is True
    assert report["hard_blocker_count"] == 0
    assert report["manual_review"]["required_for_internal_iteration"] is False
    assert report["manual_review"]["pending_view_count"] == 1


def test_rapid_candidate_softens_only_explicit_iteration_debt() -> None:
    failures = [
        {"code": "critical_stages_missing"},
        {"code": "fixed_view_image_sha256_mismatch"},
    ]

    hard, warnings = classify_candidate_failures(failures)

    assert [item["code"] for item in warnings] == ["critical_stages_missing"]
    assert [item["code"] for item in hard] == [
        "fixed_view_image_sha256_mismatch"
    ]


def test_rapid_candidate_still_blocks_damaged_model(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    paths["model"].write_text("tampered\n", encoding="utf-8")

    report = audit_rapid_candidate(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        lineage_path=paths["lineage"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
    )

    assert report["passed"] is False
    assert report["hard_blocker_count"] >= 1
    assert "manifest_artifact_sha256_mismatch" in {
        item["code"] for item in report["hard_blockers"]
    }


def test_artifact_tampering_fails_hash_closure(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    paths["model"].write_text("tampered\n", encoding="utf-8")

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert report["passed"] is False
    assert "manifest_artifact_sha256_mismatch" in codes
    assert "gate_artifact_sha256_mismatch" in codes
    assert "registry_model_sha256_mismatch" in codes


def test_release_and_parent_lineage_mismatch_fail(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)

    report = audit_candidate_release(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        expected_release_id="wrong-release",
        expected_parent_release_id="wrong-parent",
    )

    codes = {failure["code"] for failure in report["failures"]}
    assert "manifest_release_id_mismatch" in codes
    assert "registry_parent_release_id_mismatch" in codes
    assert "lineage_parent_release_id_mismatch" in codes


def test_parent_model_and_registry_dependencies_are_hash_bound(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    policy = CandidateReleasePolicy(require_parent_artifact_hashes=True)

    passing = audit_candidate_release(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
        policy=policy,
    )
    assert passing["passed"] is True

    parent_model = tmp_path / PARENT_ID / "model" / "parent.obj"
    parent_model.write_text("tampered parent\n", encoding="utf-8")
    failing = audit_candidate_release(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
        policy=policy,
    )
    codes = {failure["code"] for failure in failing["failures"]}
    assert "parent_manifest_artifact_sha256_mismatch" in codes
    assert "lineage_parent_artifact_sha256_mismatch" in codes


def test_registry_asset_and_relation_promotion_fail_closed(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    registry["assets"][0]["status"] = "accepted"
    registry["assets"][0]["evidence_status"] = "supported"
    registry["relations"][0]["status"] = "formal_relation"
    _write_json(paths["registry"], registry)

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert "registry_asset_status_promoted" in codes
    assert "registry_asset_evidence_status_promoted" in codes
    assert "registry_relation_status_promoted" in codes


def test_parent_states_cannot_be_silently_rewritten(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    registry["assets"][0]["evidence_status"] = "candidate_reclassified"
    _write_json(paths["registry"], registry)

    report = audit_candidate_release(
        paths["root"],
        paths["manifest"],
        paths["gate"],
        expected_release_id=RELEASE_ID,
        expected_parent_release_id=PARENT_ID,
        policy=CandidateReleasePolicy(require_parent_state_preservation=True),
    )

    changes = [
        failure
        for failure in report["failures"]
        if failure["code"] == "parent_registry_state_changed_without_allowlist"
    ]
    assert changes == [
        {
            "code": "parent_registry_state_changed_without_allowlist",
            "collection": "assets",
            "identifier": "asset-1",
            "field": "evidence_status",
            "parent": "candidate_pending_review",
            "candidate": "candidate_reclassified",
        }
    ]


def test_glb_release_or_asset_state_mismatch_fails(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    data = paths["hq_glb"].read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))
    document["extras"]["sourceRelease"] = "another-release"
    document["extras"]["assetRegistry"][0]["parameters"]["status"] = "accepted"
    _write_glb(paths["hq_glb"], document)

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert "glb_release_id_mismatch" in codes
    assert "glb_asset_state_mismatch" in codes


def test_fixed_view_source_and_image_hash_mismatch_fail(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    fixed = json.loads(paths["fixed"].read_text(encoding="utf-8"))
    fixed["source_obj_sha256"] = "0" * 64
    fixed["views"][0]["image_sha256"] = "1" * 64
    _write_json(paths["fixed"], fixed)

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert "fixed_views_source_sha256_mismatch" in codes
    assert "fixed_view_image_sha256_mismatch" in codes


def test_failed_critical_stage_is_not_treated_as_candidate_success(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    stage_path = Path(gate["critical_stages"][0]["report"])
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["status"] = "failed"
    _write_json(stage_path, stage)
    gate["critical_stages"][0]["status"] = "failed"
    gate["critical_stages"][0]["sha256"] = sha256_file(stage_path)
    _write_json(paths["gate"], gate)

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert "critical_stage_failed" in codes
    assert "critical_stage_report_failed" in codes


def test_manifest_asset_and_fixed_view_counts_must_match_sources(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["asset_count"] = 99
    manifest["relation_count"] = 99
    manifest["fixed_view_count"] = 2
    manifest["fixed_view_counts"] = {"views": 2}
    manifest["status_promotion_count"] = 1
    _write_json(paths["manifest"], manifest)

    report = _audit(paths)

    codes = {failure["code"] for failure in report["failures"]}
    assert "manifest_registry_asset_count_mismatch" in codes
    assert "manifest_registry_relation_count_mismatch" in codes
    assert "manifest_fixed_view_count_mismatch" in codes
    assert "manifest_fixed_view_counts_mismatch" in codes
    assert "manifest_status_promotion_count_not_zero" in codes


def test_cli_returns_nonzero_on_failure_and_refuses_overwrite(tmp_path: Path) -> None:
    paths = _candidate_fixture(tmp_path)
    output = tmp_path / "audit.json"
    arguments = [
        "--release-directory",
        str(paths["root"]),
        "--manifest",
        str(paths["manifest"]),
        "--delivery-gate",
        str(paths["gate"]),
        "--expected-release-id",
        "wrong-release",
        "--output",
        str(output),
    ]

    assert main(arguments) == 2
    original = output.read_bytes()
    assert main(arguments) == 3
    assert output.read_bytes() == original
