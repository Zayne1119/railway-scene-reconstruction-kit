from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway_recon.cli import build_parser, main


def _asset(asset_id: str) -> dict:
    return {
        "id": asset_id,
        "status": "review_candidate",
        "evidence_status": "direct_observation_candidate",
        "geometry": {"file": "candidate.obj", "node": asset_id},
    }


def _claim(*, residual_m: float = 0.008, owner_resolved: bool = True) -> dict:
    return {
        "id": "REL-COLUMN-RESTS-ON-OWNER",
        "type": "rests_on",
        "source": "COLUMN",
        "target": "OWNER",
        "status": "review_candidate",
        "evidence_status": "direct_interface_candidate",
        "support_evidence": {
            "direct_contact_observed": True,
            "contact_residual_m": residual_m,
            "owner_semantics_resolved": owner_resolved,
            "owner_identification_method": "observed_boundary",
            "evidence_kinds": ["full_density_point_cloud", "manual_panorama_review"],
            "proximity_only": False,
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_support_relationship_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "audit-support-relationships",
            "--registry",
            "registry.json",
            "--patch",
            "patch.json",
            "--output",
            "audit.json",
            "--maximum-contact-residual-m",
            "0.02",
            "--minimum-direct-evidence-kinds",
            "2",
        ]
    )

    assert args.command == "audit-support-relationships"
    assert args.maximum_contact_residual_m == 0.02
    assert args.minimum_direct_evidence_kinds == 2
    assert args.allow_unmodeled_owner is False


def test_support_relationship_cli_writes_pass_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "registry.json"
    patch = tmp_path / "patch.json"
    output = tmp_path / "audit.json"
    _write(registry, {"assets": [_asset("COLUMN"), _asset("OWNER")]})
    _write(
        patch,
        {
            "formal_release": False,
            "delivery_allowed": False,
            "status": "review_candidate",
            "relations": [_claim()],
        },
    )

    main(
        [
            "audit-support-relationships",
            "--registry",
            str(registry),
            "--patch",
            str(patch),
            "--output",
            str(output),
            "--minimum-direct-evidence-kinds",
            "2",
        ]
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["relationship_patch_allowed"] is True
    assert result["formal_acceptance"] is False

    with pytest.raises(SystemExit, match="1"):
        main(
            [
                "audit-support-relationships",
                "--registry",
                str(registry),
                "--patch",
                str(patch),
                "--output",
                str(output),
            ]
        )
    assert "ERROR:" in capsys.readouterr().err


def test_support_relationship_cli_emits_blocked_audit_with_nonzero_exit(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    patch = tmp_path / "patch.json"
    output = tmp_path / "audit.json"
    _write(registry, {"assets": [_asset("COLUMN"), _asset("OWNER")]})
    _write(
        patch,
        {
            "formal_release": False,
            "delivery_allowed": False,
            "status": "review_candidate",
            "relations": [_claim(residual_m=0.078, owner_resolved=False)],
        },
    )

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "audit-support-relationships",
                "--registry",
                str(registry),
                "--patch",
                str(patch),
                "--output",
                str(output),
                "--maximum-contact-residual-m",
                "0.05",
            ]
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["relationship_patch_allowed"] is False
    codes = {item["code"] for item in result["claims"][0]["failures"]}
    assert {"contact_residual_exceeds_tolerance", "owner_semantics_unresolved"} <= codes
