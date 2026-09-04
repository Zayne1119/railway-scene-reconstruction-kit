from __future__ import annotations

import json
from pathlib import Path

import pytest

from railway_recon.fixed_view_review import (
    audit_fixed_view_review,
    build_fixed_view_review_template,
    main,
    parse_expected_group_counts,
    write_fixed_view_review_gate,
    write_fixed_view_review_template,
)
from railway_recon.io import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    model = tmp_path / "candidate.obj"
    registry = tmp_path / "registry.json"
    model.write_text("v 0 0 0\n", encoding="utf-8")
    registry.write_text("{}\n", encoding="utf-8")
    views = []
    for index in range(12):
        image = tmp_path / f"view_{index:02d}.png"
        image.write_bytes(f"image-{index}".encode())
        views.append(
            {
                "id": f"view_{index:02d}",
                "title": f"View {index:02d}",
                "image": image.name,
                "image_sha256": sha256_file(image),
                "review_status": "pending_human_review",
            }
        )
    manifest = {
        "schema_version": "example.fixed-views.v1",
        "release_id": "candidate-1",
        "generated_at": "2026-09-04T13:00:00+08:00",
        "formal_release": False,
        "review_status": "pending_human_review",
        "source_obj": str(model),
        "source_obj_sha256": sha256_file(model),
        "source_registry": str(registry),
        "source_registry_sha256": sha256_file(registry),
        "views": views[:8],
        "detail_views": views[8:],
    }
    manifest_path = tmp_path / "fixed_views.json"
    write_json(manifest_path, manifest)
    review = {
        "schema_version": "railway.fixed-view-review-signoff.v2",
        "release_id": manifest["release_id"],
        "source_obj_sha256": manifest["source_obj_sha256"],
        "source_registry_sha256": manifest["source_registry_sha256"],
        "render_manifest_sha256": sha256_file(manifest_path),
        "formal_release": False,
        "delivery_allowed": False,
        "status": "candidate_human_review_completed",
        "reviewer": {"type": "human", "id": "reviewer-1"},
        "reviewed_at": "2026-09-04T14:00:00+08:00",
        "view_groups": ["views", "detail_views"],
        "views": [
            {
                "group": "views" if index < 8 else "detail_views",
                "id": item["id"],
                "title": item["title"],
                "image": item["image"],
                "image_sha256": item["image_sha256"],
                "decision": "pass",
                "comment": "",
            }
            for index, item in enumerate(views)
        ],
    }
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    return manifest_path, review_path, review


def failure_codes(report: dict) -> set[str]:
    return {item["code"] for item in report["failures"]}


def test_complete_hash_bound_human_review_passes(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    report = audit_fixed_view_review(manifest, review)
    assert report["passed"] is True
    assert report["formal_release"] is False
    assert report["view_count"] == report["passed_count"] == 12


def test_rendered_images_alone_never_pass(tmp_path: Path) -> None:
    manifest, _, _ = fixture(tmp_path)
    report = audit_fixed_view_review(manifest)
    assert report["passed"] is False
    assert "human_review_record_missing" in failure_codes(report)


def test_incomplete_review_fails(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"].pop()
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "review_views_incomplete" in failure_codes(report)


def test_technical_pass_token_cannot_launder_human_acceptance(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"][0]["decision"] = "technical_pass"
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "review_decision_not_explicit_pass_or_fail" in failure_codes(report)


def test_explicit_failed_view_blocks_gate(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"][0]["decision"] = "fail"
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert report["status"] == "blocked_human_review_failed"
    assert "human_review_view_failed" in failure_codes(report)


def test_automated_reviewer_cannot_sign_off(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["reviewer"] = {"type": "automation", "id": "render-bot"}
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "reviewer_is_not_explicitly_human" in failure_codes(report)


def test_release_and_source_identity_must_match(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["release_id"] = "other-release"
    review["source_obj_sha256"] = "0" * 64
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "review_release_id_mismatch" in failure_codes(report)
    assert "review_source_hash_mismatch" in failure_codes(report)


def test_review_image_hash_must_match_manifest(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"][0]["image_sha256"] = "0" * 64
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "review_image_hash_mismatch" in failure_codes(report)


def test_manifest_image_tamper_is_detected(tmp_path: Path) -> None:
    manifest, review_path, _ = fixture(tmp_path)
    (tmp_path / "view_00.png").write_bytes(b"changed")
    report = audit_fixed_view_review(manifest, review_path)
    assert "manifest_image_hash_mismatch" in failure_codes(report)


def test_review_timestamp_requires_timezone(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["reviewed_at"] = "2026-09-04T14:00:00"
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest, review_path)
    assert "review_timestamp_missing_or_not_timezone_aware" in failure_codes(report)


def test_review_timestamp_cannot_precede_render_manifest(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["reviewed_at"] = "2026-09-04T12:59:59+08:00"
    write_json(review_path, review)

    report = audit_fixed_view_review(manifest, review_path)

    assert "review_timestamp_precedes_manifest" in failure_codes(report)


def test_review_must_bind_the_render_manifest_hash(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review.pop("render_manifest_sha256")
    write_json(review_path, review)

    report = audit_fixed_view_review(manifest, review_path)

    assert "review_manifest_hash_mismatch" in failure_codes(report)


def test_duplicate_and_unknown_review_views_fail(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"].append(dict(review["views"][0]))
    unknown = dict(review["views"][1])
    unknown["id"] = "unknown-view"
    review["views"].append(unknown)
    write_json(review_path, review)

    report = audit_fixed_view_review(manifest, review_path)
    failures = failure_codes(report)

    assert "review_view_id_duplicate" in failures
    assert "review_contains_unknown_views" in failures


def test_group_title_and_image_are_hash_bound_review_fields(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["views"][0]["group"] = "detail_views"
    review["views"][0]["title"] = "substituted title"
    review["views"][0]["image"] = "substituted.png"
    write_json(review_path, review)

    report = audit_fixed_view_review(manifest, review_path)

    assert "review_view_binding_mismatch" in failure_codes(report)


def test_review_status_and_formal_fields_cannot_launder_acceptance(tmp_path: Path) -> None:
    manifest, review_path, review = fixture(tmp_path)
    review["status"] = "formally_accepted"
    review["formal_release"] = True
    review["delivery_allowed"] = True
    review["views"][0]["review_status"] = "accepted"
    write_json(review_path, review)

    report = audit_fixed_view_review(manifest, review_path)
    failures = failure_codes(report)

    assert "review_status_not_completed_candidate" in failures
    assert "review_formal_release_not_false" in failures
    assert "review_delivery_allowed_not_false" in failures
    assert "review_state_pollution" in failures


def test_manifest_status_cannot_claim_acceptance(tmp_path: Path) -> None:
    manifest_path, review_path, _ = fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_release"] = True
    manifest["review_status"] = "accepted"
    manifest["views"][0]["review_status"] = "formally_approved"
    write_json(manifest_path, manifest)

    report = audit_fixed_view_review(manifest_path, review_path)
    failures = failure_codes(report)

    assert "manifest_formal_release_not_false" in failures
    assert "manifest_review_status_not_pending" in failures
    assert "manifest_view_status_not_pending" in failures
    assert "manifest_state_pollution" in failures


def test_expected_group_counts_fail_on_missing_unknown_or_wrong_count(tmp_path: Path) -> None:
    manifest, review_path, _ = fixture(tmp_path)

    report = audit_fixed_view_review(
        manifest,
        review_path,
        expected_group_counts={"views": 8, "required_details": 4},
    )
    failure = next(
        item
        for item in report["failures"]
        if item["code"] == "manifest_view_group_contract_mismatch"
    )

    assert failure["missing_groups"] == ["required_details"]
    assert failure["unknown_groups"] == ["detail_views"]


def test_malformed_row_is_not_hidden_by_group_discovery(tmp_path: Path) -> None:
    manifest_path, review_path, _ = fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"][0] = "not-an-object"
    write_json(manifest_path, manifest)

    report = audit_fixed_view_review(manifest_path, review_path)

    assert "manifest_view_row_malformed" in failure_codes(report)


def test_manifest_image_must_stay_below_manifest_directory(tmp_path: Path) -> None:
    manifest_path, review_path, _ = fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"][0]["image"] = "../outside.png"
    write_json(manifest_path, manifest)

    report = audit_fixed_view_review(manifest_path, review_path)

    assert "manifest_image_path_outside_manifest_directory" in failure_codes(report)


def test_gate_output_is_immutable(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    output = tmp_path / "gate.json"
    report = write_fixed_view_review_gate(output, manifest, review)
    original = output.read_bytes()
    assert report["passed"] is True
    assert main(["--manifest", str(manifest), "--review", str(review), "--output", str(output)]) == 3
    assert output.read_bytes() == original


def test_cli_returns_blocked_code_for_missing_review(tmp_path: Path) -> None:
    manifest, _, _ = fixture(tmp_path)
    output = tmp_path / "pending_gate.json"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["formal_release"] is False


def test_expected_group_arguments_are_strict_and_ordered() -> None:
    assert parse_expected_group_counts(["views=8", "detail_views=4"]) == {
        "views": 8,
        "detail_views": 4,
    }


@pytest.mark.parametrize(
    "value",
    [
        "views",
        "=12",
        "views=",
        "views=0",
        "views=-1",
        "views=1.5",
        "views=+1",
        "views=01",
        "view group=1",
        "views=1=2",
    ],
)
def test_expected_group_argument_rejects_malformed_or_nonpositive_count(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        parse_expected_group_counts([value])


def test_expected_group_argument_rejects_duplicate_name() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_expected_group_counts(["views=8", "views=8"])


def test_cli_expected_groups_are_passed_to_audit(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    output = tmp_path / "gate.json"

    result = main(
        [
            "--manifest",
            str(manifest),
            "--review",
            str(review),
            "--output",
            str(output),
            "--expected-group",
            "views=8",
            "--expected-group",
            "detail_views=4",
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["view_group_counts"] == {"views": 8, "detail_views": 4}


def test_cli_group_count_mismatch_blocks_gate(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    output = tmp_path / "gate.json"

    result = main(
        [
            "--manifest",
            str(manifest),
            "--review",
            str(review),
            "--output",
            str(output),
            "--expected-group",
            "views=7",
            "--expected-group",
            "detail_views=5",
        ]
    )

    assert result == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "manifest_view_group_contract_mismatch" in failure_codes(report)


def test_cli_invalid_expected_group_fails_before_writing_output(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    output = tmp_path / "gate.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--manifest",
                str(manifest),
                "--review",
                str(review),
                "--output",
                str(output),
                "--expected-group",
                "views=0",
            ]
        )

    assert error.value.code == 2
    assert not output.exists()


def test_cli_duplicate_expected_group_fails_before_writing_output(tmp_path: Path) -> None:
    manifest, review, _ = fixture(tmp_path)
    output = tmp_path / "gate.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--manifest",
                str(manifest),
                "--review",
                str(review),
                "--output",
                str(output),
                "--expected-group",
                "views=8",
                "--expected-group",
                "views=8",
            ]
        )

    assert error.value.code == 2
    assert not output.exists()


def test_review_template_is_complete_but_non_accepting(tmp_path: Path) -> None:
    manifest, _, _ = fixture(tmp_path)
    template = build_fixed_view_review_template(manifest)
    assert len(template["views"]) == 12
    assert all(item["decision"] is None for item in template["views"])
    assert template["reviewer"] == {"type": "human", "id": ""}
    assert template["formal_release"] is False
    assert template["delivery_allowed"] is False
    assert template["status"] == "pending_human_review"
    assert template["schema_version"] == "railway.fixed-view-review-signoff.v2"
    assert set(template["views"][0]) == {
        "comment",
        "decision",
        "group",
        "id",
        "image",
        "image_sha256",
        "title",
    }


def test_review_template_output_is_immutable(tmp_path: Path) -> None:
    manifest, _, _ = fixture(tmp_path)
    output = tmp_path / "review_template.json"
    write_fixed_view_review_template(output, manifest)
    original = output.read_bytes()
    try:
        write_fixed_view_review_template(output, manifest)
    except FileExistsError:
        pass
    else:
        raise AssertionError("template writer overwrote an existing output")
    assert output.read_bytes() == original


def test_malformed_identifiers_fail_without_crashing(tmp_path: Path) -> None:
    manifest_path, review_path, review = fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"][0]["id"] = {"not": "text"}
    write_json(manifest_path, manifest)
    review["views"][0]["id"] = ["not", "text"]
    write_json(review_path, review)
    report = audit_fixed_view_review(manifest_path, review_path)
    assert "manifest_view_id_missing" in failure_codes(report)
    assert "review_view_id_missing" in failure_codes(report)
