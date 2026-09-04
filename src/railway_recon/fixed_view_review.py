"""Fail-closed validation for human review of rendered fixed views.

Rendering an image is not visual acceptance.  This module keeps those two
events separate by validating an immutable render manifest against a distinct
human sign-off record.  A successful result is still only a review gate; it
never promotes a candidate or declares a formal release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file

ALLOWED_DECISIONS = {"pass", "fail"}
REVIEW_SCHEMA_VERSION = "railway.fixed-view-review-signoff.v2"
COMPLETED_REVIEW_STATUS = "candidate_human_review_completed"
PENDING_REVIEW_STATUS = "pending_human_review"
VIEW_REQUIRED_FIELDS = {
    "id",
    "title",
    "image",
    "image_sha256",
    "review_status",
}
FORMAL_STATUS_TOKENS = {"accepted", "approved", "final", "formal", "released"}
EXPECTED_GROUP_ARGUMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)=([1-9][0-9]*)")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_expected_group_counts(values: Sequence[str]) -> dict[str, int]:
    """Parse repeated ``NAME=COUNT`` values without silently replacing a key."""

    result: dict[str, int] = {}
    for value in values:
        if not isinstance(value, str) or EXPECTED_GROUP_ARGUMENT.fullmatch(value) is None:
            raise ValueError(
                "--expected-group must be NAME=COUNT with a positive integer COUNT "
                "and a NAME containing only letters, digits, '_', '.', or '-'"
            )
        name, count_text = value.split("=", maxsplit=1)
        if name in result:
            raise ValueError(f"Duplicate --expected-group name: {name}")
        result[name] = int(count_text)
    return result


def _timezone_aware_iso8601(value: Any) -> bool:
    if not _nonempty_text(value):
        return False
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _parse_timezone_aware_iso8601(value: Any) -> datetime | None:
    if not _timezone_aware_iso8601(value):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _status_tokens(value: Any) -> set[str]:
    return set(filter(None, re.split(r"[^a-z0-9]+", str(value or "").lower())))


def _state_pollution(value: Any, *, path: str = "review") -> list[str]:
    polluted: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized_key = str(key).strip().lower()
            if normalized_key in {
                "acceptance_status",
                "evidence_status",
                "release_status",
                "review_status",
                "status",
            } and _status_tokens(child) & FORMAL_STATUS_TOKENS:
                polluted.append(child_path)
            if normalized_key == "formal_release" and child is not False:
                polluted.append(child_path)
            if normalized_key == "delivery_allowed" and child is not False:
                polluted.append(child_path)
            polluted.extend(_state_pollution(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            polluted.extend(_state_pollution(child, path=f"{path}[{index}]"))
    return polluted


def _looks_like_view_group(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, Mapping)
        and bool(set(item) & {"id", "image", "image_sha256", "review_status", "title"})
        for item in value
    )


def _manifest_views(
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any]]], list[str], list[dict[str, Any]]]:
    """Collect legacy top-level view groups without hiding malformed rows."""

    views: list[tuple[str, dict[str, Any]]] = []
    groups: list[str] = []
    malformed: list[dict[str, Any]] = []
    for key, value in manifest.items():
        if not _looks_like_view_group(value):
            continue
        groups.append(key)
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                malformed.append({"group": key, "index": index, "reason": "not_object"})
                continue
            views.append((key, item))
    return views, groups, malformed


def audit_fixed_view_review(
    manifest_path: str | Path,
    review_path: str | Path | None = None,
    *,
    minimum_view_count: int = 12,
    expected_group_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Audit render integrity and an optional, separate human sign-off record.

    The returned ``passed`` value means that all bound images received an
    explicit human ``pass`` decision.  It does not imply release acceptance.
    """

    if minimum_view_count < 1:
        raise ValueError("minimum_view_count must be positive")

    manifest_file = Path(manifest_path).resolve()
    manifest = load_json(manifest_file)
    grouped_views, groups, malformed_manifest_rows = _manifest_views(manifest)
    views = [item for _, item in grouped_views]
    failures: list[dict[str, Any]] = []

    def fail(code: str, **details: Any) -> None:
        failures.append({"code": code, **details})

    release_id = manifest.get("release_id")
    if not _nonempty_text(manifest.get("schema_version")):
        fail("manifest_schema_version_missing")
    if not _nonempty_text(release_id):
        fail("manifest_release_id_missing")
    manifest_generated_at = _parse_timezone_aware_iso8601(manifest.get("generated_at"))
    if manifest_generated_at is None:
        fail("manifest_timestamp_missing_or_not_timezone_aware")
    if manifest.get("formal_release") is not False:
        fail("manifest_formal_release_not_false")
    if manifest.get("review_status") != PENDING_REVIEW_STATUS:
        fail(
            "manifest_review_status_not_pending",
            actual=manifest.get("review_status"),
            expected=PENDING_REVIEW_STATUS,
        )
    if polluted := _state_pollution(manifest, path="manifest"):
        fail("manifest_state_pollution", fields=polluted)
    if malformed_manifest_rows:
        fail("manifest_view_row_malformed", rows=malformed_manifest_rows)

    group_counts = Counter(group for group, _ in grouped_views)
    if expected_group_counts is not None:
        expected = dict(expected_group_counts)
        if any(not _nonempty_text(key) for key in expected):
            raise ValueError("expected_group_counts keys must be non-empty strings")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in expected.values()
        ):
            raise ValueError("expected_group_counts values must be positive integers")
        missing_groups = sorted(set(expected) - set(groups))
        unknown_groups = sorted(set(groups) - set(expected))
        count_mismatches = {
            group: {"expected": expected[group], "actual": group_counts.get(group, 0)}
            for group in sorted(set(expected) & set(groups))
            if expected[group] != group_counts.get(group, 0)
        }
        if missing_groups or unknown_groups or count_mismatches:
            fail(
                "manifest_view_group_contract_mismatch",
                missing_groups=missing_groups,
                unknown_groups=unknown_groups,
                count_mismatches=count_mismatches,
            )

    identifiers = [item.get("id") for item in views]
    images = [item.get("image") for item in views]
    valid_identifiers = [value for value in identifiers if _nonempty_text(value)]
    valid_images = [value for value in images if _nonempty_text(value)]
    if len(views) < minimum_view_count:
        fail(
            "manifest_view_count_below_minimum",
            actual=len(views),
            minimum=minimum_view_count,
        )
    if any(not _nonempty_text(value) for value in identifiers):
        fail("manifest_view_id_missing")
    if len(set(valid_identifiers)) != len(valid_identifiers):
        fail("manifest_view_id_duplicate")
    if any(not _nonempty_text(value) for value in images):
        fail("manifest_image_path_missing")
    if len(set(valid_images)) != len(valid_images):
        fail("manifest_image_path_duplicate")

    source_bindings = (
        ("source_obj", "source_obj_sha256"),
        ("source_registry", "source_registry_sha256"),
    )
    for path_key, hash_key in source_bindings:
        value = manifest.get(path_key)
        expected = manifest.get(hash_key)
        if not _nonempty_text(value) or not _nonempty_text(expected):
            fail("manifest_source_binding_missing", path_field=path_key, hash_field=hash_key)
            continue
        source = Path(value).resolve()
        if not source.is_file():
            fail("manifest_source_missing", field=path_key, path=str(source))
        elif sha256_file(source) != expected:
            fail("manifest_source_hash_mismatch", field=path_key, path=str(source))

    manifest_by_id: dict[str, dict[str, Any]] = {}
    manifest_group_by_id: dict[str, str] = {}
    manifest_root = manifest_file.parent
    for group, item in grouped_views:
        identifier = item.get("id")
        image_value = item.get("image")
        expected_hash = item.get("image_sha256")
        missing_fields = sorted(VIEW_REQUIRED_FIELDS - set(item))
        if missing_fields:
            fail(
                "manifest_view_fields_missing",
                view_id=identifier,
                group=group,
                missing_fields=missing_fields,
            )
        if not _nonempty_text(item.get("title")):
            fail("manifest_view_title_missing", view_id=identifier, group=group)
        if item.get("review_status") != PENDING_REVIEW_STATUS:
            fail(
                "manifest_view_status_not_pending",
                view_id=identifier,
                group=group,
                actual=item.get("review_status"),
            )
        if not _nonempty_text(identifier) or not _nonempty_text(image_value):
            continue
        manifest_by_id[identifier] = item
        manifest_group_by_id[identifier] = group
        if not _nonempty_text(expected_hash):
            fail("manifest_image_hash_missing", view_id=identifier)
            continue
        image_path = Path(image_value)
        image = (manifest_root / image_path).resolve()
        if image_path.is_absolute() or not image.is_relative_to(manifest_root):
            fail(
                "manifest_image_path_outside_manifest_directory",
                view_id=identifier,
                path=str(image),
            )
            continue
        if not image.is_file():
            fail("manifest_image_missing", view_id=identifier, path=str(image))
        elif sha256_file(image) != expected_hash:
            fail("manifest_image_hash_mismatch", view_id=identifier, path=str(image))

    review: dict[str, Any] | None = None
    reviewed_count = 0
    passed_count = 0
    failed_count = 0
    if review_path is None:
        fail("human_review_record_missing")
    else:
        review_file = Path(review_path).resolve()
        if not review_file.is_file():
            fail("human_review_record_missing", path=str(review_file))
        else:
            review = load_json(review_file)
            if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
                fail(
                    "review_schema_version_mismatch",
                    actual=review.get("schema_version"),
                    expected=REVIEW_SCHEMA_VERSION,
                )
            if review.get("release_id") != release_id:
                fail("review_release_id_mismatch")
            for hash_key in ("source_obj_sha256", "source_registry_sha256"):
                if review.get(hash_key) != manifest.get(hash_key):
                    fail("review_source_hash_mismatch", field=hash_key)
            actual_manifest_hash = sha256_file(manifest_file)
            if review.get("render_manifest_sha256") != actual_manifest_hash:
                fail("review_manifest_hash_mismatch")
            if review.get("formal_release") is not False:
                fail("review_formal_release_not_false")
            if review.get("delivery_allowed") is not False:
                fail("review_delivery_allowed_not_false")
            if review.get("status") != COMPLETED_REVIEW_STATUS:
                fail(
                    "review_status_not_completed_candidate",
                    actual=review.get("status"),
                    expected=COMPLETED_REVIEW_STATUS,
                )
            if polluted := _state_pollution(review):
                fail("review_state_pollution", fields=polluted)
            review_groups = review.get("view_groups")
            if review_groups != groups:
                fail(
                    "review_view_groups_mismatch",
                    expected=groups,
                    actual=review_groups,
                )

            reviewer = review.get("reviewer")
            if not isinstance(reviewer, dict) or reviewer.get("type") != "human":
                fail("reviewer_is_not_explicitly_human")
            elif not _nonempty_text(reviewer.get("id")):
                fail("reviewer_id_missing")
            reviewed_at = _parse_timezone_aware_iso8601(review.get("reviewed_at"))
            if reviewed_at is None:
                fail("review_timestamp_missing_or_not_timezone_aware")
            elif manifest_generated_at is not None and reviewed_at < manifest_generated_at:
                fail("review_timestamp_precedes_manifest")

            review_views = review.get("views")
            if not isinstance(review_views, Sequence) or isinstance(review_views, (str, bytes)):
                fail("review_views_missing")
                review_views = []
            review_ids = [item.get("id") for item in review_views if isinstance(item, dict)]
            valid_review_ids = [value for value in review_ids if _nonempty_text(value)]
            if len(review_ids) != len(review_views) or any(
                not _nonempty_text(value) for value in review_ids
            ):
                fail("review_view_id_missing")
            if len(set(valid_review_ids)) != len(valid_review_ids):
                fail("review_view_id_duplicate")
            expected_ids = set(manifest_by_id)
            actual_ids = set(valid_review_ids)
            if missing := sorted(expected_ids - actual_ids):
                fail("review_views_incomplete", missing_view_ids=missing)
            if extra := sorted(actual_ids - expected_ids):
                fail("review_contains_unknown_views", unknown_view_ids=extra)

            for item in review_views:
                if not isinstance(item, dict):
                    continue
                identifier = item.get("id")
                if not _nonempty_text(identifier) or identifier not in manifest_by_id:
                    continue
                reviewed_count += 1
                required_fields = {
                    "comment",
                    "decision",
                    "group",
                    "id",
                    "image",
                    "image_sha256",
                    "title",
                }
                missing_fields = sorted(required_fields - set(item))
                if missing_fields:
                    fail(
                        "review_view_fields_missing",
                        view_id=identifier,
                        missing_fields=missing_fields,
                    )
                expected_view = manifest_by_id[identifier]
                bindings = {
                    "group": manifest_group_by_id[identifier],
                    "title": expected_view.get("title"),
                    "image": expected_view.get("image"),
                    "image_sha256": expected_view.get("image_sha256"),
                }
                mismatched_bindings = [
                    field for field, expected in bindings.items() if item.get(field) != expected
                ]
                if mismatched_bindings:
                    fail(
                        "review_view_binding_mismatch",
                        view_id=identifier,
                        fields=mismatched_bindings,
                    )
                if item.get("image_sha256") != expected_view.get("image_sha256"):
                    fail("review_image_hash_mismatch", view_id=identifier)
                if not isinstance(item.get("comment"), str):
                    fail("review_view_comment_not_text", view_id=identifier)
                decision = item.get("decision")
                if decision not in ALLOWED_DECISIONS:
                    fail(
                        "review_decision_not_explicit_pass_or_fail",
                        view_id=identifier,
                        decision=decision,
                    )
                    continue
                if decision == "pass":
                    passed_count += 1
                else:
                    failed_count += 1
                    if not _nonempty_text(item.get("comment")):
                        fail("failed_review_view_comment_missing", view_id=identifier)
                    fail("human_review_view_failed", view_id=identifier)

    if not failures:
        status = "technical_pass_human_signoff_complete"
    elif failed_count:
        status = "blocked_human_review_failed"
    else:
        status = "blocked_pending_or_invalid_human_review"
    return {
        "schema_version": "railway.fixed-view-review-gate.v1",
        "release_id": release_id,
        "formal_release": False,
        "passed": not failures,
        "status": status,
        "render_manifest": str(manifest_file),
        "render_manifest_sha256": sha256_file(manifest_file),
        "review_record": str(Path(review_path).resolve()) if review_path is not None else None,
        "view_groups": groups,
        "view_group_counts": dict(group_counts),
        "view_count": len(views),
        "reviewed_count": reviewed_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "failures": failures,
        "policy": (
            "A rendered view is not an accepted view. This gate requires one explicit human "
            "decision per hash-bound image and never declares a formal release."
        ),
    }


def build_fixed_view_review_template(manifest_path: str | Path) -> dict[str, Any]:
    """Build a non-accepting review form bound to every rendered image."""

    manifest_file = Path(manifest_path).resolve()
    manifest = load_json(manifest_file)
    grouped_views, groups, _ = _manifest_views(manifest)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "release_id": manifest.get("release_id"),
        "source_obj_sha256": manifest.get("source_obj_sha256"),
        "source_registry_sha256": manifest.get("source_registry_sha256"),
        "render_manifest_sha256": sha256_file(manifest_file),
        "formal_release": False,
        "delivery_allowed": False,
        "status": PENDING_REVIEW_STATUS,
        "reviewer": {"type": "human", "id": ""},
        "reviewed_at": "",
        "view_groups": groups,
        "views": [
            {
                "group": group,
                "id": item.get("id"),
                "title": item.get("title"),
                "image": item.get("image"),
                "image_sha256": item.get("image_sha256"),
                "decision": None,
                "comment": "",
            }
            for group, item in grouped_views
        ],
        "instructions": (
            "Inspect every bound image. Set every decision to pass or fail, retain all "
            "group/title/image/hash bindings, identify the human reviewer, provide a "
            "timezone-aware reviewed_at value, and set status to "
            f"{COMPLETED_REVIEW_STATUS!r}. This form cannot promote, deliver, or formally "
            "release a candidate."
        ),
    }


def write_fixed_view_review_template(
    output_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Write one immutable, initially non-accepting review form."""

    template = build_fixed_view_review_template(manifest_path)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return template


def write_fixed_view_review_gate(
    output_path: str | Path,
    manifest_path: str | Path,
    review_path: str | Path | None = None,
    *,
    minimum_view_count: int = 12,
    expected_group_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write one immutable gate report and return its value."""

    report = audit_fixed_view_review(
        manifest_path,
        review_path,
        minimum_view_count=minimum_view_count,
        expected_group_counts=expected_group_counts,
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of hash-bound human fixed-view review"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-view-count", type=int, default=12)
    parser.add_argument(
        "--expected-group",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Require an exact manifest view-group count; repeat once per group.",
    )
    args = parser.parse_args(argv)
    try:
        expected_group_counts = parse_expected_group_counts(args.expected_group)
    except ValueError as error:
        parser.error(str(error))
    try:
        report = write_fixed_view_review_gate(
            args.output,
            args.manifest,
            args.review,
            minimum_view_count=args.minimum_view_count,
            expected_group_counts=expected_group_counts or None,
        )
    except FileExistsError:
        print(f"Refusing to overwrite fixed-view gate: {args.output.resolve()}", file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
