"""Automated triage for fixed-view render manifests.

This module removes repetitive image-integrity work from solo iteration.  It
checks every rendered image, computes lightweight diagnostics, and produces a
small risk-ranked spot-check queue.  It deliberately does not impersonate a
human sign-off or authorize delivery/formal release.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io import load_json, sha256_file

AUTO_REVIEW_SCHEMA_VERSION = "railway.fixed-view-auto-review.v1"
RISK_KEYWORDS = {
    "access": 8,
    "boundary": 7,
    "canopy": 10,
    "catenary": 9,
    "column": 7,
    "end": 5,
    "exception": 12,
    "facade": 6,
    "gap": 12,
    "interface": 10,
    "platform": 7,
    "positioner": 8,
    "rail": 9,
    "refit": 9,
    "roof": 8,
    "seam": 11,
    "sleeper": 6,
    "stair": 8,
    "track": 8,
}


def _manifest_views(
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    """Collect view-like rows from legacy and current top-level groups."""

    rows: list[tuple[str, dict[str, Any]]] = []
    malformed: list[dict[str, Any]] = []
    for group, value in manifest.items():
        if not isinstance(value, list):
            continue
        looks_like_group = any(
            isinstance(item, dict)
            and bool(set(item) & {"id", "image", "image_sha256", "review_status"})
            for item in value
        )
        if not looks_like_group:
            continue
        for index, item in enumerate(value):
            if isinstance(item, dict):
                rows.append((group, item))
            else:
                malformed.append(
                    {"code": "manifest_view_row_malformed", "group": group, "index": index}
                )
    return rows, malformed


def _safe_image_path(manifest_directory: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return None
    resolved = (manifest_directory / candidate).resolve()
    try:
        resolved.relative_to(manifest_directory)
    except ValueError:
        return None
    return resolved


def _image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        width, height = source.size
        rgb = source.convert("RGB")
        rgb.thumbnail((512, 512))
        pixels = np.asarray(rgb, dtype=np.float32)

    luminance = (
        0.2126 * pixels[:, :, 0]
        + 0.7152 * pixels[:, :, 1]
        + 0.0722 * pixels[:, :, 2]
    )
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    detail_energy = float(
        (horizontal.mean() if horizontal.size else 0.0)
        + (vertical.mean() if vertical.size else 0.0)
    ) / 2.0
    return {
        "width": width,
        "height": height,
        "luminance_mean": round(float(luminance.mean()), 3),
        "luminance_std": round(float(luminance.std()), 3),
        "detail_energy": round(detail_energy, 3),
        "dark_clip_ratio": round(float(np.mean(luminance <= 3.0)), 6),
        "bright_clip_ratio": round(float(np.mean(luminance >= 252.0)), 6),
    }


def _diagnostic_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if metrics["width"] < 640 or metrics["height"] < 360:
        warnings.append("low_resolution")
    if metrics["luminance_std"] < 2.0:
        warnings.append("near_blank_frame")
    elif metrics["luminance_std"] < 8.0:
        warnings.append("low_contrast")
    if metrics["detail_energy"] < 0.5:
        warnings.append("very_low_detail")
    if metrics["dark_clip_ratio"] > 0.95:
        warnings.append("mostly_black")
    if metrics["bright_clip_ratio"] > 0.95:
        warnings.append("mostly_white")
    return warnings


def _semantic_risk(group: str, view: dict[str, Any]) -> tuple[int, list[str]]:
    text = " ".join(
        str(value).lower()
        for value in (group, view.get("id"), view.get("title"))
        if value is not None
    )
    matched = sorted(keyword for keyword in RISK_KEYWORDS if keyword in text)
    score = sum(RISK_KEYWORDS[keyword] for keyword in matched)
    if view.get("hidden_for_diagnostic"):
        score += 6
        matched.append("diagnostic_visibility_override")
    if view.get("only_visible_tokens"):
        score += 6
        matched.append("isolated_asset_view")
    return score, matched


def audit_fixed_view_images(
    manifest_path: str | Path,
    *,
    maximum_spot_checks: int = 6,
) -> dict[str, Any]:
    """Check every image and return a bounded, risk-ranked manual queue."""

    if maximum_spot_checks < 0:
        raise ValueError("maximum_spot_checks must be non-negative")

    manifest_file = Path(manifest_path).resolve()
    manifest = load_json(manifest_file)
    manifest_directory = manifest_file.parent
    grouped_views, malformed = _manifest_views(manifest)
    hard_blockers: list[dict[str, Any]] = list(malformed)
    results: list[dict[str, Any]] = []
    identifiers: list[str] = []
    image_values: list[str] = []

    for group, view in grouped_views:
        identifier = view.get("id")
        image_value = view.get("image")
        if isinstance(identifier, str) and identifier.strip():
            identifiers.append(identifier)
        else:
            hard_blockers.append({"code": "view_id_missing", "group": group})
            identifier = f"{group}:missing-id"
        if isinstance(image_value, str) and image_value.strip():
            image_values.append(image_value)

        result: dict[str, Any] = {
            "id": identifier,
            "group": group,
            "title": view.get("title"),
            "image": image_value,
            "status": "hard_blocker",
            "warnings": [],
        }
        image_path = _safe_image_path(manifest_directory, image_value)
        if image_path is None:
            hard_blockers.append(
                {
                    "code": "image_path_missing_or_outside_manifest_directory",
                    "view_id": identifier,
                    "image": image_value,
                }
            )
            results.append(result)
            continue
        if not image_path.is_file():
            hard_blockers.append(
                {"code": "image_missing", "view_id": identifier, "image": image_value}
            )
            results.append(result)
            continue

        expected_hash = view.get("image_sha256")
        actual_hash = sha256_file(image_path)
        result["image_sha256"] = actual_hash
        if not isinstance(expected_hash, str) or not expected_hash.strip():
            hard_blockers.append({"code": "image_hash_missing", "view_id": identifier})
        elif actual_hash != expected_hash:
            hard_blockers.append(
                {
                    "code": "image_hash_mismatch",
                    "view_id": identifier,
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )

        try:
            metrics = _image_metrics(image_path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            hard_blockers.append(
                {"code": "image_decode_failed", "view_id": identifier, "error": str(error)}
            )
            results.append(result)
            continue

        warnings = _diagnostic_warnings(metrics)
        semantic_score, semantic_reasons = _semantic_risk(group, view)
        result.update(
            {
                "status": "warning" if warnings else "automated_clear",
                "warnings": warnings,
                "metrics": metrics,
                "semantic_risk_score": semantic_score,
                "semantic_risk_reasons": semantic_reasons,
            }
        )
        results.append(result)

    for value, code in ((identifiers, "view_id_duplicate"), (image_values, "image_path_duplicate")):
        duplicates = sorted(key for key, count in Counter(value).items() if count > 1)
        if duplicates:
            hard_blockers.append({"code": code, "values": duplicates})
    if not grouped_views:
        hard_blockers.append({"code": "manifest_contains_no_fixed_views"})

    queue_candidates = [row for row in results if row["status"] != "hard_blocker"]
    for row in queue_candidates:
        warning_score = 100 * len(row["warnings"])
        row["review_priority_score"] = warning_score + row["semantic_risk_score"]
    queue_candidates.sort(
        key=lambda row: (-row["review_priority_score"], str(row["id"]))
    )
    review_queue = [
        {
            "id": row["id"],
            "group": row["group"],
            "title": row["title"],
            "image": row["image"],
            "priority": "attention" if row["warnings"] else "spot_check",
            "priority_score": row["review_priority_score"],
            "reasons": row["warnings"] or row["semantic_risk_reasons"],
        }
        for row in queue_candidates[:maximum_spot_checks]
    ]
    warning_count = sum(row["status"] == "warning" for row in results)
    clear_count = sum(row["status"] == "automated_clear" for row in results)
    passed = not hard_blockers
    return {
        "schema_version": AUTO_REVIEW_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "release_id": manifest.get("release_id"),
        "formal_release": False,
        "delivery_allowed": False,
        "passed": passed,
        "automated_integrity_clear": passed,
        "status": "ready_for_internal_iteration" if passed else "blocked_broken_render_set",
        "render_manifest": str(manifest_file),
        "render_manifest_sha256": sha256_file(manifest_file),
        "summary": {
            "view_count": len(grouped_views),
            "automated_clear_count": clear_count,
            "warning_view_count": warning_count,
            "hard_blocker_count": len(hard_blockers),
            "recommended_manual_spot_check_count": len(review_queue),
        },
        "hard_blockers": hard_blockers,
        "review_queue": review_queue,
        "views": results,
        "policy": {
            "manual_review_required_for_internal_iteration": False,
            "internal_manual_scope": "review_queue_only_when_owner_wants_a_visual_spot_check",
            "all_images_hash_and_decode_checked": True,
            "semantic_defects_are_not_claimed_as_automatically_accepted": True,
            "formal_release_requires_explicit_owner_confirmation": True,
        },
    }


def write_fixed_view_auto_review(
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    maximum_spot_checks: int = 6,
) -> dict[str, Any]:
    report = audit_fixed_view_images(
        manifest_path, maximum_spot_checks=maximum_spot_checks
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automate fixed-view integrity checks and create a bounded spot-check queue"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-spot-checks", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        report = write_fixed_view_auto_review(
            args.output,
            args.manifest,
            maximum_spot_checks=args.maximum_spot_checks,
        )
    except FileExistsError:
        print(f"Refusing to overwrite auto review: {args.output.resolve()}", file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
