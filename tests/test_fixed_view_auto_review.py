from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from railway_recon.fixed_view_auto_review import (
    audit_fixed_view_images,
    write_fixed_view_auto_review,
)
from railway_recon.io import sha256_file


def _render(path: Path, *, blank: bool = False, seed: int = 1) -> None:
    if blank:
        pixels = np.zeros((450, 800, 3), dtype=np.uint8)
    else:
        y, x = np.indices((450, 800))
        pixels = np.stack(
            (
                (x + seed * 17) % 256,
                (y * 2 + seed * 29) % 256,
                ((x // 8 + y // 8) % 2) * 180 + 40,
            ),
            axis=-1,
        ).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _manifest(tmp_path: Path, views: list[dict[str, object]]) -> Path:
    path = tmp_path / "fixed_views_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "test.fixed-views.v1",
                "release_id": "candidate-01",
                "formal_release": False,
                "review_status": "pending_human_review",
                "views": views,
            }
        ),
        encoding="utf-8",
    )
    return path


def _view(path: Path, identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "title": identifier.replace("_", " "),
        "image": path.name,
        "image_sha256": sha256_file(path),
        "review_status": "pending_human_review",
    }


def test_auto_review_checks_every_image_without_claiming_human_signoff(tmp_path: Path) -> None:
    image = tmp_path / "overview.png"
    _render(image)
    report = audit_fixed_view_images(_manifest(tmp_path, [_view(image, "overview")]))

    assert report["passed"] is True
    assert report["formal_release"] is False
    assert report["delivery_allowed"] is False
    assert report["summary"]["view_count"] == 1
    assert report["summary"]["automated_clear_count"] == 1
    assert report["policy"]["manual_review_required_for_internal_iteration"] is False
    assert report["policy"]["semantic_defects_are_not_claimed_as_automatically_accepted"]


def test_auto_review_routes_blank_render_to_attention_queue(tmp_path: Path) -> None:
    image = tmp_path / "canopy_gap.png"
    _render(image, blank=True)
    report = audit_fixed_view_images(_manifest(tmp_path, [_view(image, "canopy_gap")]))

    assert report["passed"] is True
    assert report["summary"]["warning_view_count"] == 1
    assert report["review_queue"][0]["priority"] == "attention"
    assert "near_blank_frame" in report["review_queue"][0]["reasons"]
    assert "mostly_black" in report["review_queue"][0]["reasons"]


def test_auto_review_hash_mismatch_is_a_hard_blocker(tmp_path: Path) -> None:
    image = tmp_path / "overview.png"
    _render(image)
    view = _view(image, "overview")
    view["image_sha256"] = "0" * 64
    report = audit_fixed_view_images(_manifest(tmp_path, [view]))

    assert report["passed"] is False
    assert report["summary"]["hard_blocker_count"] == 1
    assert report["hard_blockers"][0]["code"] == "image_hash_mismatch"


def test_auto_review_bounds_spot_checks_and_prioritizes_risky_views(tmp_path: Path) -> None:
    views = []
    for index, identifier in enumerate(
        ["overview", "platform_seam", "canopy_gap_exception", "track_boundary"]
    ):
        image = tmp_path / f"{identifier}.png"
        _render(image, seed=index + 1)
        views.append(_view(image, identifier))

    report = audit_fixed_view_images(
        _manifest(tmp_path, views), maximum_spot_checks=2
    )

    assert len(report["review_queue"]) == 2
    assert report["review_queue"][0]["id"] == "canopy_gap_exception"
    assert report["review_queue"][0]["priority"] == "spot_check"


def test_auto_review_rejects_image_path_escape(tmp_path: Path) -> None:
    image = tmp_path.parent / "outside.png"
    _render(image)
    manifest = _manifest(
        tmp_path,
        [
            {
                "id": "outside",
                "title": "outside",
                "image": "../outside.png",
                "image_sha256": sha256_file(image),
                "review_status": "pending_human_review",
            }
        ],
    )

    report = audit_fixed_view_images(manifest)

    assert report["passed"] is False
    assert report["hard_blockers"][0]["code"] == (
        "image_path_missing_or_outside_manifest_directory"
    )


def test_auto_review_output_is_immutable(tmp_path: Path) -> None:
    image = tmp_path / "overview.png"
    _render(image)
    manifest = _manifest(tmp_path, [_view(image, "overview")])
    output = tmp_path / "auto-review.json"

    write_fixed_view_auto_review(output, manifest)
    with pytest.raises(FileExistsError):
        write_fixed_view_auto_review(output, manifest)
