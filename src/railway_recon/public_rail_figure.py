"""Deterministic, explicitly exploratory figures for a public training sample.

The renderer never fits a model or selects parameters.  Labels are only used for
display after predictions and full-point evaluations have been supplied.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_MODES = ("height_prominence", "paired_geometry")
_COLORS = {
    "background": (250, 251, 253),
    "text": (31, 42, 57),
    "other": (135, 144, 156),
    "ignored": (210, 215, 221),
    "tp": (21, 136, 91),
    "fp": (201, 48, 65),
    "fn": (216, 133, 16),
}


def _class_ids(values: tuple[int, ...], name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else ''}tuple")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
        for value in values
    ):
        raise ValueError(f"{name} must contain non-negative integer class IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate class IDs")
    return tuple(sorted(int(value) for value in values))


def _display_label(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 120
        or any(character in value for character in ("/", "\\"))
        or not value.isprintable()
    ):
        raise ValueError(f"{name} must be a short printable label, not a path or URL")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 has only the fixed-size built-in font.
        return ImageFont.load_default()


def _validate_evaluation(
    evaluation: dict[str, Any], reference: np.ndarray, prediction: np.ndarray, ignored: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Refuse a plot caption whose full-point evaluation does not match its masks."""
    if not isinstance(evaluation, dict):
        raise TypeError("Every evaluation must be a dictionary")
    checked: dict[str, dict[str, Any]] = {}
    for scope, included in (
        ("all_points", np.ones(len(reference), dtype=bool)),
        ("annotated_only", ~ignored),
    ):
        reported = evaluation.get(scope)
        if not isinstance(reported, dict):
            raise TypeError(f"Evaluation must include the {scope} full-point scope")
        counts = {
            "tp": int(np.count_nonzero(included & reference & prediction)),
            "fp": int(np.count_nonzero(included & ~reference & prediction)),
            "fn": int(np.count_nonzero(included & reference & ~prediction)),
            "tn": int(np.count_nonzero(included & ~reference & ~prediction)),
            "evaluated_points": int(np.count_nonzero(included)),
        }
        for key, actual in counts.items():
            value = reported.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value != actual:
                raise ValueError(f"Evaluation {scope}.{key} does not match full-point masks")
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        metrics: dict[str, float | None] = {}
        for name, numerator, denominator in (
            ("precision", tp, tp + fp),
            ("recall", tp, tp + fn),
            ("f1", 2 * tp, 2 * tp + fp + fn),
            ("iou", tp, tp + fp + fn),
        ):
            value = reported.get(name)
            expected = numerator / denominator if denominator else None
            if expected is None:
                if value is not None:
                    raise ValueError(f"Undefined evaluation {scope}.{name} must be None")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError(f"Evaluation {scope}.{name} does not match full-point masks")
            metrics[name] = value
        checked[scope] = {**counts, **metrics}
    return checked


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_public_rail_comparison(
    xyz: np.ndarray,
    classification: np.ndarray,
    predictions: dict[str, np.ndarray],
    evaluations: dict[str, dict[str, Any]],
    output: Path,
    *,
    target_class_ids: tuple[int, ...],
    ignored_class_ids: tuple[int, ...] = (),
    source_label: str = "public training sample",
    max_display_points: int = 100000,
    target_label: str = "numeric reference target",
    mapping_status: str = "not_verified",
) -> dict[str, Any]:
    """Write one new PNG and return its JSON-safe display manifest.

    All three panels share the same deterministic uniform source-index subsample
    and full-cloud XY bounds.  Full-point metrics are supplied by the evaluator
    and checked here; they are never calculated from the display subsample.
    """
    if (
        not isinstance(xyz, np.ndarray)
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(xyz) == 0
        or xyz.dtype.kind not in "iuf"
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("xyz must be a non-empty, finite numeric ndarray with shape (N, 3)")
    if (
        not isinstance(classification, np.ndarray)
        or classification.shape != (len(xyz),)
        or classification.dtype.kind not in "iu"
        or np.any(classification < 0)
    ):
        raise ValueError("classification must be a non-negative integer ndarray of length N")
    targets = _class_ids(target_class_ids, "target_class_ids", allow_empty=False)
    ignored_ids = _class_ids(ignored_class_ids, "ignored_class_ids", allow_empty=True)
    if set(targets) & set(ignored_ids):
        raise ValueError("Target and ignored class IDs must be disjoint")
    if (
        isinstance(max_display_points, bool)
        or not isinstance(max_display_points, int)
        or max_display_points < 1
    ):
        raise ValueError("max_display_points must be a positive integer")
    _display_label(source_label, "source_label")
    _display_label(target_label, "target_label")
    if mapping_status not in ("not_verified", "verified_from_source"):
        raise ValueError("mapping_status must be not_verified or verified_from_source")
    if not isinstance(predictions, dict) or set(predictions) != set(_MODES):
        raise ValueError(f"Predictions must contain exactly these modes: {_MODES}")
    if not isinstance(evaluations, dict) or set(evaluations) != set(_MODES):
        raise ValueError(f"Evaluations must contain exactly these modes: {_MODES}")
    reference = np.isin(classification, targets)
    ignored = np.isin(classification, ignored_ids)
    verified: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in _MODES:
        prediction = predictions[mode]
        if (
            not isinstance(prediction, np.ndarray)
            or prediction.shape != (len(xyz),)
            or prediction.dtype != np.dtype(bool)
        ):
            raise ValueError(f"Prediction {mode} must be a boolean ndarray of length N")
        verified[mode] = _validate_evaluation(evaluations[mode], reference, prediction, ignored)
    image_path = Path(output)
    if image_path.suffix.lower() != ".png":
        raise ValueError("Comparison output must have a .png extension")
    if image_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison figure: {image_path}")

    displayed_count = min(len(xyz), max_display_points)
    indices = np.linspace(0, len(xyz) - 1, num=displayed_count, dtype=np.int64)
    minimum = np.min(xyz[:, :2], axis=0).astype(np.float64)
    maximum = np.max(xyz[:, :2], axis=0).astype(np.float64)
    span = maximum - minimum
    if not np.isfinite(span).all():
        raise ValueError("XY coordinate range is too large for a finite display projection")
    center = minimum + span / 2.0
    panel_width, margin, plot_width, plot_height = 600, 28, 544, 520
    plot_top = 260
    image = Image.new("RGB", (1800, 916), _COLORS["background"])
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(24), _font(17), _font(15)
    draw.text(
        (28, 20),
        "PUBLIC TRAIN sample: exploratory candidate extraction",
        font=title_font,
        fill=_COLORS["text"],
    )
    draw.text(
        (28, 54),
        "Not held-out evaluation; not absolute geometric accuracy",
        font=body_font,
        fill=_COLORS["fp"],
    )
    draw.text((28, 84), f"Source: {source_label}", font=body_font, fill=_COLORS["text"])
    mapping_description = (
        "semantic mapping NOT VERIFIED"
        if mapping_status == "not_verified"
        else "semantic mapping verified from source contract"
    )
    draw.text(
        (28, 112),
        f"Target: {target_label} (IDs {','.join(map(str, targets))}); {mapping_description}",
        font=body_font,
        fill=_COLORS["fp"] if mapping_status == "not_verified" else _COLORS["text"],
    )
    draw.text(
        (28, 142),
        f"Display: {displayed_count:,} / {len(xyz):,} points; "
        "same deterministic index sample in every panel; metrics use full point arrays.",
        font=body_font,
        fill=_COLORS["text"],
    )
    scale = min(
        (plot_width - 16) / max(float(span[0]), 1e-9),
        (plot_height - 16) / max(float(span[1]), 1e-9),
    )
    centered = xyz[indices, :2].astype(np.float64) - center
    pixel_x = np.rint(centered[:, 0] * scale + plot_width / 2.0).astype(int)
    pixel_y = np.rint(-centered[:, 1] * scale + plot_height / 2.0).astype(int)
    display_reference, display_ignored = reference[indices], ignored[indices]
    panel_specs: list[dict[str, Any]] = []
    for panel_index, mode in enumerate(("reference", *_MODES)):
        offset_x = panel_index * panel_width + margin
        heading = "Raw points + reference target" if mode == "reference" else mode
        draw.text((offset_x, 188), heading, font=body_font, fill=_COLORS["text"])
        if mode == "reference":
            draw.text(
                (offset_x, 219),
                f"Target labels: {int(reference.sum()):,}; ignored labels: {int(ignored.sum()):,}",
                font=small_font,
                fill=_COLORS["text"],
            )
            layers = [
                (~display_ignored, "other"),
                (display_ignored, "ignored"),
                (display_reference, "tp"),
            ]
        else:
            metric = verified[mode]["annotated_only"]
            draw.text(
                (offset_x, 218),
                f"Full annotated-only: TP {metric['tp']:,}  "
                f"FP {metric['fp']:,}  FN {metric['fn']:,}",
                font=small_font,
                fill=_COLORS["text"],
            )
            draw.text(
                (offset_x, 239),
                f"P {_metric(metric['precision'])}  "
                f"R {_metric(metric['recall'])}  F1 {_metric(metric['f1'])}  "
                f"IoU {_metric(metric['iou'])}",
                font=small_font,
                fill=_COLORS["text"],
            )
            prediction = predictions[mode][indices]
            included = ~display_ignored
            layers = [
                (included, "other"),
                (display_ignored, "ignored"),
                (included & display_reference & prediction, "tp"),
                (included & display_reference & ~prediction, "fn"),
                (included & ~display_reference & prediction, "fp"),
            ]
        pixels = np.full((plot_height + 1, plot_width + 1, 3), 255, dtype=np.uint8)
        for mask, color in layers:
            pixels[pixel_y[mask], pixel_x[mask]] = _COLORS[color]
        image.paste(Image.fromarray(pixels), (offset_x, plot_top))
        draw.rectangle(
            (offset_x, plot_top, offset_x + plot_width, plot_top + plot_height),
            outline=(182, 191, 203),
            width=1,
        )
        draw.text(
            (offset_x, plot_top + plot_height + 12),
            f"Full XY extent: {span[0]:.2f} x {span[1]:.2f} coordinate units; equal scale",
            font=small_font,
            fill=_COLORS["text"],
        )
        panel_specs.append(
            {"mode": mode, "metric_scope": None if mode == "reference" else "annotated_only"}
        )
    legend = (
        ("other", "Other reference classes"),
        ("ignored", "Ignored labels (gray)"),
        ("tp", "TP / reference target"),
        ("fp", "FP (red)"),
        ("fn", "FN (orange)"),
    )
    cursor_x = 28
    for color, label in legend:
        draw.rectangle((cursor_x, 840, cursor_x + 14, 854), fill=_COLORS[color])
        draw.text((cursor_x + 22, 838), label, font=small_font, fill=_COLORS["text"])
        cursor_x += int(draw.textlength(label, font=small_font)) + 54
    ignored_description = ",".join(map(str, ignored_ids)) or "none"
    draw.text(
        (28, 875),
        f"Ignored IDs: {ignored_description}. Ignored points are displayed, "
        "excluded only from annotated-only metrics; all-point metrics remain in the report.",
        font=small_font,
        fill=_COLORS["text"],
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    with image_path.open("xb") as stream:
        image.save(stream, format="PNG")
    return {
        "schema_version": "railway.public-rail-comparison-figure.v1",
        "visualization_role": "public_training_sample_exploratory_not_held_out",
        "source_label": source_label,
        "image_filename": image_path.name,
        "source_point_count": len(xyz),
        "display_point_count": displayed_count,
        "display_is_subsample": displayed_count < len(xyz),
        "sampling": "deterministic_uniform_source_index_linspace",
        "display_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "identical_display_indices_across_panels": True,
        "projection": "xy_equal_scale_full_cloud_bounds",
        "full_bounds_xy": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "target_class_ids": list(targets),
        "target_label": target_label,
        "mapping_status": mapping_status,
        "ignored_class_ids": list(ignored_ids),
        "metrics_use_full_point_arrays": True,
        "metrics": verified,
        "panels": panel_specs,
        "limitations": [
            "Public training sample only; not held-out evaluation or absolute geometric accuracy.",
            "A deterministic display subsample can omit rare classes and small errors.",
            "Semantic reference labels are not independent rail-centerline or connection truth.",
            "Ignored classes are shown in gray, not silently relabeled as negative truth.",
            "Both all-point and annotated-only evaluation scopes remain in the metrics report.",
        ],
    }
