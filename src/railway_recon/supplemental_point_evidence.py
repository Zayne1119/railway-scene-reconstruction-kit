from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import write_json


@dataclass(frozen=True)
class GridBounds:
    minimum_x: float
    minimum_y: float
    maximum_x: float
    maximum_y: float

    @classmethod
    def from_las(cls, path: str | Path) -> GridBounds:
        with laspy.open(Path(path)) as reader:
            minimum = reader.header.mins
            maximum = reader.header.maxs
        return cls(
            minimum_x=float(minimum[0]),
            minimum_y=float(minimum[1]),
            maximum_x=float(maximum[0]),
            maximum_y=float(maximum[1]),
        )

    def grid_shape(self, cell_size_m: float) -> tuple[int, int]:
        width = max(1, int(np.ceil((self.maximum_x - self.minimum_x) / cell_size_m)))
        height = max(1, int(np.ceil((self.maximum_y - self.minimum_y) / cell_size_m)))
        return height, width

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x >= self.minimum_x)
            & (x <= self.maximum_x)
            & (y >= self.minimum_y)
            & (y <= self.maximum_y)
        )

    def to_json(self) -> dict[str, list[float]]:
        return {
            "minimum_xy": [self.minimum_x, self.minimum_y],
            "maximum_xy": [self.maximum_x, self.maximum_y],
        }


def accumulate_density(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounds: GridBounds,
    cell_size_m: float,
) -> int:
    inside = bounds.contains(x, y)
    if not np.any(inside):
        return 0
    x_inside = x[inside]
    y_inside = y[inside]
    column = np.floor((x_inside - bounds.minimum_x) / cell_size_m).astype(np.int64)
    row = np.floor((y_inside - bounds.minimum_y) / cell_size_m).astype(np.int64)
    np.clip(column, 0, grid.shape[1] - 1, out=column)
    np.clip(row, 0, grid.shape[0] - 1, out=row)
    counts = np.bincount(row * grid.shape[1] + column, minlength=grid.size)
    grid += counts.reshape(grid.shape).astype(grid.dtype, copy=False)
    return len(x_inside)


def _header_summary(path: Path) -> dict[str, Any]:
    with laspy.open(path) as reader:
        header = reader.header
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "point_count": int(header.point_count),
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "minimum_xyz": [float(value) for value in header.mins],
            "maximum_xyz": [float(value) for value in header.maxs],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
        }


def _grid_stats(grid: np.ndarray, cell_size_m: float) -> dict[str, Any]:
    occupied = grid[grid > 0]
    total_cells = int(grid.size)
    occupied_count = len(occupied)
    return {
        "point_count": int(np.sum(grid, dtype=np.int64)),
        "cell_size_m": cell_size_m,
        "cell_area_m2": cell_size_m**2,
        "grid_shape_yx": [int(grid.shape[0]), int(grid.shape[1])],
        "occupied_cell_count": occupied_count,
        "occupied_fraction": occupied_count / total_cells,
        "occupied_area_m2": occupied_count * cell_size_m**2,
        "points_per_occupied_cell": {
            "p50": float(np.percentile(occupied, 50)) if occupied_count else 0.0,
            "p90": float(np.percentile(occupied, 90)) if occupied_count else 0.0,
            "p95": float(np.percentile(occupied, 95)) if occupied_count else 0.0,
            "p99": float(np.percentile(occupied, 99)) if occupied_count else 0.0,
            "maximum": int(np.max(occupied)) if occupied_count else 0,
        },
    }


def scan_density(
    path: str | Path,
    bounds: GridBounds,
    *,
    cell_size_m: float = 0.25,
    z_bands: Iterable[tuple[str, float, float]] = (),
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    source = Path(path).resolve()
    shape = bounds.grid_shape(cell_size_m)
    density = np.zeros(shape, dtype=np.uint64)
    bands = {
        name: np.zeros(shape, dtype=np.uint64)
        for name, _, _ in z_bands
    }
    scanned = 0
    retained = 0
    band_counts = {name: 0 for name in bands}
    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            scanned += len(x)
            inside = bounds.contains(x, y)
            retained += accumulate_density(density, x, y, bounds, cell_size_m)
            if not np.any(inside):
                continue
            for name, minimum_z, maximum_z in z_bands:
                selected = inside & (z >= minimum_z) & (z < maximum_z)
                band_counts[name] += accumulate_density(
                    bands[name], x[selected], y[selected], bounds, cell_size_m
                )
    report = {
        "source": _header_summary(source),
        "scanned_point_count": scanned,
        "retained_point_count": retained,
        "density": _grid_stats(density, cell_size_m),
        "z_band_point_counts": band_counts,
    }
    return density, bands, report


def _colorize(values: np.ndarray, *, palette: str) -> Image.Image:
    positive = values[values > 0]
    scale = float(np.percentile(np.log1p(positive), 99)) if len(positive) else 1.0
    normalized = np.clip(np.log1p(values) / max(scale, 1e-12), 0.0, 1.0)
    if palette == "cyan":
        rgb = np.stack(
            (
                12 + 30 * normalized,
                24 + 190 * normalized,
                34 + 220 * normalized,
            ),
            axis=-1,
        )
    elif palette == "lime":
        rgb = np.stack(
            (
                8 + 155 * normalized,
                22 + 230 * normalized,
                28 + 65 * normalized,
            ),
            axis=-1,
        )
    elif palette == "orange":
        rgb = np.stack(
            (
                16 + 239 * normalized,
                22 + 110 * normalized,
                30 + 20 * normalized,
            ),
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported palette: {palette}")
    return Image.fromarray(rgb.astype(np.uint8))


def render_density_comparison(
    old_density: np.ndarray,
    new_density: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Supplemental point evidence / 150 m review",
) -> Path:
    if old_density.shape != new_density.shape:
        raise ValueError("Density grid shapes differ")
    positive_delta = np.maximum(
        new_density.astype(np.int64) - old_density.astype(np.int64), 0
    )
    panels = [
        ("OLD TILE 5 / CLIPPED", old_density, "cyan"),
        ("NEW MERGED V2 / CLIP", new_density, "lime"),
        ("ADDED EVIDENCE (NEW - OLD)", positive_delta, "orange"),
    ]
    target_height = 760
    margin = 28
    header_height = 86
    label_height = 42
    source_height, source_width = old_density.shape
    panel_width = max(260, int(target_height * source_width / max(source_height, 1)))
    canvas = Image.new(
        "RGB",
        (margin * 4 + panel_width * 3, header_height + target_height + label_height + margin),
        (4, 18, 26),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, 24), title, fill=(224, 242, 247), font=font)
    draw.text(
        (margin, 48),
        "Log-density view; orange marks point support that was absent from the old tile.",
        fill=(125, 168, 180),
        font=font,
    )
    for index, (label, values, palette) in enumerate(panels):
        panel = _colorize(np.flipud(values), palette=palette).resize(
            (panel_width, target_height), Image.Resampling.NEAREST
        )
        x = margin + index * (panel_width + margin)
        canvas.paste(panel, (x, header_height))
        draw.rectangle(
            (x, header_height, x + panel_width - 1, header_height + target_height - 1),
            outline=(35, 91, 108),
            width=1,
        )
        draw.text(
            (x + 8, header_height + target_height + 13),
            label,
            fill=(178, 255, 65) if index == 1 else (207, 227, 232),
            font=font,
        )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def compare_supplemental_density(
    old_path: str | Path,
    new_clip_path: str | Path,
    output_directory: str | Path,
    *,
    cell_size_m: float = 0.25,
    z_bands: Iterable[tuple[str, float, float]] = (
        ("track_and_ground", 19.0, 22.0),
        ("platform_and_facade", 22.0, 26.0),
        ("canopy_and_overhead", 26.0, 31.0),
    ),
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bounds = GridBounds.from_las(new_clip_path)
    old_density, old_bands, old_report = scan_density(
        old_path, bounds, cell_size_m=cell_size_m, z_bands=z_bands
    )
    new_density, new_bands, new_report = scan_density(
        new_clip_path, bounds, cell_size_m=cell_size_m, z_bands=z_bands
    )
    old_count = int(np.sum(old_density, dtype=np.int64))
    new_count = int(np.sum(new_density, dtype=np.int64))
    delta = new_density.astype(np.int64) - old_density.astype(np.int64)
    positive_delta = np.maximum(delta, 0)
    report = {
        "schema_version": "railway.supplemental-point-evidence.v1",
        "policy": {
            "merged_v2_role": "replacement_for_old_tile_5",
            "clip_role": "150m_incremental_refinement_and_qa_only",
            "prohibited_operation": "do_not_append_old_tile_5_or_clip_to_merged_v2",
        },
        "bounds": bounds.to_json(),
        "cell_size_m": cell_size_m,
        "old": old_report,
        "new": new_report,
        "comparison": {
            "old_point_count_in_clip": old_count,
            "new_point_count_in_clip": new_count,
            "net_point_increase": new_count - old_count,
            "density_multiplier": new_count / old_count if old_count else None,
            "positive_added_evidence_points": int(np.sum(positive_delta)),
            "cells_with_positive_added_evidence": int(np.count_nonzero(positive_delta)),
        },
        "z_bands": {
            name: {
                "old_point_count": int(np.sum(old_bands[name], dtype=np.int64)),
                "new_point_count": int(np.sum(new_bands[name], dtype=np.int64)),
                "net_point_increase": int(
                    np.sum(new_bands[name], dtype=np.int64)
                    - np.sum(old_bands[name], dtype=np.int64)
                ),
            }
            for name in old_bands
        },
    }
    report_path = output / "supplemental_density_evidence.json"
    arrays_path = output / "supplemental_density_grids.npz"
    image_path = output / "supplemental_density_comparison.png"
    write_json(report_path, report)
    np.savez_compressed(
        arrays_path,
        old_density=old_density,
        new_density=new_density,
        positive_delta=positive_delta,
        **{f"old_{name}": value for name, value in old_bands.items()},
        **{f"new_{name}": value for name, value in new_bands.items()},
    )
    render_density_comparison(old_density, new_density, image_path)
    outputs = {"report": report_path, "arrays": arrays_path, "image": image_path}
    for name in old_bands:
        band_path = output / f"supplemental_density_{name}.png"
        render_density_comparison(
            old_bands[name],
            new_bands[name],
            band_path,
            title=f"Supplemental point evidence / {name.replace('_', ' ')}",
        )
        outputs[f"image_{name}"] = band_path
    return outputs
