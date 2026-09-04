from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from .io import load_json, write_json


@dataclass(frozen=True)
class ObjModel:
    vertices: np.ndarray
    faces_by_object: dict[str, list[tuple[int, ...]]]


def parse_obj_model(path: str | Path) -> ObjModel:
    vertices: list[tuple[float, float, float]] = []
    faces: dict[str, list[tuple[int, ...]]] = {}
    current_object = "UNNAMED"
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith(("o ", "g ")):
                current_object = line[2:].strip() or "UNNAMED"
                faces.setdefault(current_object, [])
            elif line.startswith("f "):
                indices: list[int] = []
                for token in line.split()[1:]:
                    value = int(token.split("/", maxsplit=1)[0])
                    index = value - 1 if value > 0 else len(vertices) + value
                    indices.append(index)
                if len(indices) >= 3:
                    faces.setdefault(current_object, []).append(tuple(indices))
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return ObjModel(np.asarray(vertices, dtype=np.float64), faces)


def object_vertex_indices(model: ObjModel, name: str) -> np.ndarray:
    faces = model.faces_by_object.get(name, [])
    if not faces:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.fromiter((index for face in faces for index in face), dtype=np.int64))


def sample_object_surfaces(
    model: ObjModel,
    *,
    origin_xyz: np.ndarray | None = None,
    maximum_samples_per_object: int = 5_000,
) -> dict[str, np.ndarray]:
    if maximum_samples_per_object <= 0:
        raise ValueError("maximum_samples_per_object must be positive")
    origin = (
        np.zeros(3, dtype=np.float64)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float64)
    )
    samples: dict[str, np.ndarray] = {}
    for name, faces in model.faces_by_object.items():
        if not faces:
            continue
        vertex_indices = object_vertex_indices(model, name)
        face_centroids = np.asarray(
            [np.mean(model.vertices[np.asarray(face, dtype=np.int64)], axis=0) for face in faces],
            dtype=np.float64,
        )
        combined = np.vstack((model.vertices[vertex_indices], face_centroids))
        if len(combined) > maximum_samples_per_object:
            selected = np.linspace(
                0, len(combined) - 1, maximum_samples_per_object, dtype=np.int64
            )
            combined = combined[selected]
        samples[name] = combined + origin
    return samples


def load_cloud_sample(
    path: str | Path,
    *,
    target_point_count: int = 3_000_000,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, dict[str, int]]:
    if target_point_count <= 0:
        raise ValueError("target_point_count must be positive")
    source = Path(path).resolve()
    chunks: list[np.ndarray] = []
    offset = 0
    with laspy.open(source) as reader:
        point_count = int(reader.header.point_count)
        stride = max(1, int(np.ceil(point_count / target_point_count)))
        for points in reader.chunk_iterator(chunk_size):
            start = (-offset) % stride
            x = np.asarray(points.x, dtype=np.float64)[start::stride]
            y = np.asarray(points.y, dtype=np.float64)[start::stride]
            z = np.asarray(points.z, dtype=np.float64)[start::stride]
            chunks.append(np.column_stack((x, y, z)))
            offset += len(points)
    cloud = np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float64)
    return cloud, {
        "source_point_count": point_count,
        "sample_point_count": len(cloud),
        "systematic_stride": stride,
    }


def _metrics(distances: np.ndarray) -> dict[str, float | int | str]:
    p50 = float(np.percentile(distances, 50))
    p90 = float(np.percentile(distances, 90))
    p95 = float(np.percentile(distances, 95))
    coverage_010 = float(np.mean(distances <= 0.10))
    coverage_020 = float(np.mean(distances <= 0.20))
    if p90 <= 0.10 and coverage_010 >= 0.75:
        disposition = "supported_keep"
    elif p90 <= 0.20 and coverage_020 >= 0.75:
        disposition = "minor_refit_review"
    else:
        disposition = "evidence_gap_or_geometry_review"
    return {
        "sample_count": len(distances),
        "p50_m": p50,
        "p90_m": p90,
        "p95_m": p95,
        "maximum_m": float(np.max(distances)),
        "coverage_at_0_05m": float(np.mean(distances <= 0.05)),
        "coverage_at_0_10m": coverage_010,
        "coverage_at_0_20m": coverage_020,
        "disposition": disposition,
    }


def evaluate_model_support(
    object_samples: dict[str, np.ndarray], cloud_sample: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if not len(cloud_sample):
        raise ValueError("cloud_sample is empty")
    tree = cKDTree(cloud_sample)
    reports: list[dict[str, Any]] = []
    distances_by_object: dict[str, np.ndarray] = {}
    for name in sorted(object_samples):
        points = object_samples[name]
        distances, indices = tree.query(points, workers=-1)
        distances = np.asarray(distances, dtype=np.float64)
        deltas = cloud_sample[np.asarray(indices, dtype=np.int64)] - points
        distances_by_object[name] = distances
        reports.append(
            {
                "object_name": name,
                **_metrics(distances),
                "nearest_delta_median_xyz_m": [
                    float(value) for value in np.median(deltas, axis=0)
                ],
                "nearest_delta_abs_p90_xyz_m": [
                    float(value) for value in np.percentile(np.abs(deltas), 90, axis=0)
                ],
            }
        )
    return reports, distances_by_object


def _asset_metadata(registry_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if registry_path is None:
        return {}
    registry = load_json(Path(registry_path))
    return {
        str(asset["id"]): {
            "asset_type": asset.get("type"),
            "evidence_level": asset.get("evidence_level"),
            "confidence": asset.get("confidence"),
            "status": asset.get("status"),
        }
        for asset in registry.get("assets", [])
        if asset.get("id")
    }


def _distance_color(distance: float) -> tuple[int, int, int]:
    if distance <= 0.05:
        return 129, 255, 52
    if distance <= 0.10:
        return 43, 214, 227
    if distance <= 0.20:
        return 255, 221, 66
    if distance <= 0.50:
        return 255, 133, 35
    return 240, 56, 74


def render_support_heatmap(
    object_samples: dict[str, np.ndarray],
    distances_by_object: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    width: int = 1500,
    height: int = 950,
) -> Path:
    all_points = np.vstack(list(object_samples.values()))
    minimum = np.min(all_points[:, :2], axis=0)
    maximum = np.max(all_points[:, :2], axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    margin = 70
    plot_width = width - margin * 2
    plot_height = height - margin * 2 - 50
    scale = min(plot_width / span[0], plot_height / span[1])
    image = Image.new("RGB", (width, height), (4, 18, 26))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (margin, 24),
        "150 m model-to-supplemental-cloud support / top view",
        fill=(225, 243, 247),
        font=font,
    )
    draw.text(
        (margin, 44),
        "green <= 5 cm | cyan <= 10 cm | yellow <= 20 cm | orange <= 50 cm | red > 50 cm",
        fill=(142, 181, 191),
        font=font,
    )
    center = (minimum + maximum) / 2.0
    for name in sorted(object_samples):
        points = object_samples[name]
        distances = distances_by_object[name]
        x = (points[:, 0] - center[0]) * scale + width / 2.0
        y = height / 2.0 - (points[:, 1] - center[1]) * scale + 30
        for px, py, distance in zip(x, y, distances, strict=True):
            draw.point((int(px), int(py)), fill=_distance_color(float(distance)))
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def audit_model_point_support(
    obj_path: str | Path,
    model_origin_path: str | Path,
    cloud_path: str | Path,
    output_directory: str | Path,
    *,
    asset_registry_path: str | Path | None = None,
    target_cloud_points: int = 3_000_000,
    maximum_samples_per_object: int = 5_000,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    model = parse_obj_model(obj_path)
    object_samples = sample_object_surfaces(
        model,
        origin_xyz=origin,
        maximum_samples_per_object=maximum_samples_per_object,
    )
    cloud_sample, cloud_sampling = load_cloud_sample(
        cloud_path, target_point_count=target_cloud_points
    )
    object_reports, distances_by_object = evaluate_model_support(object_samples, cloud_sample)
    metadata = _asset_metadata(asset_registry_path)
    for item in object_reports:
        item.update(metadata.get(str(item["object_name"]), {}))
    object_reports.sort(
        key=lambda item: (float(item["p90_m"]), -int(item["sample_count"])), reverse=True
    )
    counts: dict[str, int] = {}
    for item in object_reports:
        disposition = str(item["disposition"])
        counts[disposition] = counts.get(disposition, 0) + 1
    report = {
        "schema_version": "railway.model-point-support.v1",
        "method": {
            "cloud_sampling": "deterministic_systematic_record_sampling",
            "model_sampling": "unique_face_vertices_plus_face_centroids",
            "distance": "nearest_3d_euclidean_model_sample_to_cloud_sample",
            "interpretation": (
                "A large distance can mean wrong geometry or an unobserved surface; it is a review "
                "trigger, not automatic deletion authority."
            ),
        },
        "inputs": {
            "obj": str(Path(obj_path).resolve()),
            "model_origin": str(Path(model_origin_path).resolve()),
            "cloud": str(Path(cloud_path).resolve()),
            "asset_registry": (
                str(Path(asset_registry_path).resolve()) if asset_registry_path else None
            ),
        },
        "sampling": {
            **cloud_sampling,
            "model_object_count": len(object_samples),
            "model_sample_count": int(sum(len(value) for value in object_samples.values())),
            "maximum_samples_per_object": maximum_samples_per_object,
        },
        "disposition_counts": counts,
        "objects": object_reports,
    }
    report_path = output / "model_point_support.json"
    csv_path = output / "model_point_support.csv"
    heatmap_path = output / "model_point_support_heatmap.png"
    write_json(report_path, report)
    fieldnames = sorted({key for item in object_reports for key in item})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                **item,
                "nearest_delta_median_xyz_m": ";".join(
                    f"{value:.6f}" for value in item["nearest_delta_median_xyz_m"]
                ),
                "nearest_delta_abs_p90_xyz_m": ";".join(
                    f"{value:.6f}" for value in item["nearest_delta_abs_p90_xyz_m"]
                ),
            }
            for item in object_reports
        )
    render_support_heatmap(object_samples, distances_by_object, heatmap_path)
    return {"report": report_path, "csv": csv_path, "heatmap": heatmap_path}
