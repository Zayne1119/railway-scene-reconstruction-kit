from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .camera import load_camera_rows
from .config import ProjectConfig
from .io import load_json, write_json


def signed_permutations() -> list[tuple[str, np.ndarray]]:
    result: list[tuple[str, np.ndarray]] = []
    names = "xyz"
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for output_axis, input_axis in enumerate(permutation):
                matrix[output_axis, input_axis] = signs[output_axis]
            label = ",".join(
                f"{names[input_axis]}{'+' if sign > 0 else '-'}"
                for input_axis, sign in zip(permutation, signs)
            )
            result.append((label, matrix))
    return result


def project_equirectangular(
    canonical_vectors: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(canonical_vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    safe_norms = np.where(valid, norms, 1.0)
    longitude = np.arctan2(vectors[:, 0], vectors[:, 2])
    latitude = np.arcsin(np.clip(vectors[:, 1] / safe_norms, -1.0, 1.0))
    u = np.mod((longitude / (2.0 * np.pi) + 0.5) * width, width)
    v = np.clip((0.5 - latitude / np.pi) * height, 0, height - 1)
    u[~valid] = np.nan
    v[~valid] = np.nan
    return u, v


def _find_photo(root: Path, reference: str) -> Path:
    relative = Path(reference)
    for candidate in (root / relative, root / relative.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Panorama not found for reference: {reference}")


def _canonical_vectors(
    world_vectors: np.ndarray,
    base_rotation: np.ndarray,
    direction: str,
    permutation: np.ndarray,
) -> np.ndarray:
    if direction == "camera_to_world":
        local = world_vectors @ base_rotation
    elif direction == "world_to_camera":
        local = world_vectors @ base_rotation.T
    else:
        raise ValueError(f"Unknown pose direction: {direction}")
    return local @ permutation.T


def _color_score(canonical: np.ndarray, point_colors: np.ndarray, image: np.ndarray) -> float:
    height, width = image.shape[:2]
    u, v = project_equirectangular(canonical, width, height)
    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return float("inf")
    pixels = image[v[valid].astype(np.int64), u[valid].astype(np.int64)].astype(np.float32)
    return float(np.median(np.mean(np.abs(pixels - point_colors[valid]), axis=1)))


def _save_overlay(
    image: np.ndarray,
    world_vectors: np.ndarray,
    distances: np.ndarray,
    base_rotation: np.ndarray,
    direction: str,
    permutation: np.ndarray,
    output: Path,
) -> None:
    canonical = _canonical_vectors(world_vectors, base_rotation, direction, permutation)
    height, width = image.shape[:2]
    u, v = project_equirectangular(canonical, width, height)
    valid = np.isfinite(u) & np.isfinite(v)
    order = np.argsort(distances[valid])[::-1]
    x = u[valid][order].astype(np.int64)
    y = v[valid][order].astype(np.int64)
    overlay = (image.astype(np.float32) * 0.62).astype(np.uint8)
    marker = np.asarray([20, 255, 110], dtype=np.uint8)
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        overlay[np.clip(y + dy, 0, height - 1), np.mod(x + dx, width)] = marker
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    Image.fromarray(overlay, mode="RGB").save(temporary, quality=92)
    os.replace(temporary, output)


def calibrate_projection(
    project: ProjectConfig,
    segment_id: str,
    camera_index: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_json(project.resolve(project.value["algorithms"]["projection_calibration"]))
    point_cloud_path = project.workspace_path("segments") / f"{segment_id}.laz"
    if not point_cloud_path.is_file():
        raise FileNotFoundError(point_cloud_path)
    camera_path = project.input_path("camera_csv")
    panorama_root = project.input_path("panorama_root")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    if panorama_root is None or not panorama_root.is_dir():
        raise FileNotFoundError(panorama_root)
    rows = load_camera_rows(camera_path)
    row = next((item for item in rows if int(item["index"]) == camera_index), None)
    if row is None:
        raise ValueError(f"Camera index not found: {camera_index}")
    missing_rotation = [
        name for name in ("rot_x", "rot_y", "rot_z") if not (row.get(name) or "").strip()
    ]
    if missing_rotation:
        raise ValueError(f"Camera pose is missing rotation fields: {missing_rotation}")
    photo_path = _find_photo(panorama_root, row["file"])
    output_root = project.workspace_path("reports") / "projection"
    report_path = output_root / f"{segment_id}_camera_{camera_index}_calibration.json"
    overlay_path = output_root / f"{segment_id}_camera_{camera_index}_overlay.jpg"
    for path in (report_path, overlay_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    image = np.asarray(Image.open(photo_path).convert("RGB"))
    cloud = laspy.read(point_cloud_path)
    points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    camera_position = np.asarray(
        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
    )
    world_vectors = points - camera_position
    distances = np.linalg.norm(world_vectors, axis=1)
    dimensions = set(cloud.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dimensions):
        raise ValueError("Projection calibration requires RGB point attributes")
    colors = np.column_stack((cloud.red, cloud.green, cloud.blue)).astype(np.float32)
    if colors.max(initial=0) > 255:
        colors /= 257.0
    valid = (
        (distances >= float(config["minimum_distance_m"]))
        & (distances <= float(config["maximum_distance_m"]))
        & np.any(colors > 0, axis=1)
    )
    world_vectors = world_vectors[valid]
    distances = distances[valid]
    colors = colors[valid]
    if not len(world_vectors):
        raise ValueError("No colored points remain in the projection calibration radius")
    random = np.random.default_rng(int(config["random_seed"]))
    sample_count = int(config["sample_count"])
    if len(world_vectors) > sample_count:
        selected = random.choice(len(world_vectors), sample_count, replace=False)
        sample_vectors = world_vectors[selected]
        sample_colors = colors[selected]
    else:
        sample_vectors = world_vectors
        sample_colors = colors

    angle_map = {axis: float(row[f"rot_{axis}"]) for axis in "xyz"}
    degrees = bool(config.get("rotation_degrees", False))
    matrices: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    results: list[dict[str, Any]] = []
    for order_lower in config.get("euler_orders", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]):
        ordered_angles = [angle_map[axis] for axis in order_lower]
        for order in (order_lower, order_lower.upper()):
            base = Rotation.from_euler(order, ordered_angles, degrees=degrees).as_matrix()
            for direction in config.get("pose_directions", ["camera_to_world", "world_to_camera"]):
                for label, permutation in signed_permutations():
                    score = _color_score(
                        _canonical_vectors(sample_vectors, base, direction, permutation),
                        sample_colors,
                        image,
                    )
                    record = {
                        "score_median_rgb_mae": score,
                        "euler_order": order,
                        "pose_direction": direction,
                        "canonical_axes_from_local": label,
                    }
                    results.append(record)
                    matrices[(order, direction, label)] = (base, permutation)
    results.sort(key=lambda item: item["score_median_rgb_mae"])
    top = results[: int(config.get("top_result_count", 20))]
    best = top[0]
    base, permutation = matrices[
        (best["euler_order"], best["pose_direction"], best["canonical_axes_from_local"])
    ]
    _save_overlay(
        image, world_vectors, distances, base, best["pose_direction"], permutation, overlay_path
    )
    report = {
        "schema_version": "railway.projection-calibration.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "camera_index": camera_index,
        "camera_file": row["file"],
        "point_cloud": str(point_cloud_path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "candidate_point_count": len(world_vectors),
        "sample_point_count": len(sample_vectors),
        "angles": angle_map,
        "rotation_degrees": degrees,
        "best": best,
        "best_base_rotation": base.tolist(),
        "best_permutation": permutation.tolist(),
        "top_results": top,
        "overlay": str(overlay_path),
        "status": "automatic_rgb_hypothesis_visual_acceptance_required",
        "limitations": [
            "The lowest RGB error is a hypothesis, not proof of a correct pose convention.",
            "Accept only after multiple cameras and visible structural edges agree.",
        ],
    }
    write_json(report_path, report)
    return report

