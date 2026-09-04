from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .camera import load_camera_rows
from .config import ProjectConfig
from .io import load_json, sha256_json, write_json
from .multi_source import effective_camera_csv_path

_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _convention_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["euler_order"]),
        str(record["pose_direction"]),
        str(record["canonical_axes_from_local"]),
    )


def aggregate_consensus_scores(
    camera_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-camera hypotheses without allowing one camera to set the convention."""
    if len(camera_results) < 3:
        raise ValueError("Projection consensus requires at least three camera samples")
    result_maps: list[dict[tuple[str, str, str], float]] = []
    rank_maps: list[dict[tuple[str, str, str], float]] = []
    sample_ids: list[str] = []
    for camera in camera_results:
        sample_id = str(camera["sample_id"])
        records = list(camera["results"])
        if not records:
            raise ValueError(f"Projection sample {sample_id} has no candidate results")
        score_map = {
            _convention_key(record): float(record["score_median_rgb_mae"])
            for record in records
        }
        ordered = sorted(score_map, key=lambda key: (score_map[key], key))
        denominator = max(1, len(ordered) - 1)
        tied_ranks: dict[tuple[str, str, str], float] = {}
        start = 0
        while start < len(ordered):
            end = start + 1
            score = score_map[ordered[start]]
            while end < len(ordered) and score_map[ordered[end]] == score:
                end += 1
            average_rank = ((start + end - 1) / 2.0) / denominator
            for key in ordered[start:end]:
                tied_ranks[key] = average_rank
            start = end
        result_maps.append(score_map)
        rank_maps.append(tied_ranks)
        sample_ids.append(sample_id)
    shared_keys = set(result_maps[0])
    for score_map in result_maps[1:]:
        shared_keys.intersection_update(score_map)
    if not shared_keys:
        raise ValueError("Projection samples have no shared convention candidates")

    aggregated: list[dict[str, Any]] = []
    for key in shared_keys:
        scores = [score_map[key] for score_map in result_maps]
        ranks = [rank_map[key] for rank_map in rank_maps]
        aggregated.append(
            {
                "euler_order": key[0],
                "pose_direction": key[1],
                "canonical_axes_from_local": key[2],
                "consensus_median_rank_fraction": float(np.median(ranks)),
                "consensus_mean_rank_fraction": float(np.mean(ranks)),
                "aggregate_median_rgb_mae": float(np.median(scores)),
                "aggregate_mean_rgb_mae": float(np.mean(scores)),
                "aggregate_max_rgb_mae": float(np.max(scores)),
                "per_camera": [
                    {
                        "sample_id": sample_id,
                        "score_median_rgb_mae": score,
                        "rank_fraction": rank,
                    }
                    for sample_id, score, rank in zip(sample_ids, scores, ranks)
                ],
            }
        )
    aggregated.sort(
        key=lambda item: (
            item["consensus_median_rank_fraction"],
            item["consensus_mean_rank_fraction"],
            item["aggregate_median_rgb_mae"],
            item["aggregate_max_rgb_mae"],
            item["euler_order"],
            item["pose_direction"],
            item["canonical_axes_from_local"],
        )
    )
    return aggregated


def _prepare_consensus_sample(
    project: ProjectConfig,
    segment_id: str,
    camera_index: int,
    rows: list[dict[str, str]],
    panorama_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    point_cloud_path = project.workspace_path("segments") / f"{segment_id}.laz"
    if not point_cloud_path.is_file():
        raise FileNotFoundError(point_cloud_path)
    row = next((item for item in rows if int(item["index"]) == camera_index), None)
    if row is None:
        raise ValueError(f"Camera index not found: {camera_index}")
    missing_rotation = [
        name for name in ("rot_x", "rot_y", "rot_z") if not (row.get(name) or "").strip()
    ]
    if missing_rotation:
        raise ValueError(f"Camera pose is missing rotation fields: {missing_rotation}")
    photo_path = _find_photo(panorama_root, row["file"])
    image = np.asarray(Image.open(photo_path).convert("RGB"))
    cloud = laspy.read(point_cloud_path)
    dimensions = set(cloud.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dimensions):
        raise ValueError("Projection calibration requires RGB point attributes")
    points = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    camera_position = np.asarray(
        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
    )
    world_vectors = points - camera_position
    distances = np.linalg.norm(world_vectors, axis=1)
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
        raise ValueError(
            f"No colored points remain for projection sample {segment_id}={camera_index}"
        )
    random = np.random.default_rng(int(config["random_seed"]) + camera_index)
    sample_count = int(config["sample_count"])
    if len(world_vectors) > sample_count:
        selected = random.choice(len(world_vectors), sample_count, replace=False)
        sample_vectors = world_vectors[selected]
        sample_distances = distances[selected]
        sample_colors = colors[selected]
    else:
        sample_vectors = world_vectors
        sample_distances = distances
        sample_colors = colors
    del points, cloud, world_vectors, distances, colors
    return {
        "sample_id": f"{segment_id}:{camera_index}",
        "segment_id": segment_id,
        "camera_index": camera_index,
        "row": row,
        "photo_path": photo_path,
        "point_cloud_path": point_cloud_path,
        "image": image,
        "candidate_point_count": int(np.count_nonzero(valid)),
        "sample_vectors": sample_vectors,
        "sample_distances": sample_distances,
        "sample_colors": sample_colors,
    }


def _score_consensus_sample(
    sample: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    row = sample["row"]
    angle_map = {axis: float(row[f"rot_{axis}"]) for axis in "xyz"}
    degrees = bool(config.get("rotation_degrees", False))
    results: list[dict[str, Any]] = []
    for order_lower in config.get(
        "euler_orders", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    ):
        ordered_angles = [angle_map[axis] for axis in order_lower]
        for order in (order_lower, order_lower.upper()):
            base = Rotation.from_euler(order, ordered_angles, degrees=degrees).as_matrix()
            for direction in config.get(
                "pose_directions", ["camera_to_world", "world_to_camera"]
            ):
                for label, permutation in signed_permutations():
                    score = _color_score(
                        _canonical_vectors(
                            sample["sample_vectors"], base, direction, permutation
                        ),
                        sample["sample_colors"],
                        sample["image"],
                    )
                    results.append(
                        {
                            "score_median_rgb_mae": score,
                            "euler_order": order,
                            "pose_direction": direction,
                            "canonical_axes_from_local": label,
                        }
                    )
    results.sort(key=lambda item: (item["score_median_rgb_mae"], _convention_key(item)))
    return results


def _matrices_for_convention(
    row: dict[str, str], convention: dict[str, Any], degrees: bool
) -> tuple[np.ndarray, np.ndarray]:
    angle_map = {axis: float(row[f"rot_{axis}"]) for axis in "xyz"}
    order = str(convention["euler_order"])
    ordered_angles = [angle_map[axis] for axis in order.lower()]
    base = Rotation.from_euler(order, ordered_angles, degrees=degrees).as_matrix()
    permutations = dict(signed_permutations())
    return base, permutations[str(convention["canonical_axes_from_local"])]


def calibrate_projection_consensus(
    project: ProjectConfig,
    samples: list[tuple[str, int]],
    output_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Choose one pose/projection convention using three or more independent frames."""
    if not _SAFE_OUTPUT_NAME.fullmatch(output_name):
        raise ValueError(
            "Projection consensus output name may contain only letters, numbers, . _ and -"
        )
    if len(samples) < 3 or len({camera_index for _, camera_index in samples}) < 3:
        raise ValueError("Projection consensus requires at least three unique camera indexes")
    if len(set(samples)) != len(samples):
        raise ValueError("Projection consensus samples must be unique")
    config = load_json(project.resolve(project.value["algorithms"]["projection_calibration"]))
    camera_path = effective_camera_csv_path(project)
    panorama_root = project.input_path("panorama_root")
    if panorama_root is None or not panorama_root.is_dir():
        raise FileNotFoundError(panorama_root)
    output_root = project.workspace_path("reports") / "projection"
    report_path = output_root / f"{output_name}_consensus.json"
    overlay_paths = [
        output_root / f"{output_name}_{segment_id}_camera_{camera_index}_consensus.jpg"
        for segment_id, camera_index in samples
    ]
    for path in [report_path, *overlay_paths]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    rows = load_camera_rows(camera_path)
    prepared: list[dict[str, Any]] = []
    camera_results: list[dict[str, Any]] = []
    for segment_id, camera_index in samples:
        sample = _prepare_consensus_sample(
            project, segment_id, camera_index, rows, panorama_root, config
        )
        prepared.append(sample)
        camera_results.append(
            {
                "sample_id": sample["sample_id"],
                "results": _score_consensus_sample(sample, config),
            }
        )

    aggregated = aggregate_consensus_scores(camera_results)
    top_count = int(config.get("top_result_count", 20))
    top = aggregated[:top_count]
    best = top[0]
    aggregate_fields = (
        "consensus_median_rank_fraction",
        "consensus_mean_rank_fraction",
        "aggregate_median_rgb_mae",
        "aggregate_max_rgb_mae",
    )
    equivalent_best = [
        item
        for item in aggregated
        if all(
            np.isclose(float(item[field]), float(best[field]), rtol=0.0, atol=1e-12)
            for field in aggregate_fields
        )
    ]
    best_key = _convention_key(best)
    degrees = bool(config.get("rotation_degrees", False))
    per_camera: list[dict[str, Any]] = []
    local_best_match_count = 0
    for sample, result_group, overlay_path in zip(
        prepared, camera_results, overlay_paths
    ):
        local_best = result_group["results"][0]
        agrees = _convention_key(local_best) == best_key
        local_best_match_count += int(agrees)
        consensus_record = next(
            item for item in result_group["results"] if _convention_key(item) == best_key
        )
        base, permutation = _matrices_for_convention(sample["row"], best, degrees)
        _save_overlay(
            sample["image"],
            sample["sample_vectors"],
            sample["sample_distances"],
            base,
            str(best["pose_direction"]),
            permutation,
            overlay_path,
        )
        per_camera.append(
            {
                "sample_id": sample["sample_id"],
                "segment_id": sample["segment_id"],
                "camera_index": sample["camera_index"],
                "camera_file": sample["row"]["file"],
                "photo": str(sample["photo_path"]),
                "point_cloud": str(sample["point_cloud_path"]),
                "image_size": [
                    int(sample["image"].shape[1]),
                    int(sample["image"].shape[0]),
                ],
                "candidate_point_count": sample["candidate_point_count"],
                "sample_point_count": len(sample["sample_vectors"]),
                "local_best": local_best,
                "local_best_matches_consensus": agrees,
                "consensus_score_median_rgb_mae": consensus_record[
                    "score_median_rgb_mae"
                ],
                "consensus_base_rotation": base.tolist(),
                "consensus_overlay": str(overlay_path),
            }
        )

    report = {
        "schema_version": "railway.projection-consensus.v1",
        "project_id": project.project_id,
        "sample_count": len(samples),
        "camera_index_count": len({camera_index for _, camera_index in samples}),
        "camera_csv": str(camera_path),
        "settings_sha256": sha256_json(config),
        "aggregation_method": (
            "minimum median per-camera rank fraction, then mean rank, median RGB MAE "
            "and maximum RGB MAE"
        ),
        "best_shared_convention": best,
        "equivalent_best_count": len(equivalent_best),
        "equivalent_best_conventions": [
            {
                "euler_order": item["euler_order"],
                "pose_direction": item["pose_direction"],
                "canonical_axes_from_local": item["canonical_axes_from_local"],
            }
            for item in equivalent_best
        ],
        "convention_identifiability": (
            "unique_at_current_sampling_resolution"
            if len(equivalent_best) == 1
            else "multiple_numerically_equivalent_hypotheses"
        ),
        "local_best_match_count": local_best_match_count,
        "all_local_bests_match": local_best_match_count == len(samples),
        "per_camera": per_camera,
        "top_shared_conventions": top,
        "report_path": str(report_path),
        "status": "automatic_multi_camera_hypothesis_visual_acceptance_required",
        "limitations": [
            "A shared low RGB error removes per-frame convention drift but is not pose proof.",
            "Accept only after all consensus overlays agree on visible structural edges.",
            "Camera pose translation and rotation bias are not optimized by this command.",
        ],
    }
    write_json(report_path, report)
    return report


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
    camera_path = effective_camera_csv_path(project)
    panorama_root = project.input_path("panorama_root")
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
