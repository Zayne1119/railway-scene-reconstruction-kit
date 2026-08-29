from __future__ import annotations

import copy
import os
import re
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json

OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
HASH_BUCKETS = 1_000_000


def spatial_voxel_holdout_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    voxel_size_m: float,
    holdout_fraction: float,
    seed: int,
) -> np.ndarray:
    """Assign complete spatial voxels to holdout with a stable 64-bit hash."""

    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between zero and one")
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y and z must have identical lengths")
    ix = np.floor(np.asarray(x, dtype=np.float64) / voxel_size_m).astype(np.int64)
    iy = np.floor(np.asarray(y, dtype=np.float64) / voxel_size_m).astype(np.int64)
    iz = np.floor(np.asarray(z, dtype=np.float64) / voxel_size_m).astype(np.int64)
    ux = ix.astype(np.uint64)
    uy = iy.astype(np.uint64)
    uz = iz.astype(np.uint64)
    value = np.bitwise_xor(
        np.multiply(ux, np.uint64(0x9E3779B185EBCA87), dtype=np.uint64),
        np.multiply(uy, np.uint64(0xC2B2AE3D27D4EB4F), dtype=np.uint64),
    )
    value = np.bitwise_xor(
        value,
        np.multiply(uz, np.uint64(0x165667B19E3779F9), dtype=np.uint64),
    )
    value = np.bitwise_xor(value, np.uint64(seed & ((1 << 64) - 1)))
    value = np.bitwise_xor(value, value >> np.uint64(30))
    value = np.multiply(value, np.uint64(0xBF58476D1CE4E5B9), dtype=np.uint64)
    value = np.bitwise_xor(value, value >> np.uint64(27))
    value = np.multiply(value, np.uint64(0x94D049BB133111EB), dtype=np.uint64)
    value = np.bitwise_xor(value, value >> np.uint64(31))
    threshold = round(holdout_fraction * HASH_BUCKETS)
    return value % np.uint64(HASH_BUCKETS) < np.uint64(threshold)


def _inside(points: laspy.ScaleAwarePointRecord, bounds: dict[str, float]) -> np.ndarray:
    return (
        (points.x >= bounds["min_x"])
        & (points.x <= bounds["max_x"])
        & (points.y >= bounds["min_y"])
        & (points.y <= bounds["max_y"])
        & (points.z >= bounds["min_z"])
        & (points.z <= bounds["max_z"])
    )


def _source_record(benchmark_root: Path, source: Path) -> dict[str, Any]:
    dataset = load_json(benchmark_root / "manifests" / "dataset-manifest.json")
    for scene in dataset.get("scenes", []):
        for item in scene.get("inputs", []):
            reference = Path(str(item.get("path", "")))
            resolved = (
                reference.resolve()
                if reference.is_absolute()
                else (benchmark_root / reference).resolve()
            )
            if item.get("kind") == "point_cloud" and resolved == source:
                return {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": item.get("sha256"),
                    "hash_source": "dataset_manifest_declared",
                }
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": None,
        "hash_source": "not_declared",
    }


def split_point_cloud_holdout(
    project: ProjectConfig,
    benchmark_root: str | Path,
    segment_ids: set[str],
    output_name: str = "point_holdout_v1",
    holdout_fraction: float = 0.20,
    voxel_size_m: float = 0.50,
    seed: int = 20260827,
) -> Path:
    """Crop requested segments and split complete voxels into train/holdout LAZ files."""

    if not segment_ids:
        raise ValueError("At least one segment id is required")
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("Invalid output_name")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie strictly between zero and one")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")

    benchmark = Path(benchmark_root).resolve()
    output_root = benchmark / "holdouts" / output_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite point holdout: {output_root}")
    output_root.mkdir(parents=True)
    source = project.input_path("point_cloud")
    if source is None or not source.is_file():
        raise FileNotFoundError(source)
    segment_manifest_path = project.workspace_path("segment_manifest")
    segment_manifest = load_json(segment_manifest_path)
    by_id = {str(item["id"]): item for item in segment_manifest.get("segments", [])}
    unknown = segment_ids - set(by_id)
    if unknown:
        raise ValueError(f"Unknown segment ids: {sorted(unknown)}")
    segments = [by_id[segment_id] for segment_id in sorted(segment_ids)]
    for segment in segments:
        if not isinstance(segment.get("bounds"), dict):
            raise TypeError(f"Segment has no crop bounds: {segment['id']}")

    roles = ("train", "holdout")
    outputs = {
        (segment["id"], role): output_root / f"{segment['id']}.{role}.laz"
        for segment in segments
        for role in roles
    }
    temporary = {
        key: path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial.laz")
        for key, path in outputs.items()
    }
    counts = {
        (segment["id"], role): 0 for segment in segments for role in roles
    }
    selected_counts = {segment["id"]: 0 for segment in segments}
    source_point_count = 0
    try:
        with laspy.open(source) as reader, ExitStack() as stack:
            source_point_count = int(reader.header.point_count)
            writers = {
                key: stack.enter_context(
                    laspy.open(
                        path,
                        mode="w",
                        header=copy.deepcopy(reader.header),
                        do_compress=True,
                    )
                )
                for key, path in temporary.items()
            }
            chunk_size = int(project.value["segmentation"]["chunk_size_points"])
            for points in reader.chunk_iterator(chunk_size):
                for segment in segments:
                    segment_id = str(segment["id"])
                    selected = points[_inside(points, segment["bounds"])]
                    if not len(selected):
                        continue
                    selected_counts[segment_id] += len(selected)
                    holdout = spatial_voxel_holdout_mask(
                        np.asarray(selected.x),
                        np.asarray(selected.y),
                        np.asarray(selected.z),
                        voxel_size_m,
                        holdout_fraction,
                        seed,
                    )
                    for role, mask in (("train", ~holdout), ("holdout", holdout)):
                        subset = selected[mask]
                        if len(subset):
                            writers[(segment_id, role)].write_points(subset)
                            counts[(segment_id, role)] += len(subset)
        for key, path in outputs.items():
            os.replace(temporary[key], path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    segment_records: list[dict[str, Any]] = []
    for segment in segments:
        segment_id = str(segment["id"])
        train_path = outputs[(segment_id, "train")]
        holdout_path = outputs[(segment_id, "holdout")]
        train_count = counts[(segment_id, "train")]
        holdout_count = counts[(segment_id, "holdout")]
        selected_count = selected_counts[segment_id]
        segment_records.append(
            {
                "segment_id": segment_id,
                "chainage_range_m": [
                    segment["chainage_start_m"],
                    segment["chainage_end_m"],
                ],
                "bounds": segment["bounds"],
                "selected_point_count": selected_count,
                "train": {
                    "path": str(train_path),
                    "point_count": train_count,
                    "bytes": train_path.stat().st_size,
                    "sha256": sha256_file(train_path),
                },
                "holdout": {
                    "path": str(holdout_path),
                    "point_count": holdout_count,
                    "bytes": holdout_path.stat().st_size,
                    "sha256": sha256_file(holdout_path),
                },
                "realized_holdout_fraction": (
                    holdout_count / selected_count if selected_count else None
                ),
            }
        )

    manifest = {
        "schema_version": "railway.point-holdout.v1",
        "holdout_id": output_name,
        "created_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "strategy": "spatial_voxel_group_hash",
        "source_point_cloud": _source_record(benchmark, source),
        "source_point_count": source_point_count,
        "segment_manifest": str(segment_manifest_path),
        "segment_manifest_sha256": sha256_file(segment_manifest_path),
        "voxel_size_m": voxel_size_m,
        "requested_holdout_fraction": holdout_fraction,
        "seed": seed,
        "segments": segment_records,
        "limitations": [
            "All points in one voxel receive the same train/holdout assignment.",
            "Overlapping segment context keeps the same global voxel assignment.",
            "A model trained on the original unsplit cloud is not eligible for holdout accuracy.",
        ],
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def split_existing_point_cloud_holdout(
    segment_sources: list[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    holdout_id: str = "nested_point_holdout_v1",
    holdout_fraction: float = 0.25,
    voxel_size_m: float = 0.50,
    seed: int = 20260828,
    chunk_size_points: int = 2_000_000,
) -> Path:
    """Split already-cropped segment clouds for nested train-only tuning."""

    if not segment_sources:
        raise ValueError("At least one segment source is required")
    if not OUTPUT_NAME_PATTERN.fullmatch(holdout_id):
        raise ValueError("Invalid holdout_id")
    if not 0.0 < holdout_fraction < 1.0 or voxel_size_m <= 0:
        raise ValueError("Invalid holdout fraction or voxel size")
    if chunk_size_points < 1:
        raise ValueError("chunk_size_points must be positive")
    sources = {segment: Path(path).resolve() for segment, path in segment_sources}
    if len(sources) != len(segment_sources):
        raise ValueError("Duplicate segment source id")
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite nested holdout: {root}")
    root.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    for segment_id, source in sorted(sources.items()):
        train_path = root / f"{segment_id}.train.laz"
        holdout_path = root / f"{segment_id}.holdout.laz"
        train_temp = train_path.with_name(
            f".{train_path.stem}.{uuid.uuid4().hex}.partial.laz"
        )
        holdout_temp = holdout_path.with_name(
            f".{holdout_path.stem}.{uuid.uuid4().hex}.partial.laz"
        )
        train_count = holdout_count = 0
        try:
            with laspy.open(source) as reader, laspy.open(
                train_temp,
                mode="w",
                header=copy.deepcopy(reader.header),
                do_compress=True,
            ) as train_writer, laspy.open(
                holdout_temp,
                mode="w",
                header=copy.deepcopy(reader.header),
                do_compress=True,
            ) as holdout_writer:
                for points in reader.chunk_iterator(chunk_size_points):
                    holdout = spatial_voxel_holdout_mask(
                        np.asarray(points.x),
                        np.asarray(points.y),
                        np.asarray(points.z),
                        voxel_size_m,
                        holdout_fraction,
                        seed,
                    )
                    if np.any(~holdout):
                        train_writer.write_points(points[~holdout])
                        train_count += int(np.count_nonzero(~holdout))
                    if np.any(holdout):
                        holdout_writer.write_points(points[holdout])
                        holdout_count += int(np.count_nonzero(holdout))
            os.replace(train_temp, train_path)
            os.replace(holdout_temp, holdout_path)
        finally:
            train_temp.unlink(missing_ok=True)
            holdout_temp.unlink(missing_ok=True)
        source_count = train_count + holdout_count
        records.append(
            {
                "segment_id": segment_id,
                "source": {
                    "path": str(source),
                    "point_count": source_count,
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                },
                "selected_point_count": source_count,
                "train": {
                    "path": str(train_path),
                    "point_count": train_count,
                    "bytes": train_path.stat().st_size,
                    "sha256": sha256_file(train_path),
                },
                "holdout": {
                    "path": str(holdout_path),
                    "point_count": holdout_count,
                    "bytes": holdout_path.stat().st_size,
                    "sha256": sha256_file(holdout_path),
                },
                "realized_holdout_fraction": (
                    holdout_count / source_count if source_count else None
                ),
            }
        )
    manifest = {
        "schema_version": "railway.point-holdout.v1",
        "holdout_id": holdout_id,
        "created_at": datetime.now(UTC).isoformat(),
        "project_id": "nested_train_only_tuning",
        "strategy": "spatial_voxel_group_hash_existing_sources",
        "voxel_size_m": voxel_size_m,
        "requested_holdout_fraction": holdout_fraction,
        "seed": seed,
        "segments": records,
        "limitations": [
            "This nested split is derived only from the outer training partition.",
            "It is for parameter selection and is not an additional independent test site.",
            "All points in one voxel receive the same nested assignment.",
        ],
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def validate_point_cloud_holdout(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_json(source)
    errors: list[str] = []
    if manifest.get("schema_version") != "railway.point-holdout.v1":
        errors.append("Unsupported holdout schema")
    segment_manifest_value = manifest.get("segment_manifest")
    if segment_manifest_value:
        segment_manifest_path = Path(str(segment_manifest_value))
        if not segment_manifest_path.is_file():
            errors.append(f"Missing segment manifest: {segment_manifest_path}")
        elif sha256_file(segment_manifest_path) != manifest.get("segment_manifest_sha256"):
            errors.append("Segment manifest hash changed")
    total_train = total_holdout = 0
    for segment in manifest.get("segments", []):
        selected = int(segment.get("selected_point_count", 0))
        role_counts: dict[str, int] = {}
        for role in ("train", "holdout"):
            record = segment.get(role, {})
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                errors.append(f"{segment.get('segment_id')} missing {role} file")
                continue
            if sha256_file(path) != record.get("sha256"):
                errors.append(f"{segment.get('segment_id')} {role} hash mismatch")
            with laspy.open(path) as reader:
                actual_count = int(reader.header.point_count)
            declared_count = int(record.get("point_count", -1))
            if actual_count != declared_count:
                errors.append(
                    f"{segment.get('segment_id')} {role} point count mismatch"
                )
            role_counts[role] = actual_count
        if sum(role_counts.values()) != selected:
            errors.append(f"{segment.get('segment_id')} split is not complementary")
        total_train += role_counts.get("train", 0)
        total_holdout += role_counts.get("holdout", 0)
    result = {
        "schema_version": "railway.point-holdout-validation.v1",
        "holdout_id": manifest.get("holdout_id"),
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "segment_count": len(manifest.get("segments", [])),
        "train_point_count": total_train,
        "holdout_point_count": total_holdout,
        "errors": errors,
    }
    write_json(source.parent / "validation.json", result)
    return result
