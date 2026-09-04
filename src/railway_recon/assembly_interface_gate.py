from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree

from .io import sha256_file
from .model_point_support import ObjModel, parse_obj_model

VerticalSide = Literal["top", "bottom"]
ContactMode = Literal["area", "localized"]


def _points_xyz(value: np.ndarray | Sequence[Sequence[float]], *, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3)")
    if not len(points):
        raise ValueError(f"{label} is empty")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains non-finite values")
    return points


def sample_object_triangle_surface(
    model: ObjModel,
    object_name: str,
    *,
    origin_xyz: np.ndarray | Sequence[float] | None = None,
    spacing_m: float = 0.04,
    maximum_samples: int = 100_000,
) -> np.ndarray:
    """Return deterministic, approximately uniform samples from one OBJ object.

    The existing lightweight OBJ sampler is sufficient for model-to-cloud QA, but
    assembly interfaces need samples along triangle interiors and edges as well.
    A barycentric lattice makes the result repeatable without a random seed.
    """

    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    origin = (
        np.zeros(3, dtype=np.float64)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float64)
    )
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_xyz must contain three finite values")

    triangles = [
        model.vertices[[face[0], face[index], face[index + 1]]]
        for face in model.faces_by_object.get(object_name, [])
        for index in range(1, len(face) - 1)
    ]
    if not triangles:
        raise ValueError(f"OBJ object has no sampleable faces: {object_name}")
    samples: list[np.ndarray] = []
    samples_per_triangle = max(3, maximum_samples // len(triangles))
    maximum_resolution = max(
        1,
        int(np.floor((np.sqrt(8.0 * samples_per_triangle + 1.0) - 3.0) / 2.0)),
    )
    for triangle in triangles:
        edge_lengths = (
            np.linalg.norm(triangle[1] - triangle[0]),
            np.linalg.norm(triangle[2] - triangle[1]),
            np.linalg.norm(triangle[0] - triangle[2]),
        )
        resolution = min(
            maximum_resolution,
            max(1, int(np.ceil(max(edge_lengths) / spacing_m))),
        )
        barycentric = np.asarray(
            [
                (first / resolution, second / resolution)
                for first in range(resolution + 1)
                for second in range(resolution + 1 - first)
            ],
            dtype=np.float64,
        )
        first_weight = barycentric[:, :1]
        second_weight = barycentric[:, 1:2]
        samples.append(
            triangle[0]
            + first_weight * (triangle[1] - triangle[0])
            + second_weight * (triangle[2] - triangle[0])
        )
    combined = np.vstack(samples)
    if len(combined) > maximum_samples:
        selected = np.linspace(0, len(combined) - 1, maximum_samples, dtype=np.int64)
        combined = combined[selected]
    return combined + origin


def evaluate_vertical_interface(
    source_surface_points: np.ndarray | Sequence[Sequence[float]],
    target_surface_points: np.ndarray | Sequence[Sequence[float]],
    *,
    source_side: VerticalSide,
    interface_band_m: float = 0.04,
    maximum_horizontal_distance_m: float = 0.08,
    contact_tolerance_m: float = 0.03,
    insertion_allowed: bool = False,
    maximum_insertion_m: float = 0.08,
    minimum_horizontal_coverage_ratio: float = 0.80,
    minimum_contact_ratio: float = 0.75,
    local_target_neighbors: int = 32,
    contact_mode: ContactMode = "area",
) -> dict[str, Any]:
    """Evaluate a nominally vertical bearing/contact relationship.

    Positive signed separation always means an open gap. Negative separation
    means overlap/insertion. Shallow insertion only passes when the relationship
    explicitly allows it; this keeps the gate conservative for unknown joints.
    The result is a candidate-geometry check and never a formal acceptance.
    """

    source = _points_xyz(source_surface_points, label="source_surface_points")
    target = _points_xyz(target_surface_points, label="target_surface_points")
    if source_side not in {"top", "bottom"}:
        raise ValueError("source_side must be 'top' or 'bottom'")
    if contact_mode not in {"area", "localized"}:
        raise ValueError("contact_mode must be 'area' or 'localized'")
    if interface_band_m <= 0.0:
        raise ValueError("interface_band_m must be positive")
    if maximum_horizontal_distance_m <= 0.0:
        raise ValueError("maximum_horizontal_distance_m must be positive")
    if contact_tolerance_m < 0.0:
        raise ValueError("contact_tolerance_m cannot be negative")
    if maximum_insertion_m < contact_tolerance_m:
        raise ValueError("maximum_insertion_m cannot be less than contact_tolerance_m")
    if not 0.0 <= minimum_horizontal_coverage_ratio <= 1.0:
        raise ValueError("minimum_horizontal_coverage_ratio must be within [0, 1]")
    if not 0.0 <= minimum_contact_ratio <= 1.0:
        raise ValueError("minimum_contact_ratio must be within [0, 1]")
    if local_target_neighbors <= 0:
        raise ValueError("local_target_neighbors must be positive")

    interface_z = (
        float(np.max(source[:, 2])) if source_side == "top" else float(np.min(source[:, 2]))
    )
    if source_side == "top":
        interface_points = source[source[:, 2] >= interface_z - interface_band_m].copy()
    else:
        interface_points = source[source[:, 2] <= interface_z + interface_band_m].copy()
    # Compare against the nominal end plane. Side-face samples in the band would
    # otherwise manufacture a small gap even when the actual end face is flush.
    interface_points[:, 2] = interface_z

    xy_tree = cKDTree(target[:, :2])
    neighbor_count = min(local_target_neighbors, len(target))
    horizontal_distances, target_indices = xy_tree.query(
        interface_points[:, :2], k=neighbor_count, workers=-1
    )
    if neighbor_count == 1:
        horizontal_distances = horizontal_distances[:, None]
        target_indices = target_indices[:, None]
    horizontal_distances = np.asarray(horizontal_distances, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    horizontal_valid = horizontal_distances <= maximum_horizontal_distance_m

    candidate_target_z = target[target_indices, 2]
    vertical_distance = np.abs(candidate_target_z - interface_z)
    vertical_distance[~horizontal_valid] = np.inf
    selected_columns = np.argmin(vertical_distance, axis=1)
    row_indices = np.arange(len(interface_points), dtype=np.int64)
    has_horizontal_target = np.any(horizontal_valid, axis=1)
    selected_target_z = candidate_target_z[row_indices, selected_columns]
    selected_horizontal_distance = horizontal_distances[row_indices, selected_columns]

    if source_side == "top":
        signed_separation = selected_target_z - interface_z
    else:
        signed_separation = interface_z - selected_target_z
    valid_separation = signed_separation[has_horizontal_target]
    allowed_insertion = maximum_insertion_m if insertion_allowed else contact_tolerance_m

    valid_contact = (valid_separation <= contact_tolerance_m) & (
        valid_separation >= -allowed_insertion
    )
    horizontal_coverage_ratio = float(np.mean(has_horizontal_target))
    contact_ratio = float(np.sum(valid_contact) / len(interface_points))
    if len(valid_separation):
        open_gap = np.maximum(valid_separation, 0.0)
        insertion = np.maximum(-valid_separation, 0.0)
        metrics: dict[str, float | int | None] = {
            "closest_signed_separation_m": float(
                valid_separation[np.argmin(np.abs(valid_separation))]
            ),
            "signed_separation_p10_m": float(np.percentile(valid_separation, 10)),
            "signed_separation_p50_m": float(np.percentile(valid_separation, 50)),
            "signed_separation_p90_m": float(np.percentile(valid_separation, 90)),
            "absolute_separation_p90_m": float(np.percentile(np.abs(valid_separation), 90)),
            "open_gap_p90_m": float(np.percentile(open_gap, 90)),
            "maximum_open_gap_m": float(np.max(open_gap)),
            "insertion_p90_m": float(np.percentile(insertion, 90)),
            "maximum_insertion_m": float(np.max(insertion)),
            "selected_horizontal_distance_p90_m": float(
                np.percentile(
                    selected_horizontal_distance[has_horizontal_target],
                    90,
                )
            ),
        }
    else:
        metrics = {
            "closest_signed_separation_m": None,
            "signed_separation_p10_m": None,
            "signed_separation_p50_m": None,
            "signed_separation_p90_m": None,
            "absolute_separation_p90_m": None,
            "open_gap_p90_m": None,
            "maximum_open_gap_m": None,
            "insertion_p90_m": None,
            "maximum_insertion_m": None,
            "selected_horizontal_distance_p90_m": None,
        }

    if contact_mode == "localized" and len(valid_separation):
        closest_separation = float(valid_separation[np.argmin(np.abs(valid_separation))])
        gap_passed = closest_separation <= contact_tolerance_m
        insertion_passed = closest_separation >= -allowed_insertion
    else:
        gap_passed = bool(
            len(valid_separation)
            and float(np.percentile(np.maximum(valid_separation, 0.0), 90)) <= contact_tolerance_m
        )
        insertion_passed = bool(
            len(valid_separation)
            and float(np.percentile(np.maximum(-valid_separation, 0.0), 90)) <= allowed_insertion
        )
    gates = {
        "horizontal_coverage": horizontal_coverage_ratio >= minimum_horizontal_coverage_ratio,
        "open_gap": gap_passed,
        "insertion": insertion_passed,
        "contact_fraction": contact_ratio >= minimum_contact_ratio,
    }
    candidate_geometry_passed = all(gates.values())
    failed_gates = [name for name, passed in gates.items() if not passed]
    return {
        "status": (
            "candidate_geometry_interface_pass_pending_review"
            if candidate_geometry_passed
            else "review_required_interface_gate_failed"
        ),
        "candidate_geometry_passed": candidate_geometry_passed,
        "formal_acceptance": False,
        "source_side": source_side,
        "contact_mode": contact_mode,
        "insertion_allowed": insertion_allowed,
        "source_surface_sample_count": len(source),
        "source_interface_sample_count": len(interface_points),
        "target_surface_sample_count": len(target),
        "target_overlap_sample_count": int(np.sum(has_horizontal_target)),
        "interface_z_m": interface_z,
        "horizontal_coverage_ratio": horizontal_coverage_ratio,
        "contact_ratio": contact_ratio,
        "metrics": metrics,
        "thresholds": {
            "interface_band_m": interface_band_m,
            "maximum_horizontal_distance_m": maximum_horizontal_distance_m,
            "contact_tolerance_m": contact_tolerance_m,
            "maximum_insertion_m": allowed_insertion,
            "minimum_horizontal_coverage_ratio": minimum_horizontal_coverage_ratio,
            "minimum_contact_ratio": minimum_contact_ratio,
            "local_target_neighbors": local_target_neighbors,
            "contact_mode": contact_mode,
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "interpretation": "candidate_geometry_check_only_not_formal_acceptance",
    }


def audit_obj_vertical_interfaces(
    obj_path: str | Path,
    interface_specs: Sequence[dict[str, Any]],
    *,
    origin_xyz: np.ndarray | Sequence[float] | None = None,
    sampling_spacing_m: float = 0.04,
    maximum_samples_per_object: int = 100_000,
) -> dict[str, Any]:
    """Evaluate several declared interfaces in one OBJ while reusing samples."""

    source = Path(obj_path).resolve()
    model = parse_obj_model(source)
    identifiers = [str(spec.get("interface_id", "")) for spec in interface_specs]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every interface spec must have a non-empty interface_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("interface_id values must be unique")

    cache: dict[str, np.ndarray] = {}

    def samples(object_name: str) -> np.ndarray:
        if object_name not in cache:
            cache[object_name] = sample_object_triangle_surface(
                model,
                object_name,
                origin_xyz=origin_xyz,
                spacing_m=sampling_spacing_m,
                maximum_samples=maximum_samples_per_object,
            )
        return cache[object_name]

    results: list[dict[str, Any]] = []
    for spec in interface_specs:
        interface_id = str(spec["interface_id"])
        source_object = str(spec.get("source_object", ""))
        target_objects = [str(value) for value in spec.get("target_objects", [])]
        missing_objects = [
            name
            for name in [source_object, *target_objects]
            if not name or not model.faces_by_object.get(name)
        ]
        if not target_objects:
            missing_objects.append("<no-target-objects-declared>")
        common = {
            "interface_id": interface_id,
            "declared_relation": spec.get("declared_relation"),
            "source_object": source_object,
            "target_objects": target_objects,
        }
        if missing_objects:
            results.append(
                {
                    **common,
                    "status": "review_required_missing_interface_objects",
                    "candidate_geometry_passed": False,
                    "formal_acceptance": False,
                    "missing_objects": sorted(set(missing_objects)),
                    "interpretation": "candidate_geometry_check_only_not_formal_acceptance",
                }
            )
            continue
        target_samples = np.vstack([samples(name) for name in target_objects])
        evaluation_keys = {
            "source_side",
            "interface_band_m",
            "maximum_horizontal_distance_m",
            "contact_tolerance_m",
            "insertion_allowed",
            "maximum_insertion_m",
            "minimum_horizontal_coverage_ratio",
            "minimum_contact_ratio",
            "local_target_neighbors",
            "contact_mode",
        }
        settings = {key: spec[key] for key in evaluation_keys if key in spec}
        results.append(
            {
                **common,
                **evaluate_vertical_interface(
                    samples(source_object),
                    target_samples,
                    **settings,
                ),
            }
        )

    passed_count = sum(bool(result["candidate_geometry_passed"]) for result in results)
    return {
        "schema": "railway_recon.obj_vertical_interface_audit/v1",
        "source_obj": str(source),
        "source_obj_sha256": sha256_file(source),
        "sampling": {
            "method": "deterministic_triangle_barycentric_lattice",
            "spacing_m": sampling_spacing_m,
            "maximum_samples_per_object": maximum_samples_per_object,
        },
        "summary": {
            "interface_count": len(results),
            "candidate_geometry_pass_count": passed_count,
            "review_required_count": len(results) - passed_count,
            "formal_acceptance": False,
        },
        "interfaces": results,
        "interpretation": "candidate_geometry_check_only_not_formal_acceptance",
    }
