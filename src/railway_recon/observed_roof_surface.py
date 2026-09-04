from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import CorridorFrame


def _validate_range(name: str, value: list[float] | tuple[float, float]) -> tuple[float, float]:
    low, high = (float(item) for item in value)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"{name} must contain two increasing finite values")
    return low, high


def fit_observed_roof_surface(
    longitudinal: np.ndarray,
    cross: np.ndarray,
    elevation: np.ndarray,
    *,
    longitudinal_range_m: list[float] | tuple[float, float],
    cross_range_m: list[float] | tuple[float, float],
    initial_cross_line: list[float] | tuple[float, float],
    initial_residual_m: float = 0.12,
    fit_residual_m: float = 0.10,
    final_residual_m: float = 0.08,
    profile_length_m: float = 1.0,
    cross_bin_m: float = 0.25,
    minimum_profile_points: int = 80,
    minimum_cross_coverage_ratio: float = 0.70,
    maximum_profile_residual_p90_m: float = 0.04,
    minimum_passed_profile_ratio: float = 0.90,
    refinement_iterations: int = 5,
) -> dict[str, Any]:
    """Fit one reviewed roof plane and preserve unsupported longitudinal gaps.

    The caller must provide a reviewed cross-section seed. This function deliberately
    does not discover or join unrelated roof bands.
    """
    s = np.asarray(longitudinal, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    z = np.asarray(elevation, dtype=np.float64)
    if s.shape != c.shape or s.shape != z.shape or s.ndim != 1:
        raise ValueError("longitudinal, cross and elevation must be matching 1D arrays")
    if not s.size:
        raise ValueError("At least one point is required")
    s_low, s_high = _validate_range("longitudinal_range_m", longitudinal_range_m)
    c_low, c_high = _validate_range("cross_range_m", cross_range_m)
    if min(initial_residual_m, fit_residual_m, final_residual_m) <= 0.0:
        raise ValueError("Residual thresholds must be positive")
    if profile_length_m <= 0.0 or cross_bin_m <= 0.0:
        raise ValueError("Profile and cross-bin sizes must be positive")
    if not 0.0 <= minimum_cross_coverage_ratio <= 1.0:
        raise ValueError("minimum_cross_coverage_ratio must be within 0..1")
    if not 0.0 <= minimum_passed_profile_ratio <= 1.0:
        raise ValueError("minimum_passed_profile_ratio must be within 0..1")

    initial_b, initial_d = (float(value) for value in initial_cross_line)
    in_envelope = (
        (s >= s_low)
        & (s <= s_high)
        & (c >= c_low)
        & (c <= c_high)
        & np.isfinite(s)
        & np.isfinite(c)
        & np.isfinite(z)
    )
    initial_residual = np.abs(z - (initial_b * c + initial_d))
    candidate = in_envelope & (initial_residual <= initial_residual_m)
    if int(np.count_nonzero(candidate)) < 3:
        raise ValueError("Reviewed seed has fewer than three candidate points")

    fit_mask = candidate.copy()
    coefficients = np.asarray([0.0, initial_b, initial_d], dtype=np.float64)
    for _ in range(max(1, int(refinement_iterations))):
        design = np.column_stack((s[fit_mask], c[fit_mask], np.ones(np.count_nonzero(fit_mask))))
        coefficients = np.linalg.lstsq(design, z[fit_mask], rcond=None)[0]
        residual = np.abs(z - (coefficients[0] * s + coefficients[1] * c + coefficients[2]))
        refined = candidate & (residual <= fit_residual_m)
        if int(np.count_nonzero(refined)) < 3:
            raise ValueError("Roof refinement rejected all candidate points")
        if np.array_equal(refined, fit_mask):
            break
        fit_mask = refined

    residual = np.abs(z - (coefficients[0] * s + coefficients[1] * c + coefficients[2]))
    inliers = candidate & (residual <= final_residual_m)
    inlier_residual = residual[inliers]
    if not inlier_residual.size:
        raise ValueError("No roof inliers remain after final residual gate")

    profile_count = math.ceil((s_high - s_low) / profile_length_m)
    expected_cross_bins = math.ceil((c_high - c_low) / cross_bin_m)
    profiles: list[dict[str, Any]] = []
    for index in range(profile_count):
        low = s_low + index * profile_length_m
        high = min(s_high, low + profile_length_m)
        profile_mask = inliers & (s >= low) & (
            s <= high if index == profile_count - 1 else s < high
        )
        point_count = int(np.count_nonzero(profile_mask))
        occupied_bins = set(
            np.floor((c[profile_mask] - c_low) / cross_bin_m).astype(np.int64).tolist()
        )
        coverage = len(
            {value for value in occupied_bins if 0 <= value < expected_cross_bins}
        ) / expected_cross_bins
        residual_p90 = (
            float(np.percentile(residual[profile_mask], 90.0)) if point_count else None
        )
        passed = (
            point_count >= minimum_profile_points
            and coverage >= minimum_cross_coverage_ratio
            and residual_p90 is not None
            and residual_p90 <= maximum_profile_residual_p90_m
        )
        profiles.append(
            {
                "longitudinal_range_m": [low, high],
                "point_count": point_count,
                "cross_bin_coverage_ratio": coverage,
                "absolute_residual_p90_m": residual_p90,
                "passed": passed,
            }
        )

    runs: list[list[dict[str, Any]]] = []
    for profile in profiles:
        if not profile["passed"]:
            continue
        if not runs or float(profile["longitudinal_range_m"][0]) > float(
            runs[-1][-1]["longitudinal_range_m"][1]
        ) + 1.0e-8:
            runs.append([profile])
        else:
            runs[-1].append(profile)
    supported_runs = [
        [float(run[0]["longitudinal_range_m"][0]), float(run[-1]["longitudinal_range_m"][1])]
        for run in runs
    ]
    passed_count = sum(1 for profile in profiles if profile["passed"])
    passed_ratio = passed_count / len(profiles)
    passed = passed_ratio >= minimum_passed_profile_ratio and bool(supported_runs)
    return {
        "plane_z_equals_a_s_plus_b_c_plus_d": [float(value) for value in coefficients],
        "longitudinal_range_m": [s_low, s_high],
        "cross_range_m": [c_low, c_high],
        "candidate_point_count": int(np.count_nonzero(candidate)),
        "inlier_point_count": int(np.count_nonzero(inliers)),
        "inlier_absolute_residual_p50_m": float(np.median(inlier_residual)),
        "inlier_absolute_residual_p90_m": float(np.percentile(inlier_residual, 90.0)),
        "profiles": profiles,
        "profile_count": len(profiles),
        "passed_profile_count": passed_count,
        "passed_profile_ratio": passed_ratio,
        "supported_runs_m": supported_runs,
        "gate_thresholds": {
            "initial_residual_m": initial_residual_m,
            "fit_residual_m": fit_residual_m,
            "final_residual_m": final_residual_m,
            "profile_length_m": profile_length_m,
            "cross_bin_m": cross_bin_m,
            "minimum_profile_points": minimum_profile_points,
            "minimum_cross_coverage_ratio": minimum_cross_coverage_ratio,
            "maximum_profile_residual_p90_m": maximum_profile_residual_p90_m,
            "minimum_passed_profile_ratio": minimum_passed_profile_ratio,
        },
        "unsupported_gaps_filled": False,
        "passed": passed,
        "status": "pass" if passed else "review_required",
    }


def roof_slab_mesh(
    frame: CorridorFrame,
    plane: list[float] | tuple[float, float, float],
    longitudinal_range_m: list[float] | tuple[float, float],
    cross_range_m: list[float] | tuple[float, float],
    *,
    thickness_m: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Create one closed sloped slab for an already accepted observed run."""
    s_low, s_high = _validate_range("longitudinal_range_m", longitudinal_range_m)
    c_low, c_high = _validate_range("cross_range_m", cross_range_m)
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    a, b, d = (float(value) for value in plane)
    local = np.asarray(
        [
            [s_low, c_low],
            [s_high, c_low],
            [s_high, c_high],
            [s_low, c_high],
        ],
        dtype=np.float64,
    )
    xy = frame.world_xy(local[:, 0], local[:, 1])
    top_z = a * local[:, 0] + b * local[:, 1] + d
    bottom = np.column_stack((xy, top_z - thickness_m))
    top = np.column_stack((xy, top_z))
    vertices = np.vstack((bottom, top))
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def roof_profile_slab_mesh(
    frame: CorridorFrame,
    profiles_station_cross_z: np.ndarray | list[list[float]],
    *,
    thickness_m: float,
    cap_start: bool = True,
    cap_end: bool = True,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Create a slab from ordered two-edge roof profiles.

    Each profile is ``[station, low_cross, low_top_z, high_cross, high_top_z]``.
    This is useful for a directly observed roof run whose endpoints vary along the
    corridor, and for joining an evidence-fitted run to an exact existing boundary.
    ``cap_start`` and ``cap_end`` may be disabled when that boundary is already
    owned by an adjoining closed slab. The caller remains responsible for proving
    every interval and open assembly seam is supported.
    """

    profiles = np.asarray(profiles_station_cross_z, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[1] != 5:
        raise ValueError("profiles_station_cross_z must have shape (N, 5)")
    if len(profiles) < 2:
        raise ValueError("At least two ordered profiles are required")
    if not np.all(np.isfinite(profiles)):
        raise ValueError("Profiles contain non-finite values")
    if np.any(np.diff(profiles[:, 0]) <= 0.0):
        raise ValueError("Profile stations must be strictly increasing")
    if np.any(profiles[:, 3] <= profiles[:, 1]):
        raise ValueError("Every profile must have increasing cross coordinates")
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")

    station = np.repeat(profiles[:, 0], 2)
    cross = profiles[:, [1, 3]].reshape(-1)
    top_z = profiles[:, [2, 4]].reshape(-1)
    xy = frame.world_xy(station, cross)
    top = np.column_stack((xy, top_z))
    bottom = top.copy()
    bottom[:, 2] -= thickness_m
    vertices = np.vstack((top, bottom))
    offset = len(top)
    faces: list[tuple[int, ...]] = []
    for index in range(len(profiles) - 1):
        low_a, high_a = 2 * index, 2 * index + 1
        low_b, high_b = low_a + 2, high_a + 2
        faces.extend(
            (
                (low_a, low_b, high_b, high_a),
                (
                    offset + low_a,
                    offset + high_a,
                    offset + high_b,
                    offset + low_b,
                ),
                (low_a, offset + low_a, offset + low_b, low_b),
                (high_a, high_b, offset + high_b, offset + high_a),
            )
        )
    final_low, final_high = len(top) - 2, len(top) - 1
    if cap_start:
        faces.append((0, 1, offset + 1, offset))
    if cap_end:
        faces.append(
            (
                final_low,
                offset + final_low,
                offset + final_high,
                final_high,
            )
        )
    return vertices, faces
