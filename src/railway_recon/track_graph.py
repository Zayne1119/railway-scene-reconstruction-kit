from __future__ import annotations

import math
from datetime import UTC, datetime
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import load_json, sha256_file, sha256_json, write_json

ACCEPTED_REVIEW_STATUSES = {
    "accepted",
    "reviewed_accepted",
    "owner_override_accepted",
}
CONFIRMED_INTERIOR_BOUNDARIES = {"confirmed_turnout", "confirmed_buffer_stop"}


class RouteSampler:
    """Interpolate the canonical route by cumulative camera-chainage."""

    def __init__(
        self,
        trajectory: list[dict[str, Any]],
        tangent_window_m: float = 5.0,
    ) -> None:
        if len(trajectory) < 2:
            raise ValueError("TrackGraph requires at least two camera poses")
        samples: list[dict[str, Any]] = []
        for item in trajectory:
            if samples and math.isclose(
                float(item["distance_m"]), float(samples[-1]["distance_m"]), abs_tol=1e-9
            ):
                samples[-1] = item
            else:
                samples.append(item)
        if len(samples) < 2:
            raise ValueError("Camera trajectory has no non-zero route length")
        self.chainage = np.asarray([item["distance_m"] for item in samples], dtype=np.float64)
        self.xyz = np.asarray(
            [[item["x"], item["y"], item["z"]] for item in samples], dtype=np.float64
        )
        self.tangent_window_m = float(tangent_window_m)
        if self.tangent_window_m <= 0:
            raise ValueError("Route tangent window must be positive")

    @property
    def start_m(self) -> float:
        return float(self.chainage[0])

    @property
    def end_m(self) -> float:
        return float(self.chainage[-1])

    def _sample_point(self, chainage_m: float) -> np.ndarray:
        value = float(np.clip(chainage_m, self.start_m, self.end_m))
        right = int(np.searchsorted(self.chainage, value, side="right"))
        right = min(max(right, 1), len(self.chainage) - 1)
        left = right - 1
        span = float(self.chainage[right] - self.chainage[left])
        fraction = 0.0 if span <= 0 else (value - float(self.chainage[left])) / span
        return self.xyz[left] + fraction * (self.xyz[right] - self.xyz[left])

    def sample(self, chainage_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = float(np.clip(chainage_m, self.start_m, self.end_m))
        point = self._sample_point(value)
        before = max(self.start_m, value - self.tangent_window_m)
        after = min(self.end_m, value + self.tangent_window_m)
        tangent = self._sample_point(after)[:2] - self._sample_point(before)[:2]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-12:
            raise ValueError(f"Camera trajectory tangent is undefined at {value:.3f} m")
        tangent /= norm
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        return point, tangent, normal


def load_track_graph_settings(project: ProjectConfig) -> dict[str, Any]:
    configured = project.value.get("algorithms", {}).get("track_graph")
    if configured:
        return load_json(project.resolve(configured))
    resource = resources.files("railway_recon.resources").joinpath("track-graph.default.json")
    with resource.open("r", encoding="utf-8") as stream:
        import json

        return json.load(stream)


def validate_track_graph_bindings(
    project: ProjectConfig, graph: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    camera_path = project.input_path("camera_csv")
    expected_camera_hash = graph.get("route", {}).get("camera_csv_sha256")
    if camera_path is None or not camera_path.is_file():
        failures.append({"reason": "camera_csv_missing", "path": str(camera_path)})
    elif expected_camera_hash != sha256_file(camera_path):
        failures.append(
            {
                "reason": "camera_csv_hash_mismatch",
                "expected": expected_camera_hash,
                "actual": sha256_file(camera_path),
            }
        )
    current_settings_hash = sha256_json(load_track_graph_settings(project))
    if graph.get("settings_sha256") != current_settings_hash:
        failures.append(
            {
                "reason": "track_graph_settings_hash_mismatch",
                "expected": graph.get("settings_sha256"),
                "actual": current_settings_hash,
            }
        )
    return failures


def _source_review_status(report: dict[str, Any]) -> str:
    review = str(report.get("review_status", "")).strip().lower()
    status = str(report.get("status", "")).strip().lower()
    return "accepted" if review in ACCEPTED_REVIEW_STATUSES or status == "accepted" else "pending"


def _source_review_mode(report: dict[str, Any]) -> str:
    review_status = str(report.get("review_status", "")).strip().lower()
    review = report.get("review", {})
    if review_status == "owner_override_accepted":
        return "project_owner_override"
    if review_status in {"accepted", "reviewed_accepted"}:
        return str(review.get("mode", "independent_review"))
    if str(report.get("status", "")).strip().lower() == "accepted":
        return "legacy_accepted"
    return "pending"


def _closest_line(report: dict[str, Any], cross_m: float) -> dict[str, Any]:
    lines = report.get("rail_lines", [])
    if not lines:
        raise ValueError(f"Rail report has no rail_lines: {report.get('segment_id')}")
    return min(lines, key=lambda item: abs(float(item["cross_position_m"]) - cross_m))


def _closest_peak_coverage(report: dict[str, Any], cross_m: float) -> float | None:
    peaks = report.get("peaks", [])
    if not peaks:
        return None
    peak = min(peaks, key=lambda item: abs(float(item["cross_position_m"]) - cross_m))
    return float(peak["coverage"])


def _fitted_line_value(
    line: dict[str, Any],
    prefix: str,
    longitudinal_m: float,
    fallback: float,
) -> float:
    slope = line.get(f"{prefix}_fit_slope_m_per_m")
    intercept = line.get(f"{prefix}_fit_intercept_m")
    if slope is None or intercept is None:
        return fallback
    return float(slope) * longitudinal_m + float(intercept)


def _observation_from_pair(
    segment: dict[str, Any],
    report: dict[str, Any],
    pair: dict[str, Any],
    route: RouteSampler,
    settings: dict[str, Any],
) -> dict[str, Any]:
    start = float(segment["chainage_start_m"])
    end = float(segment["chainage_end_m"])
    midpoint = (start + end) / 2.0
    frame = report["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross_axis = np.asarray(frame["cross_xy"], dtype=np.float64)
    if origin.shape != (2,) or along.shape != (2,) or cross_axis.shape != (2,):
        raise ValueError(f"Invalid corridor frame in {report.get('segment_id')}")
    along_norm = float(np.linalg.norm(along))
    cross_norm = float(np.linalg.norm(cross_axis))
    if along_norm <= 1e-12 or cross_norm <= 1e-12:
        raise ValueError(f"Degenerate corridor frame in {report.get('segment_id')}")
    along /= along_norm
    cross_axis /= cross_norm
    determinant = float(np.linalg.det(np.column_stack((along, cross_axis))))

    positions = [float(value) for value in pair["cross_positions_m"]]
    if len(positions) != 2:
        raise ValueError(f"Rail pair must contain two rails: {pair}")
    longitudinal = report.get("longitudinal_range_m")
    if not isinstance(longitudinal, list) or len(longitudinal) != 2:
        raise ValueError(f"Missing longitudinal_range_m in {report.get('segment_id')}")

    lines = [_closest_line(report, position) for position in positions]
    z_values = [line.get("median_z_m") for line in lines]
    if any(value is None for value in z_values):
        raise ValueError(f"Rail pair has no median height: {report.get('segment_id')} {pair}")
    rail_top_z_values = [float(value) for value in z_values]
    route_start, _, normal_start = route.sample(start)
    route_midpoint, route_tangent, normal_midpoint = route.sample(midpoint)
    route_end, _, normal_end = route.sample(end)
    route_values = [route_start, route_midpoint, route_end]
    route_normals = [normal_start, normal_midpoint, normal_end]
    local_values = [float(np.dot(value[:2] - origin, along)) for value in route_values]
    lateral_values: list[float] = []
    world_fit_values: list[np.ndarray] = []
    z_fit_values: list[float] = []
    crosslevel_values: list[float] = []
    for local_value, route_value, route_normal in zip(local_values, route_values, route_normals):
        line_cross = [
            _fitted_line_value(line, "cross", local_value, position)
            for line, position in zip(lines, positions)
        ]
        line_z = [
            _fitted_line_value(line, "z", local_value, fallback)
            for line, fallback in zip(lines, rail_top_z_values)
        ]
        candidate_world = origin + local_value * along + float(np.mean(line_cross)) * cross_axis
        world_fit_values.append(candidate_world)
        lateral_values.append(float(np.dot(candidate_world - route_value[:2], route_normal)))
        z_fit_values.append(float(np.mean(line_z)))
        crosslevel_values.append(abs(line_z[1] - line_z[0]))
    lateral_start, lateral_offset, lateral_end = lateral_values
    world_start, world_midpoint, world_end = world_fit_values
    z_start, z_m, z_end = z_fit_values
    direction_dot = float(np.dot(along, route_tangent))
    rail_center_spacing = float(pair.get("rail_center_spacing_m", pair["separation_m"]))
    gauge = float(
        pair.get(
            "gauge_m",
            rail_center_spacing - float(settings.get("rail_head_width_m", 0.073)),
        )
    )
    evidence_level = str(pair.get("evidence_level", "observed"))
    coverages = [_closest_peak_coverage(report, position) for position in positions]
    support = (
        None
        if evidence_level != "observed" or any(value is None for value in coverages)
        else float(min(coverages))
    )
    joint_support = pair.get("joint_support_ratio")
    asymmetric_support = pair.get("asymmetric_support_ratio")
    maximum_internal_joint_gap = pair.get("maximum_internal_joint_gap_m")

    return {
        "id": f"{segment['id']}:{pair['track_id']}",
        "segment_id": str(segment["id"]),
        "local_track_id": str(pair["track_id"]),
        "global_track_id": None,
        "chainage_start_m": start,
        "chainage_end_m": end,
        "lateral_offset_m": lateral_offset,
        "lateral_offset_start_m": lateral_start,
        "lateral_offset_end_m": lateral_end,
        "center_world_start_x_m": float(world_start[0]),
        "center_world_start_y_m": float(world_start[1]),
        "center_world_x_m": float(world_midpoint[0]),
        "center_world_y_m": float(world_midpoint[1]),
        "center_world_end_x_m": float(world_end[0]),
        "center_world_end_y_m": float(world_end[1]),
        "rail_cross_positions_local_m": positions,
        "gauge_m": gauge,
        "rail_center_spacing_m": rail_center_spacing,
        "observed_rail_center_spacing_m": float(
            pair.get("observed_rail_center_spacing_m", rail_center_spacing)
        ),
        "rail_position_correction_m": float(pair.get("rail_position_correction_m", 0.0)),
        "rail_top_z_m": z_m,
        "rail_top_start_z_m": z_start,
        "rail_top_end_z_m": z_end,
        "rail_top_crosslevel_m": max(crosslevel_values),
        "direction_dot": direction_dot,
        "frame_determinant": determinant,
        "support_ratio": support,
        "paired_support_evaluated": joint_support is not None,
        "joint_support_ratio": (
            None if joint_support is None else float(joint_support)
        ),
        "asymmetric_support_ratio": (
            None if asymmetric_support is None else float(asymmetric_support)
        ),
        "maximum_internal_joint_gap_m": (
            None
            if maximum_internal_joint_gap is None
            else float(maximum_internal_joint_gap)
        ),
        "pair_continuity_status": pair.get("pair_continuity_status"),
        "point_count": int(sum(int(line.get("point_count", 0)) for line in lines)),
        "pair_score": float(pair.get("score", 0.0)),
        "evidence_level": evidence_level,
        "review_status": _source_review_status(report),
        "start_boundary_reason": pair.get("start_boundary_reason"),
        "end_boundary_reason": pair.get("end_boundary_reason"),
    }


def _match_observations(
    batches: list[list[dict[str, Any]]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    matching = settings["matching"]
    next_track = 1
    events: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        batch.sort(key=lambda item: item["lateral_offset_m"])
        if not previous:
            for item in batch:
                item["global_track_id"] = f"TRACK-{next_track:04d}"
                next_track += 1
            previous = batch
            continue

        identity_gap = float(batch[0]["chainage_start_m"] - previous[0]["chainage_end_m"])
        eligible_identity = abs(identity_gap) <= float(matching["maximum_identity_gap_m"])
        assignments: list[tuple[int, int, float]] = []
        if eligible_identity and previous and batch:
            impossible = 1e9
            costs = np.full((len(previous), len(batch)), impossible, dtype=np.float64)
            for row, prior in enumerate(previous):
                for column, current in enumerate(batch):
                    lateral = abs(
                        float(current["lateral_offset_m"]) - float(prior["lateral_offset_m"])
                    )
                    vertical = abs(float(current["rail_top_z_m"]) - float(prior["rail_top_z_m"]))
                    gauge = abs(float(current["gauge_m"]) - float(prior["gauge_m"]))
                    if (
                        lateral <= float(matching["maximum_lateral_difference_m"])
                        and vertical <= float(matching["maximum_vertical_difference_m"])
                        and gauge <= float(matching["maximum_gauge_difference_m"])
                        and float(prior["direction_dot"]) > 0
                        and float(current["direction_dot"]) > 0
                    ):
                        costs[row, column] = (
                            lateral / float(matching["maximum_lateral_difference_m"])
                            + vertical / float(matching["maximum_vertical_difference_m"])
                            + gauge / float(matching["maximum_gauge_difference_m"])
                        )
            rows, columns = linear_sum_assignment(costs)
            assignments = [
                (int(row), int(column), float(costs[row, column]))
                for row, column in zip(rows, columns)
                if costs[row, column] < impossible
            ]

        assigned_current: set[int] = set()
        match_records: list[dict[str, Any]] = []
        for row, column, cost in assignments:
            current = batch[column]
            prior = previous[row]
            current["global_track_id"] = prior["global_track_id"]
            assigned_current.add(column)
            match_records.append(
                {
                    "global_track_id": current["global_track_id"],
                    "previous_observation_id": prior["id"],
                    "current_observation_id": current["id"],
                    "previous_lateral_m": prior["lateral_offset_m"],
                    "current_lateral_m": current["lateral_offset_m"],
                    "cost": cost,
                }
            )
        for column, current in enumerate(batch):
            if column not in assigned_current:
                current["global_track_id"] = f"TRACK-{next_track:04d}"
                next_track += 1

        ordered = sorted(match_records, key=lambda item: item["previous_lateral_m"])
        crossing = any(
            right["current_lateral_m"] <= left["current_lateral_m"]
            for left, right in pairwise(ordered)
        )
        events.append(
            {
                "from_segment_id": previous[0]["segment_id"],
                "to_segment_id": batch[0]["segment_id"],
                "signed_chainage_gap_m": identity_gap,
                "identity_matching_attempted": eligible_identity,
                "match_count": len(match_records),
                "previous_count": len(previous),
                "current_count": len(batch),
                "order_crossing": crossing,
                "matches": match_records,
            }
        )
        previous = batch
    return events


def _classify_pair_continuity_recovery(
    observations: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Find adjacent same-track evidence without changing rail geometry."""

    quality = settings["quality"]
    maximum_lateral = float(
        quality.get("maximum_pair_recovery_lateral_difference_m", 0.30)
    )
    maximum_gauge = float(
        quality.get("maximum_pair_recovery_gauge_difference_m", 0.03)
    )
    maximum_chainage_gap = float(
        quality.get("maximum_pair_recovery_chainage_gap_m", 5.0)
    )
    maximum_rail_top = float(
        quality.get("maximum_pair_recovery_rail_top_difference_m", 0.15)
    )
    results: list[dict[str, Any]] = []
    for observation in observations:
        if not observation.get("paired_support_evaluated"):
            observation["pair_recovery_status"] = "not_evaluated"
            continue
        if observation.get("pair_continuity_status") == "pass":
            observation["pair_recovery_status"] = "not_needed"
            continue

        candidates: list[dict[str, Any]] = []
        for other in observations:
            if other is observation:
                continue
            if other.get("global_track_id") != observation.get("global_track_id"):
                continue
            if other.get("pair_continuity_status") != "pass":
                continue
            if float(other["chainage_end_m"]) <= float(observation["chainage_start_m"]):
                chainage_gap = float(observation["chainage_start_m"]) - float(
                    other["chainage_end_m"]
                )
                rail_top_difference = abs(
                    float(observation["rail_top_start_z_m"])
                    - float(other["rail_top_end_z_m"])
                )
            elif float(observation["chainage_end_m"]) <= float(other["chainage_start_m"]):
                chainage_gap = float(other["chainage_start_m"]) - float(
                    observation["chainage_end_m"]
                )
                rail_top_difference = abs(
                    float(other["rail_top_start_z_m"])
                    - float(observation["rail_top_end_z_m"])
                )
            else:
                chainage_gap = 0.0
                rail_top_difference = abs(
                    float(other["rail_top_z_m"])
                    - float(observation["rail_top_z_m"])
                )
            lateral_difference = abs(
                float(other["lateral_offset_m"])
                - float(observation["lateral_offset_m"])
            )
            gauge_difference = abs(
                float(other["gauge_m"]) - float(observation["gauge_m"])
            )
            if (
                chainage_gap <= maximum_chainage_gap
                and lateral_difference <= maximum_lateral
                and gauge_difference <= maximum_gauge
                and rail_top_difference <= maximum_rail_top
            ):
                candidates.append(
                    {
                        "observation_id": other["id"],
                        "chainage_gap_m": chainage_gap,
                        "lateral_difference_m": lateral_difference,
                        "gauge_difference_m": gauge_difference,
                        "rail_top_difference_m": rail_top_difference,
                    }
                )
        candidates.sort(
            key=lambda item: (
                float(item["chainage_gap_m"]),
                float(item["lateral_difference_m"]),
                float(item["rail_top_difference_m"]),
            )
        )
        selected = candidates[0] if candidates else None
        status = "adjacent_track_evidence" if selected else "unresolved"
        observation["pair_recovery_status"] = status
        observation["pair_recovery_support_observation_id"] = (
            None if selected is None else selected["observation_id"]
        )
        results.append(
            {
                "observation_id": observation["id"],
                "global_track_id": observation["global_track_id"],
                "status": status,
                "recommended_lateral_window_m": [
                    float(observation["lateral_offset_m"]) - maximum_lateral,
                    float(observation["lateral_offset_m"]) + maximum_lateral,
                ],
                "support": selected,
            }
        )
    return results


def _reconcile_observation_seams(
    observations: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Make small, evidence-backed segment-boundary corrections explicit."""

    quality = settings["quality"]
    boundary_tolerance = float(quality["boundary_chainage_tolerance_m"])
    maximum_correction = float(quality["maximum_automatic_correction_m"])
    by_track: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_track.setdefault(str(observation["global_track_id"]), []).append(observation)

    reconciliations: list[dict[str, Any]] = []
    for track_id, members in by_track.items():
        members.sort(key=lambda item: float(item["chainage_start_m"]))
        for left, right in pairwise(members):
            gap = float(right["chainage_start_m"] - left["chainage_end_m"])
            left_lateral = float(left.get("lateral_offset_end_m", left["lateral_offset_m"]))
            right_lateral = float(right.get("lateral_offset_start_m", right["lateral_offset_m"]))
            left_z = float(left.get("rail_top_end_z_m", left["rail_top_z_m"]))
            right_z = float(right.get("rail_top_start_z_m", right["rail_top_z_m"]))
            lateral_difference = abs(right_lateral - left_lateral)
            vertical_difference = abs(right_z - left_z)
            raw_endpoint_difference = math.hypot(lateral_difference, vertical_difference)
            endpoint_correction = 0.5 * raw_endpoint_difference
            eligible = abs(gap) <= boundary_tolerance and endpoint_correction <= maximum_correction
            record = {
                "id": f"RECONCILE:{left['id']}->{right['id']}",
                "track_id": track_id,
                "from_observation_id": left["id"],
                "to_observation_id": right["id"],
                "signed_chainage_gap_m": gap,
                "raw_lateral_difference_m": lateral_difference,
                "raw_vertical_difference_m": vertical_difference,
                "raw_endpoint_difference_m": raw_endpoint_difference,
                "maximum_endpoint_correction_m": endpoint_correction,
                "maximum_allowed_correction_m": maximum_correction,
                "applied": eligible,
            }
            if eligible:
                consensus_lateral = 0.5 * (left_lateral + right_lateral)
                consensus_z = 0.5 * (left_z + right_z)
                left["lateral_offset_end_m"] = consensus_lateral
                right["lateral_offset_start_m"] = consensus_lateral
                left["rail_top_end_z_m"] = consensus_z
                right["rail_top_start_z_m"] = consensus_z
                record.update(
                    {
                        "status": "consensus_applied",
                        "consensus_lateral_offset_m": consensus_lateral,
                        "consensus_rail_top_z_m": consensus_z,
                    }
                )
            else:
                record["status"] = "outside_automatic_correction_limit"
            reconciliations.append(record)
    return reconciliations


def _world_endpoint(
    route: RouteSampler, chainage_m: float, lateral_m: float, z_m: float
) -> list[float]:
    point, _, normal = route.sample(chainage_m)
    xy = point[:2] + lateral_m * normal
    return [float(xy[0]), float(xy[1]), float(z_m)]


def _build_geometry(
    observations: list[dict[str, Any]],
    route: RouteSampler,
    reconciliations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    by_track: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        track_id = str(observation["global_track_id"])
        by_track.setdefault(track_id, []).append(observation)
        start_node = f"{observation['id']}:START"
        end_node = f"{observation['id']}:END"
        start_xyz = _world_endpoint(
            route,
            float(observation["chainage_start_m"]),
            float(observation.get("lateral_offset_start_m", observation["lateral_offset_m"])),
            float(observation.get("rail_top_start_z_m", observation["rail_top_z_m"])),
        )
        end_xyz = _world_endpoint(
            route,
            float(observation["chainage_end_m"]),
            float(observation.get("lateral_offset_end_m", observation["lateral_offset_m"])),
            float(observation.get("rail_top_end_z_m", observation["rail_top_z_m"])),
        )
        nodes.extend(
            [
                {
                    "id": start_node,
                    "track_id": track_id,
                    "chainage_m": observation["chainage_start_m"],
                    "xyz": start_xyz,
                },
                {
                    "id": end_node,
                    "track_id": track_id,
                    "chainage_m": observation["chainage_end_m"],
                    "xyz": end_xyz,
                },
            ]
        )
        edges.append(
            {
                "id": f"EDGE:{observation['id']}",
                "track_id": track_id,
                "from_node": start_node,
                "to_node": end_node,
                "chainage_start_m": observation["chainage_start_m"],
                "chainage_end_m": observation["chainage_end_m"],
                "evidence_level": observation["evidence_level"],
                "source_observation_id": observation["id"],
            }
        )

    reconciliation_by_pair = {
        (str(item["from_observation_id"]), str(item["to_observation_id"])): item
        for item in (reconciliations or [])
    }
    seams: list[dict[str, Any]] = []
    for track_id, members in by_track.items():
        members.sort(key=lambda item: item["chainage_start_m"])
        for left, right in pairwise(members):
            left_xyz = np.asarray(
                _world_endpoint(
                    route,
                    float(left["chainage_end_m"]),
                    float(left.get("lateral_offset_end_m", left["lateral_offset_m"])),
                    float(left.get("rail_top_end_z_m", left["rail_top_z_m"])),
                )
            )
            right_xyz = np.asarray(
                _world_endpoint(
                    route,
                    float(right["chainage_start_m"]),
                    float(right.get("lateral_offset_start_m", right["lateral_offset_m"])),
                    float(right.get("rail_top_start_z_m", right["rail_top_z_m"])),
                )
            )
            _, left_tangent, _ = route.sample(float(left["chainage_end_m"]))
            _, right_tangent, _ = route.sample(float(right["chainage_start_m"]))
            reconciliation = reconciliation_by_pair.get((str(left["id"]), str(right["id"])))
            seam = {
                "id": f"SEAM:{left['id']}->{right['id']}",
                "track_id": track_id,
                "from_observation_id": left["id"],
                "to_observation_id": right["id"],
                "signed_chainage_gap_m": float(right["chainage_start_m"] - left["chainage_end_m"]),
                "lateral_difference_m": abs(
                    float(
                        right.get("lateral_offset_start_m", right["lateral_offset_m"])
                        - left.get("lateral_offset_end_m", left["lateral_offset_m"])
                    )
                ),
                "vertical_difference_m": abs(
                    float(
                        right.get("rail_top_start_z_m", right["rail_top_z_m"])
                        - left.get("rail_top_end_z_m", left["rail_top_z_m"])
                    )
                ),
                "endpoint_3d_difference_m": float(np.linalg.norm(right_xyz - left_xyz)),
                "tangent_dot": float(np.dot(left_tangent, right_tangent)),
                "required_correction_m": float(
                    reconciliation["maximum_endpoint_correction_m"]
                    if reconciliation is not None
                    else np.linalg.norm(right_xyz - left_xyz)
                ),
            }
            if reconciliation is not None:
                seam["reconciliation"] = reconciliation
            seams.append(seam)
    return nodes, edges, seams


def _union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _maximum_gap(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    return max(
        [max(0.0, right[0] - left[1]) for left, right in pairwise(ordered)],
        default=0.0,
    )


def _maximum_union_span(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    maximum = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            maximum = max(maximum, end - start)
            start, end = next_start, next_end
    return max(maximum, end - start)


def _check(check_id: str, failed: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "fail" if failed else "pass",
        "summary": summary,
        "failure_count": len(failed),
        "failures": failed,
    }


def _schema_failures(graph: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return []
    resource = resources.files("railway_recon.resources").joinpath("track-graph.schema.json")
    with resource.open("r", encoding="utf-8") as stream:
        import json

        schema = json.load(stream)
    validator = Draft202012Validator(schema)
    return [
        {
            "path": ".".join(str(part) for part in error.absolute_path) or "<root>",
            "reason": error.message,
        }
        for error in sorted(validator.iter_errors(graph), key=lambda item: list(item.path))
    ]


def audit_track_graph(graph: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    quality = settings["quality"]
    observations = list(graph.get("observations", []))
    tracks = list(graph.get("tracks", []))
    seams = list(graph.get("seams", []))
    checks: list[dict[str, Any]] = []

    empty_sources = [
        {
            "segment_id": source.get("segment_id"),
            "rail_pair_count": source.get("rail_pair_count"),
            "reason": "no_rail_pair_candidates",
        }
        for source in graph.get("sources", [])
        if int(source.get("rail_pair_count", 0)) <= 0
    ]
    checks.append(
        _check(
            "source_rail_pair_presence",
            empty_sources,
            "Every supplied segment contains at least one reviewed rail-pair candidate",
        )
    )

    structural: list[dict[str, Any]] = _schema_failures(graph)
    observation_ids = [str(item.get("id")) for item in observations]
    if len(observation_ids) != len(set(observation_ids)):
        structural.append({"reason": "duplicate_observation_id"})
    track_ids = {str(item.get("id")) for item in tracks}
    observation_by_id = {str(item.get("id")): item for item in observations}
    node_ids = [str(item.get("id")) for item in graph.get("nodes", [])]
    if len(node_ids) != len(set(node_ids)):
        structural.append({"reason": "duplicate_node_id"})
    node_id_set = set(node_ids)
    edge_ids = [str(item.get("id")) for item in graph.get("edges", [])]
    if len(edge_ids) != len(set(edge_ids)):
        structural.append({"reason": "duplicate_edge_id"})
    for edge in graph.get("edges", []):
        for field in ("from_node", "to_node"):
            if edge.get(field) not in node_id_set:
                structural.append({"edge_id": edge.get("id"), "reason": f"unknown_{field}"})
        source_id = edge.get("source_observation_id")
        if source_id:
            source = observation_by_id.get(str(source_id))
            if source is None:
                structural.append(
                    {"edge_id": edge.get("id"), "reason": "unknown_source_observation"}
                )
            elif (
                edge.get("track_id") != source.get("global_track_id")
                or not math.isclose(
                    float(edge.get("chainage_start_m", math.nan)),
                    float(source["chainage_start_m"]),
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(edge.get("chainage_end_m", math.nan)),
                    float(source["chainage_end_m"]),
                    abs_tol=1e-9,
                )
                or edge.get("evidence_level") != source.get("evidence_level")
            ):
                structural.append(
                    {"edge_id": edge.get("id"), "reason": "source_observation_mismatch"}
                )
    for item in observations:
        if item.get("global_track_id") not in track_ids:
            structural.append({"observation_id": item.get("id"), "reason": "unknown_track"})
        if float(item.get("chainage_end_m", 0)) <= float(item.get("chainage_start_m", 0)):
            structural.append({"observation_id": item.get("id"), "reason": "invalid_interval"})
    checks.append(
        _check("track_graph_structure", structural, "Unique IDs and valid graph references")
    )

    node_degree = {node_id: 0 for node_id in node_ids}
    observation_nodes: dict[str, tuple[str, str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("from_node") in node_degree:
            node_degree[str(edge["from_node"])] += 1
        if edge.get("to_node") in node_degree:
            node_degree[str(edge["to_node"])] += 1
        source_id = edge.get("source_observation_id")
        if source_id:
            observation_nodes[str(source_id)] = (
                str(edge.get("from_node")),
                str(edge.get("to_node")),
            )
    topology_failures: list[dict[str, Any]] = []
    for seam in seams:
        left_nodes = observation_nodes.get(str(seam.get("from_observation_id")))
        right_nodes = observation_nodes.get(str(seam.get("to_observation_id")))
        if left_nodes is None or right_nodes is None:
            topology_failures.append({"seam_id": seam.get("id"), "reason": "unknown_observation"})
            continue
        node_degree[left_nodes[1]] += 1
        node_degree[right_nodes[0]] += 1
    for node_id, degree in node_degree.items():
        if degree not in {1, 2}:
            topology_failures.append(
                {"node_id": node_id, "degree": degree, "reason": "invalid_ordinary_degree"}
            )
    checks.append(
        _check(
            "topology_degree",
            topology_failures,
            "Ordinary track nodes have boundary degree 1 or continuous degree 2",
        )
    )

    orientation = [
        {
            "observation_id": item["id"],
            "direction_dot": item["direction_dot"],
            "frame_determinant": item["frame_determinant"],
        }
        for item in observations
        if float(item["direction_dot"]) < float(quality["minimum_direction_dot"])
        or abs(float(item["frame_determinant"]) - 1.0)
        > float(quality["maximum_frame_determinant_error"])
    ]
    checks.append(
        _check("canonical_direction", orientation, "No reversed or reflected segment frames")
    )

    nominal_gauge = float(settings["nominal_gauge_m"])
    gauge = [
        {
            "observation_id": item["id"],
            "gauge_m": item["gauge_m"],
            "error_m": abs(float(item["gauge_m"]) - nominal_gauge),
        }
        for item in observations
        if abs(float(item["gauge_m"]) - nominal_gauge) > float(quality["maximum_gauge_error_m"])
    ]
    checks.append(_check("rail_gauge", gauge, "Gauge remains within the project tolerance"))

    crosslevel = [
        {
            "observation_id": item["id"],
            "rail_top_crosslevel_m": item.get("rail_top_crosslevel_m"),
            "reason": (
                "missing_rail_top_crosslevel"
                if item.get("rail_top_crosslevel_m") is None
                else "crosslevel_exceeds_project_limit"
            ),
        }
        for item in observations
        if item.get("rail_top_crosslevel_m") is None
        or float(item["rail_top_crosslevel_m"]) > float(quality["maximum_rail_top_crosslevel_m"])
    ]
    checks.append(
        _check(
            "rail_top_crosslevel",
            crosslevel,
            "Paired rail-top heights remain physically plausible for this project",
        )
    )

    candidate_corrections = [
        {
            "observation_id": item["id"],
            "rail_position_correction_m": item.get("rail_position_correction_m"),
        }
        for item in observations
        if float(item.get("rail_position_correction_m", 0.0))
        > float(quality["maximum_candidate_rail_position_correction_m"])
    ]
    checks.append(
        _check(
            "candidate_rail_position_correction",
            candidate_corrections,
            "Nominal-gauge projection remains within the frozen project limit",
        )
    )

    support = [
        {"observation_id": item["id"], "support_ratio": item.get("support_ratio")}
        for item in observations
        if item.get("evidence_level") == "observed"
        and (
            item.get("support_ratio") is None
            or float(item["support_ratio"])
            < float(quality["minimum_observation_bin_coverage"])
        )
    ]
    checks.append(
        _check("observation_support", support, "Rail observations have longitudinal point support")
    )

    paired_continuity = [
        {
            "observation_id": item["id"],
            "joint_support_ratio": item.get("joint_support_ratio"),
            "asymmetric_support_ratio": item.get("asymmetric_support_ratio"),
            "maximum_internal_joint_gap_m": item.get(
                "maximum_internal_joint_gap_m"
            ),
            "pair_recovery_status": item.get("pair_recovery_status"),
            "pair_recovery_support_observation_id": item.get(
                "pair_recovery_support_observation_id"
            ),
        }
        for item in observations
        if item.get("paired_support_evaluated")
        and item.get("pair_recovery_status")
        != "fixed_center_evidence_recovered"
        and (
            item.get("joint_support_ratio") is None
            or float(item["joint_support_ratio"])
            < float(quality.get("minimum_pair_joint_support_ratio", 0.10))
            or item.get("asymmetric_support_ratio") is None
            or float(item["asymmetric_support_ratio"])
            > float(quality.get("maximum_pair_asymmetric_support_ratio", 0.25))
            or item.get("maximum_internal_joint_gap_m") is None
            or float(item["maximum_internal_joint_gap_m"])
            > float(quality.get("maximum_pair_internal_joint_gap_m", 5.0))
        )
    ]
    if bool(quality.get("paired_rail_continuity_gate_enabled", False)):
        checks.append(
            _check(
                "paired_rail_continuity",
                paired_continuity,
                "Both rails have simultaneous longitudinal support without a hidden internal gap",
            )
        )
    else:
        checks.append(
            {
                "id": "paired_rail_continuity",
                "status": "pass",
                "summary": "Paired rail continuity is recorded in diagnostic-only mode",
                "failure_count": 0,
                "failures": [],
                "diagnostic_count": len(paired_continuity),
                "diagnostics": paired_continuity,
                "enforcement": "diagnostic_only",
            }
        )

    seam_failures = [
        seam
        for seam in seams
        if abs(float(seam["signed_chainage_gap_m"])) > float(quality["maximum_seam_chainage_gap_m"])
        or float(seam["lateral_difference_m"]) > float(quality["maximum_seam_lateral_error_m"])
        or float(seam["vertical_difference_m"]) > float(quality["maximum_seam_vertical_error_m"])
        or float(seam["endpoint_3d_difference_m"]) > float(quality["maximum_seam_3d_error_m"])
        or float(seam["tangent_dot"]) < float(quality["minimum_seam_tangent_dot"])
        or float(seam["required_correction_m"]) > float(quality["maximum_automatic_correction_m"])
    ]
    checks.append(
        _check(
            "segment_seams", seam_failures, "Adjacent observations meet without hidden connectors"
        )
    )

    crossing = [event for event in graph.get("identity_events", []) if event.get("order_crossing")]
    checks.append(_check("track_identity_order", crossing, "Cross-track ordering is preserved"))

    global_start = float(graph["route"]["source_chainage_start_m"])
    global_end = float(graph["route"]["source_chainage_end_m"])
    coverage_failures: list[dict[str, Any]] = []
    termination_failures: list[dict[str, Any]] = []
    inferred_failures: list[dict[str, Any]] = []
    for track in tracks:
        intervals = [
            (float(item["chainage_start_m"]), float(item["chainage_end_m"]))
            for item in observations
            if item["global_track_id"] == track["id"]
        ]
        if not intervals:
            continue
        span_start = min(value[0] for value in intervals)
        span_end = max(value[1] for value in intervals)
        span = span_end - span_start
        ratio = _union_length(intervals) / span if span > 0 else 0.0
        maximum_gap = _maximum_gap(intervals)
        if ratio < float(quality["minimum_track_observation_coverage"]) or maximum_gap > float(
            quality["maximum_continuous_unobserved_m"]
        ):
            coverage_failures.append(
                {
                    "track_id": track["id"],
                    "coverage_ratio": ratio,
                    "maximum_gap_m": maximum_gap,
                }
            )
        members = sorted(
            [item for item in observations if item["global_track_id"] == track["id"]],
            key=lambda item: item["chainage_start_m"],
        )
        start_confirmed = members[0].get("start_boundary_reason") in CONFIRMED_INTERIOR_BOUNDARIES
        end_confirmed = members[-1].get("end_boundary_reason") in CONFIRMED_INTERIOR_BOUNDARIES
        boundary_tolerance = float(quality["boundary_chainage_tolerance_m"])
        if span_start > global_start + boundary_tolerance and not start_confirmed:
            termination_failures.append(
                {
                    "track_id": track["id"],
                    "chainage_m": span_start,
                    "reason": "unconfirmed_interior_start",
                }
            )
        if span_end < global_end - boundary_tolerance and not end_confirmed:
            termination_failures.append(
                {
                    "track_id": track["id"],
                    "chainage_m": span_end,
                    "reason": "unconfirmed_interior_end",
                }
            )

        inferred_intervals = [
            (float(edge["chainage_start_m"]), float(edge["chainage_end_m"]))
            for edge in graph.get("edges", [])
            if edge.get("track_id") == track["id"] and edge.get("evidence_level") == "rule_inferred"
        ]
        inferred_length = sum(end - start for start, end in inferred_intervals)
        authoritative_span = max(span, 1e-12)
        if _maximum_union_span(inferred_intervals) > float(
            quality["maximum_continuous_inferred_m"]
        ) or inferred_length / authoritative_span > float(quality["maximum_inferred_ratio"]):
            inferred_failures.append(
                {
                    "track_id": track["id"],
                    "inferred_length_m": inferred_length,
                    "inferred_ratio": inferred_length / authoritative_span,
                }
            )
    checks.append(
        _check(
            "track_observation_coverage", coverage_failures, "Track spans are observation-backed"
        )
    )
    checks.append(
        _check(
            "interior_track_termination",
            termination_failures,
            "Track births and deaths require topology evidence",
        )
    )
    checks.append(
        _check("inferred_spans", inferred_failures, "Long inferred rails are not authoritative")
    )

    duplicates: list[dict[str, Any]] = []
    for index, left in enumerate(tracks):
        for right in tracks[index + 1 :]:
            overlap = max(
                0.0,
                min(float(left["chainage_end_m"]), float(right["chainage_end_m"]))
                - max(float(left["chainage_start_m"]), float(right["chainage_start_m"])),
            )
            distance = abs(
                float(left["median_lateral_offset_m"]) - float(right["median_lateral_offset_m"])
            )
            if overlap > float(quality["duplicate_minimum_overlap_m"]) and distance < float(
                quality["duplicate_maximum_center_distance_m"]
            ):
                duplicates.append(
                    {
                        "left_track_id": left["id"],
                        "right_track_id": right["id"],
                        "overlap_m": overlap,
                        "center_distance_m": distance,
                    }
                )
    checks.append(
        _check("duplicate_tracks", duplicates, "No overlapping same-role track centerlines")
    )

    source_pending = [
        source for source in graph.get("sources", []) if source.get("review_status") != "accepted"
    ]
    owner_overrides = [
        source
        for source in graph.get("sources", [])
        if source.get("review_mode") == "project_owner_override"
    ]
    checks.append(
        {
            "id": "source_review",
            "status": "review_required" if source_pending else "pass",
            "summary": (
                "Source rail candidates are accepted; project-owner overrides are explicit"
                if owner_overrides
                else "Source rail candidates are independently reviewed"
            ),
            "failure_count": len(source_pending),
            "failures": source_pending,
            "owner_override_count": len(owner_overrides),
            "independent_review_complete": not owner_overrides and not source_pending,
        }
    )

    status = (
        "fail"
        if any(item["status"] == "fail" for item in checks)
        else (
            "review_required"
            if any(item["status"] == "review_required" for item in checks)
            else "pass"
        )
    )
    return {
        "schema_version": "railway.track-graph-audit.v1",
        "project_id": graph.get("project_id"),
        "graph_schema_version": graph.get("schema_version"),
        "status": status,
        "passed": status == "pass",
        "check_counts": {
            value: sum(item["status"] == value for item in checks)
            for value in ("pass", "fail", "review_required")
        },
        "checks": checks,
        "threshold_policy": "project_defaults_not_industry_regulations",
        "settings": settings,
    }


def _track_summaries(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[str, list[dict[str, Any]]] = {}
    for item in observations:
        by_track.setdefault(str(item["global_track_id"]), []).append(item)
    tracks: list[dict[str, Any]] = []
    for track_id, members in sorted(by_track.items()):
        tracks.append(
            {
                "id": track_id,
                "status": "candidate",
                "observation_ids": [
                    item["id"]
                    for item in sorted(members, key=lambda value: value["chainage_start_m"])
                ],
                "chainage_start_m": min(float(item["chainage_start_m"]) for item in members),
                "chainage_end_m": max(float(item["chainage_end_m"]) for item in members),
                "median_lateral_offset_m": float(
                    np.median([float(item["lateral_offset_m"]) for item in members])
                ),
                "median_gauge_m": float(np.median([float(item["gauge_m"]) for item in members])),
                "median_rail_center_spacing_m": float(
                    np.median([float(item["rail_center_spacing_m"]) for item in members])
                ),
            }
        )
    return tracks


def build_track_graph(
    project: ProjectConfig,
    segment_sources: list[tuple[str, str | Path]],
    overwrite: bool = False,
    output_value: str | Path | None = None,
    audit_value: str | Path | None = None,
) -> dict[str, Any]:
    if not segment_sources:
        raise ValueError("At least one SEGMENT=REPORT source is required")
    output_path = (
        Path(output_value).resolve()
        if output_value is not None
        else project.workspace_path("derived") / "track_graph.json"
    )
    audit_path = (
        Path(audit_value).resolve()
        if audit_value is not None
        else project.workspace_path("reports") / "track_graph_audit.json"
    )
    for path in (output_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    manifest_path = project.workspace_path("segment_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    segment_lookup = {str(item["id"]): item for item in manifest.get("segments", [])}
    camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    trajectory = camera_trajectory(load_camera_rows(camera_path))
    settings = load_track_graph_settings(project)
    route = RouteSampler(
        trajectory,
        float(settings.get("route_frame_tangent_window_m", 5.0)),
    )

    seen_segments: set[str] = set()
    sources: list[dict[str, Any]] = []
    batches: list[list[dict[str, Any]]] = []
    for segment_id, source_value in segment_sources:
        if segment_id in seen_segments:
            raise ValueError(f"Duplicate TrackGraph source segment: {segment_id}")
        seen_segments.add(segment_id)
        if segment_id not in segment_lookup:
            raise ValueError(f"TrackGraph source is absent from segment manifest: {segment_id}")
        source_path = Path(source_value).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        report = load_json(source_path)
        if report.get("schema_version") != "railway.rail-candidates.v1":
            raise ValueError(f"Unsupported rail candidate schema: {source_path}")
        if report.get("segment_id") != segment_id:
            raise ValueError(f"Source segment mismatch for {source_path}: expected {segment_id}")
        review_status = _source_review_status(report)
        sources.append(
            {
                "segment_id": segment_id,
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "review_status": review_status,
                "review_mode": _source_review_mode(report),
                "source_status": report.get("status"),
                "rail_pair_count": len(report.get("rail_pairs", [])),
                "candidate_point_count": int(report.get("candidate_point_count", 0)),
            }
        )
        batches.append(
            [
                _observation_from_pair(segment_lookup[segment_id], report, pair, route, settings)
                for pair in report.get("rail_pairs", [])
            ]
        )
    paired = sorted(
        zip(sources, batches),
        key=lambda item: float(segment_lookup[item[0]["segment_id"]]["chainage_start_m"]),
    )
    sources = [item[0] for item in paired]
    batches = [item[1] for item in paired]
    identity_events = _match_observations([batch for batch in batches if batch], settings)
    observations = [item for batch in batches for item in batch]
    seam_reconciliations = _reconcile_observation_seams(observations, settings)
    pair_continuity_recovery = _classify_pair_continuity_recovery(
        observations, settings
    )
    nodes, edges, seams = _build_geometry(observations, route, seam_reconciliations)
    tracks = _track_summaries(observations)
    source_segments = [segment_lookup[item["segment_id"]] for item in sources]
    graph = {
        "schema_version": "railway.track-graph.v1",
        "project_id": project.project_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "candidate",
        "settings_sha256": sha256_json(settings),
        "canonical_axis": "camera_trajectory_chainage",
        "route": {
            "camera_csv_sha256": sha256_file(camera_path),
            "trajectory_sample_count": len(trajectory),
            "trajectory_length_m": route.end_m,
            "source_chainage_start_m": min(
                float(item["chainage_start_m"]) for item in source_segments
            ),
            "source_chainage_end_m": max(float(item["chainage_end_m"]) for item in source_segments),
        },
        "sources": sources,
        "observations": observations,
        "nodes": nodes,
        "edges": edges,
        "seams": seams,
        "tracks": tracks,
        "identity_events": identity_events,
        "pair_continuity_recovery": pair_continuity_recovery,
        "seam_reconciliations": seam_reconciliations,
        "policy": {
            "mesh_may_only_use_observed_or_approved_inferred_edges": True,
            "unmatched_tracks_are_not_force_connected": True,
            "thresholds_are_project_defaults_not_industry_regulations": True,
        },
    }
    audit = audit_track_graph(graph, settings)
    graph["status"] = audit["status"]
    graph["validation_summary"] = audit["check_counts"]
    write_json(output_path, graph)
    audit["graph_path"] = str(output_path)
    audit["graph_sha256"] = sha256_file(output_path)
    write_json(audit_path, audit)
    return {
        "status": audit["status"],
        "track_count": len(tracks),
        "observation_count": len(observations),
        "seam_count": len(seams),
        "output_graph_path": str(output_path),
        "output_audit_path": str(audit_path),
    }


def validate_track_graph(
    project: ProjectConfig, graph_value: str | Path, output: str | Path | None = None
) -> dict[str, Any]:
    graph_path = Path(graph_value).resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    graph = load_json(graph_path)
    if graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError(f"Unsupported TrackGraph schema: {graph_path}")
    if graph.get("project_id") != project.project_id:
        raise ValueError("TrackGraph project_id does not match project")
    binding_failures = validate_track_graph_bindings(project, graph)
    if binding_failures:
        raise ValueError(f"TrackGraph project bindings are stale: {binding_failures}")
    report = audit_track_graph(graph, load_track_graph_settings(project))
    report["graph_path"] = str(graph_path)
    report["graph_sha256"] = sha256_file(graph_path)
    output_path = (
        Path(output).resolve()
        if output is not None
        else project.workspace_path("reports") / "track_graph_validation.json"
    )
    write_json(output_path, report)
    return {
        "status": report["status"],
        "passed": report["passed"],
        "check_counts": report["check_counts"],
        "output_report_path": str(output_path),
    }
