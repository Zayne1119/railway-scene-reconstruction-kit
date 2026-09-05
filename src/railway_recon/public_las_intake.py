"""Offline, hash-pinned LAS/LAZ field intake without assigning ground truth.

The caller selects a legitimately available input. A hash proves file identity,
not a license, public provenance, coordinate accuracy or label interpretation.
Only this explicit file is opened; no dataset discovery or network access occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import laspy
import numpy as np
from jsonschema import validate

SCHEMA_VERSION = "railway.public-las-intake.v1"
DEFAULT_MAX_POINTS = 5_000_000
DEFAULT_CHUNK_SIZE = 100_000


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_INTEGER = {"type": "integer", "minimum": 0}
_VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_HISTOGRAM = _object(
    {
        "present": {"type": "boolean"},
        "unique_values_including_zero": _INTEGER,
        "counts": {
            "type": "object",
            "patternProperties": {"^[0-9]+$": _INTEGER},
            "additionalProperties": False,
        },
        "interpretation": {"type": "string"},
    }
)
PUBLIC_LAS_INTAKE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **_object(
        {
            "schema_version": {"const": SCHEMA_VERSION},
            "source": _object(
                {
                    "input_filename": {"type": "string"},
                    "bytes": _INTEGER,
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "sha256_verified_before_and_after_decode": {"const": True},
                    "provenance": {
                        "const": "caller_selected_file_not_inferred_from_content_or_hash"
                    },
                }
            ),
            "reader": _object(
                {
                    "laspy_version": {"type": "string"},
                    "numpy_version": {"type": "string"},
                    "chunk_size_points": {"type": "integer", "minimum": 1},
                    "maximum_points": {"type": "integer", "minimum": 1},
                    "chunks_decoded": _INTEGER,
                }
            ),
            "header": _object(
                {
                    "las_version": {"type": "string"},
                    "point_format_id": _INTEGER,
                    "point_record_bytes": {"type": "integer", "minimum": 1},
                    "compressed": {"type": "boolean"},
                    "declared_point_count": _INTEGER,
                    "scales": _VECTOR,
                    "offsets": _VECTOR,
                    "minimum": _VECTOR,
                    "maximum": _VECTOR,
                    "dimensions": {
                        "type": "array",
                        "items": _object(
                            {
                                "name": {"type": "string"},
                                "dtype": {"type": ["string", "null"]},
                                "num_bits": {"type": "integer", "minimum": 1},
                                "num_elements": {"type": "integer", "minimum": 1},
                                "extra_dimension": {"type": "boolean"},
                            }
                        ),
                    },
                }
            ),
            "decoded": _object(
                {
                    "point_count": _INTEGER,
                    "all_coordinates_finite": {"const": True},
                    "all_floating_dimensions_finite": {"const": True},
                    "minimum": {"anyOf": [_VECTOR, {"type": "null"}]},
                    "maximum": {"anyOf": [_VECTOR, {"type": "null"}]},
                    "header_bounds_match": {"type": ["boolean", "null"]},
                    "classification": _HISTOGRAM,
                    "point_source_id": _HISTOGRAM,
                }
            ),
            "field_interface": _object(
                {
                    "coordinates": {
                        "const": "scaled_xyz = raw_XYZ * header.scales + header.offsets"
                    },
                    "coordinate_units": {"const": "not_asserted_by_this_intake"},
                    "coordinate_reference_system": {"const": "not_resolved_by_this_intake"},
                    "classification": {
                        "const": "raw_numeric_code_without_automatic_semantic_mapping"
                    },
                    "point_source_id": {
                        "const": "raw_file_local_value_without_instance_or_cross_file_identity_claim"
                    },
                    "candidate_geometry": {"const": "not_extracted"},
                    "relation_ground_truth": {"const": "not_provided_or_inferred"},
                }
            ),
            "model_evaluation_performed": {"const": False},
            "limitations": {"type": "array", "items": {"type": "string"}},
        }
    ),
}


def _sha256(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_vector(name: str, values: Any, *, positive: bool = False) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 3 or any(not math.isfinite(value) for value in result):
        raise ValueError(f"Header {name} must contain three finite values")
    if positive and any(value <= 0 for value in result):
        raise ValueError(f"Header {name} must be positive")
    return result


def _summarize_histogram(counter: Counter, present: bool, interpretation: str) -> dict:
    return {
        "present": present,
        "unique_values_including_zero": len(counter),
        "counts": {str(key): counter[key] for key in sorted(counter)},
        "interpretation": interpretation,
    }


def inspect_public_las(
    input_path: str | Path,
    expected_sha256: str,
    output: str | Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Decode a hash-pinned file in chunks and write one new JSON field report.

    The point limit is a rejection threshold, not a sampling count. Inputs above
    it are rejected before point decoding. No report is created on checksum,
    malformed header, non-finite data, size-limit or truncated-point failure.
    Existing output files/directories/symlinks are always refused.
    """
    source, destination = Path(input_path), Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    expected_sha256 = expected_sha256.lower()
    _positive_integer("max_points", max_points)
    _positive_integer("chunk_size", chunk_size)
    if chunk_size > 1_000_000:
        raise ValueError("chunk_size must not exceed 1,000,000 points")
    if not source.is_file():
        raise FileNotFoundError(f"LAS/LAZ input is not a file: {source}")

    with source.open("rb") as stream:
        actual_sha256 = _sha256(stream)
        source_bytes = stream.tell()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Input SHA256 does not match expected_sha256; decode was not attempted"
            )
        stream.seek(0)
        with laspy.open(stream, closefd=False, read_evlrs=False) as reader:
            header = reader.header
            declared_count = int(header.point_count)
            if declared_count > max_points:
                raise ValueError(
                    f"Declared point count {declared_count} exceeds max_points={max_points}"
                )
            scales = _finite_vector("scales", header.scales, positive=True)
            offsets = _finite_vector("offsets", header.offsets)
            header_min = _finite_vector("minimum", header.mins)
            header_max = _finite_vector("maximum", header.maxs)
            if any(low > high for low, high in zip(header_min, header_max)):
                raise ValueError("Header minimum exceeds maximum")
            extra_names = set(header.point_format.extra_dimension_names)
            dimensions = [
                {
                    "name": dimension.name,
                    "dtype": str(dimension.dtype) if dimension.dtype is not None else None,
                    "num_bits": int(dimension.num_bits),
                    "num_elements": int(dimension.num_elements),
                    "extra_dimension": dimension.name in extra_names,
                }
                for dimension in header.point_format.dimensions
            ]
            dimension_names = {dimension["name"] for dimension in dimensions}
            counters: dict[str, Counter] = {
                "classification": Counter(),
                "point_source_id": Counter(),
            }
            decoded_count = 0
            chunks = 0
            actual_min, actual_max = np.full(3, np.inf), np.full(3, -np.inf)
            for points in reader.chunk_iterator(min(chunk_size, max_points)):
                chunks += 1
                decoded_count += len(points)
                if decoded_count > max_points:
                    raise ValueError("Decoded point count exceeds max_points")
                with np.errstate(over="ignore", invalid="ignore"):
                    xyz = np.column_stack((points.x, points.y, points.z))
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Non-finite decoded XYZ in chunk {chunks}")
                for dimension in dimensions:
                    values = np.asarray(points[dimension["name"]])
                    if values.dtype.kind == "f" and not np.isfinite(values).all():
                        raise ValueError(
                            f"Non-finite floating dimension {dimension['name']} in chunk {chunks}"
                        )
                if len(points):
                    actual_min = np.minimum(actual_min, xyz.min(axis=0))
                    actual_max = np.maximum(actual_max, xyz.max(axis=0))
                for name, counter in counters.items():
                    if name in dimension_names:
                        unique, counts = np.unique(np.asarray(points[name]), return_counts=True)
                        counter.update({int(key): int(count) for key, count in zip(unique, counts)})
            if decoded_count != declared_count:
                raise ValueError(
                    f"Decoded {decoded_count} points but header declares {declared_count}"
                )
            if decoded_count:
                # Half a storage step tolerates harmless header decimal rounding.
                tolerance = (
                    np.asarray(scales) / 2
                    + np.finfo(float).eps
                    * np.maximum(1, np.maximum(np.abs(actual_min), np.abs(actual_max)))
                    * 4
                )
                bounds_match: bool | None = bool(
                    np.all(np.abs(actual_min - header_min) <= tolerance)
                    and np.all(np.abs(actual_max - header_max) <= tolerance)
                )
            else:
                bounds_match = None
            report = {
                "schema_version": SCHEMA_VERSION,
                "source": {
                    "input_filename": source.name,
                    "bytes": source_bytes,
                    "sha256": actual_sha256,
                    "sha256_verified_before_and_after_decode": True,
                    "provenance": "caller_selected_file_not_inferred_from_content_or_hash",
                },
                "reader": {
                    "laspy_version": laspy.__version__,
                    "numpy_version": np.__version__,
                    "chunk_size_points": min(chunk_size, max_points),
                    "maximum_points": max_points,
                    "chunks_decoded": chunks,
                },
                "header": {
                    "las_version": str(header.version),
                    "point_format_id": header.point_format.id,
                    "point_record_bytes": header.point_format.size,
                    "compressed": bool(header.are_points_compressed),
                    "declared_point_count": declared_count,
                    "scales": scales,
                    "offsets": offsets,
                    "minimum": header_min,
                    "maximum": header_max,
                    "dimensions": dimensions,
                },
                "decoded": {
                    "point_count": decoded_count,
                    "all_coordinates_finite": True,
                    "all_floating_dimensions_finite": True,
                    "minimum": actual_min.tolist() if decoded_count else None,
                    "maximum": actual_max.tolist() if decoded_count else None,
                    "header_bounds_match": bounds_match,
                    "classification": _summarize_histogram(
                        counters["classification"],
                        "classification" in dimension_names,
                        "Numeric classification codes only; semantic names require a dataset-specific mapping",
                    ),
                    "point_source_id": _summarize_histogram(
                        counters["point_source_id"],
                        "point_source_id" in dimension_names,
                        "Raw values in this file, including zero; not automatically instance identifiers",
                    ),
                },
                "field_interface": {
                    "coordinates": "scaled_xyz = raw_XYZ * header.scales + header.offsets",
                    "coordinate_units": "not_asserted_by_this_intake",
                    "coordinate_reference_system": "not_resolved_by_this_intake",
                    "classification": "raw_numeric_code_without_automatic_semantic_mapping",
                    "point_source_id": "raw_file_local_value_without_instance_or_cross_file_identity_claim",
                    "candidate_geometry": "not_extracted",
                    "relation_ground_truth": "not_provided_or_inferred",
                },
                "model_evaluation_performed": False,
                "limitations": [
                    "SHA256 pins bytes; it does not establish a data license or public provenance.",
                    "LAS scales encode quantization increments, not measured accuracy or precision.",
                    "Coordinate units and CRS need independent source documentation; metres are not assumed.",
                    "Classification and point_source_id codes are not independent geometric or connection truth.",
                    "No cross-file object identity, candidate reconstruction or model performance is evaluated.",
                    "The complete declared point record set is decoded; max_points never silently subsamples.",
                    "All numeric floating point dimensions are required finite, including optional GPS/extra fields.",
                ],
            }
        if _sha256(stream) != expected_sha256:
            raise ValueError("Input changed during decoding; no report was written")
    validate(instance=report, schema=PUBLIC_LAS_INTAKE_SCHEMA)
    encoded = (
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a concurrent run from overwriting an output.
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return report
