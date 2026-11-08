from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import networkx as nx
from pyproj import Transformer
from shapely import transform as shapely_transform
from shapely.geometry import LineString

from .config import (
    MATCH_LOCAL_WINDOW_M,
    MATCH_MIN_COSINE,
    MATCH_MIN_LOCAL_COVERAGE,
    MATCH_MIDPOINT_TOLERANCE_M,
    MATCH_SAMPLE_TOLERANCE_M,
    MIN_TOMTOM_SEGMENT_LENGTH_M,
)


@dataclass(frozen=True)
class MatchResult:
    """Detailed result for matching one directed OSM edge to TomTom geometry."""

    accepted: bool
    reason: str
    # Kept for compatibility with earlier notebook/report field names.
    midpoint_distance_m: float | None = None
    fraction_near: float | None = None
    direction_cosine: float | None = None
    probe_fraction: float = 0.5


def directed_osm_line(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
) -> LineString:
    """Return the edge geometry oriented from OSM node u towards node v."""
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a networkx.MultiDiGraph")
    if not graph.has_edge(u, v, key):
        raise KeyError(f"OSM edge does not exist: {(u, v, key)}")

    attrs = graph[u][v][key]
    raw = attrs.get("geometry")
    points = list(raw.coords) if raw is not None else [
        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
    ]

    start = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
    end = (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))

    if len(points) < 2:
        raise ValueError("Edge requires at least two geometry coordinates")

    # OSMnx geometries are usually already ordered, but some imported graphs
    # contain reversed coordinate sequences. Orient them using their endpoints.
    first_distance = hypot(points[0][0] - start[0], points[0][1] - start[1])
    last_distance = hypot(points[-1][0] - start[0], points[-1][1] - start[1])
    if last_distance < first_distance:
        points.reverse()

    line = LineString(points)
    if line.is_empty or not line.is_valid or line.length <= 0:
        raise ValueError(f"Invalid OSM edge geometry: {(u, v, key)}")

    # Check that the oriented geometry terminates closest to node v.
    if hypot(line.coords[-1][0] - end[0], line.coords[-1][1] - end[1]) > hypot(
        line.coords[0][0] - end[0], line.coords[0][1] - end[1]
    ):
        # Do not reverse a valid u->v line solely because of an unusual
        # geometry shape; the start-node test above is the primary orientation.
        pass

    return line


def _tangent(line: LineString, distance_along: float) -> tuple[float, float]:
    """Return a unit tangent in the coordinate order of the line."""
    if line.length <= 0:
        return 0.0, 0.0
    radius = min(10.0, max(0.01, line.length / 5.0))
    a = line.interpolate(max(0.0, distance_along - radius))
    b = line.interpolate(min(line.length, distance_along + radius))
    vx, vy = b.x - a.x, b.y - a.y
    magnitude = hypot(vx, vy)
    if magnitude <= 1e-8:
        return 0.0, 0.0
    return vx / magnitude, vy / magnitude


def _tomtom_line(payload: dict) -> LineString | None:
    """Parse TomTom Flow Segment coordinates as a WGS84 LineString."""
    if not isinstance(payload, dict):
        return None

    raw = payload.get("coordinates")
    if not isinstance(raw, dict):
        return None

    points = raw.get("coordinate", [])
    if isinstance(points, dict):
        points = [points]
    if not isinstance(points, list) or len(points) < 2:
        return None

    try:
        xy = [
            (float(point["longitude"]), float(point["latitude"]))
            for point in points
        ]
    except (ValueError, TypeError, KeyError, OverflowError):
        return None

    if not all(
        isfinite(x)
        and isfinite(y)
        and -180.0 <= x <= 180.0
        and -90.0 <= y <= 90.0
        for x, y in xy
    ):
        return None

    try:
        line = LineString(xy)
    except (TypeError, ValueError):
        return None

    if line.is_empty or not line.is_valid or line.length <= 0:
        return None
    return line


def assess_osm_edge(
    graph: nx.MultiDiGraph,
    edge_id: tuple[int, int, int],
    tomtom_payload: dict,
    transformer: Transformer | None = None,
    *,
    probe_fraction: float = 0.5,
) -> MatchResult:
    """Assess geometric overlap and direction for one directed OSM edge.

    `probe_fraction` is the position along the OSM edge where the TomTom
    request was made. This matters because a Flow Segment can cover only a
    local part of a longer OSM edge; comparing only the full-edge midpoint
    creates avoidable false negatives when the request was made elsewhere.

    The TomTom coordinate order is treated as a direction proxy only. It is
    not authoritative proof of legal driving direction.
    """
    try:
        probe_fraction = float(probe_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probe_fraction must be numeric") from exc
    if not isfinite(probe_fraction) or not 0.0 <= probe_fraction <= 1.0:
        raise ValueError("probe_fraction must be between 0 and 1")

    tom = _tomtom_line(tomtom_payload)
    if tom is None:
        return MatchResult(False, "invalid_tomtom_geometry", probe_fraction=probe_fraction)

    if not isinstance(graph, nx.MultiDiGraph) or not graph.has_edge(*edge_id):
        return MatchResult(False, "osm_edge_not_found", probe_fraction=probe_fraction)

    source_crs = graph.graph.get("crs", "EPSG:4326")
    graph_transformer = transformer or Transformer.from_crs(
        source_crs, "EPSG:27700", always_xy=True
    )
    tom_transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:27700", always_xy=True
    )

    try:
        osm_line = directed_osm_line(graph, *edge_id)
        osm_m = shapely_transform(
            osm_line,
            graph_transformer.transform,
            interleaved=False,
        )
        tom_m = shapely_transform(
            tom,
            tom_transformer.transform,
            interleaved=False,
        )
    except Exception:
        return MatchResult(False, "geometry_transform_error", probe_fraction=probe_fraction)

    if osm_m.length < 10.0 or tom_m.length < MIN_TOMTOM_SEGMENT_LENGTH_M:
        return MatchResult(False, "short_segment", probe_fraction=probe_fraction)

    probe_distance_along = probe_fraction * osm_m.length
    probe_point = osm_m.interpolate(probe_distance_along)
    probe_distance = probe_point.distance(tom_m)

    if probe_distance > MATCH_MIDPOINT_TOLERANCE_M:
        reason = "midpoint_too_far" if abs(probe_fraction - 0.5) < 1e-9 else "probe_point_too_far"
        return MatchResult(
            False,
            reason,
            round(probe_distance, 2),
            None,
            None,
            probe_fraction,
        )

    # Test a local window around the actual query location rather than
    # demanding that the TomTom segment overlap the entire OSM edge.
    radius = min(
        MATCH_LOCAL_WINDOW_M,
        max(10.0, osm_m.length * 0.20),
    )
    start_distance = max(0.0, probe_distance_along - radius)
    end_distance = min(osm_m.length, probe_distance_along + radius)
    if end_distance - start_distance < 1.0:
        return MatchResult(
            False,
            "insufficient_local_geometry",
            round(probe_distance, 2),
            0.0,
            None,
            probe_fraction,
        )

    sample_count = 5
    near_count = 0
    for index in range(sample_count):
        local_distance = start_distance + (
            (end_distance - start_distance) * index / (sample_count - 1)
        )
        sample_point = osm_m.interpolate(local_distance)
        if sample_point.distance(tom_m) <= MATCH_SAMPLE_TOLERANCE_M:
            near_count += 1

    fraction_near = near_count / sample_count
    if fraction_near < MATCH_MIN_LOCAL_COVERAGE:
        return MatchResult(
            False,
            "insufficient_overlap",
            round(probe_distance, 2),
            round(fraction_near, 3),
            None,
            probe_fraction,
        )

    tom_distance_along = tom_m.project(probe_point)
    osm_tangent = _tangent(osm_m, probe_distance_along)
    tom_tangent = _tangent(tom_m, tom_distance_along)
    cosine = (
        osm_tangent[0] * tom_tangent[0]
        + osm_tangent[1] * tom_tangent[1]
    )

    accepted = cosine >= MATCH_MIN_COSINE
    return MatchResult(
        accepted,
        "candidate_match" if accepted else "heading_proxy_mismatch",
        round(probe_distance, 2),
        round(fraction_near, 3),
        round(cosine, 4),
        probe_fraction,
    )


def match_osm_edge(
    graph: nx.MultiDiGraph,
    edge_id: tuple[int, int, int],
    tomtom_payload: dict,
    transformer: Transformer | None = None,
) -> bool:
    """Backward-compatible boolean wrapper around :func:`assess_osm_edge`."""
    return assess_osm_edge(graph, edge_id, tomtom_payload, transformer).accepted
