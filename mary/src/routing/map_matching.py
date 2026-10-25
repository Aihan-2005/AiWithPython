
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import networkx as nx
from pyproj import Transformer
from shapely import transform as shapely_transform
from shapely.geometry import LineString

from .config import (
    MATCH_MIDPOINT_TOLERANCE_M,
    MATCH_SAMPLE_TOLERANCE_M,
    MATCH_MIN_COSINE,
)


@dataclass(frozen=True)
class MatchResult:
    accepted: bool
    reason: str
    midpoint_distance_m: float | None = None
    fraction_near: float | None = None
    direction_cosine: float | None = None


def directed_osm_line(graph: nx.MultiDiGraph, u: int, v: int, key: int) -> LineString:
    attrs = graph[u][v][key]
    raw = attrs.get("geometry")
    points = list(raw.coords) if raw is not None else [
        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
    ]
    start = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
    if len(points) < 2:
        raise ValueError("Edge requires at least two geometry coordinates")
    if hypot(points[-1][0] - start[0], points[-1][1] - start[1]) < hypot(
        points[0][0] - start[0], points[0][1] - start[1]
    ):
        points.reverse()
    return LineString(points)


def _tangent(line: LineString, distance_along: float) -> tuple[float, float]:
    radius = min(10.0, max(0.01, line.length / 5))
    a = line.interpolate(max(0.0, distance_along - radius))
    b = line.interpolate(min(line.length, distance_along + radius))
    vx, vy = b.x - a.x, b.y - a.y
    magnitude = hypot(vx, vy)
    return (vx / magnitude, vy / magnitude) if magnitude > 1e-8 else (0.0, 0.0)


def _tomtom_line(payload: dict) -> LineString | None:
    raw = payload.get("coordinates")
    if not isinstance(raw, dict):
        return None
    points = raw.get("coordinate", [])
    if isinstance(points, dict):
        points = [points]
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xy = [(float(p["longitude"]), float(p["latitude"])) for p in points]
    except (ValueError, TypeError, KeyError):
        return None
    if not all(isfinite(x) and isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90 for x, y in xy):
        return None
    line = LineString(xy)
    return None if line.is_empty or not line.is_valid or line.length == 0 else line


def assess_osm_edge(
    graph: nx.MultiDiGraph,
    edge_id: tuple[int, int, int],
    tomtom_payload: dict,
    transformer: Transformer | None = None,
) -> MatchResult:

    tom = _tomtom_line(tomtom_payload)
    if tom is None:
        return MatchResult(False, "invalid_tomtom_geometry")
    transformer = transformer or Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    osm_m = shapely_transform(directed_osm_line(graph, *edge_id), transformer.transform, interleaved=False)
    tom_m = shapely_transform(tom, transformer.transform, interleaved=False)
    if osm_m.length < 10 or tom_m.length < 10:
        return MatchResult(False, "short_segment")
    midpoint = osm_m.interpolate(0.5, normalized=True)
    midpoint_m = midpoint.distance(tom_m)
    if midpoint_m > MATCH_MIDPOINT_TOLERANCE_M:
        return MatchResult(False, "midpoint_too_far", midpoint_distance_m=round(midpoint_m, 2))
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    near = sum(
        osm_m.interpolate(f, normalized=True).distance(tom_m) <= MATCH_SAMPLE_TOLERANCE_M
        for f in fractions
    ) / len(fractions)
    if near < 0.8:
        return MatchResult(False, "insufficient_overlap", round(midpoint_m, 2), near)
    d1 = _tangent(osm_m, osm_m.length / 2)
    d2 = _tangent(tom_m, tom_m.project(midpoint))
    cosine = d1[0] * d2[0] + d1[1] * d2[1]
    accepted = cosine >= MATCH_MIN_COSINE
    return MatchResult(
        accepted,
        "candidate_match" if accepted else "heading_proxy_mismatch",
        round(midpoint_m, 2),
        near,
        round(cosine, 4),
    )


def match_osm_edge(
    graph: nx.MultiDiGraph,
    edge_id: tuple[int, int, int],
    tomtom_payload: dict,
    transformer: Transformer | None = None,
) -> bool:
    return assess_osm_edge(graph, edge_id, tomtom_payload, transformer).accepted
