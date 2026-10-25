

from __future__ import annotations

from math import hypot

import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import transform

from .config import (
    MATCH_MIDPOINT_TOLERANCE_M,
    MATCH_SAMPLE_TOLERANCE_M,
    MATCH_MIN_COSINE,
)


def directed_osm_line(graph: nx.MultiDiGraph, u: int, v: int, key: int) -> LineString:

    attrs = graph[u][v][key]
    if attrs.get("geometry") is not None:
        points = list(attrs["geometry"].coords)
    else:
        points = [
            (graph.nodes[u]["x"], graph.nodes[u]["y"]),
            (graph.nodes[v]["x"], graph.nodes[v]["y"]),
        ]
    start = (graph.nodes[u]["x"], graph.nodes[u]["y"])
    if hypot(points[-1][0] - start[0], points[-1][1] - start[1]) < hypot(
        points[0][0] - start[0], points[0][1] - start[1]
    ):
        points.reverse()
    return LineString(points)


def _tangent(line: LineString, at: float) -> tuple[float, float]:
    radius = min(10.0, max(0.01, line.length / 5))
    a = line.interpolate(max(0.0, at - radius))
    b = line.interpolate(min(line.length, at + radius))
    vx, vy = b.x - a.x, b.y - a.y
    length = hypot(vx, vy)
    return (vx / length, vy / length) if length > 1e-8 else (0.0, 0.0)


def match_osm_edge(
    graph: nx.MultiDiGraph,
    edge_id: tuple[int, int, int],
    tomtom_payload: dict,
    transformer: Transformer | None = None,
) -> bool:
  
    coordinate_obj = tomtom_payload.get("coordinates")
    if not isinstance(coordinate_obj, dict):
        return False
    points = coordinate_obj.get("coordinate", [])
    if isinstance(points, dict):
        points = [points]
    if not isinstance(points, list) or len(points) < 2:
        return False
    try:
        line = LineString([
            (float(p["longitude"]), float(p["latitude"])) for p in points
        ])
    except (ValueError, TypeError, KeyError):
        return False
    if line.is_empty or line.length == 0:
        return False
    transformer = transformer or Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    osm_m = transform(transformer.transform, directed_osm_line(graph, *edge_id))
    tom_m = transform(transformer.transform, line)
    if osm_m.length < 10 or tom_m.length < 10:
        return False
    midpoint = osm_m.interpolate(0.5, normalized=True)
    if midpoint.distance(tom_m) > MATCH_MIDPOINT_TOLERANCE_M:
        return False
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    near_count = sum(
        osm_m.interpolate(f, normalized=True).distance(tom_m) <= MATCH_SAMPLE_TOLERANCE_M
        for f in fractions
    )
    if near_count / len(fractions) < 0.8:
        return False
    osm_dir = _tangent(osm_m, osm_m.length / 2)
    tom_dir = _tangent(tom_m, tom_m.project(midpoint))
    cosine = osm_dir[0] * tom_dir[0] + osm_dir[1] * tom_dir[1]
    return cosine >= MATCH_MIN_COSINE
