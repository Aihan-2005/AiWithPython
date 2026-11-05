
from __future__ import annotations

from math import isfinite
from typing import Mapping

import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString, Point

from .config import ORIGINS, DESTINATION
from .map_matching import directed_osm_line


_ROAD_PRIORITY = {
    "motorway": 0, "trunk": 1, "primary": 2,
    "secondary": 3, "tertiary": 4, "unclassified": 5,
    "residential": 6, "living_street": 7,
}
_EXCLUDED = {"service", "track", "path", "footway", "pedestrian", "cycleway", "steps"}


def _road_class(value) -> str | None:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    values = [str(v) for v in values if v is not None]
    allowed = [v for v in values if v in _ROAD_PRIORITY]
    if any(v in _EXCLUDED for v in values) and not allowed:
        return None
    return min(allowed, key=lambda v: _ROAD_PRIORITY[v]) if allowed else None


def plan_corridor_probes(
    graph: nx.MultiDiGraph,
    *,
    max_probes: int = 12,
    origins: Mapping[str, tuple[float, float]] | None = None,
    destination: tuple[float, float] = DESTINATION,
    corridor_width_m: float = 350.0,
    min_length_m: float = 35.0,
    min_spacing_m: float = 65.0,
) -> list[dict]:


    if not isinstance(max_probes, int) or max_probes < 0:
        raise ValueError("max_probes must be a nonnegative integer")
    if corridor_width_m <= 0 or min_length_m < 0 or min_spacing_m < 0:
        raise ValueError("Invalid corridor/length/spacing parameter")
    if max_probes == 0:
        return []
    origins = ORIGINS if origins is None else origins
    if not origins:
        raise ValueError("At least one origin is required")
    crs = graph.graph.get("crs", "EPSG:4326")
    xy = Transformer.from_crs(crs, "EPSG:27700", always_xy=True)
    def point(location):
        lat, lon = location
        xx, yy = xy.transform(lon, lat)
        return (xx, yy)

    end = point(destination)
    corridors = {
        str(name): LineString([point(start), end])
        for name, start in sorted(origins.items())
        if point(start) != end
    }
    if not corridors:
        raise ValueError("Origin and destination cannot all coincide")

    candidates = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        road_class = _road_class(attrs.get("highway"))
        length = float(attrs.get("length", 0))
        if road_class is None or not isfinite(length) or length < min_length_m:
            continue
        osm_line = directed_osm_line(graph, u, v, key)
        midpoint = osm_line.interpolate(.5, normalized=True)
        mx, my = xy.transform(midpoint.x, midpoint.y)
        mid_point = Point(mx, my)
        distances = {name: mid_point.distance(line) for name, line in corridors.items()}
        if min(distances.values()) > corridor_width_m:
            continue
        candidates.append({
            "edge_id": (u, v, key), "latitude": round(midpoint.y, 7),
            "longitude": round(midpoint.x, 7), "length_m": round(length, 2),
            "road_class": road_class, "_mx": mx, "_my": my,
            "_distances": distances,
        })

    chosen = []
    selected = set()

    while len(chosen) < max_probes:
        advanced = False
        for name in corridors:
            if len(chosen) >= max_probes:
                break
            options = [
                row for row in candidates
                if row["edge_id"] not in selected
                and all(
                    (row["_mx"] - old["_mx"]) ** 2 +
                    (row["_my"] - old["_my"]) ** 2 >= min_spacing_m**2
                    for old in chosen
                )
            ]
            if not options:
                continue
            winner = min(options, key=lambda row: (
                round(row["_distances"][name], 4),
                _ROAD_PRIORITY[row["road_class"]],
                -row["length_m"],
                tuple(map(str, row["edge_id"])),
            ))
            chosen.append(winner)
            selected.add(winner["edge_id"])
            winner["chosen_for"] = name
            advanced = True
        if not advanced:
            break

    return [
        {
            "edge_id": row["edge_id"],
            "latitude": row["latitude"], "longitude": row["longitude"],
            "length_m": row["length_m"], "road_class": row["road_class"],
            "corridor": row["chosen_for"],
            "corridor_offset_m": round(row["_distances"][row["chosen_for"]], 1),
        }
        for row in chosen
    ]


def audit_traffic_snapshot(
    base_graph: nx.MultiDiGraph,
    snapshot: nx.MultiDiGraph,
    summary: dict | None = None,
) -> dict:


    base_edges = set(base_graph.edges(keys=True))
    snap_edges = set(snapshot.edges(keys=True))
    identical_edges = base_edges == snap_edges and set(base_graph.nodes) == set(snapshot.nodes)
    costs_ok = True
    wrong_length = 0
    observed, imputed, missing, sampled = 0, 0, 0, 0
    for u, v, key, attrs in snapshot.edges(keys=True, data=True):
        ident = (u, v, key)
        if ident not in base_edges:
            costs_ok = False
            continue
        try:
            edge_length = float(attrs["length"])
            base_length = float(base_graph[u][v][key]["length"])
        except (KeyError, TypeError, ValueError):
            edge_length = float("nan")
            base_length = float("nan")
        if not isfinite(edge_length) or edge_length < 0:
            costs_ok = False
        if edge_length != base_length:
            wrong_length += 1
        if bool(attrs.get("traffic_sampled")):
            sampled += 1
        if attrs.get("traffic_status") == "missing":
            missing += 1
        else:
            observed += 1
        if attrs.get("time_is_imputed") is True:
            imputed += 1
        value = attrs.get("travel_time_s")
        if attrs.get("is_closed"):
            if value is not None:
                costs_ok = False
        else:
            try:
                numeric_time = float(value)
            except (TypeError, ValueError):
                numeric_time = float("nan")
            if not isfinite(numeric_time) or numeric_time < 0:
                costs_ok = False

    total = len(snap_edges)
    coverage = round(100 * observed / total, 3) if total else 0.0
    warnings = []
    if not identical_edges or wrong_length:
        warnings.append("The traffic snapshot does not match the base OSM graph exactly")
    if not costs_ok:
        warnings.append("Some nonclosed edge travel times are invalid")
    if coverage < 5:
        warnings.append("Observed traffic coverage is sparse; most travel times are not measured")
    warnings.append("TomTom geometry point order is not authoritative legal direction proof")
    if summary and summary.get("request_errors", 0):
        warnings.append(f"TomTom request errors: {summary['request_errors']}")
    return {
        "graph_identity_ok": identical_edges and wrong_length == 0,
        "costs_ok": costs_ok,
        "base_nodes": base_graph.number_of_nodes(),
        "snapshot_nodes": snapshot.number_of_nodes(),
        "total_directed_edges": total,
        "sampled_directed_edges": sampled,
        "observed_edges": observed,
        "missing_edges": missing,
        "imputed_edges": imputed,
        "observed_coverage_percent": coverage,
        "algorithm_smoke_test_ready": identical_edges and wrong_length == 0 and costs_ok and total > 0,
        "final_traffic_comparison_ready": False,
        "warnings": warnings,
    }
