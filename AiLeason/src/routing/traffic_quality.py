from __future__ import annotations

from math import isfinite
from typing import Mapping

import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString, Point

from .config import ORIGINS, DESTINATION
from .map_matching import directed_osm_line


# Lower values mean a more important road class for probe selection.
_ROAD_PRIORITY = {
    "motorway": 0,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 3,
    "primary": 4,
    "primary_link": 5,
    "secondary": 6,
    "secondary_link": 7,
    "tertiary": 8,
    "tertiary_link": 9,
    "unclassified": 10,
    "residential": 11,
    "living_street": 12,
}

_EXCLUDED = {
    "service",
    "track",
    "path",
    "footway",
    "pedestrian",
    "cycleway",
    "steps",
}

_USABLE_TRAFFIC_STATUSES = {"red", "yellow", "clear", "closed"}
_HISTORY_WEIGHT = 4.0


def _road_class(value: object) -> str | None:
    """Normalize an OSM highway tag and return its supported road class."""
    values = value if isinstance(value, (list, tuple, set)) else [value]
    normalized = [
        str(item).strip().lower()
        for item in values
        if item is not None
    ]

    allowed = [item for item in normalized if item in _ROAD_PRIORITY]

    # Excluded classes are ignored unless a supported class is also present.
    if any(item in _EXCLUDED for item in normalized) and not allowed:
        return None

    if not allowed:
        return None

    return min(allowed, key=lambda item: _ROAD_PRIORITY[item])


def _safe_nonnegative_int(value: object) -> int:
    """Convert a value to a non-negative integer; invalid values become zero."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _history_failure_rate(
    history: Mapping | None,
    edge_id: tuple,
) -> float:
    """
    Return a Beta(1, 1)-smoothed failure estimate for a directed edge.

    Direct usable observations count as one success. A usable reverse-inferred
    observation counts as half a success. Global API errors are excluded from
    the denominator because they do not describe the road candidate itself.
    """
    if not isinstance(history, Mapping):
        return 0.5

    edges = history.get("edges", {})
    if not isinstance(edges, Mapping):
        return 0.5

    key = "|".join(map(str, edge_id))
    item = edges.get(key, {})
    if not isinstance(item, Mapping):
        return 0.5

    attempts = _safe_nonnegative_int(item.get("attempts", 0))
    global_errors = _safe_nonnegative_int(item.get("global_errors", 0))
    effective_attempts = max(0, attempts - global_errors)

    direct_successes = _safe_nonnegative_int(item.get("direct_usable", 0))
    reverse_successes = _safe_nonnegative_int(item.get("reverse_rescued", 0))
    successes = min(
        float(effective_attempts),
        direct_successes + 0.5 * reverse_successes,
    )
    failures = max(0.0, effective_attempts - successes)

    # Beta(1, 1) prior prevents a single observation from dominating.
    return (failures + 1.0) / (effective_attempts + 2.0)


def _history_effective_attempts(
    history: Mapping | None,
    edge_id: tuple,
) -> int:
    """Return road-specific attempts, excluding global API failures."""
    if not isinstance(history, Mapping):
        return 0
    edges = history.get("edges", {})
    if not isinstance(edges, Mapping):
        return 0
    item = edges.get("|".join(map(str, edge_id)), {})
    if not isinstance(item, Mapping):
        return 0
    attempts = _safe_nonnegative_int(item.get("attempts", 0))
    global_errors = _safe_nonnegative_int(item.get("global_errors", 0))
    return max(0, attempts - global_errors)


def _coordinate_pair(location: object, *, label: str) -> tuple[float, float]:
    """Validate a (latitude, longitude) pair."""
    if not isinstance(location, (tuple, list)) or len(location) != 2:
        raise ValueError(f"{label} must be a (latitude, longitude) pair")

    try:
        latitude = float(location[0])
        longitude = float(location[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} coordinates must be numeric") from exc

    if (
        not isfinite(latitude)
        or not isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        raise ValueError(f"{label} contains invalid latitude/longitude")

    return latitude, longitude


def plan_corridor_probes(
    graph: nx.MultiDiGraph,
    *,
    max_probes: int = 12,
    origins: Mapping[str, tuple[float, float]] | None = None,
    destination: tuple[float, float] = DESTINATION,
    corridor_width_m: float = 350.0,
    min_length_m: float = 35.0,
    min_spacing_m: float = 65.0,
    history: Mapping | None = None,
) -> list[dict]:
    """
    Select spatially separated, corridor-relevant directed OSM edges.

    Selection order:
      1. Prefer more important road classes.
      2. Avoid repeatedly probing the same edge.
      3. Add an extra penalty to edges with poor historical outcomes.
      4. Prefer edges closer to the active corridor.
      5. Prefer longer edges as a deterministic tie-breaker.

    The optional history argument is backward-compatible: omitting it gives
    all candidates a neutral historical failure estimate of 0.5.
    """
    if (
        isinstance(max_probes, bool)
        or not isinstance(max_probes, int)
        or max_probes < 0
    ):
        raise ValueError("max_probes must be a nonnegative integer")

    try:
        corridor_width_m = float(corridor_width_m)
        min_length_m = float(min_length_m)
        min_spacing_m = float(min_spacing_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Corridor, length, and spacing values must be numeric"
        ) from exc

    if (
        not isfinite(corridor_width_m)
        or not isfinite(min_length_m)
        or not isfinite(min_spacing_m)
        or corridor_width_m <= 0
        or min_length_m < 0
        or min_spacing_m < 0
    ):
        raise ValueError("Invalid corridor/length/spacing parameter")

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a networkx.MultiDiGraph")

    if max_probes == 0:
        return []

    origins = ORIGINS if origins is None else origins
    if not isinstance(origins, Mapping) or not origins:
        raise ValueError("At least one origin is required")

    destination = _coordinate_pair(destination, label="destination")

    graph_crs = graph.graph.get("crs", "EPSG:4326")
    graph_to_meters = Transformer.from_crs(
        graph_crs, "EPSG:27700", always_xy=True
    )
    wgs84_to_meters = Transformer.from_crs(
        "EPSG:4326", "EPSG:27700", always_xy=True
    )
    graph_to_wgs84 = Transformer.from_crs(
        graph_crs, "EPSG:4326", always_xy=True
    )

    def project_wgs84(location: tuple[float, float]) -> tuple[float, float]:
        latitude, longitude = location
        return wgs84_to_meters.transform(longitude, latitude)

    end_x, end_y = project_wgs84(destination)

    corridors: dict[str, LineString] = {}
    for name, start in sorted(origins.items(), key=lambda item: str(item[0])):
        start = _coordinate_pair(start, label=f"origin {name!r}")
        start_x, start_y = project_wgs84(start)

        if (start_x, start_y) != (end_x, end_y):
            corridors[str(name)] = LineString(
                [(start_x, start_y), (end_x, end_y)]
            )

    if not corridors:
        raise ValueError("Origin and destination cannot all coincide")

    candidates: list[dict] = []

    for u, v, key, attrs in graph.edges(keys=True, data=True):
        road_class = _road_class(attrs.get("highway"))
        if road_class is None:
            continue

        try:
            length = float(attrs.get("length", 0))
        except (TypeError, ValueError, OverflowError):
            continue

        if not isfinite(length) or length < min_length_m:
            continue

        osm_line = directed_osm_line(graph, u, v, key)
        midpoint = osm_line.interpolate(0.5, normalized=True)

        mx, my = graph_to_meters.transform(midpoint.x, midpoint.y)
        if not isfinite(mx) or not isfinite(my):
            continue

        longitude, latitude = graph_to_wgs84.transform(
            midpoint.x, midpoint.y
        )

        midpoint_point = Point(mx, my)
        distances = {
            name: midpoint_point.distance(corridor)
            for name, corridor in corridors.items()
        }

        if min(distances.values()) > corridor_width_m:
            continue

        candidates.append(
            {
                "edge_id": (u, v, key),
                "latitude": round(latitude, 7),
                "longitude": round(longitude, 7),
                "length_m": round(length, 2),
                "road_class": road_class,
                "_mx": mx,
                "_my": my,
                "_distances": distances,
            }
        )

    chosen: list[dict] = []
    selected: set[tuple] = set()

    while len(chosen) < max_probes:
        advanced = False

        for corridor_name in corridors:
            if len(chosen) >= max_probes:
                break

            options = [
                row
                for row in candidates
                if row["edge_id"] not in selected
                and row["_distances"][corridor_name] <= corridor_width_m
                and all(
                    (row["_mx"] - old["_mx"]) ** 2
                    + (row["_my"] - old["_my"]) ** 2
                    >= min_spacing_m**2
                    for old in chosen
                )
            ]

            if not options:
                continue

            def selection_key(row: dict) -> tuple:
                failure_rate = _history_failure_rate(
                    history, row["edge_id"]
                )
                effective_attempts = _history_effective_attempts(
                    history, row["edge_id"]
                )
                repeat_probe_penalty = 0.25 * min(effective_attempts, 4)
                failure_penalty = _HISTORY_WEIGHT * max(
                    0.0, failure_rate - 0.5
                )
                adjusted_priority = (
                    _ROAD_PRIORITY[row["road_class"]]
                    + repeat_probe_penalty
                    + failure_penalty
                )

                return (
                    round(adjusted_priority, 4),
                    round(row["_distances"][corridor_name], 4),
                    -row["length_m"],
                    tuple(map(str, row["edge_id"])),
                )

            winner = min(options, key=selection_key)
            chosen.append(winner)
            selected.add(winner["edge_id"])
            winner["chosen_for"] = corridor_name
            advanced = True

        if not advanced:
            break

    return [
        {
            "edge_id": row["edge_id"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "length_m": row["length_m"],
            "road_class": row["road_class"],
            "corridor": row["chosen_for"],
            "corridor_offset_m": round(
                row["_distances"][row["chosen_for"]], 1
            ),
        }
        for row in chosen
    ]


def audit_traffic_snapshot(
    base_graph: nx.MultiDiGraph,
    snapshot: nx.MultiDiGraph,
    summary: dict | None = None,
) -> dict:
    """
    Audit graph identity, travel-time costs, and traffic-observation provenance.

    `observed_edges` counts usable traffic statuses, whether direct or reverse
    inferred. Direct and reverse observations are also reported separately.
    A geometry match with low confidence is not counted as an observed edge.
    """
    if not isinstance(base_graph, nx.MultiDiGraph):
        raise TypeError("base_graph must be a networkx.MultiDiGraph")
    if not isinstance(snapshot, nx.MultiDiGraph):
        raise TypeError("snapshot must be a networkx.MultiDiGraph")

    base_edges = set(base_graph.edges(keys=True))
    snapshot_edges = set(snapshot.edges(keys=True))
    identical_edges = (
        base_edges == snapshot_edges
        and set(base_graph.nodes) == set(snapshot.nodes)
    )

    costs_ok = True
    wrong_length = 0
    invalid_status_edges = 0

    observed = 0
    direct_observed = 0
    reverse_observed = 0
    imputed = 0
    missing = 0
    sampled = 0
    matched = 0
    directly_queried = 0

    for u, v, key, attrs in snapshot.edges(keys=True, data=True):
        edge_id = (u, v, key)

        if edge_id not in base_edges:
            costs_ok = False
            continue

        try:
            edge_length = float(attrs["length"])
            base_length = float(base_graph[u][v][key]["length"])
        except (KeyError, TypeError, ValueError, OverflowError):
            edge_length = float("nan")
            base_length = float("nan")

        if (
            not isfinite(edge_length)
            or edge_length < 0
            or not isfinite(base_length)
            or base_length < 0
        ):
            costs_ok = False

        if edge_length != base_length:
            wrong_length += 1

        if bool(attrs.get("traffic_sampled")):
            sampled += 1
        if bool(attrs.get("traffic_directly_queried")):
            directly_queried += 1
        if bool(attrs.get("traffic_matched")):
            matched += 1

        status = attrs.get("traffic_status", "missing")
        if status in _USABLE_TRAFFIC_STATUSES:
            observed += 1
            source = attrs.get("traffic_match_source")
            if source == "direct_probe":
                direct_observed += 1
            elif source == "reverse_inferred":
                reverse_observed += 1
        elif status == "missing":
            missing += 1
        else:
            invalid_status_edges += 1
            missing += 1
            costs_ok = False

        if attrs.get("time_is_imputed") is True:
            imputed += 1

        value = attrs.get("travel_time_s")
        is_closed = attrs.get("is_closed") is True or status == "closed"

        if is_closed:
            if (
                attrs.get("is_closed") is not True
                or status != "closed"
                or value is not None
            ):
                costs_ok = False
        else:
            try:
                numeric_time = float(value)
            except (TypeError, ValueError, OverflowError):
                numeric_time = float("nan")

            if not isfinite(numeric_time) or numeric_time < 0:
                costs_ok = False

    total = len(snapshot_edges)

    def percent(count: int) -> float:
        return round(100.0 * count / total, 3) if total else 0.0

    coverage = percent(observed)
    direct_coverage = percent(direct_observed)
    reverse_coverage = percent(reverse_observed)

    warnings: list[str] = []
    if not identical_edges or wrong_length:
        warnings.append(
            "The traffic snapshot does not match the base OSM graph exactly"
        )
    if not costs_ok:
        warnings.append("Some edge travel times or traffic statuses are invalid")
    if coverage < 5:
        warnings.append(
            "Observed traffic coverage is sparse; most travel times are not measured"
        )
    if direct_observed == 0:
        warnings.append("No usable direct traffic observations are present")
    if invalid_status_edges:
        warnings.append(
            f"Unknown traffic status on {invalid_status_edges} directed edges"
        )

    warnings.append(
        "TomTom geometry point order is not authoritative legal direction proof"
    )
    if summary and summary.get("request_errors", 0):
        warnings.append(f"TomTom request errors: {summary['request_errors']}")

    return {
        "graph_identity_ok": identical_edges and wrong_length == 0,
        "costs_ok": costs_ok,
        "base_nodes": base_graph.number_of_nodes(),
        "snapshot_nodes": snapshot.number_of_nodes(),
        "total_directed_edges": total,
        "sampled_directed_edges": sampled,
        "directly_queried_directed_edges": directly_queried,
        "matched_edges": matched,
        "observed_edges": observed,
        "direct_observed_edges": direct_observed,
        "reverse_inferred_observed_edges": reverse_observed,
        "missing_edges": missing,
        "imputed_edges": imputed,
        "invalid_status_edges": invalid_status_edges,
        "observed_coverage_percent": coverage,
        "direct_observed_coverage_percent": direct_coverage,
        "reverse_observed_coverage_percent": reverse_coverage,
        # This is only a software smoke-test gate, not a real-data benchmark gate.
        "algorithm_smoke_test_ready": (
            identical_edges
            and wrong_length == 0
            and costs_ok
            and total > 0
        ),
        # Keep false until route-level observed coverage and benchmark criteria
        # have been defined and met.
        "final_traffic_comparison_ready": False,
        "warnings": warnings,
    }
