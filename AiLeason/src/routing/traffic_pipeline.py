from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import math
import time

import networkx as nx
from pyproj import Transformer

from .config import (
    DEFAULT_UNOBSERVED_SPEED_KPH,
    MAX_TRAFFIC_REQUESTS,
    MIN_TOMTOM_CONFIDENCE,
    TRAFFIC_REQUEST_PAUSE_S,
    RESULTS_DIR,
    MAP_CENTER,
    MIN_PROBE_EDGE_LENGTH_M,
)
from .map_matching import assess_osm_edge, directed_osm_line
from .traffic_api import TomTomClient, TomTomAPIError


_USABLE_OBSERVED_STATUSES = {"red", "yellow", "clear", "closed"}
_ALL_TRAFFIC_STATUSES = {"missing", "red", "yellow", "clear", "closed"}


def _positive(value: object) -> float | None:
    """Return a finite positive float, or None for invalid/non-positive input."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None
def _initialize_costs(
    graph: nx.MultiDiGraph,
    *,
    preserve_existing: bool = True,
) -> None:
    """Initialize routing costs without discarding existing observations.

    When ``preserve_existing`` is true, previously collected traffic fields are
    retained and only missing defaults are filled. Set it to false only when a
    caller explicitly wants a clean traffic layer on a graph copy.
    """
    fallback_speed = float(DEFAULT_UNOBSERVED_SPEED_KPH)
    if not math.isfinite(fallback_speed) or fallback_speed <= 0:
        raise ValueError("DEFAULT_UNOBSERVED_SPEED_KPH must be finite and positive")

    for _, _, _, edge in graph.edges(keys=True, data=True):
        try:
            length = float(edge["length"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Every OSM edge must have a numeric 'length'") from exc
        if not math.isfinite(length) or length < 0:
            raise ValueError("OSM edge length must be finite and non-negative")

        fallback_time = 3.6 * length / fallback_speed
        defaults = {
            "traffic_status": "missing",
            "traffic_reason": "not_queried",
            "traffic_sampled": False,
            "traffic_directly_queried": False,
            "traffic_matched": False,
            "traffic_match_reason": None,
            "traffic_match_source": None,
            "direction_match_method": None,
            "traffic_confidence": None,
            "traffic_match_midpoint_distance_m": None,
            "traffic_match_fraction_near": None,
            "traffic_match_direction_cosine": None,
            "traffic_probe_fraction": None,
            "traffic_probe_latitude": None,
            "traffic_probe_longitude": None,
            "traffic_attempt_count": 0,
            "current_speed_kph": None,
            "free_flow_speed_kph": None,
            "fallback_speed_kph": fallback_speed,
            "free_flow_time_s": fallback_time,
            "travel_time_s": fallback_time,
            "free_flow_is_imputed": True,
            "time_is_imputed": True,
            "is_closed": False,
            "traffic_probe_edge_id": None,
        }
        if preserve_existing:
            for name, value in defaults.items():
                edge.setdefault(name, value)
        else:
            edge.update(defaults)


def _edge_id_text(edge_id: tuple) -> str:
    """Serialize a directed OSM edge ID for JSON-friendly edge metadata."""
    return "|".join(map(str, edge_id))


def _apply_validated_flow(
    edge: dict,
    payload: dict,
    min_conf: float,
    *,
    direction_method: str = "coordinate_order_proxy_unverified",
    probe_edge_id: tuple | None = None,
    match_source: str = "direct_probe",
    probe_fraction: float = 0.5,
) -> None:
    """
    Store a geometrically matched TomTom response on one directed edge.

    A geometry match and a usable traffic observation are deliberately
    different concepts: low-confidence/invalid-speed matches are recorded
    as matched, but remain traffic_status='missing' with imputed route time.
    """
    edge["traffic_matched"] = True
    edge["traffic_match_source"] = match_source
    edge["direction_match_method"] = direction_method
    edge["traffic_probe_edge_id"] = (
        _edge_id_text(probe_edge_id) if probe_edge_id is not None else None
    )
    edge["traffic_probe_fraction"] = float(probe_fraction)

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        confidence = math.nan

    edge["traffic_confidence"] = (
        confidence if math.isfinite(confidence) else None
    )

    if not (
        math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
        and confidence >= min_conf
    ):
        edge["traffic_reason"] = (
            "low_confidence" if math.isfinite(confidence)
            else "missing_confidence"
        )
        return

    free_flow_speed = _positive(payload.get("freeFlowSpeed"))
    current_speed = _positive(payload.get("currentSpeed"))
    edge_length = float(edge["length"])

    if free_flow_speed is not None:
        edge["free_flow_speed_kph"] = free_flow_speed
        edge["free_flow_time_s"] = 3.6 * edge_length / free_flow_speed
        edge["free_flow_is_imputed"] = False

    # A closure is trusted only after geometry and confidence checks pass.
    if payload.get("roadClosure") is True:
        edge.update(
            is_closed=True,
            traffic_status="closed",
            traffic_reason="reported_closure",
            current_speed_kph=0.0,
            travel_time_s=None,
            time_is_imputed=False,
        )
        return

    if current_speed is None or free_flow_speed is None:
        edge["traffic_reason"] = "invalid_or_zero_speed"
        return

    speed_ratio = current_speed / free_flow_speed
    edge.update(
        traffic_status=(
            "red" if speed_ratio < 0.5
            else "yellow" if speed_ratio < 0.8
            else "clear"
        ),
        traffic_reason="observed",
        current_speed_kph=current_speed,
        travel_time_s=3.6 * edge_length / current_speed,
        time_is_imputed=False,
    )


def _select_edges(graph: nx.MultiDiGraph, limit: int) -> list[tuple]:
    """
    Select spatially distributed directed-edge midpoints.

    The first probe is closest to MAP_CENTER. Remaining probes are selected
    with a deterministic farthest-point strategy and a 15 m minimum spacing.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0 or graph.number_of_edges() == 0:
        return []

    graph_crs = graph.graph.get("crs")
    if not graph_crs:
        raise ValueError("OSM graph must declare graph.graph['crs']")

    graph_to_meters = Transformer.from_crs(
        graph_crs, "EPSG:27700", always_xy=True
    )
    center_to_meters = Transformer.from_crs(
        "EPSG:4326", "EPSG:27700", always_xy=True
    )

    candidates = []
    for edge_id in sorted(
        graph.edges(keys=True),
        key=lambda item: tuple(map(str, item)),
    ):
        edge_attrs = graph[edge_id[0]][edge_id[1]][edge_id[2]]
        edge_length = _positive(edge_attrs.get("length"))
        if edge_length is None or edge_length < MIN_PROBE_EDGE_LENGTH_M:
            continue
        line = directed_osm_line(graph, *edge_id)
        midpoint = line.interpolate(0.5, normalized=True)
        x, y = graph_to_meters.transform(midpoint.x, midpoint.y)
        if math.isfinite(x) and math.isfinite(y):
            candidates.append((edge_id, x, y))

    if not candidates:
        return []

    center_x, center_y = center_to_meters.transform(
        MAP_CENTER[1], MAP_CENTER[0]
    )
    closest = min(
        range(len(candidates)),
        key=lambda i: (
            (candidates[i][1] - center_x) ** 2
            + (candidates[i][2] - center_y) ** 2,
            i,
        ),
    )

    selected_indices = [closest]
    chosen_indices = {closest}
    min_distance_sq = [
        (candidate[1] - candidates[closest][1]) ** 2
        + (candidate[2] - candidates[closest][2]) ** 2
        for candidate in candidates
    ]

    target_count = min(limit, len(candidates))
    for _ in range(target_count - 1):
        choices = (
            i for i in range(len(candidates))
            if i not in chosen_indices and min_distance_sq[i] >= 15.0**2
        )
        best = max(
            choices,
            key=lambda i: (min_distance_sq[i], -i),
            default=None,
        )
        if best is None:
            break

        chosen_indices.add(best)
        selected_indices.append(best)
        best_x, best_y = candidates[best][1:]

        for i, (_, x, y) in enumerate(candidates):
            distance_sq = (x - best_x) ** 2 + (y - best_y) ** 2
            if distance_sq < min_distance_sq[i]:
                min_distance_sq[i] = distance_sq

    return [candidates[i][0] for i in selected_indices]


def _reverse_edge_ids(
    graph: nx.MultiDiGraph,
    edge_id: tuple,
) -> list[tuple]:
    """Return existing directed edges from v to u; never create reverse edges."""
    if not isinstance(edge_id, (tuple, list)) or len(edge_id) != 3:
        raise ValueError("edge_id must be a directed (u, v, key) tuple")

    u, v, _ = edge_id
    reverse_data = graph.get_edge_data(v, u, default={})
    return [(v, u, key) for key in sorted(reverse_data, key=str)]


def preview_probes(
    graph: nx.MultiDiGraph,
    limit: int = 5,
) -> list[dict]:
    """Return a preview of the automatically selected TomTom probe locations."""
    rows = []
    for edge_id in _select_edges(graph, limit):
        midpoint = directed_osm_line(
            graph, *edge_id
        ).interpolate(0.5, normalized=True)
        edge_attrs = graph[edge_id[0]][edge_id[1]][edge_id[2]]
        rows.append(
            {
                "edge_id": edge_id,
                "latitude": round(midpoint.y, 7),
                "longitude": round(midpoint.x, 7),
                "length_m": round(float(edge_attrs["length"]), 1),
            }
        )
    return rows


def _usable_observation(attrs: Mapping) -> bool:
    return (
        attrs.get("traffic_matched") is True
        and attrs.get("time_is_imputed") is False
        and attrs.get("traffic_status") in _USABLE_OBSERVED_STATUSES
    )


def _store_match_diagnostics(
    attrs: dict,
    match,
    *,
    probe_fraction: float,
    probe_latitude: float,
    probe_longitude: float,
    prefix: str = "traffic_match_",
) -> None:
    """Store all map-matching diagnostics, including rejected matches."""
    attrs[f"{prefix}reason"] = match.reason
    attrs[f"{prefix}midpoint_distance_m"] = match.midpoint_distance_m
    attrs[f"{prefix}fraction_near"] = match.fraction_near
    attrs[f"{prefix}direction_cosine"] = match.direction_cosine
    attrs["traffic_probe_fraction"] = float(probe_fraction)
    attrs["traffic_probe_latitude"] = float(probe_latitude)
    attrs["traffic_probe_longitude"] = float(probe_longitude)


def _apply_flow_preserving_usable_observation(
    edge: dict,
    payload: dict,
    min_confidence: float,
    *,
    previous_usable: bool,
    direction_method: str,
    probe_edge_id: tuple,
    match_source: str,
    probe_fraction: float,
) -> None:
    """Apply a fresh flow response without destroying an older valid reading."""
    candidate = dict(edge)
    fallback_speed = float(edge.get("fallback_speed_kph", DEFAULT_UNOBSERVED_SPEED_KPH))
    edge_length = float(edge["length"])
    candidate.update(
        traffic_status="missing",
        traffic_reason="not_queried",
        traffic_matched=False,
        traffic_match_source=None,
        traffic_confidence=None,
        current_speed_kph=None,
        free_flow_speed_kph=None,
        travel_time_s=3.6 * edge_length / fallback_speed,
        free_flow_time_s=3.6 * edge_length / fallback_speed,
        time_is_imputed=True,
        free_flow_is_imputed=True,
        is_closed=False,
    )
    # A successful geometry match supersedes the previous match diagnostics.
    candidate["traffic_match_reason"] = edge.get(
        "traffic_last_attempt_match_reason", "candidate_match"
    )
    for canonical, last_attempt in (
        ("traffic_match_midpoint_distance_m", "traffic_last_attempt_match_midpoint_distance_m"),
        ("traffic_match_fraction_near", "traffic_last_attempt_match_fraction_near"),
        ("traffic_match_direction_cosine", "traffic_last_attempt_match_direction_cosine"),
    ):
        if edge.get(last_attempt) is not None:
            candidate[canonical] = edge[last_attempt]
    _apply_validated_flow(
        candidate,
        payload,
        min_confidence,
        direction_method=direction_method,
        probe_edge_id=probe_edge_id,
        match_source=match_source,
        probe_fraction=probe_fraction,
    )
    if previous_usable and not _usable_observation(candidate):
        edge["traffic_last_attempt_reason"] = candidate.get("traffic_reason")
        edge["traffic_last_attempt_confidence"] = candidate.get("traffic_confidence")
        edge["traffic_last_attempt_match_source"] = match_source
        return
    edge.update(candidate)


def collect_traffic_snapshot(
    osm_graph: nx.MultiDiGraph,
    client: TomTomClient | None,
    *,
    max_requests: int = MAX_TRAFFIC_REQUESTS,
    pause_s: float = TRAFFIC_REQUEST_PAUSE_S,
    confidence_threshold: float = MIN_TOMTOM_CONFIDENCE,
    selected_edge_ids: Iterable[tuple] | None = None,
    probe_fractions: Mapping[tuple, float] | None = None,
) -> tuple[nx.MultiDiGraph, dict]:
    """Collect a bounded TomTom traffic snapshot for directed OSM edges.

    A request point can be moved along an edge with ``probe_fractions``. This
    supports controlled retries after a geometric miss: first probe the edge
    midpoint, then try quarter/three-quarter positions. The map matcher checks
    overlap around the actual request location rather than always comparing
    against the full-edge midpoint.

    Existing observations are preserved if the input is already a traffic
    snapshot. Unmatched or low-confidence responses remain imputed.
    """
    if not isinstance(osm_graph, nx.MultiDiGraph):
        raise TypeError("osm_graph must be a networkx.MultiDiGraph")
    if isinstance(max_requests, bool) or not isinstance(max_requests, int):
        raise TypeError("max_requests must be an integer")
    if max_requests < 0:
        raise ValueError("max_requests must be non-negative")

    try:
        confidence_threshold = float(confidence_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence_threshold must be numeric") from exc
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and between 0 and 1")

    try:
        pause_s = float(pause_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pause_s must be numeric") from exc
    if not math.isfinite(pause_s) or pause_s < 0:
        raise ValueError("pause_s must be finite and non-negative")

    graph = osm_graph.copy()
    _initialize_costs(graph, preserve_existing=True)

    graph_crs = graph.graph.get("crs", "EPSG:4326")
    transformer = Transformer.from_crs(
        graph_crs, "EPSG:27700", always_xy=True
    )

    rejected_short_edges: list[dict] = []
    if selected_edge_ids is None:
        targets = _select_edges(graph, max_requests)
        sampling_strategy = "deterministic_farthest_midpoints_min_length"
    else:
        targets = []
        seen = set()
        for raw_edge_id in selected_edge_ids:
            if not isinstance(raw_edge_id, (tuple, list)) or len(raw_edge_id) != 3:
                raise ValueError("Each planned probe must be a directed (u, v, key) edge")
            edge_id = tuple(raw_edge_id)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            if not graph.has_edge(*edge_id):
                raise ValueError(f"Probe edge not found in graph: {edge_id}")
            attrs = graph[edge_id[0]][edge_id[1]][edge_id[2]]
            length = _positive(attrs.get("length"))
            if length is None or length < MIN_PROBE_EDGE_LENGTH_M:
                rejected_short_edges.append({
                    "edge_id": list(edge_id),
                    "length_m": length,
                    "reason": "below_min_probe_edge_length",
                })
                continue
            if len(targets) < max_requests:
                targets.append(edge_id)
        sampling_strategy = "provided_reviewed_edge_plan"

    fraction_map: dict[tuple, float] = {}
    if probe_fractions is not None:
        if not isinstance(probe_fractions, Mapping):
            raise TypeError("probe_fractions must be a mapping of edge_id to fraction")
        for raw_edge_id, raw_fraction in probe_fractions.items():
            edge_id = tuple(raw_edge_id)
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid probe fraction for {edge_id}") from exc
            if not math.isfinite(fraction) or not 0.05 <= fraction <= 0.95:
                raise ValueError(f"Probe fraction for {edge_id} must be between 0.05 and 0.95")
            fraction_map[edge_id] = fraction

    if targets and client is None:
        raise ValueError("A TomTom client is required when probes are selected")

    target_set = set(targets)
    started_at = datetime.now(timezone.utc).isoformat()
    requests_made = 0
    request_errors = 0
    stopped_http_status = None
    reverse_direction_reassignments = 0
    reverse_candidate_checks = 0
    api_failures = Counter()
    api_error_examples = {}
    probe_diagnostics: list[dict] = []
    assessed_edge_ids: set[tuple] = set()

    for edge_id in targets:
        u, v, key = edge_id
        attrs = graph[u][v][key]
        line = directed_osm_line(graph, u, v, key)
        fraction = fraction_map.get(edge_id, 0.5)
        probe_point = line.interpolate(fraction, normalized=True)
        probe_latitude = float(probe_point.y)
        probe_longitude = float(probe_point.x)
        previous_usable = _usable_observation(attrs)

        assessed_edge_ids.add(edge_id)
        attrs["traffic_sampled"] = True
        attrs["traffic_directly_queried"] = True
        try:
            previous_attempt_count = int(attrs.get("traffic_attempt_count", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            previous_attempt_count = 0
        attrs["traffic_attempt_count"] = max(0, previous_attempt_count) + 1
        attrs["traffic_probe_fraction"] = fraction
        attrs["traffic_probe_latitude"] = probe_latitude
        attrs["traffic_probe_longitude"] = probe_longitude
        attrs["traffic_last_checked_at_utc"] = datetime.now(timezone.utc).isoformat()

        try:
            requests_made += 1
            payload = client.fetch_flow(lat=probe_latitude, lon=probe_longitude)
        except TomTomAPIError as exc:
            request_errors += 1
            status_code = exc.status_code
            error_code = getattr(exc, "error_code", None)
            detail = getattr(exc, "detail", None)
            reason = (
                f"api_http_{status_code}"
                if status_code is not None
                else "api_network_error"
            )
            attrs["traffic_last_attempt_reason"] = reason
            if not previous_usable:
                attrs["traffic_reason"] = reason
            api_failures[(status_code, error_code)] += 1
            if len(api_error_examples) < 5 and (status_code, error_code) not in api_error_examples:
                api_error_examples[(status_code, error_code)] = detail
            probe_diagnostics.append({
                "edge_id": list(edge_id),
                "probe_fraction": fraction,
                "latitude": round(probe_latitude, 7),
                "longitude": round(probe_longitude, 7),
                "match_reason": None,
                "traffic_reason": reason,
                "midpoint_distance_m": None,
                "fraction_near": None,
                "direction_cosine": None,
            })
            if status_code in (401, 403, 429):
                stopped_http_status = status_code
                break
        else:
            match = assess_osm_edge(
                graph,
                edge_id,
                payload,
                transformer,
                probe_fraction=fraction,
            )
            _store_match_diagnostics(
                attrs,
                match,
                probe_fraction=fraction,
                probe_latitude=probe_latitude,
                probe_longitude=probe_longitude,
                prefix=("traffic_last_attempt_match_" if previous_usable else "traffic_match_"),
            )

            if match.accepted:
                _apply_flow_preserving_usable_observation(
                    attrs,
                    payload,
                    confidence_threshold,
                    previous_usable=previous_usable,
                    direction_method="coordinate_order_proxy_unverified",
                    probe_edge_id=edge_id,
                    match_source="direct_probe",
                    probe_fraction=fraction,
                )
            elif match.reason == "heading_proxy_mismatch":
                reverse_ids = _reverse_edge_ids(graph, edge_id)
                reverse_matches = []
                for reverse_id in reverse_ids:
                    reverse_candidate_checks += 1
                    assessed_edge_ids.add(reverse_id)
                    reverse_attrs = graph[reverse_id[0]][reverse_id[1]][reverse_id[2]]
                    reverse_attrs["traffic_sampled"] = True
                    reverse_fraction = 1.0 - fraction
                    reverse_match = assess_osm_edge(
                        graph,
                        reverse_id,
                        payload,
                        transformer,
                        probe_fraction=reverse_fraction,
                    )
                    _store_match_diagnostics(
                        reverse_attrs,
                        reverse_match,
                        probe_fraction=reverse_fraction,
                        probe_latitude=probe_latitude,
                        probe_longitude=probe_longitude,
                        prefix=("traffic_last_attempt_match_" if _usable_observation(reverse_attrs) else "traffic_match_"),
                    )
                    if reverse_match.accepted:
                        reverse_matches.append((reverse_id, reverse_match, reverse_fraction))

                if len(reverse_matches) > 1:
                    attrs["traffic_reason"] = "ambiguous_reverse_match" if not previous_usable else attrs.get("traffic_reason")
                    attrs["traffic_last_attempt_reason"] = "ambiguous_reverse_match"
                    for reverse_id, _, _ in reverse_matches:
                        reverse_attrs = graph[reverse_id[0]][reverse_id[1]][reverse_id[2]]
                        reverse_attrs["traffic_reason"] = "ambiguous_reverse_match"
                        reverse_attrs["traffic_match_reason"] = "ambiguous_reverse_candidate"
                elif len(reverse_matches) == 1:
                    reverse_id, reverse_match, reverse_fraction = reverse_matches[0]
                    reverse_attrs = graph[reverse_id[0]][reverse_id[1]][reverse_id[2]]
                    if reverse_id in target_set:
                        if not previous_usable:
                            attrs["traffic_reason"] = "reverse_edge_selected_for_direct_query"
                    elif _usable_observation(reverse_attrs):
                        if not previous_usable:
                            attrs["traffic_reason"] = "reverse_edge_already_observed"
                    else:
                        reverse_attrs["traffic_match_reason"] = "reverse_candidate_match"
                        _apply_flow_preserving_usable_observation(
                            reverse_attrs,
                            payload,
                            confidence_threshold,
                            previous_usable=False,
                            direction_method="reverse_coordinate_order_proxy_unverified",
                            probe_edge_id=edge_id,
                            match_source="reverse_inferred",
                            probe_fraction=reverse_fraction,
                        )
                        attrs["traffic_reason"] = "matched_on_reverse_edge" if not previous_usable else attrs.get("traffic_reason")
                        attrs["traffic_last_attempt_reason"] = "matched_on_reverse_edge"
                        reverse_direction_reassignments += 1
                else:
                    if not previous_usable:
                        attrs["traffic_reason"] = match.reason
                    attrs["traffic_last_attempt_reason"] = match.reason
            else:
                if not previous_usable:
                    attrs["traffic_reason"] = match.reason
                attrs["traffic_last_attempt_reason"] = match.reason

            probe_diagnostics.append({
                "edge_id": list(edge_id),
                "probe_fraction": fraction,
                "latitude": round(probe_latitude, 7),
                "longitude": round(probe_longitude, 7),
                "match_reason": match.reason,
                "traffic_reason": attrs.get("traffic_last_attempt_reason", attrs.get("traffic_reason")),
                "midpoint_distance_m": match.midpoint_distance_m,
                "fraction_near": match.fraction_near,
                "direction_cosine": match.direction_cosine,
            })

        if pause_s:
            time.sleep(pause_s)

    finished_at = datetime.now(timezone.utc).isoformat()
    all_edges = [data for _, _, _, data in graph.edges(keys=True, data=True)]
    counts_counter = Counter(edge.get("traffic_status", "missing") for edge in all_edges)
    counts = {
        status: counts_counter[status]
        for status in ("missing", "red", "yellow", "clear", "closed")
    }
    reasons = dict(Counter(edge.get("traffic_reason", "not_recorded") for edge in all_edges))
    match_reasons = dict(Counter(
        edge.get("traffic_match_reason")
        for edge in all_edges
        if edge.get("traffic_match_reason") is not None
    ))
    total_edges = graph.number_of_edges()
    matched_edges = sum(bool(edge.get("traffic_matched")) for edge in all_edges)
    usable_edges = sum(counts[status] for status in _USABLE_OBSERVED_STATUSES)
    sampled_edges = len(assessed_edge_ids)
    directly_queried_edges = requests_made
    direct_observed_edges = sum(
        edge.get("traffic_status") in _USABLE_OBSERVED_STATUSES
        and edge.get("traffic_match_source") == "direct_probe"
        and edge.get("time_is_imputed") is False
        for edge in all_edges
    )
    reverse_observed_edges = sum(
        edge.get("traffic_status") in _USABLE_OBSERVED_STATUSES
        and edge.get("traffic_match_source") == "reverse_inferred"
        and edge.get("time_is_imputed") is False
        for edge in all_edges
    )

    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "requests_made": requests_made,
        "request_errors": request_errors,
        "stopped_http_status": stopped_http_status,
        "total_directed_edges": total_edges,
        "sampled_directed_edges": sampled_edges,
        "directly_queried_directed_edges": directly_queried_edges,
        "reverse_candidate_checks": reverse_candidate_checks,
        "reverse_direction_reassignments": reverse_direction_reassignments,
        "geometry_and_heading_candidate_matches": matched_edges,
        "usable_observed_edges": usable_edges,
        "direct_observed_edges": direct_observed_edges,
        "reverse_inferred_observed_edges": reverse_observed_edges,
        "matched_coverage_percent": round(100.0 * matched_edges / total_edges, 3) if total_edges else 0.0,
        "usable_coverage_percent": round(100.0 * usable_edges / total_edges, 3) if total_edges else 0.0,
        "direct_observed_coverage_percent": round(100.0 * direct_observed_edges / total_edges, 3) if total_edges else 0.0,
        "reverse_observed_coverage_percent": round(100.0 * reverse_observed_edges / total_edges, 3) if total_edges else 0.0,
        "usable_among_sampled_percent": round(100.0 * usable_edges / sampled_edges, 3) if sampled_edges else 0.0,
        "counts": counts,
        "reasons": reasons,
        "map_matching_diagnostics": match_reasons,
        "probe_diagnostics": probe_diagnostics,
        "rejected_short_edges": rejected_short_edges,
        "probe_fractions": {"|".join(map(str, edge_id)): fraction_map.get(edge_id, 0.5) for edge_id in targets},
        "api_error_groups": [
            {
                "http_status": status_code,
                "error_code": error_code,
                "count": count,
                "example_message_redacted": api_error_examples.get((status_code, error_code)),
            }
            for (status_code, error_code), count in sorted(
                api_failures.items(),
                key=lambda item: (str(item[0][0]), str(item[0][1])),
            )
        ],
        "sampling_strategy": sampling_strategy,
        "probe_plan_count": len(targets),
        "fallback_speed_kph": DEFAULT_UNOBSERVED_SPEED_KPH,
        "minimum_confidence": confidence_threshold,
        "minimum_probe_edge_length_m": MIN_PROBE_EDGE_LENGTH_M,
        "heading_policy": (
            "TomTom geometry coordinate order is used only as a direction proxy; "
            "reverse-edge matching does not prove legal travel direction."
        ),
        "notes": (
            "Unqueried, unmatched, ambiguous, and low-confidence edges remain "
            "missing with imputed fallback costs. Probe diagnostics include the "
            "actual request fraction and local geometric-match measurements."
        ),
    }

    graph.graph["traffic_collection_start_utc"] = started_at
    graph.graph["traffic_collection_end_utc"] = finished_at
    graph.graph["traffic_collection_summary"] = summary.copy()
    return graph, summary


def save_summary(summary: dict) -> None:
    """Persist a traffic collection summary as UTF-8 JSON."""
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "traffic_collection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_routing_graph(
    snapshot: nx.MultiDiGraph,
    objective: str = "time",
    *,
    scenario: str = "measured",
    imposed_closure: tuple | None = None,
) -> nx.MultiDiGraph:
    """
    Build a routing graph for a time/distance objective and traffic scenario.

    Scenarios:
      - free_flow: use free_flow_time_s for time routing; retain closures.
      - measured: use travel_time_s; missing edges keep imputed fallback costs.
      - closure: measured scenario plus one simulated directed-edge closure.
      - observed_only: retain only usable direct and reverse observations.
      - direct_observed_only: retain only usable direct observations.

    The two observed-only scenarios require objective='time'. They exclude
    fallback/imputed edges so they cannot silently masquerade as measured data.
    """
    if not isinstance(snapshot, nx.MultiDiGraph):
        raise TypeError("snapshot must be a networkx.MultiDiGraph")

    if objective not in ("time", "distance"):
        raise ValueError("objective must be 'time' or 'distance'")

    allowed_scenarios = {
        "free_flow",
        "measured",
        "closure",
        "observed_only",
        "direct_observed_only",
    }
    if scenario not in allowed_scenarios:
        raise ValueError(
            "scenario must be one of: free_flow, measured, closure, "
            "observed_only, direct_observed_only"
        )

    observed_only_scenarios = {"observed_only", "direct_observed_only"}
    if scenario in observed_only_scenarios and objective != "time":
        raise ValueError(
            "Observed-only scenarios require objective='time'"
        )

    if scenario == "closure":
        if (
            imposed_closure is None
            or not isinstance(imposed_closure, (tuple, list))
            or len(imposed_closure) != 3
            or not snapshot.has_edge(*imposed_closure)
        ):
            raise ValueError(
                "closure scenario requires an existing directed "
                "(u, v, key) imposed_closure"
            )
        imposed_closure = tuple(imposed_closure)
    elif imposed_closure is not None:
        raise ValueError(
            "imposed_closure is only valid for scenario='closure'"
        )

    graph = snapshot.copy()

    observed_before_filter = sum(
        attrs.get("traffic_status") in _USABLE_OBSERVED_STATUSES
        and attrs.get("time_is_imputed") is False
        and attrs.get("traffic_match_source")
        in {"direct_probe", "reverse_inferred"}
        for _, _, _, attrs in graph.edges(keys=True, data=True)
    )

    # Reported closures are removed from every traffic-based scenario except
    # free_flow, which intentionally retains the pre-existing behavior.
    if scenario in {
        "measured",
        "closure",
        "observed_only",
        "direct_observed_only",
    }:
        closed_edges = [
            (u, v, key)
            for u, v, key, attrs in graph.edges(keys=True, data=True)
            if (
                attrs.get("is_closed") is True
                or attrs.get("traffic_status") == "closed"
            )
        ]
        graph.remove_edges_from(closed_edges)

    if scenario in observed_only_scenarios:
        allowed_sources = (
            {"direct_probe"}
            if scenario == "direct_observed_only"
            else {"direct_probe", "reverse_inferred"}
        )

        remove_edges = []
        for u, v, key, attrs in graph.edges(keys=True, data=True):
            usable_observation = (
                attrs.get("traffic_status") in _USABLE_OBSERVED_STATUSES
                and attrs.get("time_is_imputed") is False
                and attrs.get("traffic_matched") is True
                and attrs.get("traffic_match_source") in allowed_sources
            )
            if not usable_observation:
                remove_edges.append((u, v, key))

        graph.remove_edges_from(remove_edges)

    if scenario == "closure" and graph.has_edge(*imposed_closure):
        graph.remove_edge(*imposed_closure)

    graph.graph["routing_scenario"] = scenario
    graph.graph["routing_objective"] = objective
    graph.graph["routing_weight"] = (
        "length"
        if objective == "distance"
        else (
            "free_flow_time_s"
            if scenario == "free_flow"
            else "travel_time_s"
        )
    )
    graph.graph["traffic_observed_edges_before_filter"] = int(
        observed_before_filter
    )
    graph.graph["traffic_edges_after_filter"] = graph.number_of_edges()

    policy_by_scenario = {
        "free_flow": "free_flow_or_fallback_costs",
        "measured": "observed_plus_imputed_fallback",
        "closure": "observed_plus_imputed_fallback_with_simulated_closure",
        "observed_only": "direct_and_reverse_observed_only",
        "direct_observed_only": "direct_observations_only",
    }
    graph.graph["traffic_data_policy"] = policy_by_scenario[scenario]

    return graph


