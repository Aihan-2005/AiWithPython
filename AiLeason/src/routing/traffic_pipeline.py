from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
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
)
from .map_matching import assess_osm_edge, directed_osm_line
from .traffic_api import TomTomClient, TomTomAPIError


def _positive(value: object) -> float | None:
    """Return a finite positive float, or None for invalid/non-positive input."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _initialize_costs(graph: nx.MultiDiGraph) -> None:
    """Initialize every directed edge with explicit missing/imputed traffic costs."""
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

        assumed_time = 3.6 * length / fallback_speed
        edge.update(
            traffic_status="missing",
            traffic_reason="not_queried",
            traffic_sampled=False,
            traffic_directly_queried=False,
            traffic_matched=False,
            traffic_match_reason=None,
            traffic_match_source=None,
            direction_match_method=None,
            traffic_confidence=None,
            current_speed_kph=None,
            free_flow_speed_kph=None,
            fallback_speed_kph=fallback_speed,
            free_flow_time_s=assumed_time,
            travel_time_s=assumed_time,
            free_flow_is_imputed=True,
            time_is_imputed=True,
            is_closed=False,
            traffic_probe_edge_id=None,
        )


def _edge_id_text(edge_id: tuple[int, int, int]) -> str:
    """Serialize a directed OSM edge ID for JSON-friendly edge metadata."""
    return "|".join(map(str, edge_id))


def _apply_validated_flow(
    edge: dict,
    payload: dict,
    min_conf: float,
    *,
    direction_method: str = "coordinate_order_proxy_unverified",
    probe_edge_id: tuple[int, int, int] | None = None,
    match_source: str = "direct_probe",
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
            if i not in chosen_indices and min_distance_sq[i] >= 15.0 ** 2
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
    edge_id: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Return existing directed edges from v to u; never create reverse edges."""
    if not isinstance(edge_id, (tuple, list)) or len(edge_id) != 3:
        raise ValueError("edge_id must be a directed (u, v, key) tuple")

    u, v, _ = edge_id
    reverse_data = graph.get_edge_data(v, u, default={})

    return [
        (v, u, key)
        for key in sorted(reverse_data, key=str)
    ]


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


def collect_traffic_snapshot(
    osm_graph: nx.MultiDiGraph,
    client: TomTomClient,
    *,
    max_requests: int = MAX_TRAFFIC_REQUESTS,
    pause_s: float = TRAFFIC_REQUEST_PAUSE_S,
    confidence_threshold: float = MIN_TOMTOM_CONFIDENCE,
    selected_edge_ids: Iterable[tuple[int, int, int]] | None = None,
) -> tuple[nx.MultiDiGraph, dict]:
    """
    Collect a bounded TomTom traffic snapshot for directed OSM edges.

    If the selected edge fails only the heading check, existing reverse
    directed edges are assessed against the same TomTom response. This does
    not issue an additional API request. Traffic is assigned to a reverse
    edge only when exactly one reverse candidate passes and that edge has
    not already been observed or explicitly selected for its own API request.

    `sampled_directed_edges` counts directed candidates assessed (including
    reverse candidates); `directly_queried_directed_edges` counts actual
    requested edges. `requests_made` is the number of API calls attempted.
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
    if (
        not math.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be finite and between 0 and 1")

    try:
        pause_s = float(pause_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pause_s must be numeric") from exc
    if not math.isfinite(pause_s) or pause_s < 0:
        raise ValueError("pause_s must be finite and non-negative")

    graph = osm_graph.copy()
    _initialize_costs(graph)

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:27700", always_xy=True
    )

    if selected_edge_ids is None:
        targets = _select_edges(graph, max_requests)
        sampling_strategy = "deterministic_farthest_midpoints_15m_dedup"
    else:
        targets = []
        seen = set()

        for edge_id in selected_edge_ids:
            if len(targets) >= max_requests:
                break
            if not isinstance(edge_id, (tuple, list)) or len(edge_id) != 3:
                raise ValueError(
                    "Each planned probe must be a directed (u, v, key) edge"
                )

            normalized_id = tuple(edge_id)
            if not graph.has_edge(*normalized_id):
                raise ValueError(
                    f"Probe edge not found in graph: {normalized_id}"
                )

            if normalized_id not in seen:
                seen.add(normalized_id)
                targets.append(normalized_id)

        sampling_strategy = "provided_reviewed_edge_plan"

    target_set = set(targets)
    started_at = datetime.now(timezone.utc).isoformat()

    requests_made = 0
    request_errors = 0
    stopped_http_status = None
    reverse_direction_reassignments = 0
    reverse_candidate_checks = 0

    api_failures = Counter()
    api_error_examples = {}

    for u, v, key in targets:
        edge_id = (u, v, key)
        attrs = graph[u][v][key]

        midpoint = directed_osm_line(
            graph, u, v, key
        ).interpolate(0.5, normalized=True)

        attrs["traffic_sampled"] = True
        attrs["traffic_directly_queried"] = True

        try:
            requests_made += 1
            payload = client.fetch_flow(
                lat=midpoint.y,
                lon=midpoint.x,
            )
        except TomTomAPIError as exc:
            request_errors += 1
            status_code = exc.status_code
            error_code = getattr(exc, "error_code", None)
            detail = getattr(exc, "detail", None)

            attrs["traffic_reason"] = (
                f"api_http_{status_code}"
                if status_code is not None
                else "api_network_error"
            )

            api_failures[(status_code, error_code)] += 1
            if (
                len(api_error_examples) < 5
                and (status_code, error_code) not in api_error_examples
            ):
                api_error_examples[(status_code, error_code)] = detail

            if status_code in (401, 403, 429):
                stopped_http_status = status_code
                break

        else:
            # First assess the edge that was actually queried.
            match = assess_osm_edge(
                graph,
                edge_id,
                payload,
                transformer,
            )
            attrs["traffic_match_reason"] = match.reason

            if match.accepted:
                _apply_validated_flow(
                    attrs,
                    payload,
                    confidence_threshold,
                    direction_method="coordinate_order_proxy_unverified",
                    probe_edge_id=edge_id,
                    match_source="direct_probe",
                )

            elif match.reason == "heading_proxy_mismatch":
                # Only a heading mismatch can be rescued by testing
                # an existing reverse directed edge.
                reverse_ids = _reverse_edge_ids(graph, edge_id)
                reverse_matches = []

                for reverse_id in reverse_ids:
                    reverse_candidate_checks += 1
                    reverse_attrs = graph[
                        reverse_id[0]
                    ][
                        reverse_id[1]
                    ][
                        reverse_id[2]
                    ]

                    reverse_attrs["traffic_sampled"] = True

                    reverse_match = assess_osm_edge(
                        graph,
                        reverse_id,
                        payload,
                        transformer,
                    )
                    reverse_attrs["traffic_match_reason"] = (
                        f"reverse_{reverse_match.reason}"
                    )

                    if reverse_match.accepted:
                        reverse_matches.append(
                            (reverse_id, reverse_match)
                        )
                    else:
                        reverse_attrs["traffic_reason"] = (
                            "geometry_or_heading_mismatch"
                        )

                # Do not choose arbitrarily if multiple reverse edges
                # share sufficiently similar geometry and heading.
                if len(reverse_matches) > 1:
                    attrs["traffic_match_reason"] = "ambiguous_reverse_match"
                    attrs["traffic_reason"] = "ambiguous_reverse_match"

                    for reverse_id, _ in reverse_matches:
                        reverse_attrs = graph[
                            reverse_id[0]
                        ][
                            reverse_id[1]
                        ][
                            reverse_id[2]
                        ]
                        reverse_attrs["traffic_match_reason"] = (
                            "ambiguous_reverse_candidate"
                        )
                        reverse_attrs["traffic_reason"] = (
                            "ambiguous_reverse_match"
                        )

                elif len(reverse_matches) == 1:
                    reverse_id, _ = reverse_matches[0]
                    reverse_attrs = graph[
                        reverse_id[0]
                    ][
                        reverse_id[1]
                    ][
                        reverse_id[2]
                    ]

                    # If the reverse edge has its own scheduled request,
                    # let that direct request determine its traffic state.
                    if reverse_id in target_set:
                        attrs["traffic_reason"] = (
                            "reverse_edge_selected_for_direct_query"
                        )
                    # Never overwrite an observation already obtained
                    # from another probe.
                    elif reverse_attrs["traffic_matched"]:
                        attrs["traffic_reason"] = (
                            "reverse_edge_already_observed"
                        )
                    else:
                        reverse_attrs["traffic_match_reason"] = (
                            "reverse_candidate_match"
                        )

                        _apply_validated_flow(
                            reverse_attrs,
                            payload,
                            confidence_threshold,
                            direction_method=(
                                "reverse_coordinate_order_proxy_unverified"
                            ),
                            probe_edge_id=edge_id,
                            match_source="reverse_inferred",
                        )

                        attrs["traffic_reason"] = "matched_on_reverse_edge"
                        reverse_direction_reassignments += 1

                else:
                    attrs["traffic_reason"] = (
                        "geometry_or_heading_mismatch"
                    )

            else:
                attrs["traffic_reason"] = "geometry_or_heading_mismatch"

        if pause_s:
            time.sleep(pause_s)

    finished_at = datetime.now(timezone.utc).isoformat()

    all_edges = [
        data for _, _, _, data in graph.edges(keys=True, data=True)
    ]

    counts_counter = Counter(
        edge["traffic_status"] for edge in all_edges
    )
    counts = {
        status: counts_counter[status]
        for status in ("missing", "red", "yellow", "clear", "closed")
    }

    reasons = dict(
        Counter(edge["traffic_reason"] for edge in all_edges)
    )
    match_reasons = dict(
        Counter(
            edge["traffic_match_reason"]
            for edge in all_edges
            if edge["traffic_match_reason"] is not None
        )
    )

    total_edges = graph.number_of_edges()
    matched_edges = sum(
        bool(edge["traffic_matched"]) for edge in all_edges
    )
    usable_edges = sum(
        counts[status] for status in ("red", "yellow", "clear", "closed")
    )
    sampled_edges = sum(
        bool(edge["traffic_sampled"]) for edge in all_edges
    )
    directly_queried_edges = sum(
        bool(edge["traffic_directly_queried"]) for edge in all_edges
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
        "matched_coverage_percent": (
            round(100.0 * matched_edges / total_edges, 2)
            if total_edges else 0.0
        ),
        "usable_coverage_percent": (
            round(100.0 * usable_edges / total_edges, 2)
            if total_edges else 0.0
        ),
        "usable_among_sampled_percent": (
            round(100.0 * usable_edges / sampled_edges, 2)
            if sampled_edges else 0.0
        ),
        "counts": counts,
        "reasons": reasons,
        "map_matching_diagnostics": match_reasons,
        "api_error_groups": [
            {
                "http_status": status_code,
                "error_code": error_code,
                "count": count,
                "example_message_redacted": api_error_examples.get(
                    (status_code, error_code)
                ),
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
        "heading_policy": (
            "TomTom geometry coordinate order is used only as a direction "
            "proxy; reverse-edge matching is supported but does not prove "
            "legal travel direction."
        ),
        "notes": (
            "Unqueried, unmatched, ambiguous, and low-confidence edges stay "
            "missing; fallback travel times are imputed. sampled_directed_edges "
            "includes reverse candidates assessed without an extra API call."
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
    imposed_closure: tuple[int, int, int] | None = None,
) -> nx.MultiDiGraph:
    """
    Build a routing graph for a time/distance objective and traffic scenario.

    Scenarios:
      - free_flow: use free_flow_time_s for time routing; retain closures.
      - measured: use travel_time_s and remove TomTom-reported closures.
      - closure: use travel_time_s, remove reported closures, then optionally
        remove one additional directed edge as a simulated closure.
    """
    if not isinstance(snapshot, nx.MultiDiGraph):
        raise TypeError("snapshot must be a networkx.MultiDiGraph")

    if objective not in ("time", "distance"):
        raise ValueError("objective must be 'time' or 'distance'")

    if scenario not in ("free_flow", "measured", "closure"):
        raise ValueError(
            "scenario must be 'free_flow', 'measured', or 'closure'"
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

    if scenario in ("measured", "closure"):
        closed_edges = [
            (u, v, key)
            for u, v, key, attrs in graph.edges(keys=True, data=True)
            if attrs.get("is_closed") is True
        ]
        graph.remove_edges_from(closed_edges)

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

    return graph
