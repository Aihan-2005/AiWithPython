

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import time

import networkx as nx
from pyproj import Transformer

from .config import (
    DEFAULT_UNOBSERVED_SPEED_KPH, MAX_TRAFFIC_REQUESTS,
    MIN_TOMTOM_CONFIDENCE, TRAFFIC_REQUEST_PAUSE_S,
    RESULTS_DIR, MAP_CENTER,
)
from .map_matching import directed_osm_line, assess_osm_edge
from .traffic_api import TomTomClient, TomTomAPIError


def _positive(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _initialize_costs(graph: nx.MultiDiGraph) -> None:
    if DEFAULT_UNOBSERVED_SPEED_KPH <= 0:
        raise ValueError("Fallback speed must be positive")
    for _, _, _, edge in graph.edges(keys=True, data=True):
        length = float(edge["length"])
        if not math.isfinite(length) or length < 0:
            raise ValueError("OSM length must be finite and non-negative")
        assumed_time = 3.6 * length / DEFAULT_UNOBSERVED_SPEED_KPH
        edge.update(
            traffic_status="missing", traffic_reason="not_queried",
            traffic_sampled=False, traffic_matched=False, traffic_match_reason=None,
            direction_match_method=None, traffic_confidence=None,
            current_speed_kph=None, free_flow_speed_kph=None,
            fallback_speed_kph=DEFAULT_UNOBSERVED_SPEED_KPH,
            free_flow_time_s=assumed_time, travel_time_s=assumed_time,
            free_flow_is_imputed=True, time_is_imputed=True,
            is_closed=False,
        )


def _apply_validated_flow(edge: dict, payload: dict, min_conf: float) -> None:
    edge["traffic_matched"] = True
    edge["direction_match_method"] = "coordinate_order_proxy_unverified"
    try:
        conf = float(payload.get("confidence"))
    except (TypeError, ValueError):
        conf = math.nan
    edge["traffic_confidence"] = conf if math.isfinite(conf) else None
    if not (math.isfinite(conf) and 0 <= conf <= 1 and conf >= min_conf):
        edge["traffic_reason"] = "low_confidence" if math.isfinite(conf) else "missing_confidence"
        return
    free = _positive(payload.get("freeFlowSpeed"))
    current = _positive(payload.get("currentSpeed"))
    if free is not None:
        edge["free_flow_speed_kph"] = free
        edge["free_flow_time_s"] = 3.6 * float(edge["length"]) / free
        edge["free_flow_is_imputed"] = False
    if payload.get("roadClosure") is True:
        edge.update(
            is_closed=True, traffic_status="closed", traffic_reason="reported_closure",
            current_speed_kph=0.0, travel_time_s=None, time_is_imputed=False,
        )
        return
    if current is None or free is None:
        edge["traffic_reason"] = "invalid_or_zero_speed"
        return
    r = current / free
    edge.update(
        traffic_status="red" if r < 0.5 else "yellow" if r < 0.8 else "clear",
        traffic_reason="observed", current_speed_kph=current,
        travel_time_s=3.6 * float(edge["length"]) / current,
        time_is_imputed=False,
    )


def _select_edges(graph: nx.MultiDiGraph, limit: int) -> list[tuple]:
 
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    candidates = []
    for edge_id in sorted(graph.edges(keys=True), key=lambda e: tuple(map(str, e))):
        line = directed_osm_line(graph, *edge_id)
        mid = line.interpolate(0.5, normalized=True)
        x, y = transformer.transform(mid.x, mid.y)
        candidates.append((edge_id, x, y))
    if not candidates:
        return []
    cx, cy = transformer.transform(MAP_CENTER[1], MAP_CENTER[0])
    n = len(candidates)
    closest = min(range(n), key=lambda i: ((candidates[i][1]-cx)**2 + (candidates[i][2]-cy)**2, i))
    selected = [closest]
    chosen = {closest}
    min_sq = [
        (c[1] - candidates[closest][1]) ** 2 + (c[2] - candidates[closest][2]) ** 2
        for c in candidates
    ]
    for _ in range(min(limit, n) - 1):

        choices = (i for i in range(n) if i not in chosen and min_sq[i] >= 15 ** 2)
        best = max(choices, key=lambda i: (min_sq[i], -i), default=None)
        if best is None:
            break
        chosen.add(best)
        selected.append(best)
        bx, by = candidates[best][1:]
        for i, (_, x, y) in enumerate(candidates):
            d = (x - bx) ** 2 + (y - by) ** 2
            if d < min_sq[i]:
                min_sq[i] = d
    return [candidates[i][0] for i in selected]


def preview_probes(graph: nx.MultiDiGraph, limit: int = 5) -> list[dict]:
    rows = []
    for edge_id in _select_edges(graph, limit):
        mid = directed_osm_line(graph, *edge_id).interpolate(0.5, normalized=True)
        rows.append({
            "edge_id": edge_id,
            "latitude": round(mid.y, 7), "longitude": round(mid.x, 7),
            "length_m": round(float(graph[edge_id[0]][edge_id[1]][edge_id[2]]["length"]), 1),
        })
    return rows


def collect_traffic_snapshot(
    osm_graph: nx.MultiDiGraph,
    client: TomTomClient,
    *,
    max_requests: int = MAX_TRAFFIC_REQUESTS,
    pause_s: float = TRAFFIC_REQUEST_PAUSE_S,
    confidence_threshold: float = MIN_TOMTOM_CONFIDENCE,
) -> tuple[nx.MultiDiGraph, dict]:
    if not isinstance(max_requests, int) or max_requests < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0,1]")
    if pause_s < 0:
        raise ValueError("pause_s must be non-negative")
    graph = osm_graph.copy()
    _initialize_costs(graph)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    targets = _select_edges(graph, max_requests)
    start = datetime.now(timezone.utc).isoformat()
    requests_made = request_errors = 0
    stop_code = None
    failures = Counter()
    examples = {}

    for u, v, k in targets:
        edge_id = (u, v, k)
        attrs = graph[u][v][k]
        midpoint = directed_osm_line(graph, u, v, k).interpolate(0.5, normalized=True)
        attrs["traffic_sampled"] = True
        try:
            requests_made += 1
            payload = client.fetch_flow(lat=midpoint.y, lon=midpoint.x)
        except TomTomAPIError as exc:
            request_errors += 1
            code = exc.status_code
            reason = f"api_http_{code}" if code is not None else "api_network_error"
            attrs["traffic_reason"] = reason
            failures[(code, exc.error_code)] += 1
            if len(examples) < 5 and (code, exc.error_code) not in examples:
                examples[(code, exc.error_code)] = exc.detail
            if code in (401, 403, 429):
                stop_code = code
                break
        else:
            match = assess_osm_edge(graph, edge_id, payload, transformer)
            attrs["traffic_match_reason"] = match.reason
            if match.accepted:
                _apply_validated_flow(attrs, payload, confidence_threshold)
            else:

                attrs["traffic_reason"] = "geometry_or_heading_mismatch"
        if pause_s:
            time.sleep(pause_s)

    end = datetime.now(timezone.utc).isoformat()
    counts = Counter(edge["traffic_status"] for *_, edge in graph.edges(keys=True, data=True))
    counts = {k: counts[k] for k in ("missing", "red", "yellow", "clear", "closed")}
    reasons = dict(Counter(edge["traffic_reason"] for *_, edge in graph.edges(keys=True, data=True)))
    match_reasons = dict(Counter(edge["traffic_match_reason"] for *_, edge in graph.edges(keys=True, data=True)
                                 if edge["traffic_match_reason"] is not None))
    total = graph.number_of_edges()
    matched = sum(bool(d["traffic_matched"]) for *_, d in graph.edges(keys=True, data=True))
    usable = sum(counts[k] for k in ("red", "yellow", "clear", "closed"))
    sampled = sum(bool(d["traffic_sampled"]) for *_, d in graph.edges(keys=True, data=True))
    summary = {
        "started_at_utc": start, "finished_at_utc": end,
        "requests_made": requests_made, "request_errors": request_errors,
        "stopped_http_status": stop_code,
        "total_directed_edges": total, "sampled_directed_edges": sampled,
        "geometry_and_heading_candidate_matches": matched,
        "usable_observed_edges": usable,
        "matched_coverage_percent": round(100 * matched / total, 2) if total else 0.0,
        "usable_coverage_percent": round(100 * usable / total, 2) if total else 0.0,
        "usable_among_sampled_percent": round(100 * usable / sampled, 2) if sampled else 0.0,
        "counts": counts, "reasons": reasons,
        "map_matching_diagnostics": match_reasons,
        "api_error_groups": [
            {"http_status": http, "error_code": c, "count": count,
             "example_message_redacted": examples.get((http, c))}
            for (http, c), count in sorted(failures.items(), key=lambda t: (str(t[0]),))
        ],
        "sampling_strategy": "deterministic_farthest_midpoints_15m_dedup",
        "fallback_speed_kph": DEFAULT_UNOBSERVED_SPEED_KPH,
        "minimum_confidence": confidence_threshold,
        "heading_policy": "TomTom geometry ordering used only as a direction proxy; not legally verified",
        "notes": "Unqueried/unmatched/low-confidence edges stay missing; fallback travel times are imputed.",
    }
    graph.graph["traffic_collection_start_utc"] = start
    graph.graph["traffic_collection_end_utc"] = end
    return graph, summary


def save_summary(summary: dict) -> None:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "traffic_collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_routing_graph(
    snapshot: nx.MultiDiGraph,
    objective: str = "time",
    *, scenario: str = "measured",
    imposed_closure: tuple[int, int, int] | None = None,
) -> nx.MultiDiGraph:
    if objective not in ("time", "distance"):
        raise ValueError("objective must be time or distance")
    if scenario not in ("free_flow", "measured", "closure"):
        raise ValueError("invalid scenario")
    if scenario == "closure":
        if imposed_closure is None or not snapshot.has_edge(*imposed_closure):
            raise ValueError("closure requires a directed existing (u,v,key)")
    elif imposed_closure is not None:
        raise ValueError("imposed_closure only valid with closure")
    graph = snapshot.copy()
    if scenario in ("measured", "closure"):
        graph.remove_edges_from([
            (u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d.get("is_closed")
        ])
    if scenario == "closure" and graph.has_edge(*imposed_closure):
        graph.remove_edge(*imposed_closure)
    graph.graph["routing_scenario"] = scenario
    graph.graph["routing_objective"] = objective
    graph.graph["routing_weight"] = (
        "length" if objective == "distance" else
        "free_flow_time_s" if scenario == "free_flow" else "travel_time_s"
    )
    return graph
