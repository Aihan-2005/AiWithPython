

from __future__ import annotations

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
)
from .map_matching import directed_osm_line, match_osm_edge
from .traffic_api import TomTomClient, TomTomAPIError


def _positive(value) -> float | None:
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _initialize_costs(graph: nx.MultiDiGraph) -> None:
     for _, _, _, edge in graph.edges(keys=True, data=True):
        length = float(edge["length"])
        if not math.isfinite(length) or length < 0:
            raise ValueError("OSM edge length must be finite and non-negative")
        assumed_time = 3.6 * length / DEFAULT_UNOBSERVED_SPEED_KPH
        edge.update(
            traffic_status="missing",
            traffic_reason="not_queried",
            traffic_sampled=False,
            traffic_matched=False,
            direction_match_method=None,
            traffic_confidence=None,
            current_speed_kph=None,
            free_flow_speed_kph=None,  
            fallback_speed_kph=DEFAULT_UNOBSERVED_SPEED_KPH,
            free_flow_time_s=assumed_time,
            travel_time_s=assumed_time,
            free_flow_is_imputed=True,
            time_is_imputed=True,
            is_closed=False,
        )


def _apply_validated_flow(edge: dict, payload: dict, min_conf: float) -> None:
    edge["traffic_matched"] = True
    edge["direction_match_method"] = "geometry_order_proxy_not_authoritative"
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = math.nan
    edge["traffic_confidence"] = confidence if math.isfinite(confidence) else None
    if not (math.isfinite(confidence) and 0 <= confidence <= 1 and confidence >= min_conf):
        edge["traffic_reason"] = "low_confidence" if math.isfinite(confidence) else "missing_confidence"
        return

    free_kph = _positive(payload.get("freeFlowSpeed"))
    current_kph = _positive(payload.get("currentSpeed"))
    if free_kph is not None:
        edge["free_flow_speed_kph"] = free_kph
        edge["free_flow_time_s"] = 3.6 * float(edge["length"]) / free_kph
        edge["free_flow_is_imputed"] = False

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

    if free_kph is None or current_kph is None:
        edge["traffic_reason"] = "invalid_or_zero_speed"
        return

    ratio = current_kph / free_kph
    edge.update(
        traffic_status="red" if ratio < 0.5 else "yellow" if ratio < 0.8 else "clear",
        traffic_reason="observed",
        current_speed_kph=current_kph,
        travel_time_s=3.6 * float(edge["length"]) / current_kph,
        time_is_imputed=False,
    )


def _spread_indices(count: int, selected: int) -> list[int]:
    if count <= 0 or selected <= 0:
        return []
    if selected >= count:
        return list(range(count))
    if selected == 1:
        return [count // 2]
    return [round(i * (count - 1) / (selected - 1)) for i in range(selected)]


def _select_edges(graph: nx.MultiDiGraph, limit: int) -> list[tuple]:
    ids = sorted(graph.edges(keys=True), key=lambda e: (str(e[0]), str(e[1]), str(e[2])))
    return [ids[i] for i in _spread_indices(len(ids), min(limit, len(ids)))]


def preview_probes(graph: nx.MultiDiGraph, limit: int = 5) -> list[dict]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    result = []
    for edge_id in _select_edges(graph, limit):
        line = directed_osm_line(graph, *edge_id)
        mid = line.interpolate(0.5, normalized=True)
        result.append({
            "edge_id": edge_id,
            "latitude": round(mid.y, 7),
            "longitude": round(mid.x, 7),
            "length_m": round(float(graph[edge_id[0]][edge_id[1]][edge_id[2]]["length"]), 1),
        })
    return result


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
        raise ValueError("confidence_threshold must be between 0 and 1")
    if pause_s < 0:
        raise ValueError("pause_s must be non-negative")
    graph = osm_graph.copy()
    _initialize_costs(graph)
    transform_m = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    targets = _select_edges(graph, max_requests)

    start = datetime.now(timezone.utc).isoformat()
    request_count = 0
    error_count = 0
    fatal_code = None
    for edge_id in targets:
        u, v, k = edge_id
        attr = graph[u][v][k]
        osm_line = directed_osm_line(graph, u, v, k)
        sample = osm_line.interpolate(0.5, normalized=True)
        attr["traffic_sampled"] = True
        try:
            request_count += 1
            payload = client.fetch_flow(lat=sample.y, lon=sample.x)
        except TomTomAPIError as ex:
            error_count += 1
            attr["traffic_reason"] = f"api_http_{ex.status_code}" if ex.status_code else "api_network_error"
            if ex.status_code in (401, 403, 429):
                fatal_code = ex.status_code
                break  
            continue
        if match_osm_edge(graph, edge_id, payload, transformer=transform_m):
            _apply_validated_flow(attr, payload, confidence_threshold)
        else:
            attr["traffic_reason"] = "geometry_or_heading_mismatch"
        if pause_s:
            time.sleep(pause_s)

    end = datetime.now(timezone.utc).isoformat()
    counts = {name: 0 for name in ("missing", "red", "yellow", "clear", "closed")}
    reasons = {}
    for _, _, _, attr in graph.edges(keys=True, data=True):
        counts[attr["traffic_status"]] += 1
        reason = attr["traffic_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
    total = graph.number_of_edges()
    matched = sum(bool(d["traffic_matched"]) for *_, d in graph.edges(keys=True, data=True))
    usable = sum(counts[k] for k in ("red", "yellow", "clear", "closed"))
    summary = {
        "started_at_utc": start,
        "finished_at_utc": end,
        "requests_made": request_count,
        "request_errors": error_count,
        "stopped_http_status": fatal_code,
        "total_directed_edges": total,
        "sampled_directed_edges": sum(bool(d["traffic_sampled"]) for *_, d in graph.edges(keys=True, data=True)),
        "geometry_and_heading_candidate_matches": matched,
        "usable_observed_edges": usable,
        "matched_coverage_percent": round(100 * matched / total, 2) if total else 0.0,
        "usable_coverage_percent": round(100 * usable / total, 2) if total else 0.0,
        "counts": counts,
        "reasons": reasons,
        "fallback_speed_kph": DEFAULT_UNOBSERVED_SPEED_KPH,
        "minimum_confidence": confidence_threshold,
        "heading_policy": "TomTom coordinate-order proxy; not authoritative direction verification",
        "notes": "Unqueried/unmatched/low-confidence edges are missing, never clear. Imputed costs are not observations.",
    }
    return graph, summary


def save_summary(summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "traffic_collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_routing_graph(
    snapshot: nx.MultiDiGraph,
    objective: str = "time",
    *,
    scenario: str = "measured",
    imposed_closure: tuple[int, int, int] | None = None,
) -> nx.MultiDiGraph:

    if objective not in ("time", "distance"):
        raise ValueError("objective must be 'time' or 'distance'")
    if scenario not in ("free_flow", "measured", "closure"):
        raise ValueError("scenario must be 'free_flow', 'measured', or 'closure'")
    if scenario == "closure":
        if imposed_closure is None or not snapshot.has_edge(*imposed_closure):
            raise ValueError("closure requires an existing directed (u,v,key) edge")
    elif imposed_closure is not None:
        raise ValueError("imposed_closure is only valid for closure scenario")

    graph = snapshot.copy()
    if scenario in ("measured", "closure"):
        graph.remove_edges_from([
            (u, v, k)
            for u, v, k, e in graph.edges(keys=True, data=True) if e.get("is_closed")
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
