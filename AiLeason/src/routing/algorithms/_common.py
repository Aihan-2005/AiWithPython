
from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import Hashable

import networkx as nx

Node = Hashable
EdgeId = tuple[Node, Node, Hashable]


def node_sort_key(value: Node) -> tuple[str, str]:

    return type(value).__name__, str(value)

def weight_attribute(graph: nx.MultiDiGraph) -> str:
    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        raise TypeError(
            "Expected a directed networkx.MultiDiGraph"
        )

    weight = graph.graph.get("routing_weight")

    allowed = (
        "length",
        "free_flow_time_s",
        "travel_time_s",
    )

    if weight not in allowed:
        raise ValueError(
            "Missing or invalid routing_weight. "
            "Call make_routing_graph() first."
        )

    return weight
 

def valid_cost(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) and number >= 0 else None


def successors(graph: nx.MultiDiGraph, source: Node, weight: str) -> list[tuple[Node, Hashable, float]]:

    result = []
    for target, keyed_edges in graph.succ[source].items():
        candidates = []
        for key, attributes in keyed_edges.items():
            cost = valid_cost(attributes.get(weight))
            if cost is not None:
                candidates.append((cost, node_sort_key(key), key))
        if candidates:
            cost, _, key = min(candidates)
            result.append((target, key, cost))
    result.sort(key=lambda arc: node_sort_key(arc[0]))
    return result


def validate_endpoints(graph: nx.MultiDiGraph, start: Node, goal: Node) -> str:
    weight = weight_attribute(graph)
    if start not in graph:
        raise ValueError(f"Origin node {start!r} is missing from routing graph")
    if goal not in graph:
        raise ValueError(f"Destination node {goal!r} is missing from routing graph")
    return weight


def assemble_result(
    graph: nx.MultiDiGraph,
    *,
    algorithm: str,
    source: Node,
    goal: Node,
    parent: dict[Node, EdgeId],
    found: bool,
    expanded: int,
    max_frontier: int,
    started_at: float,
    weight: str,
) -> dict:
    if found:
        path_edges: list[EdgeId] = []
        node = goal
        while node != source:
            if node not in parent:
                raise RuntimeError("Broken parent chain: cannot reconstruct directed path")
            edge_id = parent[node]
            if edge_id[1] != node or not graph.has_edge(*edge_id):
                raise RuntimeError("Broken parent chain: edge absent or wrong direction")
            path_edges.append(edge_id)
            node = edge_id[0]
        path_edges.reverse()
        path_nodes = [source] + [edge[1] for edge in path_edges]
        time_key = "free_flow_time_s" if graph.graph.get("routing_scenario") == "free_flow" else "travel_time_s"
        distance_m = sum(float(graph[u][v][key]["length"]) for u, v, key in path_edges)
        time_parts = [valid_cost(graph[u][v][key].get(time_key)) for u, v, key in path_edges]
        travel_time_s = sum(time_parts) if all(t is not None for t in time_parts) else None
        objective_cost = sum(float(graph[u][v][key][weight]) for u, v, key in path_edges)
        flag_key = "free_flow_is_imputed" if time_key == "free_flow_time_s" else "time_is_imputed"
        imputed = sum(bool(graph[u][v][key].get(flag_key, False)) for u, v, key in path_edges)
    else:
        path_nodes, path_edges = [], []
        distance_m = travel_time_s = objective_cost = None
        imputed = 0
    runtime_ms = (perf_counter() - started_at) * 1000
    return {
        "algorithm": algorithm,
        "found": found,
        "origin_node": source,
        "destination_node": goal,
        "routing_objective": graph.graph.get("routing_objective", "distance" if weight == "length" else "time"),
        "routing_scenario": graph.graph.get("routing_scenario"),
        "routing_weight": weight,
        "path_nodes": path_nodes,
        "path_edges": path_edges,
        "hops": len(path_edges) if found else None,
        "distance_m": distance_m,
        "travel_time_s": travel_time_s,
        "objective_cost": objective_cost,
        "imputed_route_edges": imputed,
        "expanded_nodes": expanded,
        "max_frontier_size": max_frontier,
        "runtime_ms": runtime_ms,
    }

