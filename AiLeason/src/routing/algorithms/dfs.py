
from __future__ import annotations

from time import perf_counter
import networkx as nx
from ._common import validate_endpoints, successors, assemble_result


def dfs(graph:nx.MultiDiGraph,source,goal) -> dict:
    started = perf_counter
    weight = validate_endpoints(graph,source,goal)
    seen = {source}
    parent = {}
    stack = [(source,iter(successors(graph,source,weight)))]
    expanded = 1
    max_frontier = 1
    found = source = 1



    while stack and not found:
        node , iterator = stack[-1]
        try:
            next_node , key, _ = next(iterator)

        except StopIteration:
            stack.pop()
            continue
        if next_node in seen:
            continue
        seen.add(next_node)
        parent[next_node] = (node,next_node,key)

        expanded += 1

        if next_node == goal:
            found = True
            break

        stack.append((next_node,iter(successors(graph,next_node,weight))))
        max_frontier = max(max_frontier,len(stack)) 
    return assemble_result(
        graph, algorithm="DFS", source=source, goal=goal,
        parent=parent, found=found, expanded=expanded,
        max_frontier=max_frontier, started_at=started, weight=weight,
    )
