

from __future__ import annotations

import heapq
from itertools import count
from time import perf_counter
import networkx as nx
from ._common import validate_endpoints, successors, assemble_result
from .heuristics import build_admissible_heuristic



def greedy_best_first(graph:nx.MultiDiGraph,source,goal) -> dict:

    started = perf_counter()
    weight = validate_endpoints(graph,source,goal)
    h = build_admissible_heuristic(graph,goal,weight)
    serial = count()
    frontier = [(h(source),next(serial),source)]
    discovered = {source}
    expanded_set = set()
    parent = {}
    expanded = 0
    max_frontier = 1
    found= False

    while frontier:
        _,_,node = heapq.heappop(frontier)
        if node in  expanded_set:
            continue
        expanded_set.add(node)
        expanded +=1 

        if node == goal:
            found = True
            break

        for nxt,key,_ in  successors(graph, node,weight):
            if nxt not in discovered:
                discovered.add(nxt)
                parent[nxt] = (node,nxt,key)
                heapq.heappush(frontier, (h(nxt), next(serial), nxt))

        max_frontier = max(max_frontier,len(frontier))

    return assemble_result(
        graph, algorithm="Greedy", source=source, goal=goal,
        parent=parent, found=found, expanded=expanded,
        max_frontier=max_frontier, started_at=started, weight=weight,
    )