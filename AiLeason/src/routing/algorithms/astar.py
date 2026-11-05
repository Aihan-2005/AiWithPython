

from __future__ import annotations

import heapq
from itertools import count
from math import inf
from time import perf_counter
import networkx as nx
from ._common import validate_endpoints, successors, assemble_result
from .heuristics import build_admissible_heuristic



def astar(graph:nx.MultiDiGraph,source,goal)-> dict:
    started = perf_counter()
    weight = validate_endpoints(graph,source,goal)
    h = build_admissible_heuristic(graph,source,weight)
    serial = count()

    frontier = [(h(source), 0.0, next(serial), source)]
    best_g = {source:0.0}
    closed = set()
    parent = {}
    expanded = 0
    max_frontier = 1
    found = False


    while frontier:
        _,g,_,node = heapq.heappop(frontier)
        if node in closed or g > best_g.get(node,inf):
            continue
        closed.add(node)
        expanded +=1

        if node == goal:
            found = True
            break

        for nxt,key,cost in successors(graph,node,weight):

            if nxt in  closed:
                continue
            