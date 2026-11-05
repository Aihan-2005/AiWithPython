

from __future__ import annotations

from collections import deque
from time import perf_counter
import networkx as nx
from ._common import validate_endpoints, successors, assemble_result

def bfs(graph:nx.MultiDiGraph,source,goal) ->dict:
    started =perf_counter()
    weight = validate_endpoints(graph,source,goal)
    frontier = deque([source])
    seen = {source}
    parent = {}
    expanded = 0
    max_frontier = 1
    found = False


    while frontier:
        node = frontier.popleft()
        expanded += 1

        if node == goal:
            found = True
            break

        for next_node ,key,_ in successors(graph,node,weight):
            if next_node not in seen:
                seen.add(next_node)
                parent[next_node] = (node,next_node,key)
                frontier.append(next_node)

        max_frontier = max(max_frontier,len(frontier))
    return assemble_result(
        graph,algorithm='bfs',source=source,goal=goal,
        parent=parent,found=found,expanded=expanded,
        max_frontier=max_frontier,started_at= started,weight=weight
    )

