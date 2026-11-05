from __future__ import annotations

from time import perf_counter
import networkx as nx

from ._common import (
    validate_endpoints,
    successors,
    assemble_result,
)


def dfs(graph: nx.MultiDiGraph, source, goal) -> dict:

    started = perf_counter()

    weight = validate_endpoints(
        graph, source, goal
    )

    frontier = [source]
    seen = {source}
    parent = {}

    expanded = 0
    max_frontier = 1
    found = False

    while frontier:

        node = frontier.pop()
        expanded += 1

        if node == goal:
            found = True
            break

        # Reverse order because Stack is LIFO
        for nxt, key, _ in reversed(
            successors(graph, node, weight)
        ):
            if nxt not in seen:

                seen.add(nxt)

                parent[nxt] = (
                    node,
                    nxt,
                    key
                )

                frontier.append(nxt)

        max_frontier = max(
            max_frontier,
            len(frontier)
        )

    return assemble_result(
        graph,
        algorithm="DFS",
        source=source,
        goal=goal,
        parent=parent,
        found=found,
        expanded=expanded,
        max_frontier=max_frontier,
        started_at=started,
        weight=weight,
    )
