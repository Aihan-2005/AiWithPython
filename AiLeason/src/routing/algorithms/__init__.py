
from __future__ import annotations

from .bfs import bfs
from .dfs import dfs
from .greedy import greedy_best_first
from .astar import astar

ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "Greedy": greedy_best_first,
    "A*": astar,
}


def run_all_algorithms(graph, source, goal) -> dict[str, dict]:

    return {name: method(graph, source, goal) for name, method in ALGORITHMS.items()}


__all__ = ["bfs", "dfs", "greedy_best_first", "astar", "run_all_algorithms", "ALGORITHMS"]
