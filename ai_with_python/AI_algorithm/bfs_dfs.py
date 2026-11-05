from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Generic, Hashable, Iterable, List, Optional, Set, Tuple, TypeVar

V = TypeVar("V", bound=Hashable)


class Graph(Generic[V]):


    def __init__(self, directed: bool = False) -> None:
        self._adj : Dict[V,List[V]] = {}

    def add_vertex(self,v:V) -> None:
        self._adj.setdefault(v,[])

    def add_edge(self,u:V,v:V)->None:
        self.add_vertex(u)
        self.add_vertex(v)


        if v not in self._adj[u]:
            self._adj[u].append(v)

        if not self.directed and u not in self._adj[v]:
            self._adj[v].append(u)


    def vertices(self) -> List[V]:
        return list(self._adj.keys())

    def neighbors(self,v:V) -> List[V]:
        if v not in self._adj:
            return KeyError(f'vertex {V!r} does not exist in the graph')
        return self._adj[v]


    def edges(self) -> List[Tuple[V,V]]:
        result : List[Tuple[V,V]] = []
        seen : set[frozenset[V]] = set()

        for u  in self._adj:
            for v in  self._adj:

                if self.directed:
                    result.append((u,v))

                else:
                    key = frozenset
                