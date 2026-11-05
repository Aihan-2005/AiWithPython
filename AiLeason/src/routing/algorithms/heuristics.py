

from __future__ import annotations


from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import Callable


import networkx as nx
from ._common import valid_cost

_EARTH_RADIUS_M = 6_371_009.0


def _haversine_m(a:tuple[float,float], b:tuple[float,float])-> float:
    lat1, lon1 = map(radians,a)
    lat2,lon2 = map(radians,b)
    dlat,dlon = lat2-lat1,lon2-lon1
    q = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2*_EARTH_RADIUS_M*asin(min(1.0,sqrt(q)))

def build_admissible_heuristic(
        graph:nx.MultiDiGraph,goal,weight:str
) -> Callable[[object],float]:

    coords: dict[object,tuple[float,float]] = {}

    for node,attrs in graph.nodes(data=True):
        try:
            coords[node] = float(attrs['y']),float(attrs['x'])
        except (KeyError,TypeError,ValueError):
            return lambda node: 0.0

    if goal not in coords:
        return lambda node : 0.0


    rate  = float('inf')

    for u, v, k, attr in graph.edges(keys=True, data=True):
        cost = valid_cost(attr.get(weight))

        if cost is None:
            continue

        direct = _haversine_m(coords[u],coords[v])
        if direct > 0:
            rate = min(rate,cost/direct)

    if rate == float('inf'):
        rate = 0.0

    rate = max(0.0,rate*(1.0-1e-12))


    @lru_cache(maxsize=None)
    def estimate(node) -> float:
        return _haversine_m(coords[node], coords[goal]) * rate

    return estimate
 