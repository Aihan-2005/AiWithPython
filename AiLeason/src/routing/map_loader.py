
from __future__ import annotations

import networkx as nx
import osmnx as ox
from pyproj import Transformer

from .config import GRAPH_PATH, MAP_CENTER, MAP_RADIUS_M, RAW_DIR


def load_road_graph(force_download: bool = False, *, radius_m: int | None = None) -> nx.MultiDiGraph:

  
    radius = MAP_RADIUS_M if radius_m is None else radius_m
    if not isinstance(radius, int) or radius < 200:
        raise ValueError("radius_m must be an integer >= 200")
    path = GRAPH_PATH if radius == MAP_RADIUS_M else RAW_DIR / f"south_kensington_drive_{radius}m.graphml"
    ox.settings.use_cache = True
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_download:
        graph = ox.io.load_graphml(path)
    else:
        graph = ox.graph.graph_from_point(
            MAP_CENTER, dist=radius, network_type="drive", simplify=True, retain_all=False
        )
        graph.graph["routing_source_radius_m"] = radius
        ox.io.save_graphml(graph, path)
    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        raise TypeError("Expected directed MultiDiGraph")
    if not all("length" in attrs for *_, attrs in graph.edges(keys=True, data=True)):
        raise ValueError("Missing road length in meters")
    return graph


def nearest_drive_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    x, y = transformer.transform(lon, lat)
    nearest, best = None, float("inf")
    for node, attrs in graph.nodes(data=True):
        px, py = transformer.transform(attrs["x"], attrs["y"])
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best:
            nearest, best = node, d2
    if nearest is None:
        raise ValueError("Empty road network")
    return nearest


def node_latlon(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    return float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])
