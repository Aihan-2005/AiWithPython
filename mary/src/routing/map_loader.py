"""OSM road data and snapping locations onto the drivable graph."""
from __future__ import annotations
import networkx as nx
import osmnx as ox
from pyproj import Transformer
from .config import GRAPH_PATH, MAP_CENTER, MAP_RADIUS_M


def load_road_graph(force_download: bool = False) -> nx.MultiDiGraph:
    """Get (and cache on disk) a directed drivable OSM network in WGS84."""
    ox.settings.use_cache = True  # Overpass response cache; no TomTom traffic cached here.
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    if GRAPH_PATH.exists() and not force_download:
        graph = ox.io.load_graphml(GRAPH_PATH)
    else:
        graph = ox.graph.graph_from_point(
            MAP_CENTER,
            dist=MAP_RADIUS_M,
            network_type="drive",
            simplify=True,
            retain_all=False,
        )
        ox.io.save_graphml(graph, GRAPH_PATH)
    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        raise TypeError("Expected a directed NetworkX MultiDiGraph.")
    if not all("length" in attrs for *_, attrs in graph.edges(keys=True, data=True)):
        raise ValueError("OSM road length must be available in meters.")
    return graph


def nearest_drive_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Snap to the nearest road-network node (not necessarily an accessible gate).

    For a small area a linear nearest-node scan is simpler than optional KD-tree
    dependencies. Transform WGS84 into British National Grid meters first.
    """
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    x, y = transformer.transform(lon, lat)
    nodes = graph.nodes
    best_node, best_dist_sq = None, float("inf")
    for node, attrs in nodes(data=True):
        node_x, node_y = transformer.transform(attrs["x"], attrs["y"])
        dist_sq = (node_x - x) ** 2 + (node_y - y) ** 2
        if dist_sq < best_dist_sq:
            best_node, best_dist_sq = node, dist_sq
    if best_node is None:
        raise ValueError("The OSM driving network is empty")
    return best_node


def node_latlon(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    return float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"])
