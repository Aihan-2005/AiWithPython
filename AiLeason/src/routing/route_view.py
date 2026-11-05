
from __future__ import annotations

import networkx as nx
from ipyleaflet import GeoJSON, Map
from .map_matching import directed_osm_line

ROUTE_COLORS = {
    "BFS": "#2563eb",
    "DFS": "#9333ea",
    "Greedy": "#e87913",
    "A*": "#16a34a",
}


def add_route_overlay(widget, graph: nx.MultiDiGraph, result: dict, *, name: str | None = None,
                      color: str | None = None) -> GeoJSON | None:


    if not result.get("found"):
        return None
    map_widget = widget if isinstance(widget, Map) else widget.children[-1]
    if not isinstance(map_widget, Map):
        raise TypeError("Expected interactive_map's VBox or an ipyleaflet.Map")
    name = name or f"Route {result['algorithm']}"
    color = color or ROUTE_COLORS.get(result["algorithm"], "#2563eb")
    features = []
    for u, v, key in result["path_edges"]:
        if not graph.has_edge(u, v, key):
            raise ValueError(f"Route edge absent from given OSM graph: {(u, v, key)!r}")
        line = directed_osm_line(graph, u, v, key)
        features.append({
            "type": "Feature",
            "properties": {"edge_id": f"{u},{v},{key}"},
            "geometry": {
                "type": "LineString",
                "coordinates": [list(pt) for pt in line.coords],
            },
        })
    for existing in list(map_widget.layers):
        if getattr(existing, "name", None) == name:
            map_widget.remove(existing)
    layer = GeoJSON(
        data={"type": "FeatureCollection", "features": features},
        name=name,
        style={"color": color, "weight": 6, "opacity": 0.85},
    )
    map_widget.add(layer)
    return layer
