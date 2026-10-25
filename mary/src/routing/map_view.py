
from __future__ import annotations

from math import hypot

import networkx as nx
from ipyleaflet import AwesomeIcon, GeoJSON, LayersControl, Map, Marker
from ipywidgets import HTML, Layout, VBox
from pyproj import Transformer

from .config import MAP_CENTER
from .map_loader import nearest_drive_node, node_latlon
from .map_matching import directed_osm_line


LAYER_STYLES = {
    "red": ("#d62828", 5, None),
    "yellow": ("#f0b429", 5, None),
    "missing": ("#64748b", 2.5, "3,5"),
    "closed": ("#111827", 6, "6,4"),
}


def _road_geojson(graph: nx.MultiDiGraph, status: str | None = None) -> dict:
    features = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        if status is not None and attrs.get("traffic_status") != status:
            continue
        line = directed_osm_line(graph, u, v, key)
        features.append({
            "type": "Feature",
            "properties": {"edge_id": f"{u},{v},{key}"},
            "geometry": {
                "type": "LineString",
                "coordinates": [list(xy) for xy in line.coords],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def snap_distance_m(
    graph: nx.MultiDiGraph,
    lat: float,
    lon: float,
    node: int,
) -> float:
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    point_x, point_y = transformer.transform(lon, lat)
    node_lat, node_lon = node_latlon(graph, node)
    node_x, node_y = transformer.transform(node_lon, node_lat)
    return hypot(point_x - node_x, point_y - node_y)


def interactive_map(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    traffic_graph: nx.MultiDiGraph | None = None,
    *,
    max_snap_distance_m: float = 350.0,
) -> tuple[VBox, dict]:

    if graph.number_of_nodes() == 0:
        raise ValueError("The road network is empty")
    if max_snap_distance_m <= 0:
        raise ValueError("max_snap_distance_m must be positive")

    start_node = nearest_drive_node(graph, *origin)
    dest_node = nearest_drive_node(graph, *destination)
    start_snapped = node_latlon(graph, start_node)
    dest_snapped = node_latlon(graph, dest_node)

    state = {
        "origin_node": start_node,
        "destination_node": dest_node,
        "origin_latlon": start_snapped,
        "destination_latlon": dest_snapped,
        "last_click_latlon": None,
    }

    map_widget = Map(
        center=MAP_CENTER,
        zoom=15,
        scroll_wheel_zoom=True,
        double_click_zoom=False,
        layout=Layout(height="560px", width="100%"),
    )

    map_widget.add(GeoJSON(
        data=_road_geojson(graph),
        name="OSM driving network",
        style={"color": "#64748b", "weight": 2, "opacity": 0.35},
    ))

    if traffic_graph is not None:
        for kind, label in (
            ("missing", "Unknown traffic (grey dashed)"),
            ("red", "Heavy congestion (red)"),
            ("yellow", "Moderate congestion (yellow)"),
            ("closed", "Confirmed road closure (black dashed)"),
        ):
            if not any(attrs.get("traffic_status") == kind
                       for _, _, _, attrs in traffic_graph.edges(keys=True, data=True)):
                continue
            color, weight, dash = LAYER_STYLES[kind]
            style = {"color": color, "weight": weight, "opacity": 0.9}
            if dash is not None:
                style["dashArray"] = dash
            map_widget.add(GeoJSON(
                data=_road_geojson(traffic_graph, status=kind),
                name=label,
                style=style,
            ))

    start_marker = Marker(
        location=start_snapped,
        title="Origin (double-click to change)",
        icon=AwesomeIcon(name="play", marker_color="blue"),
        draggable=False,
    )
    dest_marker = Marker(
        location=dest_snapped,
        title="Destination (fixed)",
        icon=AwesomeIcon(name="flag", marker_color="green"),
        draggable=False,
    )
    map_widget.add(start_marker)
    map_widget.add(dest_marker)
    map_widget.add(LayersControl(position="topright"))

    status = HTML(value=(
        f"<b>Origin node:</b> {start_node} | "
        f"<b>Destination node:</b> {dest_node}<br>"
        "Double-click near a road to move the blue marker. "
        "The green destination stays fixed."
    ))

    def change_origin(**event):
        if event.get("type") != "dblclick":
            return
        coordinates = event.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            return
        try:
            lat, lon = map(float, coordinates)
        except (TypeError, ValueError):
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return

        node = nearest_drive_node(graph, lat, lon)
        distance_m = snap_distance_m(graph, lat, lon, node)
        if distance_m > max_snap_distance_m:
            status.value = (
                f"<b>Ignored click:</b> nearest road node is {distance_m:.0f} m away "
                f"(limit: {max_snap_distance_m:.0f} m). Click closer to the network."
            )
            return

        snapped = node_latlon(graph, node)
        state["origin_node"] = node
        state["origin_latlon"] = snapped
        state["last_click_latlon"] = (lat, lon)
        start_marker.location = snapped
        status.value = (
            f"<b>Origin node:</b> {node} | <b>Destination node:</b> {dest_node}<br>"
            f"Origin updated (snapped {distance_m:.1f} m from clicked point). "
            "Ready for later routing algorithms."
        )

    map_widget.on_interaction(change_origin)
    return VBox([status, map_widget]), state
