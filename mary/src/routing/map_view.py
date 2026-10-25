
from __future__ import annotations

from math import hypot
import networkx as nx
from ipyleaflet import AwesomeIcon, GeoJSON, LayersControl, Map, Marker
from ipywidgets import Checkbox, HTML, Layout, VBox
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
    for u, v, k, attrs in graph.edges(keys=True, data=True):
        if status is not None and attrs.get("traffic_status") != status:
            continue
        line = directed_osm_line(graph, u, v, k)
        features.append({
            "type": "Feature", "properties": {"edge_id": f"{u},{v},{k}"},
            "geometry": {"type": "LineString", "coordinates": [list(p) for p in line.coords]},
        })
    return {"type": "FeatureCollection", "features": features}


def snap_distance_m(graph: nx.MultiDiGraph, lat: float, lon: float, node: int) -> float:
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    px, py = transformer.transform(lon, lat)
    nlat, nlon = node_latlon(graph, node)
    nx_, ny_ = transformer.transform(nlon, nlat)
    return hypot(px - nx_, py - ny_)


def interactive_map(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    traffic_graph: nx.MultiDiGraph | None = None,
    *,
    max_snap_distance_m: float = 350.0,
) -> tuple[VBox, dict]:
    if graph.number_of_nodes() == 0:
        raise ValueError("Empty road network")
    if max_snap_distance_m <= 0:
        raise ValueError("Snap distance must be positive")
    if traffic_graph is not None and (
        graph.number_of_nodes() != traffic_graph.number_of_nodes()
        or set(graph.edges(keys=True)) != set(traffic_graph.edges(keys=True))
    ):
        raise ValueError("Traffic snapshot belongs to a DIFFERENT road graph; collect a matching snapshot")

    start_node = nearest_drive_node(graph, *origin)
    dest_node = nearest_drive_node(graph, *destination)
    start_latlon = node_latlon(graph, start_node)
    dest_latlon = node_latlon(graph, dest_node)
    state = {
        "origin_node": start_node, "destination_node": dest_node,
        "origin_latlon": start_latlon, "destination_latlon": dest_latlon,
        "last_click_latlon": None,
    }

    latitudes = [d["y"] for _, d in graph.nodes(data=True)]
    longitudes = [d["x"] for _, d in graph.nodes(data=True)]
    area_span = max(max(latitudes) - min(latitudes), (max(longitudes) - min(longitudes)) * 0.65)
    zoom = 14 if area_span > 0.025 else 15
    center = ((min(latitudes) + max(latitudes)) / 2, (min(longitudes) + max(longitudes)) / 2)
    map_widget = Map(
        center=center or MAP_CENTER, zoom=zoom, double_click_zoom=False,
        scroll_wheel_zoom=True, layout=Layout(height="560px", width="100%"),
    )

    map_widget.add(GeoJSON(
        data=_road_geojson(graph), name="OSM driving network",
        style={"color": "#3b6585", "weight": 2, "opacity": 0.40},
    ))

    missing_toggle = None
    if traffic_graph is not None:
        count_missing = sum(d.get("traffic_status") == "missing" for *_, d in traffic_graph.edges(keys=True, data=True))
        traffic_status_labels = (
            ("red", "Heavy congestion - red"),
            ("yellow", "Moderate congestion - yellow"),
            ("closed", "Confirmed closure - black dashed"),
        )
        for kind, label in traffic_status_labels:
            if not any(d.get("traffic_status") == kind for *_, d in traffic_graph.edges(keys=True, data=True)):
                continue
            color, weight, dash = LAYER_STYLES[kind]
            style = {"color": color, "weight": weight, "opacity": 0.95}
            if dash:
                style["dashArray"] = dash
            map_widget.add(GeoJSON(data=_road_geojson(traffic_graph, kind), name=label, style=style))

        missing_layer = None
        if count_missing:
            missing_layer = GeoJSON(
                data=_road_geojson(traffic_graph, "missing"),
                name=f"Unknown traffic ({count_missing} edges)",
                style={"color": "#64748b", "weight": 2.2, "opacity": 0.8, "dashArray": "3,5"},
            )
        missing_toggle = Checkbox(
            value=False, description=f"Show unknown traffic ({count_missing} edges)",
            indent=False,
        )

        def set_missing(change):
            if change["name"] != "value" or missing_layer is None:
                return
            if change["new"] and missing_layer not in map_widget.layers:
                map_widget.add(missing_layer)
            elif not change["new"] and missing_layer in map_widget.layers:
                map_widget.remove(missing_layer)

        missing_toggle.observe(set_missing, names="value")

    origin_marker = Marker(
        location=start_latlon, title="Origin (double-click to change)",
        icon=AwesomeIcon(name="play", marker_color="blue"), draggable=False,
    )
    destination_marker = Marker(
        location=dest_latlon, title="Destination (fixed)",
        icon=AwesomeIcon(name="flag", marker_color="green"), draggable=False,
    )
    map_widget.add(origin_marker)
    map_widget.add(destination_marker)
    map_widget.add(LayersControl(position="topright"))

    status = HTML(value=(
        f"<b>Origin node:</b> {start_node} | <b>Destination node:</b> {dest_node}<br>"
        "Double-click within the downloaded driving network to move the blue origin. "
        "The destination is fixed."
    ))

    def change_origin(**event):
        if event.get("type") != "dblclick":
            return
        coords = event.get("coordinates")
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            return
        try:
            lat, lon = map(float, coords)
        except (ValueError, TypeError):
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return
        node = nearest_drive_node(graph, lat, lon)
        distance = snap_distance_m(graph, lat, lon, node)
        if distance > max_snap_distance_m:
            status.value = (
                f"<b>Click rejected:</b> nearest network node {distance:.0f} m away "
                f"(limit {max_snap_distance_m:.0f} m). "
                "Pan within the downloaded network or download a larger radius."
            )
            return
        point = node_latlon(graph, node)
        state.update(
            origin_node=node, origin_latlon=point, last_click_latlon=(lat, lon),
        )
        origin_marker.location = point
        status.value = (
            f"<b>Origin node:</b> {node} | <b>Destination node:</b> {dest_node}<br>"
            f"Origin updated (snapped {distance:.1f} m from click)."
        )

    map_widget.on_interaction(change_origin)
    children = [status]
    if missing_toggle is not None:
        children.append(missing_toggle)
    children.append(map_widget)
    return VBox(children), state
