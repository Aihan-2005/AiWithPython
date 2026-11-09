from __future__ import annotations

from html import escape
from math import hypot

import networkx as nx
from ipyleaflet import AwesomeIcon, GeoJSON, LayersControl, Map, Marker
from ipywidgets import Checkbox, HTML, Layout, VBox
from pyproj import Transformer

from .config import MAP_CENTER
from .map_loader import nearest_drive_node, node_latlon
from .map_matching import directed_osm_line


# Traffic colors are intentionally distinct from the blue OSM base network.
LAYER_STYLES = {
    "red": ("#dc2626", 5, None),
    "yellow": ("#f59e0b", 5, None),
    "clear": ("#16a34a", 4, None),
    "missing": ("#64748b", 2.2, "3,5"),
    "closed": ("#111827", 6, "6,4"),
}

TRAFFIC_STATUS_LABELS = {
    "red": "Heavy congestion — red",
    "yellow": "Moderate congestion — yellow",
    "clear": "Clear / free-flow traffic — green",
    "closed": "Confirmed road closure — black dashed",
}


def _road_geojson(graph: nx.MultiDiGraph, status: str | None = None) -> dict:
    """Build a GeoJSON FeatureCollection for all roads or one traffic status."""
    features = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        actual_status = attrs.get("traffic_status", "missing")
        if actual_status not in LAYER_STYLES:
            actual_status = "missing"
        if status is not None and actual_status != status:
            continue
        line = directed_osm_line(graph, u, v, key)
        features.append(
            {
                "type": "Feature",
                "properties": {"edge_id": f"{u},{v},{key}"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(point) for point in line.coords],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _traffic_counts(graph: nx.MultiDiGraph) -> dict[str, int]:
    counts = {status: 0 for status in LAYER_STYLES}
    for _, _, _, attrs in graph.edges(keys=True, data=True):
        status = attrs.get("traffic_status", "missing")
        if status not in counts:
            status = "missing"
        counts[status] += 1
    return counts


def snap_distance_m(graph: nx.MultiDiGraph, lat: float, lon: float, node: int) -> float:
    transformer = Transformer.from_crs(graph.graph["crs"], "EPSG:27700", always_xy=True)
    px, py = transformer.transform(lon, lat)
    node_lat, node_lon = node_latlon(graph, node)
    node_x, node_y = transformer.transform(node_lon, node_lat)
    return hypot(px - node_x, py - node_y)


def interactive_map(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    traffic_graph: nx.MultiDiGraph | None = None,
    *,
    max_snap_distance_m: float = 350.0,
) -> tuple[VBox, dict]:
    """Display the complete downloaded road graph and available traffic layers.

    Clear roads are shown in green. Unknown roads can be enabled with a checkbox;
    they are not silently treated as clear roads.
    """
    if not isinstance(graph, nx.MultiDiGraph) or not graph.is_directed():
        raise TypeError("graph must be a directed networkx.MultiDiGraph")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Empty road network")
    if max_snap_distance_m <= 0:
        raise ValueError("Snap distance must be positive")

    if traffic_graph is not None:
        if not isinstance(traffic_graph, nx.MultiDiGraph):
            raise TypeError("traffic_graph must be a networkx.MultiDiGraph")
        if (
            set(graph.nodes) != set(traffic_graph.nodes)
            or set(graph.edges(keys=True)) != set(traffic_graph.edges(keys=True))
        ):
            raise ValueError(
                "Traffic snapshot belongs to a DIFFERENT road graph; "
                "load the matching graph or collect a matching snapshot."
            )

    start_node = nearest_drive_node(graph, *origin)
    destination_node = nearest_drive_node(graph, *destination)
    start_latlon = node_latlon(graph, start_node)
    destination_latlon = node_latlon(graph, destination_node)

    latitudes = [float(attrs["y"]) for _, attrs in graph.nodes(data=True)]
    longitudes = [float(attrs["x"]) for _, attrs in graph.nodes(data=True)]
    bounds = (
        (min(latitudes), min(longitudes)),
        (max(latitudes), max(longitudes)),
    )
    center = (
        (bounds[0][0] + bounds[1][0]) / 2.0,
        (bounds[0][1] + bounds[1][1]) / 2.0,
    )

    map_widget = Map(
        center=center or MAP_CENTER,
        zoom=14,
        double_click_zoom=False,
        scroll_wheel_zoom=True,
        layout=Layout(height="650px", width="100%"),
    )
    # Fit the entire downloaded OSM network instead of relying on a fixed zoom.
    map_widget.fit_bounds(bounds)

    map_widget.add(
        GeoJSON(
            data=_road_geojson(graph),
            name=f"OSM driving network ({graph.number_of_edges()} directed edges)",
            style={"color": "#3b6585", "weight": 2, "opacity": 0.40},
        )
    )

    traffic_counts = {status: 0 for status in LAYER_STYLES}
    traffic_coverage_percent = 0.0
    missing_toggle = None
    if traffic_graph is not None:
        traffic_counts = _traffic_counts(traffic_graph)
        total_edges = traffic_graph.number_of_edges()
        observed_edges = sum(
            traffic_counts[status] for status in ("red", "yellow", "clear", "closed")
        )
        traffic_coverage_percent = (
            100.0 * observed_edges / total_edges if total_edges else 0.0
        )

        for status, label in TRAFFIC_STATUS_LABELS.items():
            if traffic_counts[status] == 0:
                continue
            color, weight, dash = LAYER_STYLES[status]
            style = {"color": color, "weight": weight, "opacity": 0.95}
            if dash:
                style["dashArray"] = dash
            map_widget.add(
                GeoJSON(
                    data=_road_geojson(traffic_graph, status),
                    name=f"{label} ({traffic_counts[status]})",
                    style=style,
                )
            )

        if traffic_counts["missing"]:
            missing_style = {
                "color": LAYER_STYLES["missing"][0],
                "weight": LAYER_STYLES["missing"][1],
                "opacity": 0.75,
                "dashArray": LAYER_STYLES["missing"][2],
            }
            missing_layer = GeoJSON(
                data=_road_geojson(traffic_graph, "missing"),
                name=f"Unknown traffic ({traffic_counts['missing']})",
                style=missing_style,
            )
            missing_toggle = Checkbox(
                value=False,
                description=f"Show roads without traffic data ({traffic_counts['missing']})",
                indent=False,
            )

            def set_missing(change):
                if change.get("name") != "value":
                    return
                if change.get("new") and missing_layer not in map_widget.layers:
                    map_widget.add(missing_layer)
                elif not change.get("new") and missing_layer in map_widget.layers:
                    map_widget.remove(missing_layer)

            missing_toggle.observe(set_missing, names="value")

    origin_marker = Marker(
        location=start_latlon,
        title="Origin (double-click to change)",
        icon=AwesomeIcon(name="play", marker_color="blue"),
        draggable=False,
    )
    destination_marker = Marker(
        location=destination_latlon,
        title="Destination (fixed)",
        icon=AwesomeIcon(name="flag", marker_color="green"),
        draggable=False,
    )
    map_widget.add(origin_marker)
    map_widget.add(destination_marker)
    map_widget.add(LayersControl(position="topright"))

    if traffic_graph is None:
        info = HTML(
            value=(
                f"<b>OSM roads:</b> {graph.number_of_edges():,} directed edges. "
                "<b>Traffic:</b> no snapshot supplied."
            )
        )
    else:
        info = HTML(
            value=(
                f"<b>OSM roads:</b> {graph.number_of_edges():,} directed edges. "
                f"<b>Observed traffic:</b> {observed_edges:,} / {total_edges:,} "
                f"({traffic_coverage_percent:.2f}%). "
                f"<span style='color:#16a34a'>Clear {traffic_counts['clear']:,}</span> · "
                f"<span style='color:#f59e0b'>Moderate {traffic_counts['yellow']:,}</span> · "
                f"<span style='color:#dc2626'>Heavy {traffic_counts['red']:,}</span> · "
                f"<span style='color:#111827'>Closed {traffic_counts['closed']:,}</span> · "
                f"Unknown {traffic_counts['missing']:,}."
            )
        )

    interaction_status = HTML(
        value=(
            f"<b>Origin node:</b> {escape(str(start_node))} | "
            f"<b>Destination node:</b> {escape(str(destination_node))}<br>"
            "Double-click inside the downloaded road network to move the blue origin. "
            "The destination stays fixed. Run the routing cell again to calculate a new route."
        )
    )

    state = {
        "origin_node": start_node,
        "destination_node": destination_node,
        "origin_latlon": start_latlon,
        "destination_latlon": destination_latlon,
        "last_click_latlon": None,
        "traffic_counts": traffic_counts,
        "traffic_coverage_percent": round(traffic_coverage_percent, 3),
        "map_bounds": bounds,
    }

    def change_origin(**event):
        if event.get("type") != "dblclick":
            return
        coordinates = event.get("coordinates")
        if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
            return
        try:
            lat, lon = map(float, coordinates)
        except (ValueError, TypeError):
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return

        node = nearest_drive_node(graph, lat, lon)
        distance = snap_distance_m(graph, lat, lon, node)
        if distance > max_snap_distance_m:
            interaction_status.value = (
                f"<b>Click rejected:</b> nearest road node is {distance:.0f} m away "
                f"(limit {max_snap_distance_m:.0f} m). Zoom or pan inside the downloaded network."
            )
            return

        point = node_latlon(graph, node)
        state.update(
            origin_node=node,
            origin_latlon=point,
            last_click_latlon=(lat, lon),
        )
        origin_marker.location = point
        interaction_status.value = (
            f"<b>Origin node:</b> {escape(str(node))} | "
            f"<b>Destination node:</b> {escape(str(destination_node))}<br>"
            f"Origin updated; snapped {distance:.1f} m from your click. "
            "Run the routing cell again to recalculate the route."
        )

    map_widget.on_interaction(change_origin)

    children = [info, interaction_status]
    if missing_toggle is not None:
        children.append(missing_toggle)
    children.append(map_widget)
    return VBox(children), state
