import networkx as nx
from shapely.geometry import LineString
from routing.map_matching import directed_osm_line, match_osm_edge


def road():
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    g.add_node(1, x=-0.1750, y=51.4990)
    g.add_node(2, x=-0.1740, y=51.4990)
    g.add_edge(1, 2, key=4, length=70.0,
               geometry=LineString([(-0.1740, 51.4990), (-0.1750, 51.4990)]))
    return g


def payload(reverse=False, latitude=51.4990):
    coords = [
        {"latitude": latitude, "longitude": -0.1750},
        {"latitude": latitude, "longitude": -0.1740},
    ]
    return {"coordinates": {"coordinate": list(reversed(coords)) if reverse else coords}}


def test_stored_geometry_orientation_is_corrected():
    g = road()
    line = directed_osm_line(g, 1, 2, 4)
    assert line.coords[0][0] == -0.1750
    assert match_osm_edge(g, (1, 2, 4), payload())


def test_wrong_way_fails():
    assert not match_osm_edge(road(), (1, 2, 4), payload(reverse=True))


def test_far_segment_fails():
    assert not match_osm_edge(road(), (1, 2, 4), payload(latitude=51.5010))
