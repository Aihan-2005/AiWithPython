import networkx as nx
from shapely.geometry import LineString
from routing.map_matching import assess_osm_edge, match_osm_edge
from routing.traffic_pipeline import preview_probes, collect_traffic_snapshot, make_routing_graph
from routing.traffic_api import TomTomClient, TomTomAPIError


def road():
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-0.1800, y=51.5000)
    g.add_node(2, x=-0.1790, y=51.5000)
    g.add_edge(1, 2, key=4, length=70.0, geometry=LineString([(-.179, 51.5), (-.180, 51.5)]))
    return g


def payload(reverse=False):
    points = [
        {"latitude": 51.5, "longitude": -.180},
        {"latitude": 51.5, "longitude": -.179},
    ]
    return {
        "coordinates": {"coordinate": points[::-1] if reverse else points},
        "currentSpeed": 15, "freeFlowSpeed": 40, "confidence": .95, "roadClosure": False,
    }


def test_match_and_reject_reverse():
    g = road()
    assert match_osm_edge(g, (1, 2, 4), payload())
    assert assess_osm_edge(g, (1, 2, 4), payload(reverse=True)).reason == "heading_proxy_mismatch"


def test_spatial_probe_is_stable():
    assert preview_probes(road(), 5) == preview_probes(road(), 5)
    assert len(preview_probes(road(), 5)) == 1


def test_traffic_snapshot_and_scenarios():
    class Fake:
        def fetch_flow(self, lat, lon): return payload()
    g, summary = collect_traffic_snapshot(road(), Fake(), max_requests=1, pause_s=0)
    assert summary["usable_observed_edges"] == 1
    assert g[1][2][4]["travel_time_s"] == 3.6 * 70 / 15
    assert make_routing_graph(g, objective="time").graph["routing_weight"] == "travel_time_s"
    assert make_routing_graph(g, objective="distance").graph["routing_weight"] == "length"


def test_400_grouped_no_key_leaked():
    class Fail:
        def fetch_flow(self, lat, lon):
            raise TomTomAPIError(400, error_code="INVALID_REQUEST", detail="Bad point")
    _, summary = collect_traffic_snapshot(road(), Fail(), max_requests=1, pause_s=0)
    assert summary["request_errors"] == 1
    assert summary["api_error_groups"][0]["error_code"] == "INVALID_REQUEST"


def test_http_client_scrubs_credential():
    class FakeResponse:
        ok = False
        status_code = 400
        def json(self):
            return {"detailedError": {"code": "INVALID_REQUEST", "message": "Bad key=SECRET123"}}
    class FakeSession:
        def get(self, url, params, timeout): return FakeResponse()
        def close(self): pass
    c = TomTomClient(api_key="SECRET123")
    c.session = FakeSession()
    try:
        c.fetch_flow(51.5, -.18)
        assert False
    except TomTomAPIError as exc:
        assert "SECRET123" not in str(exc)
        assert "SECRET123" not in str(exc.detail)
        assert exc.error_code == "INVALID_REQUEST"
