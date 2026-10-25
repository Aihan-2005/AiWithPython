 
import math
import networkx as nx

from routing.map_matching import match_osm_edge
from routing.traffic_api import TomTomAPIError
from routing.traffic_pipeline import (
    collect_traffic_snapshot,
    make_routing_graph,
    preview_probes,
)


def road():
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-0.18, y=51.50)
    g.add_node(2, x=-0.179, y=51.50)
    g.add_edge(1, 2, key=4, length=70.0)
    return g


def payload(*, current=15, free=40, conf=0.95, closed=False, reverse=False):
    line = [
        {"latitude": 51.50, "longitude": -0.18},
        {"latitude": 51.50, "longitude": -0.179},
    ]
    if reverse:
        line.reverse()
    return {
        "currentSpeed": current,
        "freeFlowSpeed": free,
        "confidence": conf,
        "roadClosure": closed,
        "coordinates": {"coordinate": line},
    }


class FakeClient:
    def __init__(self, item):
        self.item = item
        self.calls = []

    def fetch_flow(self, lat, lon):
        self.calls.append((lat, lon))
        return self.item


def test_candidate_matching_accepts_geometric_order():
    g = road()
    assert match_osm_edge(g, (1, 2, 4), payload())
    assert not match_osm_edge(g, (1, 2, 4), payload(reverse=True))


def test_offline_probe_is_deterministic_and_no_client_needed():
    result = preview_probes(road(), 2)
    assert len(result) == 1
    assert result[0]["edge_id"] == (1, 2, 4)
    assert result[0]["length_m"] == 70.0


def test_observed_red_cost_and_coverage():
    client = FakeClient(payload())
    g, summary = collect_traffic_snapshot(road(), client, max_requests=1, pause_s=0)
    e = g[1][2][4]
    assert len(client.calls) == 1
    assert e["traffic_status"] == "red"
    assert math.isclose(e["travel_time_s"], 3.6 * 70 / 15)
    assert math.isclose(e["free_flow_time_s"], 3.6 * 70 / 40)
    assert e["time_is_imputed"] is False
    assert e["free_flow_is_imputed"] is False
    assert summary["usable_coverage_percent"] == 100.0


def test_yellow_and_clear_thresholds():
    for speed, status in ((20, "yellow"), (31, "yellow"), (32, "clear")):
        g, _ = collect_traffic_snapshot(road(), FakeClient(payload(current=speed)), max_requests=1, pause_s=0)
        assert g[1][2][4]["traffic_status"] == status



def test_no_queries_never_means_clear():
    client = FakeClient(payload())
    g, meta = collect_traffic_snapshot(road(), client, max_requests=0, pause_s=0)
    edge = g[1][2][4]
    assert not client.calls
    assert edge["traffic_status"] == "missing"
    assert edge["current_speed_kph"] is None
    assert edge["free_flow_speed_kph"] is None
    assert edge["time_is_imputed"] and edge["free_flow_is_imputed"]
    assert math.isclose(edge["travel_time_s"], 3.6 * 70 / 30)
    assert meta["sampled_directed_edges"] == 0


def test_zero_speed_without_closure_is_missing():
    g, _ = collect_traffic_snapshot(road(), FakeClient(payload(current=0)), max_requests=1, pause_s=0)
    edge = g[1][2][4]
    assert edge["traffic_status"] == "missing"
    assert edge["traffic_reason"] == "invalid_or_zero_speed"
    assert edge["time_is_imputed"] is True
    assert edge["free_flow_is_imputed"] is False


def test_low_confidence_is_missing():
    g, meta = collect_traffic_snapshot(road(), FakeClient(payload(conf=0.3)), max_requests=1, pause_s=0)
    edge = g[1][2][4]
    assert edge["traffic_matched"]
    assert edge["traffic_status"] == "missing"
    assert edge["traffic_reason"] == "low_confidence"
    assert meta["usable_observed_edges"] == 0


def test_explicit_closure_is_removed_only_in_measured_and_closure_scenarios():
    g, _ = collect_traffic_snapshot(road(), FakeClient(payload(current=0, closed=True)), max_requests=1, pause_s=0)
    assert g[1][2][4]["traffic_status"] == "closed"
    assert g[1][2][4]["travel_time_s"] is None
    assert make_routing_graph(g, scenario="free_flow").has_edge(1, 2, 4)
    assert not make_routing_graph(g, scenario="measured").has_edge(1, 2, 4)


def test_extra_closure_removes_directed_edge_only():
    g, _ = collect_traffic_snapshot(road(), FakeClient(payload()), max_requests=1, pause_s=0)
    g.add_edge(2, 1, key=5, length=70, travel_time_s=20, free_flow_time_s=15)
    result = make_routing_graph(g, scenario="closure", imposed_closure=(1, 2, 4))
    assert not result.has_edge(1, 2, 4)
    assert result.has_edge(2, 1, 5)
    assert result.graph["routing_weight"] == "travel_time_s"
    assert make_routing_graph(g, objective="distance").graph["routing_weight"] == "length"


def test_geometry_mismatch_is_not_used():
    g, _ = collect_traffic_snapshot(road(), FakeClient(payload(reverse=True)), max_requests=1, pause_s=0)
    e = g[1][2][4]
    assert e["traffic_status"] == "missing"
    assert e["traffic_reason"] == "geometry_or_heading_mismatch"


def test_stop_when_rate_limited():
    class LimitClient:
        def fetch_flow(self, lat, lon):
            raise TomTomAPIError(429)

    g, summary = collect_traffic_snapshot(road(), LimitClient(), max_requests=1, pause_s=0)
    assert summary["stopped_http_status"] == 429
    assert summary["requests_made"] == 1
    assert g[1][2][4]["traffic_status"] == "missing"
