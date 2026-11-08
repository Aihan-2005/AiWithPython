from __future__ import annotations

import networkx as nx
from shapely.geometry import LineString

from routing.map_matching import assess_osm_edge
from routing.traffic_api import TomTomAPIError
from routing.traffic_pipeline import collect_traffic_snapshot


U = 1
V = 2
EDGE = (U, V, 0)
REVERSE = (V, U, 0)


def _graph(*, reverse=False, length=120.0):
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(U, x=-0.1800, y=51.5000)
    graph.add_node(V, x=-0.1780, y=51.5000)
    graph.add_edge(
        U,
        V,
        key=0,
        length=length,
        highway="primary",
        geometry=LineString([(-0.1800, 51.5000), (-0.1780, 51.5000)]),
    )
    if reverse:
        graph.add_edge(
            V,
            U,
            key=0,
            length=length,
            highway="primary",
            geometry=LineString([(-0.1780, 51.5000), (-0.1800, 51.5000)]),
        )
    return graph


def _payload(*, reverse=False):
    points = [
        {"longitude": -0.1800, "latitude": 51.5000},
        {"longitude": -0.1780, "latitude": 51.5000},
    ]
    if reverse:
        points.reverse()
    return {
        "frc": "FRC3",
        "currentSpeed": 18.0,
        "freeFlowSpeed": 30.0,
        "confidence": 1.0,
        "roadClosure": False,
        "coordinates": {"coordinate": points},
    }


class _Client:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def fetch_flow(self, *, lat, lon):
        self.calls.append((lat, lon))
        if self.error:
            raise self.error
        return self.payload


def test_map_match_uses_actual_probe_fraction_and_reports_metrics():
    graph = _graph()
    result = assess_osm_edge(graph, EDGE, _payload(), probe_fraction=0.25)
    assert result.accepted is True
    assert result.reason == "candidate_match"
    assert result.midpoint_distance_m is not None
    assert result.fraction_near >= 0.8
    assert result.direction_cosine > 0.8
    assert result.probe_fraction == 0.25


def test_collector_accepts_direct_flow_and_persists_diagnostics():
    graph = _graph()
    client = _Client(_payload())
    snapshot, summary = collect_traffic_snapshot(
        graph,
        client,
        max_requests=1,
        selected_edge_ids=[EDGE],
        probe_fractions={EDGE: 0.25},
        pause_s=0,
    )
    attrs = snapshot[U][V][0]
    assert summary["requests_made"] == 1
    assert summary["usable_observed_edges"] == 1
    assert attrs["traffic_status"] == "yellow"
    assert attrs["time_is_imputed"] is False
    assert attrs["traffic_probe_fraction"] == 0.25
    assert attrs["traffic_match_fraction_near"] >= 0.8
    assert len(client.calls) == 1


def test_collector_can_reassign_reversed_tomtom_geometry_to_reverse_edge():
    graph = _graph(reverse=True)
    client = _Client(_payload(reverse=True))
    snapshot, summary = collect_traffic_snapshot(
        graph,
        client,
        max_requests=1,
        selected_edge_ids=[EDGE],
        pause_s=0,
    )
    direct = snapshot[U][V][0]
    reverse = snapshot[V][U][0]
    assert summary["requests_made"] == 1
    assert summary["reverse_direction_reassignments"] == 1
    assert direct["traffic_status"] == "missing"
    assert reverse["traffic_status"] == "yellow"
    assert reverse["traffic_match_source"] == "reverse_inferred"


def test_short_edges_are_rejected_before_api_call():
    graph = _graph(length=9.76)
    client = _Client(_payload())
    snapshot, summary = collect_traffic_snapshot(
        graph,
        client,
        max_requests=1,
        selected_edge_ids=[EDGE],
        pause_s=0,
    )
    assert summary["requests_made"] == 0
    assert summary["rejected_short_edges"][0]["reason"] == "below_min_probe_edge_length"
    assert not client.calls
    assert snapshot[U][V][0]["traffic_status"] == "missing"


def test_failed_refresh_does_not_erase_previous_valid_observation():
    graph = _graph()
    attrs = graph[U][V][0]
    attrs.update(
        traffic_status="clear",
        traffic_reason="observed",
        traffic_matched=True,
        traffic_match_source="direct_probe",
        current_speed_kph=30.0,
        free_flow_speed_kph=30.0,
        travel_time_s=14.4,
        time_is_imputed=False,
        is_closed=False,
    )
    client = _Client(error=TomTomAPIError())
    snapshot, summary = collect_traffic_snapshot(
        graph,
        client,
        max_requests=1,
        selected_edge_ids=[EDGE],
        pause_s=0,
    )
    refreshed = snapshot[U][V][0]
    assert summary["request_errors"] == 1
    assert refreshed["traffic_status"] == "clear"
    assert refreshed["current_speed_kph"] == 30.0
    assert refreshed["time_is_imputed"] is False
    assert refreshed["traffic_last_attempt_reason"] == "api_network_error"
