from routing.traffic_pipeline import collect_traffic_snapshot, make_routing_graph
from test_map_matching import road, payload


class Client:
    def __init__(self, **fields):
        self.fields = fields
        self.calls = []

    def fetch_flow(self, lat, lon):
        self.calls.append((lat, lon))
        return dict(
            payload(), currentSpeed=15, freeFlowSpeed=40,
            confidence=0.95, roadClosure=False, **self.fields
        )


def test_red_congestion_and_weight():
    g, summary = collect_traffic_snapshot(road(), Client(), max_requests=1, pause_s=0)
    d = g[1][2][4]
    assert d["traffic_status"] == "red"
    assert d["travel_time_s"] == 3.6 * 70 / 15
    assert not d["time_is_imputed"]
    assert summary["usable_observed_edges"] == 1
    assert summary["matched_coverage_percent"] == 100
    assert make_routing_graph(g, "time").has_edge(1, 2, 4)


def test_unqueried_not_treated_as_clear():
    g, summary = collect_traffic_snapshot(road(), Client(), max_requests=0, pause_s=0)
    assert g[1][2][4]["traffic_status"] == "missing"
    assert g[1][2][4]["time_is_imputed"]
    assert summary["usable_observed_edges"] == 0


def test_closure_excluded_from_routing():
    class ClosedClient(Client):
        def fetch_flow(self, lat, lon):
            return dict(payload(), currentSpeed=0, freeFlowSpeed=40,
                        confidence=0.95, roadClosure=True)

    g, _ = collect_traffic_snapshot(road(), ClosedClient(), max_requests=1, pause_s=0)
    assert g[1][2][4]["traffic_status"] == "closed"
    assert not make_routing_graph(g, "time").has_edge(1, 2, 4)


def test_zero_speed_without_closure_is_missing():
    class ZeroClient(Client):
        def fetch_flow(self, lat, lon):
            return dict(payload(), currentSpeed=0, freeFlowSpeed=40,
                        confidence=0.95, roadClosure=False)

    g, _ = collect_traffic_snapshot(road(), ZeroClient(), max_requests=1, pause_s=0)
    assert g[1][2][4]["traffic_status"] == "missing"
    assert g[1][2][4]["time_is_imputed"]


def test_low_confidence_is_missing():
    class PoorClient(Client):
        def fetch_flow(self, lat, lon):
            return dict(payload(), currentSpeed=12, freeFlowSpeed=40,
                        confidence=0.20, roadClosure=False)

    g, _ = collect_traffic_snapshot(road(), PoorClient(), max_requests=1, pause_s=0)
    assert g[1][2][4]["traffic_reason"] == "low_confidence"
