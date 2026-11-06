import networkx as nx

from routing.probe_history import (
    update_probe_history,
    save_probe_history,
    load_probe_history,
)


def test_history_records_direct_success_and_failure(tmp_path):

    graph = nx.MultiDiGraph()

    graph.add_edge(
        1, 2, key=0,
        traffic_directly_queried=True,
        traffic_status="clear",
        traffic_matched=True,
        traffic_match_source="direct_probe",
        traffic_reason="observed",
        traffic_match_reason="candidate_match",
    )

    graph.add_edge(
        2, 3, key=0,
        traffic_directly_queried=True,
        traffic_status="missing",
        traffic_matched=False,
        traffic_match_source=None,
        traffic_reason="api_http_400",
        traffic_match_reason=None,
    )

    graph.add_edge(
        3, 4, key=0,
        traffic_directly_queried=False,
        traffic_status="clear",
        traffic_matched=True,
        traffic_match_source="reverse_inferred",
        traffic_reason="observed",
    )

    summary = {
        "requests_made": 2,
        "request_errors": 1,
    }

    history = update_probe_history(
        graph,
        summary,
    )

    assert history["edges"]["1|2|0"]["attempts"] == 1

    assert (
        history["edges"]["1|2|0"]["direct_usable"]
        == 1
    )

    assert (
        history["edges"]["2|3|0"]["point_errors"]
        == 1
    )

    # Reverse inferred edge was not directly queried.
    assert "3|4|0" not in history["edges"]

    path = tmp_path / "probe_history.json"

    save_probe_history(history, path)

    loaded = load_probe_history(path)

    assert loaded == history


def test_offline_run_does_not_update_history():

    history = {
        "schema_version": 1,
        "edges": {},
        "runs": [],
    }

    graph = nx.MultiDiGraph()

    result = update_probe_history(
        graph,
        {"requests_made": 0, "request_errors": 0},
        history,
    )

    assert result["runs"] == []
