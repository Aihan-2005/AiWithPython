import pytest
import networkx as nx
from shapely.geometry import LineString
from routing.traffic_quality import plan_corridor_probes, audit_traffic_snapshot
from routing.traffic_pipeline import collect_traffic_snapshot, make_routing_graph


def tiny():
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-.180, y=51.500)
    g.add_node(2, x=-.179, y=51.500)
    g.add_node(3, x=-.179, y=51.501)
    g.add_edge(1, 2, key=4, length=70., highway="secondary")
    g.add_edge(2, 1, key=3, length=70., highway="secondary")
    g.add_edge(2, 3, key=0, length=110., highway="service")
    return g


def payload():
    return {
        "currentSpeed": 15,"freeFlowSpeed":40,"confidence":.95,"roadClosure":False,
        "coordinates":{"coordinate":[
            {"latitude":51.500,"longitude":-.180},
            {"latitude":51.500,"longitude":-.179},
        ]},
    }


def test_plan_deterministic_and_avoids_service():
    g=tiny()
    opts=dict(origins={"A":(51.5,-.180)}, destination=(51.5,-.179), corridor_width_m=150)
    a=plan_corridor_probes(g,max_probes=5,**opts)
    assert a==plan_corridor_probes(g,max_probes=5,**opts)
    assert len(a)==1
    assert a[0]["road_class"]=="secondary"
    assert a[0]["edge_id"] in {(1,2,4),(2,1,3)}


def test_bad_parameters_and_zero_limit():
    assert plan_corridor_probes(tiny(), max_probes=0)==[]
    with pytest.raises(ValueError): plan_corridor_probes(tiny(), max_probes=-1)
    with pytest.raises(ValueError): plan_corridor_probes(tiny(), origins={})


def test_explicit_selected_edges_and_smoke_audit():
    class Fake:
        def __init__(self): self.calls=0
        def fetch_flow(self,lat,lon):
            self.calls+=1
            return payload()
    base=tiny()
    client=Fake()
    snap, summary=collect_traffic_snapshot(
        base,client,selected_edge_ids=[(1,2,4),(1,2,4)],max_requests=5,pause_s=0)
    assert client.calls==1
    assert summary["sampling_strategy"]=="provided_reviewed_edge_plan"
    assert summary["probe_plan_count"]==1
    assert snap[1][2][4]["traffic_status"]=="red"
    report=audit_traffic_snapshot(base,snap,summary)
    assert report["graph_identity_ok"]
    assert report["algorithm_smoke_test_ready"]
    assert not report["final_traffic_comparison_ready"]
    assert report["observed_edges"]==1
    assert report["missing_edges"]==2


def test_invalid_edge_plan_never_queries():
    class Bad:
        def fetch_flow(self,*args,**kwargs): raise AssertionError('should not call API')
    with pytest.raises(ValueError,match='not found'):
        collect_traffic_snapshot(tiny(),Bad(),selected_edge_ids=[(99,100,0)],max_requests=1)


def test_offline_audit_warns_about_missing_data():
    base=tiny()
    snap,summary=collect_traffic_snapshot(base,None,max_requests=0)
    report=audit_traffic_snapshot(base,snap,summary)
    assert report["imputed_edges"]==3
    assert report["observed_coverage_percent"]==0
    assert report["algorithm_smoke_test_ready"]
    assert len(report["warnings"])>=2
    assert make_routing_graph(snap,"distance").number_of_edges()==3


def test_zero_max_with_explicit_plan_makes_no_request():
    base=tiny()
    snap, summary=collect_traffic_snapshot(base,None,max_requests=0,selected_edge_ids=[(1,2,4)])
    assert summary["requests_made"]==0
    assert summary["sampled_directed_edges"]==0
    assert snap[1][2][4]["traffic_status"]=="missing"
