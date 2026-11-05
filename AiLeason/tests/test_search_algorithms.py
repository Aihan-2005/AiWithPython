

import math
import random

import networkx as nx
import pytest

from routing.algorithms import bfs, dfs, greedy_best_first, astar, run_all_algorithms
from routing.algorithms.heuristics import build_admissible_heuristic


def sample_graph(weight="length", scenario="measured"):
    g = nx.MultiDiGraph(
        crs="EPSG:4326", routing_weight=weight,
        routing_objective="distance" if weight == "length" else "time",
        routing_scenario=scenario,
    )
    for node, (lat, lon) in {
        1: (51.500, -.180), 2: (51.500, -.179),
        3: (51.501, -.180), 4: (51.501, -.179),
    }.items():
        g.add_node(node, x=lon, y=lat)
    roads = [
        (1, 2, 5, 90.0, 10.0, 12.0),
        (1, 2, 7, 120.0, 5.0, 8.0),   
        
        (2, 4, 0, 90.0, 10.0, 12.0),
        (1, 3, 0, 100.0, 10.0, 12.0),
        (3, 4, 0, 100.0, 10.0, 12.0),
        (1, 4, 0, 600.0, 100.0, 120.0),
    ]
    for u,v,k,length,t,free in roads:
        g.add_edge(u,v,key=k, length=length, travel_time_s=t,
                   free_flow_time_s=free, time_is_imputed=(u==1 and v==3),
                   free_flow_is_imputed=False, is_closed=False)
    return g


@pytest.mark.parametrize("search", [bfs, dfs, greedy_best_first, astar])
def test_all_searches_find_valid_directed_route(search):
    g=sample_graph()
    r=search(g,1,4)
    assert r["found"]
    assert r["path_nodes"][0]==1 and r["path_nodes"][-1]==4
    assert len(r["path_edges"])==r["hops"]
    assert len(r["path_nodes"])==len(r["path_edges"])+1
    for i,(u,v,k) in enumerate(r["path_edges"]):
        assert g.has_edge(u,v,k)
        assert u==r["path_nodes"][i] and v==r["path_nodes"][i+1]
    assert r["runtime_ms"]>=0
    assert r["expanded_nodes"]>=1
    assert r["max_frontier_size"]>=1


def test_bfs_finds_fewest_hops_even_if_longer():
    r=bfs(sample_graph(),1,4)
    assert r["path_edges"]==[(1,4,0)]
    assert r["distance_m"]==600.0
    assert r["hops"]==1


def test_parallel_edge_choice_depends_on_objective():
    distance=astar(sample_graph("length"),1,4)
    time=astar(sample_graph("travel_time_s"),1,4)
    assert distance["path_edges"]==[(1,2,5),(2,4,0)]
    assert distance["objective_cost"]==180.0
    assert time["path_edges"]==[(1,2,7),(2,4,0)]
    assert time["objective_cost"]==15.0
    assert distance["distance_m"]==180.0
    assert time["distance_m"]==210.0


def test_free_flow_scenario_time_is_from_free_flow_attribute():
    g=sample_graph("free_flow_time_s", "free_flow")
    r=astar(g,1,4)
    assert r["travel_time_s"]==r["objective_cost"]
    assert r["routing_weight"]=="free_flow_time_s"


def test_closed_directed_edge_changes_route():
    g=sample_graph()
    g.remove_edge(2,4,0)
    r=astar(g,1,4)
    assert r["found"]
    assert r["path_edges"]==[(1,3,0),(3,4,0)]


@pytest.mark.parametrize("search", [bfs, dfs, greedy_best_first, astar])
def test_unreachable_goal_is_clean(search):
    g=sample_graph()
    r=search(g,4,1)
    assert not r["found"]
    assert r["path_edges"]==[] and r["path_nodes"]==[]
    assert r["objective_cost"] is None
    assert r["expanded_nodes"]==1


@pytest.mark.parametrize("search", [bfs, dfs, greedy_best_first, astar])
def test_same_source_and_destination(search):
    r=search(sample_graph(),2,2)
    assert r["found"]
    assert r["hops"]==0 and r["path_nodes"]==[2]
    assert r["distance_m"]==0 and r["travel_time_s"]==0


def test_all_wrapper_and_metadata():
    r=run_all_algorithms(sample_graph(),1,4)
    assert list(r)==["BFS","DFS","Greedy","A*"]
    assert all(v["found"] for v in r.values())


def test_bad_inputs_fail_meaningfully():
    g=sample_graph()
    with pytest.raises(ValueError,match="Origin node"):
        astar(g,999,4)
    with pytest.raises(ValueError,match="Destination node"):
        bfs(g,1,999)
    del g.graph["routing_weight"]
    with pytest.raises(ValueError,match="make_routing_graph"):
        dfs(g,1,4)


def test_skips_invalid_nontraversable_edges():
    g=sample_graph("travel_time_s")
    g[1][2][7]["travel_time_s"]=None
    r=astar(g,1,4)
    assert (1,2,7) not in r["path_edges"]
    g.remove_edge(1,4,0)
    g.remove_edge(1,3,0)
    g[1][2][5]["travel_time_s"]=float("nan")
    assert not astar(g,1,4)["found"]


def test_astar_equals_networkx_oracle_on_many_directed_multigraphs():
    rng=random.Random(2026)
    for weight in ("length","travel_time_s"):
        for case in range(25):
            g=nx.MultiDiGraph(crs="EPSG:4326", routing_weight=weight,
                              routing_scenario="measured")
            for i in range(7):
                g.add_node(i,x=-.18+i*.001, y=51.50+(i%3)*.001)
            for u in range(7):
                for v in range(7):
                    if u != v and rng.random() < .34:
                        for k in range(rng.choice([1,1,2])):
                            g.add_edge(u,v,key=k,length=rng.uniform(20,800),
                                       travel_time_s=rng.uniform(1,100),
                                       free_flow_time_s=100)
            r=astar(g,0,6)
            try:
                oracle=nx.shortest_path_length(g,0,6,weight=weight)
            except nx.NetworkXNoPath:
                assert not r["found"]
            else:
                assert r["found"]
                assert math.isclose(r["objective_cost"],oracle,rel_tol=1e-10,abs_tol=1e-9)


def test_heuristic_is_consistent_on_a_realistic_graph():
    for weight in ("length","travel_time_s"):
        g=sample_graph(weight)
        h=build_admissible_heuristic(g,4,weight)
        assert math.isclose(h(4),0.0)
        for u,v,k,d in g.edges(keys=True,data=True):
            assert h(u) <= float(d[weight])+h(v)+1e-7
