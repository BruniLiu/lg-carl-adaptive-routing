from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import networkx as nx


def _add_directed_edge_pair(
    graph: nx.DiGraph,
    u: int,
    v: int,
    capacity: float,
    base_delay: float,
) -> None:
    attrs = {
        "capacity": float(capacity),
        "base_delay": float(base_delay),
        "failure": 0.0,
    }
    graph.add_edge(int(u), int(v), **attrs)
    graph.add_edge(int(v), int(u), **attrs)


def load_topology_json(path: str | Path) -> nx.DiGraph:
    """Load a topology JSON file and return a directed graph.

    Undirected files are converted by adding both edge directions.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    graph = nx.DiGraph(name=payload.get("name", "topology"))
    for key, value in payload.get("graph", {}).items():
        graph.graph[key] = value
    if "demands" in payload:
        graph.graph["demands"] = payload["demands"]
    graph.add_nodes_from(int(n) for n in payload["nodes"])
    directed = bool(payload.get("directed", False))

    for item in payload["edges"]:
        u = int(item["source"])
        v = int(item["target"])
        capacity = float(item.get("capacity", 50.0))
        base_delay = float(item.get("base_delay", 1.0))
        attrs = {"capacity": capacity, "base_delay": base_delay, "failure": 0.0}
        if directed:
            graph.add_edge(u, v, **attrs)
        else:
            _add_directed_edge_pair(graph, u, v, capacity, base_delay)

    add_edge_betweenness(graph)
    return graph


def generate_random_topology(
    num_nodes: int = 14,
    num_undirected_edges: int = 21,
    min_capacity: float = 20.0,
    max_capacity: float = 100.0,
    base_delay_range: tuple[float, float] = (1.0, 10.0),
    seed: int = 7,
) -> nx.DiGraph:
    """Generate a connected bidirectional random topology."""
    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)
    undirected_edges: set[tuple[int, int]] = set()

    for a, b in zip(nodes[:-1], nodes[1:]):
        undirected_edges.add(tuple(sorted((a, b))))

    while len(undirected_edges) < num_undirected_edges:
        u, v = rng.sample(range(num_nodes), 2)
        undirected_edges.add(tuple(sorted((u, v))))

    graph = nx.DiGraph(name="random")
    graph.add_nodes_from(range(num_nodes))
    for u, v in sorted(undirected_edges):
        capacity = rng.uniform(min_capacity, max_capacity)
        base_delay = rng.uniform(*base_delay_range)
        _add_directed_edge_pair(graph, u, v, capacity, base_delay)

    add_edge_betweenness(graph)
    return graph


def add_edge_betweenness(graph: nx.DiGraph) -> None:
    """Attach normalized edge betweenness centrality to every directed edge."""
    centrality = nx.edge_betweenness_centrality(graph, weight="base_delay", normalized=True)
    max_value = max(centrality.values(), default=1.0) or 1.0
    for edge in graph.edges:
        graph.edges[edge]["betweenness"] = float(centrality.get(edge, 0.0) / max_value)


def build_topology(config: dict[str, Any]) -> nx.DiGraph:
    topology_cfg = config.get("topology", {})
    name = topology_cfg.get("name", "nsfnet")
    if name == "random":
        random_cfg = topology_cfg.get("random", {})
        env_cfg = config.get("env", {})
        return generate_random_topology(
            num_nodes=int(random_cfg.get("num_nodes", 14)),
            num_undirected_edges=int(random_cfg.get("num_undirected_edges", 21)),
            min_capacity=float(env_cfg.get("min_capacity", 20.0)),
            max_capacity=float(env_cfg.get("max_capacity", 100.0)),
            seed=int(config.get("seed", 7)),
        )

    if name == "topohub":
        cache_path = topology_cfg.get("cache_path")
        if cache_path and Path(cache_path).exists():
            return load_topology_json(cache_path)
        from lgcarl.data.topohub_dataset import load_topohub_topology

        return load_topohub_topology(
            topology_cfg.get("topohub_id", "sndlib/polska"),
            capacity_scale=float(topology_cfg.get("capacity_scale", 1.0)),
        )

    return load_topology_json(topology_cfg.get("path", "data/topologies/nsfnet.json"))


def clone_topology(graph: nx.DiGraph) -> nx.DiGraph:
    return graph.copy(as_view=False)
