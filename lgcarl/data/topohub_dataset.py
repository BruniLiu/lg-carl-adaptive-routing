from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from lgcarl.graph.topology import add_edge_betweenness


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_").lower()


def _node_link_graph(payload: dict[str, Any]) -> nx.Graph:
    try:
        return nx.node_link_graph(payload, edges="edges")
    except TypeError:
        return nx.node_link_graph(payload)


def _edge_distance(data: dict[str, Any]) -> float:
    for key in ("dist", "distance", "length", "weight"):
        value = data.get(key)
        if value is not None:
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                pass
    return 100.0


def _capacity_from_attrs(data: dict[str, Any], capacity_scale: float = 1.0) -> float:
    for key in ("capacity", "cap", "bandwidth", "bw"):
        value = data.get(key)
        if value is not None:
            try:
                return max(1.0, float(value) * capacity_scale)
            except (TypeError, ValueError):
                pass

    utilizations: list[float] = []
    for direction_key in ("ecmp_fwd", "ecmp_bwd"):
        direction = data.get(direction_key, {})
        if isinstance(direction, dict):
            for value in direction.values():
                try:
                    utilizations.append(float(value))
                except (TypeError, ValueError):
                    continue
    if utilizations:
        headroom = 100.0 / max(max(utilizations), 1.0)
        return max(20.0, min(200.0, 60.0 * headroom)) * capacity_scale
    return 80.0 * capacity_scale


def _base_delay_from_distance(distance_km: float) -> float:
    # Rough propagation delay plus a small router processing floor.
    return max(1.0, distance_km / 200.0 + 1.0)


def _extract_demands(graph: nx.Graph, node_to_int: dict[Any, int]) -> list[dict[str, float | int]]:
    raw_demands = graph.graph.get("demands", {})
    demands: list[dict[str, float | int]] = []
    if not isinstance(raw_demands, dict):
        return demands

    for raw_src, dst_map in raw_demands.items():
        src_key = raw_src
        if src_key not in node_to_int and str(src_key) in node_to_int:
            src_key = str(src_key)
        if src_key not in node_to_int or not isinstance(dst_map, dict):
            continue
        for raw_dst, value in dst_map.items():
            dst_key = raw_dst
            if dst_key not in node_to_int and str(dst_key) in node_to_int:
                dst_key = str(dst_key)
            if dst_key not in node_to_int:
                continue
            try:
                demand = float(value)
            except (TypeError, ValueError):
                continue
            if demand <= 0.0:
                continue
            demands.append(
                {
                    "src": int(node_to_int[src_key]),
                    "dst": int(node_to_int[dst_key]),
                    "demand": demand,
                }
            )
    return demands


def topohub_to_lgcarl_json(topohub_id: str, capacity_scale: float = 1.0) -> dict[str, Any]:
    import topohub

    payload = topohub.get(topohub_id)
    source_graph = _node_link_graph(payload)
    if source_graph.is_directed():
        undirected_view = source_graph.to_undirected()
    else:
        undirected_view = source_graph

    nodes = sorted(undirected_view.nodes, key=lambda item: str(item))
    node_to_int = {node: idx for idx, node in enumerate(nodes)}
    converted_edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    distances: list[float] = []

    for u, v, data in undirected_view.edges(data=True):
        if u == v:
            continue
        src = node_to_int[u]
        dst = node_to_int[v]
        edge_key = tuple(sorted((src, dst)))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        distance = _edge_distance(data)
        distances.append(distance)
        converted_edges.append(
            {
                "source": int(edge_key[0]),
                "target": int(edge_key[1]),
                "capacity": round(_capacity_from_attrs(data, capacity_scale), 4),
                "base_delay": round(_base_delay_from_distance(distance), 4),
                "distance_km": round(distance, 4),
            }
        )

    demands = _extract_demands(source_graph, node_to_int)
    name = source_graph.graph.get("name", topohub_id)
    return {
        "name": str(name),
        "source": "topohub",
        "topohub_id": topohub_id,
        "directed": False,
        "nodes": list(range(len(nodes))),
        "node_metadata": [
            {
                "id": int(node_to_int[node]),
                "original_id": str(node),
                "name": str(source_graph.nodes[node].get("name", node)),
                "pos": source_graph.nodes[node].get("pos"),
            }
            for node in nodes
        ],
        "edges": converted_edges,
        "demands": demands,
        "graph": {
            "name": str(name),
            "source": "topohub",
            "topohub_id": topohub_id,
            "num_demands": len(demands),
            "avg_distance_km": sum(distances) / max(len(distances), 1),
        },
    }


def save_topohub_topology(topohub_id: str, output_path: str | Path, capacity_scale: float = 1.0) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = topohub_to_lgcarl_json(topohub_id, capacity_scale=capacity_scale)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out


def load_topohub_topology(topohub_id: str, capacity_scale: float = 1.0) -> nx.DiGraph:
    from lgcarl.graph.topology import load_topology_json
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8") as fh:
        payload = topohub_to_lgcarl_json(topohub_id, capacity_scale=capacity_scale)
        json.dump(payload, fh)
        fh.flush()
        graph = load_topology_json(fh.name)
    add_edge_betweenness(graph)
    return graph

