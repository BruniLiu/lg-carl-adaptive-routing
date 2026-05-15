from __future__ import annotations

from itertools import islice
from typing import Iterable

import networkx as nx

from lgcarl.graph.line_graph import Edge, edge_path_to_indices, node_path_to_edge_path


def edge_is_available(graph: nx.DiGraph, edge: Edge) -> bool:
    data = graph.edges[edge]
    return float(data.get("failure", 0.0)) < 0.5 and float(data.get("capacity", 0.0)) > 0.0


def path_is_available(graph: nx.DiGraph, edge_path: list[Edge]) -> bool:
    return all(graph.has_edge(*edge) and edge_is_available(graph, edge) for edge in edge_path)


def get_k_shortest_edge_paths(
    graph: nx.DiGraph,
    src: int,
    dst: int,
    k: int,
    weight: str = "base_delay",
) -> list[list[Edge]]:
    if src == dst or not graph.has_node(src) or not graph.has_node(dst):
        return []

    try:
        node_paths: Iterable[list[int]] = nx.shortest_simple_paths(graph, src, dst, weight=weight)
        paths: list[list[Edge]] = []
        for node_path in islice(node_paths, max(k * 8, k)):
            edge_path = node_path_to_edge_path(node_path)
            if path_is_available(graph, edge_path):
                paths.append(edge_path)
            if len(paths) >= k:
                break
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def pad_edge_paths(paths: list[list[Edge]], k: int) -> tuple[list[list[Edge]], list[bool]]:
    padded = list(paths[:k])
    mask = [True] * len(padded)
    while len(padded) < k:
        padded.append([])
        mask.append(False)
    return padded, mask


def candidate_indices(
    paths: list[list[Edge]],
    edge_to_idx: dict[Edge, int],
) -> list[list[int]]:
    return [edge_path_to_indices(path, edge_to_idx) for path in paths]

