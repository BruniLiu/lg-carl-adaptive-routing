from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


Edge = tuple[int, int]


@dataclass(frozen=True)
class LineGraphData:
    edge_order: list[Edge]
    edge_to_idx: dict[Edge, int]
    idx_to_edge: dict[int, Edge]
    edge_index: np.ndarray


def sorted_edges(graph: nx.DiGraph) -> list[Edge]:
    return sorted((int(u), int(v)) for u, v in graph.edges)


def build_directed_line_graph(graph: nx.DiGraph) -> LineGraphData:
    edge_order = sorted_edges(graph)
    edge_to_idx = {edge: i for i, edge in enumerate(edge_order)}
    idx_to_edge = {i: edge for edge, i in edge_to_idx.items()}

    sources: list[int] = []
    targets: list[int] = []
    for edge in edge_order:
        _, v = edge
        source_idx = edge_to_idx[edge]
        for _, w in graph.out_edges(v):
            next_edge = (int(v), int(w))
            if next_edge in edge_to_idx:
                sources.append(source_idx)
                targets.append(edge_to_idx[next_edge])

    if sources:
        edge_index = np.asarray([sources, targets], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return LineGraphData(
        edge_order=edge_order,
        edge_to_idx=edge_to_idx,
        idx_to_edge=idx_to_edge,
        edge_index=edge_index,
    )


def node_path_to_edge_path(path: list[int]) -> list[Edge]:
    return [(int(u), int(v)) for u, v in zip(path[:-1], path[1:])]


def edge_path_to_indices(edge_path: list[Edge], edge_to_idx: dict[Edge, int]) -> list[int]:
    return [edge_to_idx[edge] for edge in edge_path if edge in edge_to_idx]

