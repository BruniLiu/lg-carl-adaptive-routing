import networkx as nx

from lgcarl.graph.line_graph import build_directed_line_graph, node_path_to_edge_path


def test_line_graph_connects_consecutive_links():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, capacity=10, base_delay=1)
    graph.add_edge(1, 2, capacity=10, base_delay=1)
    graph.add_edge(1, 3, capacity=10, base_delay=1)

    line = build_directed_line_graph(graph)
    src = line.edge_to_idx[(0, 1)]
    dst_a = line.edge_to_idx[(1, 2)]
    dst_b = line.edge_to_idx[(1, 3)]
    pairs = set(zip(line.edge_index[0].tolist(), line.edge_index[1].tolist()))
    assert (src, dst_a) in pairs
    assert (src, dst_b) in pairs


def test_node_path_to_edge_path():
    assert node_path_to_edge_path([0, 2, 5]) == [(0, 2), (2, 5)]

