from __future__ import annotations

import numpy as np

from lgcarl.graph.line_graph import Edge


def jaccard_distance(path_a: list[Edge] | None, path_b: list[Edge] | None) -> float:
    if not path_a or not path_b:
        return 0.0
    set_a = set(path_a)
    set_b = set(path_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(union)


def safe_array(values: list[float], dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)

