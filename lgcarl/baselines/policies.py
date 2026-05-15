from __future__ import annotations

import random
from typing import Any, Protocol

import numpy as np

from lgcarl.rl.agent import DQNAgent


class Policy(Protocol):
    name: str

    def select_action(self, obs: dict[str, Any]) -> int:
        ...


class RandomKPolicy:
    name = "random_k"

    def select_action(self, obs: dict[str, Any]) -> int:
        valid = np.flatnonzero(np.asarray(obs["path_mask"], dtype=bool))
        return int(random.choice(valid.tolist())) if len(valid) else 0


class ShortestPathPolicy:
    name = "shortest_path"

    def select_action(self, obs: dict[str, Any]) -> int:
        valid = np.flatnonzero(np.asarray(obs["path_mask"], dtype=bool))
        if len(valid) == 0:
            return 0
        path_features = np.asarray(obs["path_features"], dtype=float)
        scores = path_features[valid, 1]
        return int(valid[int(np.argmin(scores))])


class DynamicShortestPathPolicy:
    name = "dynamic_shortest_path"

    def select_action(self, obs: dict[str, Any]) -> int:
        valid = np.flatnonzero(np.asarray(obs["path_mask"], dtype=bool))
        if len(valid) == 0:
            return 0
        path_features = np.asarray(obs["path_features"], dtype=float)
        delay = path_features[valid, 1]
        queue = path_features[valid, 2]
        max_util = path_features[valid, 3]
        risk = path_features[valid, 5]
        scores = delay + 1.5 * queue + 1.0 * max_util + 0.25 * risk
        return int(valid[int(np.argmin(scores))])


class OSPFLikePolicy:
    name = "ospf_like"

    def select_action(self, obs: dict[str, Any]) -> int:
        valid = np.flatnonzero(np.asarray(obs["path_mask"], dtype=bool))
        if len(valid) == 0:
            return 0
        path_features = np.asarray(obs["path_features"], dtype=float)
        delay = path_features[valid, 1]
        hop = path_features[valid, 0]
        min_capacity = path_features[valid, 4]
        scores = delay + 0.2 * hop - 0.1 * min_capacity
        return int(valid[int(np.argmin(scores))])


class DQNPolicy:
    name = "lgcarl"

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def select_action(self, obs: dict[str, Any]) -> int:
        return self.agent.select_action(obs, training=False)


def make_baseline_policy(name: str) -> Policy:
    if name == "random_k":
        return RandomKPolicy()
    if name == "shortest_path":
        return ShortestPathPolicy()
    if name == "dynamic_shortest_path":
        return DynamicShortestPathPolicy()
    if name == "ospf_like":
        return OSPFLikePolicy()
    raise ValueError(f"Unknown baseline policy: {name}")

