from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    obs: dict[str, Any]
    action: int
    reward: float
    next_obs: dict[str, Any]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=copy.deepcopy(obs),
                action=int(action),
                reward=float(reward),
                next_obs=copy.deepcopy(next_obs),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, int(batch_size))

