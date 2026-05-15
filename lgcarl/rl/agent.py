from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from lgcarl.models.q_network import PathQNetwork, mask_q_values
from lgcarl.rl.replay_buffer import ReplayBuffer
from lgcarl.rl.scheduler import LinearEpsilonScheduler
from lgcarl.utils import ensure_parent


class DQNAgent:
    def __init__(
        self,
        num_nodes: int,
        model_config: dict[str, Any],
        dqn_config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = PathQNetwork(num_nodes=num_nodes, **model_config).to(self.device)
        self.target_net = PathQNetwork(num_nodes=num_nodes, **model_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=float(dqn_config.get("learning_rate", 1e-4)),
        )
        self.gamma = float(dqn_config.get("gamma", 0.95))
        self.batch_size = int(dqn_config.get("batch_size", 64))
        self.target_update_interval = int(dqn_config.get("target_update_interval", 500))
        self.gradient_clip = float(dqn_config.get("gradient_clip", 5.0))
        self.replay_buffer = ReplayBuffer(int(dqn_config.get("replay_buffer_size", 50000)))
        self.epsilon = LinearEpsilonScheduler(
            start=float(dqn_config.get("epsilon_start", 1.0)),
            end=float(dqn_config.get("epsilon_end", 0.05)),
            decay_steps=int(dqn_config.get("epsilon_decay_steps", 20000)),
        )
        self.steps_done = 0

    def select_action(self, obs: dict[str, Any], training: bool = True) -> int:
        mask = np.asarray(obs["path_mask"], dtype=bool)
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return 0

        eps = self.epsilon.value(self.steps_done) if training else 0.0
        if training:
            self.steps_done += 1
        if training and random.random() < eps:
            return int(random.choice(valid_actions.tolist()))

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net.forward_obs(obs, device=self.device)
            torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            action = int(torch.argmax(mask_q_values(q_values, torch_mask)).item())
        if training:
            self.policy_net.train()
        return action

    def remember(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        done: bool,
    ) -> None:
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        self.policy_net.train()

        for transition in batch:
            q_values = self.policy_net.forward_obs(transition.obs, device=self.device)
            predictions.append(q_values[transition.action])

            with torch.no_grad():
                next_mask_np = np.asarray(transition.next_obs["path_mask"], dtype=bool)
                if transition.done or not next_mask_np.any():
                    next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    next_q = self.target_net.forward_obs(transition.next_obs, device=self.device)
                    next_mask = torch.as_tensor(next_mask_np, dtype=torch.bool, device=self.device)
                    next_value = mask_q_values(next_q, next_mask).max()
                target = torch.tensor(transition.reward, dtype=torch.float32, device=self.device)
                if not transition.done:
                    target = target + self.gamma * next_value
                targets.append(target)

        pred_tensor = torch.stack(predictions)
        target_tensor = torch.stack(targets)
        loss = F.smooth_l1_loss(pred_tensor, target_tensor)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        if self.steps_done % self.target_update_interval == 0:
            self.sync_target()

        return float(loss.detach().cpu().item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> None:
        out = ensure_parent(path)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            out,
        )

    def load(self, path: str | Path) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = int(checkpoint.get("steps_done", 0))
