from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PathMLPQNetwork(nn.Module):
    """A no-GNN DQN variant that only sees handcrafted path features."""

    def __init__(
        self,
        num_nodes: int,
        node_embedding_dim: int = 16,
        path_feature_dim: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        input_dim = path_feature_dim + node_embedding_dim * 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_obs(self, obs: dict[str, Any], device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        path_features = torch.as_tensor(obs["path_features"], dtype=torch.float32, device=device)
        demand = torch.as_tensor(obs["demand"], dtype=torch.float32, device=device).view(1)
        src = torch.as_tensor([int(obs["src"])], dtype=torch.long, device=device)
        dst = torch.as_tensor([int(obs["dst"])], dtype=torch.long, device=device)
        src_emb = self.node_embedding(src).view(-1)
        dst_emb = self.node_embedding(dst).view(-1)

        outputs: list[torch.Tensor] = []
        for features in path_features:
            x = torch.cat([features, src_emb, dst_emb, demand], dim=0)
            outputs.append(self.mlp(x).view(()))
        return torch.stack(outputs, dim=0)

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.forward_obs(obs)

