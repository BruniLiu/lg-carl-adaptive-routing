from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lgcarl.models.line_gnn import LineGraphSAGE


class PathQNetwork(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 7,
        hidden_dim: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.1,
        node_embedding_dim: int = 16,
        path_feature_dim: int = 7,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.path_feature_dim = int(path_feature_dim)
        self.gnn = LineGraphSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim)
        mlp_input_dim = hidden_dim * 2 + path_feature_dim + node_embedding_dim * 2 + 1
        self.q_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
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

        link_features = torch.as_tensor(obs["link_features"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(obs["line_graph_edge_index"], dtype=torch.long, device=device)
        path_features = torch.as_tensor(obs["path_features"], dtype=torch.float32, device=device)
        demand = torch.as_tensor(obs["demand"], dtype=torch.float32, device=device).view(1)
        src = torch.as_tensor([int(obs["src"])], dtype=torch.long, device=device)
        dst = torch.as_tensor([int(obs["dst"])], dtype=torch.long, device=device)

        link_embeddings = self.gnn(link_features, edge_index)
        src_emb = self.node_embedding(src).view(-1)
        dst_emb = self.node_embedding(dst).view(-1)

        q_values: list[torch.Tensor] = []
        for idx, edge_indices in enumerate(obs["candidate_edge_indices"]):
            if edge_indices:
                path_idx = torch.as_tensor(edge_indices, dtype=torch.long, device=device)
                path_h = link_embeddings[path_idx]
                mean_pool = path_h.mean(dim=0)
                max_pool = path_h.max(dim=0).values
            else:
                mean_pool = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)
                max_pool = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)

            features = torch.cat(
                [
                    mean_pool,
                    max_pool,
                    path_features[idx],
                    src_emb,
                    dst_emb,
                    demand,
                ],
                dim=0,
            )
            q_values.append(self.q_mlp(features).view(()))
        return torch.stack(q_values, dim=0)

    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        return self.forward_obs(obs)


def mask_q_values(q_values: torch.Tensor, mask: torch.Tensor, fill_value: float = -1e9) -> torch.Tensor:
    return q_values.masked_fill(~mask.bool(), fill_value)

