from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            neigh = torch.zeros_like(x)
            neigh.index_add_(0, dst, x[src])
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            neigh = neigh / deg.clamp_min(1.0).unsqueeze(-1)

        out = self.self_linear(x) + self.neigh_linear(neigh)
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class LineGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, link_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(link_features)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

