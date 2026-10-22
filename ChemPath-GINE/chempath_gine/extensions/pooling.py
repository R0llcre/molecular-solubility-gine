from __future__ import annotations

import torch
import torch.nn as nn

from chempath_gine.models.chem_path_gine import GraphPool


class GraphPoolSum(GraphPool):
    """GraphPool plus a size-extensive sum-pool branch.

    The original readout (attention, mean, max) ignores molecule size, so
    atom-additive quantities such as logP can only reach the model through the
    descriptor vector. The extra branch adds sum(node states) / sum_scale, where
    sum_scale (about the mean heavy-atom count) keeps the input near unit scale.
    """

    def __init__(self, hidden_dim: int, sum_scale: float = 13.0):
        super().__init__(hidden_dim)
        self.sum_scale = float(sum_scale)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        score = self.score(x).squeeze(-1)
        score = score.masked_fill(~node_mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)
        attn_pool = (x * weight).sum(dim=1)
        masked = x * node_mask[:, :, None]
        denom = node_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_pool = masked.sum(dim=1) / denom
        sum_pool = masked.sum(dim=1) / self.sum_scale
        max_x = x.masked_fill(~node_mask[:, :, None], torch.finfo(x.dtype).min)
        max_pool = max_x.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.out(torch.cat([attn_pool, mean_pool, max_pool, sum_pool], dim=-1))
