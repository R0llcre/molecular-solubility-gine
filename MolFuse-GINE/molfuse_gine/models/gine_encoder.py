from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        sizes = [119, 11, 11, 16, 8, 9, 9]
        self.embeddings = nn.ModuleList([nn.Embedding(size, hidden_dim) for size in sizes])
        self.float_proj = nn.Linear(5, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_cat: torch.Tensor, atom_float: torch.Tensor) -> torch.Tensor:
        h = self.float_proj(atom_float)
        for i, emb in enumerate(self.embeddings):
            idx = atom_cat[:, i].clamp(min=0, max=emb.num_embeddings - 1)
            h = h + emb(idx)
        return self.norm(h)


class EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bond_type = nn.Embedding(5, hidden_dim)
        self.stereo = nn.Embedding(8, hidden_dim)
        self.float_proj = nn.Linear(3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_cat: torch.Tensor, edge_float: torch.Tensor) -> torch.Tensor:
        if edge_cat.numel() == 0:
            return edge_float.new_empty((0, self.float_proj.out_features))
        h = self.float_proj(edge_float)
        h = h + self.bond_type(edge_cat[:, 0].clamp(0, 4))
        h = h + self.stereo(edge_cat[:, 1].clamp(0, 7))
        return self.norm(h)


class GINEEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, batch) -> torch.Tensor:
        h = self.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = self.edge_encoder(batch.edge_cat, batch.edge_float)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, batch.edge_index, edge_attr)
            out = F.silu(norm(out))
            out = F.dropout(out, p=self.dropout, training=self.training)
            h = h + out
        return h
