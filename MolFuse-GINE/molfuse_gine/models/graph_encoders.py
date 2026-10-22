from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, PNAConv

from molfuse_gine.models.gine_encoder import AtomEncoder, EdgeEncoder, GINEEncoder


class PNAEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.dropout = float(dropout)
        # A fixed broad histogram is sufficient for stable PNA scalers in the
        # small-molecule regime; dataset-specific histograms can be added later.
        self.register_buffer("deg", torch.ones(64, dtype=torch.long))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=["mean", "min", "max", "std"],
                    scalers=["identity", "amplification", "attenuation"],
                    deg=self.deg,
                    edge_dim=hidden_dim,
                    towers=1,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
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


class AttentiveFPEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for attentivefp encoder")
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, batch) -> torch.Tensor:
        h = self.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = self.edge_encoder(batch.edge_cat, batch.edge_float)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, batch.edge_index, edge_attr)
            out = F.elu(norm(out))
            out = F.dropout(out, p=self.dropout, training=self.training)
            h = h + out
        return h


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.dropout = float(dropout)
        self.edge_init = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _reverse_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        lookup = {(s, d): i for i, (s, d) in enumerate(zip(src, dst))}
        reverse = [lookup.get((d, s), i) for i, (s, d) in enumerate(zip(src, dst))]
        return torch.tensor(reverse, dtype=torch.long, device=edge_index.device)

    def forward(self, batch) -> torch.Tensor:
        h0 = self.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = self.edge_encoder(batch.edge_cat, batch.edge_float)
        if batch.edge_index.numel() == 0:
            return h0
        src, dst = batch.edge_index
        edge_input = torch.cat([h0[src], edge_attr], dim=-1)
        msg0 = self.edge_init(edge_input)
        msg = msg0
        reverse = self._reverse_edge_index(batch.edge_index)
        for layer in self.msg_layers:
            incoming = msg.new_zeros((h0.size(0), msg.size(-1)))
            incoming.index_add_(0, dst, msg)
            agg_to_src = incoming[src] - msg[reverse]
            out = layer(torch.cat([msg0, agg_to_src], dim=-1))
            out = F.dropout(out, p=self.dropout, training=self.training)
            msg = msg + out
        node_msg = msg.new_zeros((h0.size(0), msg.size(-1)))
        node_msg.index_add_(0, dst, msg)
        h = self.node_update(torch.cat([h0, node_msg], dim=-1))
        return h0 + F.silu(self.norm(h))


def build_graph_encoder(
    encoder_type: str,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
) -> nn.Module:
    key = encoder_type.lower()
    if key == "gine":
        return GINEEncoder(hidden_dim, num_layers, dropout)
    if key == "pna":
        return PNAEncoder(hidden_dim, num_layers, dropout)
    if key == "dmpnn":
        return DMPNNEncoder(hidden_dim, num_layers, dropout)
    if key in {"attentivefp", "gatv2"}:
        return AttentiveFPEncoder(hidden_dim, num_layers, heads, dropout)
    raise ValueError(f"Unknown graph encoder type {encoder_type!r}")
