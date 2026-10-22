"""Alternative node-level graph encoders for cross-architecture ensembles.

Selected with ``model.encoder`` (see ``apply_model_extensions``); the default
``gine`` keeps ChemPath's own ``GINEEncoder``. Every encoder here follows the
``GINEEncoder`` contract: ``forward(batch)`` reads ``atom_cat``, ``atom_float``,
``edge_index``, ``edge_cat`` and ``edge_float`` and returns node states of shape
``(num_nodes, hidden_dim)``, so ChemPath's readout, descriptor fusion and head
(and ``GaussianGINE``) are used unchanged. Input embeddings come from ChemPath's
``AtomEncoder`` / ``EdgeEncoder``.

Each encoder also splits its forward into ``embed(batch) -> (h0, edge_attr)``
and ``node_states_from_embeddings(h0, edge_attr, edge_index)``, the same split
``GaussianGINE`` uses for embedding-level integrated gradients, so attribution
works for these encoders too.

Encoders
--------
dmpnn        Directed message passing (Yang et al., 2019; ported from
             MolFuse-GINE): hidden states live on directed bonds, and the
             message into bond u->v sums the states of bonds w->u except the
             reverse bond v->u.
attentivefp  Atom stage of AttentiveFP (Xiong et al., 2020), as in
             ``torch_geometric.nn.models.AttentiveFP``: one edge-aware GATEConv
             layer, then ``num_gnn_layers - 1`` GATConv layers, each followed by
             a GRU update of the atom state. The molecule-level attention
             readout is not used; ChemPath's GraphPool reads out the node states.

Single-atom molecules carry one self-loop sentinel edge (0 -> 0, bond type 4,
see ``featurize_mol``). For D-MPNN that edge is its own reverse, so its
incoming message is exactly zero and the atom keeps its embedding plus the
initial edge state. For AttentiveFP the softmax over the single incoming edge is
1, so the atom attends to itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.models.attentive_fp import GATEConv

from chempath_gine.models.gine_encoder import AtomEncoder, EdgeEncoder


def reverse_edge_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the reverse edge (v -> u) for every directed edge (u -> v).

    Returns ``(rev, found)``; where an edge has no reverse, ``found`` is False
    and ``rev`` points at the edge itself (callers must mask it). A self loop is
    its own reverse. Vectorised (sort + searchsorted), so it does not depend on
    the order in which edges are stored and stays on the edge_index device.
    """
    num_edges = edge_index.size(1)
    arange = torch.arange(num_edges, device=edge_index.device)
    if num_edges == 0:
        return arange, torch.zeros(0, dtype=torch.bool, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    n = max(int(num_nodes), 1)
    key = src * n + dst
    rkey = dst * n + src
    sorted_key, order = torch.sort(key)
    pos = torch.searchsorted(sorted_key, rkey).clamp(max=num_edges - 1)
    found = sorted_key[pos] == rkey
    rev = torch.where(found, order[pos], arange)
    return rev, found


class DMPNNEncoder(nn.Module):
    """Directed MPNN with reverse-edge exclusion (port of MolFuse's DMPNNEncoder).

    msg0_{u->v}  = MLP([h0_u, e_{uv}])
    a_{u->v}     = sum_{w->u} msg_{w->u} - msg_{v->u}
    msg_{u->v}  += Dropout(MLP([msg0_{u->v}, a_{u->v}]))      (num_layers times)
    h_v          = h0_v + SiLU(LayerNorm(MLP([h0_v, sum_{u->v} msg_{u->v}])))
    """

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
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def embed(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = self.edge_encoder(batch.edge_cat, batch.edge_float)
        return h0, edge_attr

    def node_states_from_embeddings(
        self,
        h0: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h0
        num_nodes = h0.size(0)
        src, dst = edge_index[0], edge_index[1]
        rev, found = reverse_edge_index(edge_index, num_nodes)
        keep_rev = found.to(h0.dtype).unsqueeze(-1)
        msg0 = self.edge_init(torch.cat([h0[src], edge_attr], dim=-1))
        msg = msg0
        for layer in self.msg_layers:
            incoming = msg.new_zeros((num_nodes, msg.size(-1))).index_add(0, dst, msg)
            agg = incoming[src] - msg[rev] * keep_rev
            out = layer(torch.cat([msg0, agg], dim=-1))
            out = F.dropout(out, p=self.dropout, training=self.training)
            msg = msg + out
        node_msg = msg.new_zeros((num_nodes, msg.size(-1))).index_add(0, dst, msg)
        h = self.node_update(torch.cat([h0, node_msg], dim=-1))
        return h0 + F.silu(self.norm(h))

    def forward(self, batch) -> torch.Tensor:
        h0, edge_attr = self.embed(batch)
        return self.node_states_from_embeddings(h0, edge_attr, batch.edge_index)


class AttentiveFPEncoder(nn.Module):
    """Node-level AttentiveFP: GATEConv + (num_layers - 1) x GATConv, GRU updates.

    Mirrors the atom stage of ``torch_geometric.nn.models.AttentiveFP`` with
    ChemPath's embeddings as input. ``heads`` attention heads (concatenated,
    ``hidden_dim // heads`` channels each) are used in the GATConv layers; the
    edge-aware GATEConv is single-head as in the original model.
    """

    def __init__(self, hidden_dim: int, num_layers: int, heads: int, dropout: float):
        super().__init__()
        heads = max(1, int(heads))
        if hidden_dim % heads != 0:
            raise ValueError(
                f"attentivefp encoder needs hidden_dim divisible by heads; got {hidden_dim} and {heads}"
            )
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.dropout = float(dropout)
        self.heads = heads
        self.lin_in = nn.Linear(hidden_dim, hidden_dim)
        self.gate_conv = GATEConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, dropout=self.dropout)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.atom_convs = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(max(1, int(num_layers)) - 1):
            self.atom_convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=self.dropout,
                    add_self_loops=False,
                    negative_slope=0.01,
                )
            )
            self.atom_grus.append(nn.GRUCell(hidden_dim, hidden_dim))

    def embed(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = self.edge_encoder(batch.edge_cat, batch.edge_float)
        return h0, edge_attr

    def node_states_from_embeddings(
        self,
        h0: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = F.leaky_relu(self.lin_in(h0), negative_slope=0.01)
        if edge_index.numel() == 0:
            return x
        h = F.elu(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = F.relu(self.gru(h, x))
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.relu(gru(h, x))
        return x

    def forward(self, batch) -> torch.Tensor:
        h0, edge_attr = self.embed(batch)
        return self.node_states_from_embeddings(h0, edge_attr, batch.edge_index)


ENCODERS = ("dmpnn", "attentivefp")


def build_encoder(name: str, cfg) -> nn.Module:
    """Build a node-level encoder from ``cfg.model`` (hidden_dim, num_gnn_layers, dropout, heads)."""
    model_cfg = cfg.model
    key = str(name).lower()
    hidden_dim = int(model_cfg.hidden_dim)
    num_layers = int(model_cfg.num_gnn_layers)
    dropout = float(model_cfg.dropout)
    if key == "dmpnn":
        return DMPNNEncoder(hidden_dim, num_layers, dropout)
    if key == "attentivefp":
        heads = int(getattr(model_cfg, "heads", 1))
        return AttentiveFPEncoder(hidden_dim, num_layers, heads, dropout)
    raise ValueError(f"Unknown model.encoder={name!r}; choose 'gine' or one of {list(ENCODERS)}")


__all__ = [
    "AttentiveFPEncoder",
    "DMPNNEncoder",
    "ENCODERS",
    "build_encoder",
    "reverse_edge_index",
]
