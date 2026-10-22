from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from chempath_gine.models import ChemPathGINE
from chempath_gine.models.gine_encoder import GINEEncoder


class GaussianGINE(nn.Module):
    """A ChemPathGINE trunk whose regression head predicts (mu, log sigma^2).

    Only graph variants without the attention stack are supported (gine,
    gine_desc and the other descriptor-fusion variants), because the explicit
    forward below mirrors ChemPathGINE.forward for that case and is shared by
    training and by the embedding-level integrated-gradient code.

    Both outputs live in the standardized target space used by the dataset
    (y_scaled = (y - mean) / scale). sigma in original units is
    exp(0.5 * logvar) * scale.

    The encoder may be ChemPath's GINEEncoder (the original path, unchanged) or
    any module whose forward(batch) returns node states (num_nodes, hidden_dim),
    e.g. the alternative encoders in chempath_gine.extensions.encoders; those
    are called as base.encoder(batch) in forward/head_features. embed() and
    node_states_from_embeddings() (used by the attribution code) delegate to
    the encoder's methods of the same name and raise TypeError if it has none.
    """

    def __init__(
        self,
        base: ChemPathGINE,
        logvar_min: float = -9.0,
        logvar_max: float = 4.0,
        detach_variance_input: bool = False,
    ):
        super().__init__()
        if base.desc_only or len(base.attention) > 0:
            raise ValueError(
                f"GaussianGINE supports graph variants without attention; got {base.variant!r}"
            )
        self.base = base
        self.variant = base.variant
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        # True = "faithful" mode: the variance head reads a stop-gradient copy of
        # the shared representation, so the NLL term cannot change the mean path.
        self.detach_variance_input = bool(detach_variance_input)
        hidden_dim = base.head[-1].in_features
        # Shared head trunk (LayerNorm -> Linear -> SiLU -> Dropout), two output layers.
        self.trunk = nn.Sequential(*list(base.head.children())[:-1])
        self.mean_out = base.head[-1]
        self.logvar_out = nn.Linear(hidden_dim, 1)
        # Start homoscedastic with sigma = 1 in standardized units (= train std of y).
        nn.init.zeros_(self.logvar_out.weight)
        nn.init.zeros_(self.logvar_out.bias)
        # Drop the base head reference so its parameters are owned once.
        self.base.head = None

    # -- trunk pieces, exposed so attribution can inject embeddings ---------------
    def node_states_from_embeddings(
        self,
        h0: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.base.encoder
        if not isinstance(enc, GINEEncoder):
            return self._encoder_method("node_states_from_embeddings")(h0, edge_attr, edge_index)
        h = h0
        for conv, norm in zip(enc.convs, enc.norms):
            out = conv(h, edge_index, edge_attr)
            out = torch.nn.functional.silu(norm(out))
            out = torch.nn.functional.dropout(out, p=enc.dropout, training=self.training)
            h = h + out
        return h

    def embed(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.base.encoder
        if not isinstance(enc, GINEEncoder):
            return self._encoder_method("embed")(batch)
        h0 = enc.atom_encoder(batch.atom_cat, batch.atom_float)
        edge_attr = enc.edge_encoder(batch.edge_cat, batch.edge_float)
        return h0, edge_attr

    def _encoder_method(self, name: str):
        fn = getattr(self.base.encoder, name, None)
        if not callable(fn):
            raise TypeError(
                f"{type(self.base.encoder).__name__} has no {name}(); embedding-level attribution "
                "needs ChemPath's GINEEncoder or an encoder that splits embed / "
                "node_states_from_embeddings (chempath_gine.extensions.encoders)"
            )
        return fn

    def node_states(self, batch) -> torch.Tensor:
        """Final node states h (num_nodes, hidden_dim) for a batch."""
        if isinstance(self.base.encoder, GINEEncoder):
            h0, edge_attr = self.embed(batch)
            return self.node_states_from_embeddings(h0, edge_attr, batch.edge_index)
        return self.base.encoder(batch)

    def readout(
        self,
        h: torch.Tensor,
        batch_vec: torch.Tensor,
        desc: torch.Tensor | None,
        return_attention: bool = False,
    ):
        dense, node_mask = to_dense_batch(h, batch_vec)
        pool = self.base.pool
        graph_repr = pool(dense, node_mask)
        if desc is not None and self.training and self.base.descriptor_dropout > 0.0:
            desc = torch.nn.functional.dropout(desc, p=self.base.descriptor_dropout, training=True)
        graph_repr = self.base.fusion(graph_repr, desc)
        z = self.trunk(graph_repr)
        mu = self.mean_out(z).view(-1)
        z_var = z.detach() if self.detach_variance_input else z
        logvar = self.logvar_out(z_var).view(-1).clamp(self.logvar_min, self.logvar_max)
        if not return_attention:
            return mu, logvar
        score = pool.score(dense).squeeze(-1)
        score = score.masked_fill(~node_mask, torch.finfo(score.dtype).min)
        attn = torch.softmax(score, dim=-1)
        return mu, logvar, attn, node_mask

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_states(batch)
        desc = getattr(batch, "desc", None)
        return self.readout(h, batch.batch, desc)

    def head_features(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared head representation z (input of both output layers) and mu."""
        h = self.node_states(batch)
        dense, node_mask = to_dense_batch(h, batch.batch)
        graph_repr = self.base.fusion(self.base.pool(dense, node_mask), getattr(batch, "desc", None))
        z = self.trunk(graph_repr)
        return z, self.mean_out(z).view(-1)

    def logvar_from_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.logvar_out(z).view(-1).clamp(self.logvar_min, self.logvar_max)


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample Gaussian NLL without the constant 0.5*log(2*pi)."""
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))


def beta_nll_loss(
    mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """beta-NLL (Seitzer et al., 2022): NLL weighted by stop-grad(sigma^(2*beta))."""
    nll = gaussian_nll_loss(mu, logvar, y)
    if beta <= 0:
        return nll
    weight = torch.exp(logvar.detach() * beta)
    return weight * nll
