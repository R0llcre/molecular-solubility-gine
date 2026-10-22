from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from dataclasses import dataclass

from molfuse_gine.models.chem_path_attention import ChemPathAttentionBlock
from molfuse_gine.models.descriptor_fusion import DescriptorFusion, DescriptorPathGate
from molfuse_gine.models.graph_encoders import build_graph_encoder


class GraphPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        score = self.score(x).squeeze(-1)
        score = score.masked_fill(~node_mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1).unsqueeze(-1)
        attn_pool = (x * weight).sum(dim=1)

        denom = node_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_pool = (x * node_mask[:, :, None]).sum(dim=1) / denom

        max_x = x.masked_fill(~node_mask[:, :, None], torch.finfo(x.dtype).min)
        max_pool = max_x.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.out(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))


class DescriptorOnlyMLP(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError("desc_mlp requires descriptor_dim > 0")
        self.net = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        return self.net(desc).view(-1)


@dataclass(frozen=True)
class VariantConfig:
    encoder_type: str = "gine"
    use_attention: bool = False
    path_mode: str = "none"
    descriptor_fusion: bool = False
    descriptor_fusion_mode: str = "none"
    descriptor_path_gate: bool = False
    descriptor_modulation: str = "none"
    sparse_attention: bool = False
    residual_scale_init: float = 1.0
    path_transform: str = "none"
    descriptor_dropout: float = 0.0
    desc_only: bool = False


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "desc_mlp": VariantConfig(desc_only=True),
    "gine": VariantConfig(),
    "gine_desc": VariantConfig(descriptor_fusion=True, descriptor_fusion_mode="gated_add"),
    "gine_desc_add_gate": VariantConfig(descriptor_fusion=True, descriptor_fusion_mode="gated_add"),
    "gine_desc_concat": VariantConfig(descriptor_fusion=True, descriptor_fusion_mode="concat"),
    "gine_desc_residual_add": VariantConfig(descriptor_fusion=True, descriptor_fusion_mode="add"),
    "gine_desc_film_graph": VariantConfig(descriptor_fusion=True, descriptor_fusion_mode="film_graph"),
    "gine_desc_lowrank_bilinear": VariantConfig(
        descriptor_fusion=True,
        descriptor_fusion_mode="lowrank_bilinear",
    ),
    "gine_desc_ddrop_005": VariantConfig(
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_dropout=0.05,
    ),
    "gine_desc_ddrop_010": VariantConfig(
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_dropout=0.10,
    ),
    "gine_desc_ddrop_020": VariantConfig(
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_dropout=0.20,
    ),
    "pna_desc": VariantConfig(
        encoder_type="pna",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "pna_desc_concat": VariantConfig(
        encoder_type="pna",
        descriptor_fusion=True,
        descriptor_fusion_mode="concat",
    ),
    "dmpnn_desc": VariantConfig(
        encoder_type="dmpnn",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "dmpnn_desc_concat": VariantConfig(
        encoder_type="dmpnn",
        descriptor_fusion=True,
        descriptor_fusion_mode="concat",
    ),
    "attentivefp_desc": VariantConfig(
        encoder_type="attentivefp",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "attentivefp_desc_concat": VariantConfig(
        encoder_type="attentivefp",
        descriptor_fusion=True,
        descriptor_fusion_mode="concat",
    ),
    "gine_attn": VariantConfig(use_attention=True),
    "gine_attn_desc": VariantConfig(
        use_attention=True,
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "gine_spd": VariantConfig(use_attention=True, path_mode="spd"),
    "molfuse_gine": VariantConfig(use_attention=True, path_mode="chem"),
    "chempath_desc": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
    ),
    "chempath_desc_full": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
    ),
    "chempath_desc_fusion": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "chempath_desc_gate": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_path_gate=True,
    ),
    "chempath_desc_zero_path": VariantConfig(
        use_attention=True,
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
    ),
    "chempath_desc_zero_attr_same_params": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
        path_transform="zero_attr",
    ),
    "chempath_desc_shuffle_path": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
        path_transform="shuffle",
    ),
    "chempath_desc_shuffle_bucket": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
        path_transform="shuffle_bucket",
    ),
    "chempath_desc_distance_only": VariantConfig(
        use_attention=True,
        path_mode="spd",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
    ),
    "chempath_desc_chem_only": VariantConfig(
        use_attention=True,
        path_mode="chem_no_dist",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_path_gate=True,
    ),
    "chempath_desc_film": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_modulation="group_film",
    ),
    "chempath_desc_film_sparse": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_modulation="group_film",
        sparse_attention=True,
    ),
    "chempath_desc_film_sparse_ls": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_modulation="group_film",
        sparse_attention=True,
        residual_scale_init=1e-2,
    ),
    "chempath_desc_film_sparse_ls_ddrop": VariantConfig(
        use_attention=True,
        path_mode="chem",
        descriptor_fusion=True,
        descriptor_fusion_mode="gated_add",
        descriptor_modulation="group_film",
        sparse_attention=True,
        residual_scale_init=1e-2,
        descriptor_dropout=0.15,
    ),
}


class ChemPathGINE(nn.Module):
    VARIANTS = set(VARIANT_CONFIGS)

    @classmethod
    def variant_uses_real_path(cls, variant: str) -> bool:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant={variant!r}. Options: {sorted(VARIANT_CONFIGS)}")
        return VARIANT_CONFIGS[variant].path_mode in {
            "spd",
            "chem",
            "chem_no_dist",
            "distance_only",
            "chem_only",
            "shuffle",
            "shuffle_bucket",
        }

    def __init__(
        self,
        hidden_dim: int,
        num_gnn_layers: int,
        num_attention_layers: int,
        heads: int,
        dropout: float,
        path_dim: int,
        descriptor_dim: int,
        max_path_distance: int,
        variant: str = "molfuse_gine",
        use_descriptor_fusion: bool = False,
        use_descriptor_path_gate: bool = False,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant={variant!r}; choose one of {sorted(self.VARIANTS)}")

        self.variant = variant
        variant_cfg = VARIANT_CONFIGS[variant]
        self.variant_config = variant_cfg
        self.desc_only = variant_cfg.desc_only
        self.path_transform = variant_cfg.path_transform
        self.descriptor_dropout = variant_cfg.descriptor_dropout
        if self.desc_only:
            self.desc_mlp = DescriptorOnlyMLP(descriptor_dim, hidden_dim, dropout)
            self.encoder = None
            self.pool = None
            self.path_gate = None
            self.attention = nn.ModuleList()
            self.fusion = None
            self.head = None
            return

        self.encoder = build_graph_encoder(
            variant_cfg.encoder_type,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            heads=heads,
            dropout=dropout,
        )
        self.pool = GraphPool(hidden_dim)
        descriptor_fusion = use_descriptor_fusion or variant_cfg.descriptor_fusion
        descriptor_path_gate = use_descriptor_path_gate or variant_cfg.descriptor_path_gate
        if descriptor_fusion and descriptor_dim <= 0:
            raise ValueError(
                f"variant={variant!r} requires descriptor_dim > 0 for descriptor fusion"
            )
        if descriptor_path_gate and descriptor_dim <= 0:
            raise ValueError(
                f"variant={variant!r} requires descriptor_dim > 0 for descriptor path gating"
            )
        self.path_gate = (
            DescriptorPathGate(descriptor_dim, heads) if descriptor_path_gate else None
        )
        self.attention = nn.ModuleList()
        if variant_cfg.use_attention:
            for _ in range(num_attention_layers):
                self.attention.append(
                    ChemPathAttentionBlock(
                        hidden_dim=hidden_dim,
                        heads=heads,
                        path_dim=path_dim,
                        dropout=dropout,
                        max_path_distance=max_path_distance,
                        mode=variant_cfg.path_mode,
                        descriptor_dim=descriptor_dim,
                        descriptor_modulation=variant_cfg.descriptor_modulation,
                        sparse_attention=variant_cfg.sparse_attention,
                        sparse_max_distance=4,
                        residual_scale_init=variant_cfg.residual_scale_init,
                    )
                )

        fusion_mode = variant_cfg.descriptor_fusion_mode if descriptor_fusion else "none"
        if descriptor_fusion and fusion_mode == "none":
            fusion_mode = "gated_add"
        self.fusion = DescriptorFusion(
            hidden_dim,
            descriptor_dim if descriptor_fusion else 0,
            mode=fusion_mode,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _transform_path_attr(
        self,
        path_attr: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.path_transform == "none":
            return path_attr
        if self.path_transform == "zero_attr":
            return torch.zeros_like(path_attr)
        if path_attr.size(0) <= 1:
            return path_attr
        if self.path_transform == "shuffle":
            return torch.roll(path_attr, shifts=1, dims=0)
        if self.path_transform == "shuffle_bucket":
            out = path_attr.clone()
            node_counts = node_mask.sum(dim=1)
            for count in torch.unique(node_counts):
                idx = (node_counts == count).nonzero(as_tuple=False).view(-1)
                if idx.numel() > 1:
                    out[idx] = out[idx.roll(1)]
            return out
        raise ValueError(f"Unknown path_transform={self.path_transform!r}")

    @torch.no_grad()
    def diagnostics(self, batch=None) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "variant": self.variant,
            "path_transform": self.path_transform,
            "descriptor_dropout": float(self.descriptor_dropout),
        }
        if self.desc_only:
            return out
        out.update(self.fusion.diagnostics(getattr(batch, "desc", None) if batch is not None else None))
        for i, layer in enumerate(self.attention):
            out.update(layer.diagnostics(prefix=f"attention_{i}_"))
        if batch is not None and self.attention:
            desc = getattr(batch, "desc", None)
            dense, node_mask = to_dense_batch(self.encoder(batch), batch.batch)
            if hasattr(batch, "node_mask"):
                node_mask = batch.node_mask.to(device=dense.device)
            path_attr = self._transform_path_attr(batch.path_attr.to(dense.device), node_mask)
            if desc is not None:
                desc = desc.to(dense.device)
            for i, layer in enumerate(self.attention):
                stats = layer.path_usage_stats(path_attr, node_mask, desc=desc)
                for key, value in stats.items():
                    out[f"attention_{i}_{key}"] = value
        return out

    def forward(self, batch) -> torch.Tensor:
        desc = getattr(batch, "desc", None)
        if self.desc_only:
            if desc is None:
                raise ValueError("desc_mlp requires batch.desc")
            desc = desc.to(next(self.desc_mlp.parameters()).device)
            return self.desc_mlp(desc)

        h = self.encoder(batch)
        dense, node_mask = to_dense_batch(h, batch.batch)
        if hasattr(batch, "node_mask"):
            node_mask = batch.node_mask.to(device=dense.device)

        if desc is not None:
            desc = desc.to(dense.device)
            if self.training and self.descriptor_dropout > 0.0:
                desc = torch.nn.functional.dropout(desc, p=self.descriptor_dropout, training=True)

        if self.attention:
            path_gate = self.path_gate(desc) if self.path_gate is not None else None
            path_attr = self._transform_path_attr(batch.path_attr.to(dense.device), node_mask)
            for layer in self.attention:
                dense = layer(
                    dense,
                    path_attr,
                    node_mask,
                    descriptor_gate=path_gate,
                    desc=desc,
                )

        graph_repr = self.pool(dense, node_mask)
        graph_repr = self.fusion(graph_repr, desc)
        return self.head(graph_repr).view(-1)
