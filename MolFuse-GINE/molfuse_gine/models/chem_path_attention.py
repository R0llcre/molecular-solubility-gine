from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfuse_gine.features.chem_path import CHEM_PATH_FIELDS


_FIELD_INDEX = {name: i for i, name in enumerate(CHEM_PATH_FIELDS)}


class ChemPathAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        path_dim: int,
        dropout: float,
        max_path_distance: int,
        mode: str = "chem",
        descriptor_dim: int = 0,
        descriptor_modulation: str = "none",
        sparse_attention: bool = False,
        sparse_max_distance: int = 4,
        residual_scale_init: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        if mode not in {"none", "spd", "chem", "chem_no_dist"}:
            raise ValueError("mode must be 'none', 'spd', 'chem' or 'chem_no_dist'")
        if descriptor_modulation not in {"none", "group_film"}:
            raise ValueError("descriptor_modulation must be 'none' or 'group_film'")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.path_dim = 0 if mode == "none" else (1 if mode == "spd" else path_dim)
        self.max_path_distance = max_path_distance
        self.mode = mode
        self.descriptor_modulation = descriptor_modulation
        self.sparse_attention = bool(sparse_attention)
        self.sparse_max_distance = int(sparse_max_distance)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        if mode == "spd":
            self.dist_emb = nn.Embedding(max_path_distance + 2, heads)
            self.count_proj = None
            self.relation_emb = None
            self.path_count_proj = None
            self.path_scale = nn.Parameter(torch.tensor(1e-2))
        elif mode in {"chem", "chem_no_dist"}:
            self.dist_emb = nn.Embedding(max_path_distance + 2, heads)
            self.count_proj = nn.Sequential(
                nn.Linear(16, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, heads),
            )
            self.relation_emb = nn.Embedding(16, heads)
            self.path_count_proj = nn.Linear(1, heads)
            self.path_scale = nn.Parameter(torch.tensor(1e-2))
        else:
            self.dist_emb = None
            self.count_proj = None
            self.relation_emb = None
            self.path_count_proj = None
            self.path_scale = None
        if descriptor_modulation == "group_film":
            if descriptor_dim <= 0:
                raise ValueError("group_film descriptor modulation requires descriptor_dim > 0")
            self.desc_film = nn.Linear(descriptor_dim, heads * 5)
            nn.init.zeros_(self.desc_film.weight)
            nn.init.zeros_(self.desc_film.bias)
        else:
            self.desc_film = None
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.ffn_res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _distance_ids(self, path_attr: torch.Tensor) -> torch.Tensor:
        d = path_attr[..., _FIELD_INDEX["distance"]].round().long()
        return d.clamp(min=0, max=self.max_path_distance + 1)

    def _normalised_counts(self, path_attr: torch.Tensor) -> torch.Tensor:
        scale = max(1.0, float(self.max_path_distance))
        count_names = [
            "single_count",
            "double_count",
            "triple_count",
            "aromatic_count",
            "conjugated_count",
            "ring_bond_count",
            "hetero_atom_count",
            "aromatic_atom_count",
            "single_count_max",
            "double_count_max",
            "triple_count_max",
            "aromatic_count_max",
            "conjugated_count_max",
            "ring_bond_count_max",
            "hetero_atom_count_max",
            "aromatic_atom_count_max",
        ]
        counts = torch.stack([path_attr[..., _FIELD_INDEX[name]] for name in count_names], dim=-1)
        return counts.clamp(min=0, max=scale) / scale

    def _relation_ids(self, path_attr: torch.Tensor) -> torch.Tensor:
        donor = path_attr[..., _FIELD_INDEX["donor_acceptor_touch"]] > 0.5
        same_ring = path_attr[..., _FIELD_INDEX["same_ring"]] > 0.5
        motif = path_attr[..., _FIELD_INDEX["fg_touch"]] > 0.5
        disconnected = path_attr[..., _FIELD_INDEX["disconnected"]] > 0.5
        return (
            donor.long()
            + 2 * same_ring.long()
            + 4 * motif.long()
            + 8 * disconnected.long()
        )

    def _path_bias_components(self, path_attr: torch.Tensor) -> list[torch.Tensor] | None:
        if self.mode == "none":
            return None
        dist_bias = self.dist_emb(self._distance_ids(path_attr))
        if self.mode == "spd":
            return [dist_bias]
        count_bias = self.count_proj(self._normalised_counts(path_attr))
        relation_bias = self.relation_emb(self._relation_ids(path_attr))
        path_count = path_attr[..., _FIELD_INDEX["num_shortest_paths"]].unsqueeze(-1)
        path_count = path_count.clamp(min=0, max=math.log1p(64.0)) / math.log1p(64.0)
        path_count_bias = self.path_count_proj(path_count)
        if self.mode == "chem_no_dist":
            dist_bias = torch.zeros_like(count_bias)
        return [dist_bias, count_bias, relation_bias, path_count_bias]

    def _path_bias(
        self,
        path_attr: torch.Tensor,
        desc: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        components = self._path_bias_components(path_attr)
        if components is None:
            return None
        if len(components) == 1:
            return components[0]
        if self.desc_film is None or desc is None:
            return sum(components)

        params = self.desc_film(desc).view(desc.size(0), self.heads, 5)
        weights = 1.0 + 0.2 * torch.tanh(params[..., :4])
        beta = 0.2 * torch.tanh(params[..., 4])
        bias = sum(
            comp * weights[:, None, None, :, i]
            for i, comp in enumerate(components)
        )
        return bias + beta[:, None, None, :]

    def diagnostics(self, prefix: str = "") -> dict[str, float | str]:
        out: dict[str, float | str] = {
            f"{prefix}attention_mode": self.mode,
            f"{prefix}descriptor_modulation": self.descriptor_modulation,
            f"{prefix}sparse_attention": str(bool(self.sparse_attention)),
            f"{prefix}attn_res_scale": float(self.attn_res_scale.detach().cpu()),
            f"{prefix}ffn_res_scale": float(self.ffn_res_scale.detach().cpu()),
        }
        if self.path_scale is not None:
            out[f"{prefix}path_scale_raw"] = float(self.path_scale.detach().cpu())
            out[f"{prefix}path_scale_tanh"] = float(torch.tanh(self.path_scale).detach().cpu())
        return out

    @torch.no_grad()
    def path_usage_stats(
        self,
        path_attr: torch.Tensor,
        node_mask: torch.Tensor,
        desc: torch.Tensor | None = None,
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        path_bias = self._path_bias(path_attr, desc=desc)
        if path_bias is not None:
            scaled = torch.tanh(self.path_scale) * path_bias
            valid_pair = node_mask[:, :, None] & node_mask[:, None, :]
            valid_bias = scaled[valid_pair]
            if valid_bias.numel() > 0:
                stats.update(
                    path_bias_abs_mean=float(valid_bias.abs().mean().detach().cpu()),
                    path_bias_std=float(valid_bias.std(unbiased=False).detach().cpu()),
                )
        if self.sparse_attention and self.mode != "none":
            mask = self._sparse_mask(path_attr, node_mask)
            valid_pair = node_mask[:, :, None] & node_mask[:, None, :]
            denom = valid_pair.sum().clamp(min=1)
            stats["sparse_pair_ratio"] = float((mask & valid_pair).sum().detach().cpu() / denom.detach().cpu())
        return stats

    def _sparse_mask(self, path_attr: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        distance = path_attr[..., _FIELD_INDEX["distance"]]
        disconnected = path_attr[..., _FIELD_INDEX["disconnected"]] > 0.5
        local = distance <= float(self.sparse_max_distance)
        self_pair = distance <= 0.5
        same_ring = path_attr[..., _FIELD_INDEX["same_ring"]] > 0.5
        donor = path_attr[..., _FIELD_INDEX["donor_acceptor_touch"]] > 0.5
        motif = path_attr[..., _FIELD_INDEX["fg_touch"]] > 0.5
        aromatic = (
            path_attr[..., _FIELD_INDEX["aromatic_count"]] > 0.0
        ) | (
            path_attr[..., _FIELD_INDEX["aromatic_count_max"]] > 0.0
        )
        valid_pair = node_mask[:, :, None] & node_mask[:, None, :]
        allowed = self_pair | ((local | same_ring | donor | motif | aromatic) & ~disconnected)
        return allowed & valid_pair

    def forward(
        self,
        x: torch.Tensor,
        path_attr: torch.Tensor,
        node_mask: torch.Tensor,
        descriptor_gate: torch.Tensor | None = None,
        desc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, nodes, _ = x.shape
        residual = x
        x_norm = self.norm1(x)

        q = self.q(x_norm).view(bsz, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).view(bsz, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_norm).view(bsz, nodes, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        path_bias = self._path_bias(path_attr, desc=desc)
        if path_bias is not None:
            path_bias = path_bias.permute(0, 3, 1, 2)
            if descriptor_gate is not None:
                path_bias = path_bias * descriptor_gate[:, :, None, None]
            scores = scores + torch.tanh(self.path_scale) * path_bias

        if self.sparse_attention and self.mode != "none":
            pair_mask = self._sparse_mask(path_attr, node_mask)
            scores = scores.masked_fill(~pair_mask[:, None, :, :], torch.finfo(scores.dtype).min)

        key_mask = ~node_mask[:, None, None, :]
        scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, nodes, self.hidden_dim)
        out = self.out(out)
        out = residual + self.attn_res_scale * self.dropout(out)
        out = out.masked_fill(~node_mask[:, :, None], 0.0)

        out = out + self.ffn_res_scale * self.dropout(self.ffn(self.norm2(out)))
        out = out.masked_fill(~node_mask[:, :, None], 0.0)
        return out
