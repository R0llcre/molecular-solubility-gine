from __future__ import annotations

import torch
import torch.nn as nn


class DescriptorFusion(nn.Module):
    def __init__(self, hidden_dim: int, descriptor_dim: int, mode: str = "gated_add"):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.mode = mode if descriptor_dim > 0 else "none"
        if self.mode not in {
            "none",
            "gated_add",
            "add",
            "concat",
            "film_graph",
            "lowrank_bilinear",
        }:
            raise ValueError(f"Unknown descriptor fusion mode={mode!r}")
        if descriptor_dim > 0:
            self.desc_proj = nn.Sequential(
                nn.Linear(descriptor_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate = nn.Sequential(nn.Linear(descriptor_dim, hidden_dim), nn.Sigmoid())
            self.concat_proj = nn.Sequential(
                nn.Linear(hidden_dim + descriptor_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.film = nn.Linear(descriptor_dim, hidden_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            rank = max(16, hidden_dim // 4)
            self.graph_lowrank = nn.Linear(hidden_dim, rank)
            self.desc_lowrank = nn.Linear(descriptor_dim, rank)
            self.lowrank_out = nn.Sequential(
                nn.Linear(rank, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.desc_proj = None
            self.gate = None
            self.concat_proj = None
            self.film = None
            self.graph_lowrank = None
            self.desc_lowrank = None
            self.lowrank_out = None

    def forward(self, graph_repr: torch.Tensor, desc: torch.Tensor | None) -> torch.Tensor:
        if self.descriptor_dim == 0 or desc is None:
            return graph_repr
        if self.mode == "gated_add":
            return graph_repr + self.gate(desc) * self.desc_proj(desc)
        if self.mode == "add":
            return graph_repr + self.desc_proj(desc)
        if self.mode == "concat":
            return self.concat_proj(torch.cat([graph_repr, desc], dim=-1))
        if self.mode == "film_graph":
            gamma, beta = self.film(desc).chunk(2, dim=-1)
            return graph_repr * (1.0 + 0.2 * torch.tanh(gamma)) + 0.2 * torch.tanh(beta)
        if self.mode == "lowrank_bilinear":
            interaction = self.graph_lowrank(graph_repr) * self.desc_lowrank(desc)
            return graph_repr + self.lowrank_out(interaction)
        return graph_repr

    def diagnostics(self, desc: torch.Tensor | None = None) -> dict[str, float | str]:
        out: dict[str, float | str] = {"descriptor_fusion_mode": self.mode}
        if self.descriptor_dim == 0:
            return out
        if self.mode == "gated_add" and desc is not None:
            with torch.no_grad():
                gate = self.gate(desc)
                out.update(
                    descriptor_gate_mean=float(gate.mean().detach().cpu()),
                    descriptor_gate_std=float(gate.std(unbiased=False).detach().cpu()),
                    descriptor_gate_min=float(gate.min().detach().cpu()),
                    descriptor_gate_max=float(gate.max().detach().cpu()),
                )
        return out


class DescriptorPathGate(nn.Module):
    def __init__(self, descriptor_dim: int, heads: int):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        if descriptor_dim > 0:
            self.net = nn.Linear(descriptor_dim, heads)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = None

    def forward(self, desc: torch.Tensor | None):
        if self.net is None or desc is None:
            return None
        return 1.0 + 0.1 * torch.tanh(self.net(desc))
