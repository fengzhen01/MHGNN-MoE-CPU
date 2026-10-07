"""Pure-PyTorch MHGNN baseline and residual structure-aware MoE."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MeanGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, last: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.last = last

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.sparse.mm(adjacency, x)
        x = self.linear(torch.cat((x, neighbors), dim=1))
        x = self.norm(x)
        if not self.last:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HypergraphLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, last: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout
        self.last = last

    @staticmethod
    def vertex_to_edge(
        x: torch.Tensor, incidence_edge: torch.Tensor, incidence_vertex: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        edge_x = x.new_zeros((num_edges, x.shape[1]))
        edge_x.index_add_(0, incidence_edge, x[incidence_vertex])
        count = x.new_zeros(num_edges)
        count.index_add_(0, incidence_edge, torch.ones_like(incidence_edge, dtype=x.dtype))
        return edge_x / count.clamp_min(1.0).unsqueeze(1)

    @staticmethod
    def edge_to_vertex(
        edge_x: torch.Tensor, incidence_edge: torch.Tensor, incidence_vertex: torch.Tensor,
        num_vertices: int,
    ) -> torch.Tensor:
        vertex_x = edge_x.new_zeros((num_vertices, edge_x.shape[1]))
        vertex_x.index_add_(0, incidence_vertex, edge_x[incidence_edge])
        count = edge_x.new_zeros(num_vertices)
        count.index_add_(0, incidence_vertex, torch.ones_like(incidence_vertex, dtype=edge_x.dtype))
        return vertex_x / count.clamp_min(1.0).unsqueeze(1)

    def forward(
        self, x: torch.Tensor, incidence_edge: torch.Tensor, incidence_vertex: torch.Tensor,
        num_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.linear(x)
        edge_x = self.vertex_to_edge(projected, incidence_edge, incidence_vertex, num_edges)
        vertex_x = self.edge_to_vertex(edge_x, incidence_edge, incidence_vertex, x.shape[0])
        vertex_x = self.norm(vertex_x)
        if not self.last:
            vertex_x = F.relu(vertex_x)
            vertex_x = F.dropout(vertex_x, p=self.dropout, training=self.training)
        return vertex_x, edge_x


class PairMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class MHGNNMoE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        ppi_layers: int, hyper_layers: int, dropout: float,
        num_herbs: int, num_hyperedges: int,
        num_experts: int = 3, router_temperature: float = 1.0,
        initial_residual_strength: float = 0.1,
    ):
        super().__init__()
        if ppi_layers < 1 or hyper_layers < 1:
            raise ValueError("ppi_layers and hyper_layers must be >= 1")
        if num_experts != 3:
            raise ValueError("This interpretable implementation uses exactly 3 experts")
        self.num_herbs = num_herbs
        self.num_hyperedges = num_hyperedges
        self.router_temperature = router_temperature
        self.moe_enabled = False

        graph_layers = []
        current = input_dim
        for index in range(ppi_layers):
            graph_layers.append(MeanGraphSAGELayer(current, hidden_dim, dropout, last=index == ppi_layers - 1))
            current = hidden_dim
        self.graph_layers = nn.ModuleList(graph_layers)

        hyper_modules = []
        current = hidden_dim
        for index in range(hyper_layers):
            out = output_dim if index == hyper_layers - 1 else hidden_dim
            hyper_modules.append(HypergraphLayer(current, out, dropout, last=index == hyper_layers - 1))
            current = out
        self.hyper_layers = nn.ModuleList(hyper_modules)
        self.low_projection = nn.Linear(hidden_dim, output_dim)

        self.base_head = PairMLP(output_dim * 2, output_dim, dropout)
        self.ppi_expert = PairMLP(output_dim * 2, output_dim, dropout)
        self.hyper_expert = PairMLP(output_dim * 2, output_dim, dropout)
        self.interaction_expert = PairMLP(output_dim * 6, output_dim, dropout)
        self.router = nn.Sequential(
            nn.Linear(output_dim * 4 + 4, output_dim), nn.ReLU(), nn.Linear(output_dim, num_experts)
        )
        strength = min(max(initial_residual_strength, 1e-4), 1 - 1e-4)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(strength / (1.0 - strength))))

    def encode(
        self, features: torch.Tensor, adjacency: torch.Tensor,
        incidence_edge: torch.Tensor, incidence_vertex: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = features
        for layer in self.graph_layers:
            x = layer(x, adjacency)
        low_edges = HypergraphLayer.vertex_to_edge(
            self.low_projection(x), incidence_edge, incidence_vertex, self.num_hyperedges
        )
        high_edges = None
        for layer in self.hyper_layers:
            x, high_edges = layer(x, incidence_edge, incidence_vertex, self.num_hyperedges)
        return low_edges, high_edges

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor,
        incidence_edge: torch.Tensor, incidence_vertex: torch.Tensor,
        edges: torch.Tensor, pair_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        low, high = self.encode(features, adjacency, incidence_edge, incidence_vertex)
        return self.forward_from_encoded(low, high, edges, pair_stats)

    def forward_from_encoded(
        self, low: torch.Tensor, high: torch.Tensor, edges: torch.Tensor,
        pair_stats: torch.Tensor, base_logits_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        herb = edges[:, 0]
        symptom = edges[:, 1] + self.num_herbs
        low_h, low_s = low[herb], low[symptom]
        high_h, high_s = high[herb], high[symptom]
        low_pair = torch.cat((low_h, low_s), dim=1)
        high_pair = torch.cat((high_h, high_s), dim=1)
        base_logits = self.base_head(high_pair) if base_logits_override is None else base_logits_override
        if not self.moe_enabled:
            return {
                "logits": base_logits,
                "base_logits": base_logits,
                "router_probs": base_logits.new_zeros((len(edges), 3)),
                "balance_loss": base_logits.new_zeros(()),
                "residual_strength": base_logits.new_zeros(()),
            }

        interaction = torch.cat(
            (low_h, low_s, high_h, high_s, high_h * high_s, torch.abs(high_h - high_s)), dim=1
        )
        expert_logits = torch.stack(
            (self.ppi_expert(low_pair), self.hyper_expert(high_pair), self.interaction_expert(interaction)), dim=1
        )
        router_input = torch.cat((low_pair, high_pair, pair_stats), dim=1)
        router_probs = F.softmax(self.router(router_input) / self.router_temperature, dim=1)
        delta = (router_probs * expert_logits).sum(dim=1)
        strength = torch.sigmoid(self.residual_logit)
        logits = base_logits + strength * delta
        usage = router_probs.mean(dim=0)
        target = torch.full_like(usage, 1.0 / usage.numel())
        balance_loss = usage.numel() * torch.mean((usage - target) ** 2)
        return {
            "logits": logits,
            "base_logits": base_logits,
            "router_probs": router_probs,
            "balance_loss": balance_loss,
            "residual_strength": strength,
        }

    def set_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        if stage == "pretrain":
            self.moe_enabled = False
            for module in (self.ppi_expert, self.hyper_expert, self.interaction_expert, self.router):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self.residual_logit.requires_grad = False
        elif stage == "moe":
            self.moe_enabled = True
            for module in (self.graph_layers, self.hyper_layers, self.low_projection, self.base_head):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        elif stage == "finetune":
            self.moe_enabled = True
        else:
            raise ValueError(f"Unknown stage: {stage}")
