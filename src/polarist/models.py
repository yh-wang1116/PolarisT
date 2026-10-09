from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class PPIGCN(nn.Module):
    """The step216 two-layer GCN over the frozen PPI/CollecTRI graph."""

    def __init__(self, init_emb: torch.Tensor, hidden_dim: int = 64) -> None:
        super().__init__()
        if init_emb.ndim != 2:
            raise ValueError("init_emb must be a two-dimensional tensor")
        num_genes, emb_dim = init_emb.shape
        self.emb = nn.Embedding(num_genes, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        with torch.no_grad():
            self.emb.weight.copy_(init_emb)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.emb.weight, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class CellEncoder(nn.Module):
    """Per-cell MLP from the step216 checkpoint."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 256,
        d_feat: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.net(x)
        if x.ndim != 3:
            raise ValueError("cell features must have shape [cells, features] or [batch, cells, features]")
        batch, cells, dims = x.shape
        return self.net(x.reshape(batch * cells, dims)).reshape(batch, cells, -1)


class AttentionAggregator(nn.Module):
    """Permutation-invariant attention pooling from the step216 checkpoint."""

    def __init__(
        self,
        d_feat: int,
        attn_hidden: int = 64,
        temperature: float = 1.0,
        use_original_feature: bool = True,
    ) -> None:
        super().__init__()
        self.phi = nn.Linear(d_feat, attn_hidden, bias=True)
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_hidden * 2, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )
        self.temperature = temperature
        self.use_original_feature = use_original_feature

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        single = features.ndim == 2
        if single:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("aggregator input must have shape [cells, features] or [batch, cells, features]")
        _, cells, _ = features.shape
        projected = torch.tanh(self.phi(features))
        global_feature = projected.mean(dim=1, keepdim=True).expand(-1, cells, -1)
        attention_input = torch.cat([projected, global_feature], dim=-1)
        scores = self.attn_mlp(attention_input).squeeze(-1) / self.temperature
        weights = torch.softmax(scores, dim=1)
        base = features if self.use_original_feature else projected
        pooled = (weights.unsqueeze(-1) * base).sum(dim=1)
        if single:
            return pooled.squeeze(0), weights.squeeze(0)
        return pooled, weights


class FusionCellSetClassifier(nn.Module):
    """Late-fusion cell-set classifier matching the formal step216 model."""

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        emb_dim: int,
        d_hidden: int = 256,
        d_feat: int = 128,
        attn_hidden: int = 64,
        conndim: int = 4,
        delta_dim: int = 64,
        dropout: float = 0.2,
        initial_gene_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del n_classes
        self.encoder = CellEncoder(d_in, d_hidden, d_feat, dropout)
        self.aggregator = AttentionAggregator(d_feat, attn_hidden)
        self.feat_norm = nn.LayerNorm(d_feat)
        self.dropout = nn.Dropout(dropout)
        self.fusion_mode = "late"
        self.lambda_delta = 0.65
        self.classifier_abs = nn.utils.weight_norm(nn.Linear(d_feat, conndim, bias=False))
        self.proemb_abs = nn.utils.weight_norm(nn.Linear(emb_dim, conndim, bias=False))
        self.classifier_delta = nn.utils.weight_norm(nn.Linear(d_feat, delta_dim, bias=False))
        self.proemb_delta = nn.Linear(emb_dim, delta_dim, bias=False)
        self.classifier_delta_resid = nn.utils.weight_norm(nn.Linear(d_feat, conndim, bias=False))
        self.gate = nn.Linear(d_feat * 2, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.gate.weight)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.delta_logit_scale = nn.Parameter(torch.tensor(1.0))
        if initial_gene_embeddings is None:
            initial_gene_embeddings = torch.zeros((1, emb_dim), dtype=torch.float32)
        self.gene_emb = nn.Parameter(initial_gene_embeddings)

    def encode_set(self, x_set: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x_set)
        pooled, _ = self.aggregator(encoded)
        return self.dropout(self.feat_norm(pooled))

    def _features(
        self,
        x_ctrl: torch.Tensor,
        x_pert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pert = self.encode_set(x_pert)
        h_ctrl = self.encode_set(x_ctrl)
        if h_pert.ndim == 1:
            h_pert = h_pert.unsqueeze(0)
        if h_ctrl.ndim == 1:
            h_ctrl = h_ctrl.unsqueeze(0)
        return h_pert, h_ctrl, h_pert - h_ctrl

    def query_proj(self, x_ctrl: torch.Tensor, x_pert: torch.Tensor) -> torch.Tensor:
        h_pert, _, delta_h = self._features(x_ctrl, x_pert)
        if self.fusion_mode == "residual":
            query = self.classifier_abs(h_pert) + self.lambda_delta * self.classifier_delta_resid(delta_h)
        elif self.fusion_mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([h_pert, delta_h], dim=-1)))
            query = (1.0 - gate) * self.classifier_abs(h_pert) + gate * self.classifier_delta_resid(delta_h)
        else:
            query = self.classifier_abs(h_pert)
        return F.normalize(query, dim=-1)

    def gene_proj(self, emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proemb_abs(emb), dim=-1)

    def score_abs(self, h_pert: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        q_abs = F.normalize(self.classifier_abs(h_pert), dim=-1)
        projected_gene = F.normalize(self.proemb_abs(emb), dim=-1)
        return self.logit_scale.exp() * (q_abs @ projected_gene.T)

    def score_delta(self, delta_h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        q_delta = F.normalize(self.classifier_delta(delta_h), dim=-1)
        projected_gene = F.normalize(self.proemb_delta(emb), dim=-1)
        return self.delta_logit_scale.exp() * (q_delta @ projected_gene.T)

    def score_from_query(self, query: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * (query @ self.gene_proj(emb).T)

    def forward(
        self,
        x_ctrl: torch.Tensor,
        x_pert: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h_pert, _, delta_h = self._features(x_ctrl, x_pert)
        if self.fusion_mode == "late":
            return self.score_abs(h_pert, emb) + self.lambda_delta * self.score_delta(delta_h, emb)
        return self.score_from_query(self.query_proj(x_ctrl, x_pert), emb)
