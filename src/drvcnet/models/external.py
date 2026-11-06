"""Common-protocol causal adaptations of MMGCN, SDT, and CSS."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from drvcnet.config import ExperimentConfig
from drvcnet.constants import ModelVariant
from drvcnet.models.context import MaskedProjection
from drvcnet.models.drvc_net import ModelOutput
from drvcnet.models.visual_reference import VisualReferenceModule, VisualReferenceOutput


def _target(values: Tensor, batch: dict[str, Any]) -> Tensor:
    batch_index = torch.arange(values.shape[0], device=values.device)
    return values[batch_index, batch["target_indices"].long()]


def _causal_attention_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class _ExternalBase(nn.Module):
    """Shared frozen-feature adapters and static four-region visual input."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.experiment_config = config
        self.model_config = config.model
        h = config.model.hidden_dim
        static_config = replace(config.model, variant=ModelVariant.B2)
        self.text_projection = MaskedProjection(config.model.text_input_dim, h, config.model.dropout)
        self.audio_projection = MaskedProjection(config.model.audio_input_dim, h, config.model.dropout)
        self.visual_module = VisualReferenceModule(
            static_config, epsilon=config.video.quality_epsilon
        )
        self.role_embedding = nn.Embedding(config.model.max_roles + 1, h)

    def _inputs(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, VisualReferenceOutput]:
        valid = batch["attention_mask"].bool()
        text_mask = batch["text_available"].bool() & valid
        audio_mask = batch["audio_available"].bool() & valid
        text = self.text_projection(batch["text"].float(), text_mask)
        audio = self.audio_projection(batch["audio"].float(), audio_mask)
        visual_output = self.visual_module(batch)
        visual = visual_output.visual
        roles = batch["role"].long().clamp(max=self.model_config.max_roles)
        role = self.role_embedding(roles)
        return text, audio, visual, role, visual_output

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class _GCNIILayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self, values: Tensor, initial: Tensor, adjacency: Tensor, theta: float, alpha: float
    ) -> Tensor:
        propagated = torch.bmm(adjacency, values)
        support = torch.cat([propagated, initial], dim=-1)
        residual = (1.0 - alpha) * propagated + alpha * initial
        return theta * self.linear(support) + (1.0 - theta) * residual


class MMGCNBaseline(_ExternalBase):
    """Dense causal multimodal GCNII over shared text, audio, and visual features."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        h = config.model.hidden_dim
        self.layers = nn.ModuleList(
            [_GCNIILayer(h) for _ in range(config.model.mmgcn_layers)]
        )
        self.dropout = nn.Dropout(config.model.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(config.model.dropout)
        )
        self.classifier = nn.Linear(h, len(config.dataset.labels))

    @staticmethod
    def _similarity(first: Tensor, second: Tensor) -> Tensor:
        cosine = (first * second).sum(dim=-1).clamp(-0.99999, 0.99999)
        return 1.0 - torch.acos(cosine) / math.pi

    def _adjacency(self, modalities: Tensor, valid: Tensor) -> Tensor:
        batch_size, modality_count, length, _hidden = modalities.shape
        normalized = torch.nn.functional.normalize(modalities, dim=-1, eps=1.0e-8)
        node_count = modality_count * length
        adjacency = modalities.new_zeros(batch_size, node_count, node_count)
        causal = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=modalities.device)
        )
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2) & causal
        for modality in range(modality_count):
            start = modality * length
            similarity = torch.bmm(
                normalized[:, modality], normalized[:, modality].transpose(1, 2)
            ).clamp(-0.99999, 0.99999)
            similarity = 1.0 - torch.acos(similarity) / math.pi
            adjacency[:, start : start + length, start : start + length] = (
                similarity * pair_valid.to(similarity.dtype)
            )
        for target_modality in range(modality_count):
            target_start = target_modality * length
            for source_modality in range(modality_count):
                if target_modality == source_modality:
                    continue
                source_start = source_modality * length
                similarity = self._similarity(
                    normalized[:, target_modality], normalized[:, source_modality]
                )
                diagonal = torch.diag_embed(similarity * valid.to(similarity.dtype))
                adjacency[
                    :, target_start : target_start + length, source_start : source_start + length
                ] = diagonal
        node_valid = valid.repeat(1, modality_count)
        adjacency = adjacency * (
            node_valid.unsqueeze(-1) & node_valid.unsqueeze(-2)
        ).to(adjacency.dtype)
        adjacency = adjacency + torch.diag_embed(node_valid.to(adjacency.dtype))
        degree = adjacency.sum(dim=-1).clamp_min(1.0e-8).pow(-0.5)
        return degree.unsqueeze(-1) * adjacency * degree.unsqueeze(-2)

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        text, audio, visual, role, visual_output = self._inputs(batch)
        text = text + role
        valid = batch["attention_mask"].bool()
        modalities = torch.stack([audio, visual, text], dim=1)
        batch_size, modality_count, length, hidden = modalities.shape
        adjacency = self._adjacency(modalities, valid)
        initial = modalities.reshape(batch_size, modality_count * length, hidden)
        values = initial
        for layer_index, layer in enumerate(self.layers, start=1):
            theta = math.log(self.model_config.mmgcn_lambda / layer_index + 1.0)
            values = layer(
                self.dropout(values),
                initial,
                adjacency,
                theta,
                self.model_config.mmgcn_alpha,
            )
            values = torch.relu(values)
        separated = values.reshape(batch_size, modality_count, length, hidden)
        context = self.fusion(torch.cat(list(separated.unbind(dim=1)), dim=-1))
        logits = self.classifier(_target(context, batch))
        return ModelOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            visual=visual_output,
            context=context,
            text=text,
            audio=audio,
            shift_logits=None,
        )


class _CausalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, query: Tensor, source: Tensor, valid: Tensor) -> Tensor:
        length = query.shape[1]
        attended, _weights = self.attention(
            query,
            source,
            source,
            attn_mask=_causal_attention_mask(length, query.device),
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = self.norm1(query + attended)
        values = self.norm2(values + self.feedforward(values))
        return values * valid.unsqueeze(-1).to(values.dtype)


class SDTBaseline(_ExternalBase):
    """Causal intra/inter-modal Transformer with self-distillation heads."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        h = config.model.hidden_dim
        names = ("text", "audio", "visual")
        self.blocks = nn.ModuleDict(
            {
                f"{target}_{source}": _CausalAttentionBlock(
                    h, config.model.context_heads, config.model.dropout
                )
                for target in names
                for source in names
            }
        )
        self.gates = nn.ModuleDict(
            {f"{target}_{source}": nn.Linear(h, h, bias=False) for target in names for source in names}
        )
        self.reducers = nn.ModuleDict(
            {name: nn.Linear(3 * h, h) for name in names}
        )
        self.modality_gate = nn.Linear(h, h, bias=False)
        self.unimodal_heads = nn.ModuleDict(
            {name: nn.Linear(h, len(config.dataset.labels)) for name in names}
        )
        self.classifier = nn.Linear(h, len(config.dataset.labels))

    def _encode(self, features: dict[str, Tensor], valid: Tensor) -> dict[str, Tensor]:
        encoded: dict[str, Tensor] = {}
        for target in features:
            contributions = []
            for source in features:
                value = self.blocks[f"{target}_{source}"](
                    features[target], features[source], valid
                )
                contributions.append(torch.sigmoid(self.gates[f"{target}_{source}"](value)) * value)
            encoded[target] = self.reducers[target](torch.cat(contributions, dim=-1))
        return encoded

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        text, audio, visual, role, visual_output = self._inputs(batch)
        valid = batch["attention_mask"].bool()
        features = {"text": text + role, "audio": audio + role, "visual": visual + role}
        encoded = self._encode(features, valid)
        stacked = torch.stack([encoded[name] for name in ("text", "audio", "visual")], dim=-2)
        weights = torch.softmax(self.modality_gate(stacked), dim=-2)
        fused = (weights * stacked).sum(dim=-2)
        target_fused = _target(fused, batch)
        logits = self.classifier(target_fused)
        auxiliary = tuple(
            self.unimodal_heads[name](_target(encoded[name], batch))
            for name in ("text", "audio", "visual")
        )
        return ModelOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            visual=visual_output,
            context=fused,
            text=text,
            audio=audio,
            shift_logits=None,
            auxiliary_logits=auxiliary,
        )


class _CrossSpaceBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key_first = nn.Linear(hidden_dim, hidden_dim)
        self.key_second = nn.Linear(hidden_dim, hidden_dim)
        self.value_first = nn.Linear(hidden_dim, hidden_dim)
        self.value_second = nn.Linear(hidden_dim, hidden_dim)
        self.key_gate = nn.Linear(hidden_dim, 1)
        self.value_gate = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _heads(self, values: Tensor) -> Tensor:
        batch_size, length, _hidden = values.shape
        return values.reshape(batch_size, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, first: Tensor, second: Tensor, valid: Tensor) -> Tensor:
        key_gate = torch.sigmoid(self.key_gate(query))
        value_gate = torch.sigmoid(self.value_gate(query))
        keys = key_gate * self.key_first(first) + (1.0 - key_gate) * self.key_second(second)
        values = value_gate * self.value_first(first) + (
            1.0 - value_gate
        ) * self.value_second(second)
        queries = self._heads(self.query(query))
        keys = self._heads(keys)
        values = self._heads(values)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        length = query.shape[1]
        causal = _causal_attention_mask(length, query.device).view(1, 1, length, length)
        padding = (~valid).view(valid.shape[0], 1, 1, length)
        scores = scores.masked_fill(causal | padding, float("-inf"))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, values).transpose(1, 2).contiguous()
        output = output.reshape(query.shape[0], length, -1)
        return self.norm(query + self.output(output)) * valid.unsqueeze(-1).to(query.dtype)


class _PolynomialFusion(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, order: int) -> None:
        super().__init__()
        self.order = order
        self.text_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, rank)) for _ in range(order)]
        )
        self.audio_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, rank)) for _ in range(order)]
        )
        self.visual_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, rank)) for _ in range(order)]
        )
        self.constant_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(1, rank)) for _ in range(order)]
        )
        self.modality_gates = nn.Parameter(torch.ones(order, 3))
        self.output = nn.Parameter(torch.empty(rank, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.residual = nn.Linear(3 * hidden_dim, hidden_dim)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_normal_(parameter)

    def forward(self, text: Tensor, audio: Tensor, visual: Tensor) -> Tensor:
        batch_size, length, _hidden = text.shape
        rank = self.output.shape[0]
        product = text.new_ones(batch_size, length, rank)
        for index in range(self.order):
            gates = torch.sigmoid(self.modality_gates[index])
            constant = self.constant_weights[index].view(1, 1, rank)
            factor = (
                gates[0] * torch.nn.functional.gelu(text @ self.text_weights[index])
                + gates[1] * torch.nn.functional.gelu(audio @ self.audio_weights[index])
                + gates[2] * torch.nn.functional.gelu(visual @ self.visual_weights[index])
                + constant
            )
            product = product * factor
        normalized = torch.sign(product) * torch.sqrt(product.abs() + 1.0e-6)
        return normalized @ self.output + self.bias + self.residual(
            torch.cat([text, audio, visual], dim=-1)
        )


class CSSBaseline(_ExternalBase):
    """Causal cross-space synergy model with rank-16 polynomial fusion."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        h = config.model.hidden_dim
        self.self_blocks = nn.ModuleDict(
            {
                name: _CausalAttentionBlock(h, config.model.context_heads, config.model.dropout)
                for name in ("text", "audio", "visual")
            }
        )
        self.cross_blocks = nn.ModuleDict(
            {
                name: _CrossSpaceBlock(h, config.model.context_heads, config.model.dropout)
                for name in ("text", "audio", "visual")
            }
        )
        self.self_gates = nn.ModuleDict(
            {name: nn.Linear(h, h, bias=False) for name in ("text", "audio", "visual")}
        )
        self.cross_gates = nn.ModuleDict(
            {name: nn.Linear(h, h, bias=False) for name in ("text", "audio", "visual")}
        )
        self.reducers = nn.ModuleDict(
            {name: nn.Linear(2 * h, h) for name in ("text", "audio", "visual")}
        )
        self.fusion = _PolynomialFusion(
            h, config.model.css_rank, config.model.css_polynomial_order
        )
        self.unimodal_heads = nn.ModuleDict(
            {name: nn.Linear(h, len(config.dataset.labels)) for name in ("text", "audio", "visual")}
        )
        self.classifier = nn.Linear(h, len(config.dataset.labels))

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.text_projection,
            self.audio_projection,
            self.visual_module,
            self.self_blocks,
            self.cross_blocks,
            self.self_gates,
            self.cross_gates,
            self.reducers,
        )
        for module in modules:
            yield from module.parameters()

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        text, audio, visual, role, visual_output = self._inputs(batch)
        valid = batch["attention_mask"].bool()
        features = {"text": text + role, "audio": audio + role, "visual": visual + role}
        order = ("text", "audio", "visual")
        encoded: dict[str, Tensor] = {}
        for index, name in enumerate(order):
            other_first = features[order[(index + 1) % 3]]
            other_second = features[order[(index + 2) % 3]]
            self_value = self.self_blocks[name](features[name], features[name], valid)
            cross_value = self.cross_blocks[name](
                features[name], other_first, other_second, valid
            )
            self_value = torch.sigmoid(self.self_gates[name](self_value)) * self_value
            cross_value = torch.sigmoid(self.cross_gates[name](cross_value)) * cross_value
            encoded[name] = self.reducers[name](torch.cat([self_value, cross_value], dim=-1))
        fused = self.fusion(encoded["text"], encoded["audio"], encoded["visual"])
        logits = self.classifier(_target(fused, batch))
        auxiliary = tuple(
            self.unimodal_heads[name](_target(encoded[name], batch)) for name in order
        )
        return ModelOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            visual=visual_output,
            context=fused,
            text=text,
            audio=audio,
            shift_logits=None,
            auxiliary_logits=auxiliary,
        )
