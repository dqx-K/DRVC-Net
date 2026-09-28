"""End-to-end trainable DRVC-Net head over frozen cached features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from drvcnet.config import ExperimentConfig
from drvcnet.constants import REGIONS, ModelVariant
from drvcnet.models.context import CausalContextEncoder, MaskedProjection
from drvcnet.models.visual_reference import VisualReferenceModule, VisualReferenceOutput


@dataclass
class ModelOutput:
    logits: Tensor
    probabilities: Tensor
    visual: VisualReferenceOutput
    context: Tensor
    text: Tensor
    audio: Tensor
    shift_logits: Tensor | None


class DRVCNet(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.experiment_config = config
        model = config.model
        self.model_config = model
        h = model.hidden_dim
        self.text_projection = MaskedProjection(model.text_input_dim, h, model.dropout)
        self.audio_projection = MaskedProjection(model.audio_input_dim, h, model.dropout)
        self.visual_module = VisualReferenceModule(model, epsilon=config.video.quality_epsilon)
        self.role_embedding = nn.Embedding(model.max_roles + 1, model.role_dim)
        metadata_dim = 2 + len(REGIONS) * 10
        fusion_dim = 3 * h + model.role_dim + metadata_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, h),
            nn.LayerNorm(h),
            nn.Dropout(model.dropout),
        )
        self.context = CausalContextEncoder(model)
        self.classifier = nn.Sequential(
            nn.Linear(4 * h, model.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(model.dropout),
            nn.Linear(model.classifier_hidden_dim, len(config.dataset.labels)),
        )
        self.shift_classifier = (
            nn.Sequential(
                nn.Linear(4 * h, model.classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(model.dropout),
                nn.Linear(model.classifier_hidden_dim, 2),
            )
            if model.shift_lambda > 0
            else None
        )

    def _modality_mask(self, name: str, source: Tensor) -> Tensor:
        if name in self.model_config.enabled_modalities:
            return source.bool()
        return torch.zeros_like(source, dtype=torch.bool)

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        text_mask = self._modality_mask("text", batch["text_available"])
        audio_mask = self._modality_mask("audio", batch["audio_available"])
        text = self.text_projection(batch["text"].float(), text_mask)
        audio = self.audio_projection(batch["audio"].float(), audio_mask)
        visual = self.visual_module(batch)

        if "visual" not in self.model_config.enabled_modalities:
            visual.visual = torch.zeros_like(visual.visual)
            visual.metadata = torch.zeros_like(visual.metadata)
        if self.model_config.variant is ModelVariant.B0:
            visual.visual = torch.zeros_like(visual.visual)
            visual.metadata = torch.zeros_like(visual.metadata)

        roles = batch["role"].long()
        overflow = torch.full_like(roles, self.model_config.max_roles)
        roles = torch.where(roles < self.model_config.max_roles, roles, overflow)
        role = self.role_embedding(roles)
        metadata = torch.cat(
            [
                text_mask.to(text.dtype).unsqueeze(-1),
                audio_mask.to(text.dtype).unsqueeze(-1),
                visual.metadata.to(text.dtype),
            ],
            dim=-1,
        )
        token = self.fusion(torch.cat([text, audio, visual.visual, role, metadata], dim=-1))
        context = self.context(token, batch["attention_mask"])
        target_index = batch["lengths"].long() - 1
        batch_index = torch.arange(context.shape[0], device=context.device)
        classifier_input = torch.cat(
            [
                context[batch_index, target_index],
                text[batch_index, target_index],
                audio[batch_index, target_index],
                visual.visual[batch_index, target_index],
            ],
            dim=-1,
        )
        logits = self.classifier(classifier_input)
        shift_logits = (
            None if self.shift_classifier is None else self.shift_classifier(classifier_input)
        )
        return ModelOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            visual=visual,
            context=context,
            text=text,
            audio=audio,
            shift_logits=shift_logits,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
