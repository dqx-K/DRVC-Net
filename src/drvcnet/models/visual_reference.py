"""Regional projection, dual-reference change, gates, and visual fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from drvcnet.config import ModelConfig
from drvcnet.constants import REGIONS, ModelVariant


@dataclass
class VisualReferenceOutput:
    visual: Tensor
    metadata: Tensor
    regional: Tensor
    attention: Tensor
    recent_gate: Tensor
    historical_gate: Tensor
    recent_mask: Tensor
    historical_mask: Tensor


class ChangeMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, current: Tensor, reference: Tensor) -> Tensor:
        difference = current - reference
        inputs = torch.cat([difference, difference.abs(), current * reference], dim=-1)
        return self.network(inputs)


class ConcatenationControl(nn.Module):
    def __init__(self, hidden_dim: int, control_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, control_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(control_hidden_dim, hidden_dim),
        )

    def forward(self, current: Tensor, reference: Tensor) -> Tensor:
        return self.network(torch.cat([current, reference], dim=-1))


class QualityGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, include_quality: bool = True) -> None:
        super().__init__()
        self.include_quality = include_quality
        metadata_dim = 4 if include_quality else 2
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim + metadata_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        current: Tensor,
        proposal: Tensor,
        current_quality: Tensor,
        reference_quality: Tensor,
        count: Tensor,
        age: Tensor,
    ) -> Tensor:
        temporal = [torch.log1p(count).unsqueeze(-1), torch.log1p(age).unsqueeze(-1)]
        if self.include_quality:
            metadata = [
                current_quality.unsqueeze(-1),
                reference_quality.unsqueeze(-1),
                *temporal,
            ]
        else:
            metadata = temporal
        return torch.sigmoid(self.network(torch.cat([current, proposal, *metadata], dim=-1))).squeeze(-1)


class HistoryAttentionControl(nn.Module):
    """Current-region query over exactly the legal B6 history candidates."""

    def __init__(self, hidden_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.scale = math.sqrt(hidden_dim)

    def forward(self, current: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        # current [..., H], candidates [..., C, H], mask [..., C]
        scores = torch.einsum(
            "...h,...ch->...c", self.query(current), self.key(candidates)
        ) / self.scale
        has_candidate = mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        masked_scores = torch.where(has_candidate, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(masked_scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.einsum("...c,...ch->...h", weights, self.value(candidates))
        attended = self.output(pooled)
        return (attended + self.refine(attended)) * has_candidate.to(pooled.dtype)


class VisualReferenceModule(nn.Module):
    """Equations (3), (9)--(12), including paper controls."""

    def __init__(self, config: ModelConfig, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant
        self.epsilon = epsilon
        h = config.hidden_dim
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.visual_input_dim, h),
                    nn.LayerNorm(h),
                )
                for _ in REGIONS
            ]
        )
        proposal_type = ConcatenationControl if self.variant is ModelVariant.C2 else ChangeMLP
        if proposal_type is ConcatenationControl:
            self.recent_proposals = nn.ModuleList(
                [proposal_type(h, config.c2_hidden_dim, config.dropout) for _ in REGIONS]
            )
            self.historical_proposals = nn.ModuleList(
                [proposal_type(h, config.c2_hidden_dim, config.dropout) for _ in REGIONS]
            )
        else:
            self.recent_proposals = nn.ModuleList(
                [proposal_type(h, config.dropout) for _ in REGIONS]
            )
            self.historical_proposals = nn.ModuleList(
                [proposal_type(h, config.dropout) for _ in REGIONS]
            )
        include_quality = self.variant is not ModelVariant.G1
        self.recent_gates = nn.ModuleList(
            [QualityGate(h, config.dropout, include_quality) for _ in REGIONS]
        )
        self.historical_gates = nn.ModuleList(
            [QualityGate(h, config.dropout, include_quality) for _ in REGIONS]
        )
        self.appearance = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in REGIONS])
        self.region_attention = nn.Parameter(torch.empty(len(REGIONS), h))
        nn.init.normal_(self.region_attention, std=h**-0.5)
        self.capacity_control = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h, config.c1_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.c1_hidden_dim, h),
                )
                for _ in REGIONS
            ]
        )
        self.history_attention = nn.ModuleList(
            [HistoryAttentionControl(h, config.c3_ff_dim, config.dropout) for _ in REGIONS]
        )
        self._configure_trainable_control()

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.requires_grad_(False)

    def _configure_trainable_control(self) -> None:
        """Keep each structural control's trainable capacity explicit."""

        if not self.variant.uses_recent or self.variant is ModelVariant.C3:
            self._freeze(self.recent_proposals)
        if not self.variant.uses_historical or self.variant is ModelVariant.C3:
            self._freeze(self.historical_proposals)
        if not (self.variant.uses_recent and self.variant.uses_learned_gate):
            self._freeze(self.recent_gates)
        if not (self.variant.uses_historical and self.variant.uses_learned_gate):
            self._freeze(self.historical_gates)
        if self.variant is not ModelVariant.C1:
            self._freeze(self.capacity_control)
        if self.variant is not ModelVariant.C3:
            self._freeze(self.history_attention)
        if self.variant is ModelVariant.B0:
            self.requires_grad_(False)

    def _project(self, values: Tensor, mask: Tensor) -> Tensor:
        projected = [
            self.projections[region](values[..., region, :])
            for region in range(len(REGIONS))
        ]
        stacked = torch.stack(projected, dim=-2)
        return stacked * mask.unsqueeze(-1).to(stacked.dtype)

    def _project_history(self, values: Tensor, mask: Tensor) -> Tensor:
        # Move K before R only inside each region-specific projection.
        projected = [
            self.projections[region](values[..., region, :, :])
            for region in range(len(REGIONS))
        ]
        stacked = torch.stack(projected, dim=-3)
        return stacked * mask.unsqueeze(-1).to(stacked.dtype)

    def _reference_summary(
        self,
        batch: dict[str, Any],
        current_valid: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        recent_exists = batch["recent_mask"].bool()
        recent = self._project(batch["recent"].float(), recent_exists)
        recent_quality = batch["recent_quality"].float() * recent_exists
        recent_count = recent_exists.to(torch.float32)
        recent_age = batch["recent_age"].float() * recent_exists
        recent_branch_mask = current_valid & recent_exists
        recent_quality = recent_quality * recent_branch_mask
        recent_count = recent_count * recent_branch_mask
        recent_age = recent_age * recent_branch_mask

        member_mask = batch["historical_mask"].bool()
        members = self._project_history(batch["historical"].float(), member_mask)
        member_quality = batch["historical_quality"].float() * member_mask
        denominator = member_quality.sum(dim=-1, keepdim=True)
        weights = member_quality / denominator.clamp_min(self.epsilon)
        historical = (members * weights.unsqueeze(-1)).sum(dim=-2)
        historical_exists = member_mask.any(dim=-1)
        historical_quality = (weights * member_quality).sum(dim=-1)
        historical_count = member_mask.sum(dim=-1).to(torch.float32)
        historical_age = (weights * batch["historical_age"].float()).sum(dim=-1)
        historical_branch_mask = current_valid & historical_exists
        historical_quality = historical_quality * historical_branch_mask
        historical_count = historical_count * historical_branch_mask
        historical_age = historical_age * historical_branch_mask
        return (
            recent,
            recent_quality,
            recent_count,
            recent_age,
            recent_branch_mask,
            historical,
            historical_quality,
            historical_count,
            historical_age,
            historical_branch_mask,
            members,
            member_mask,
        )

    def _gate(
        self,
        branch: str,
        region: int,
        current: Tensor,
        proposal: Tensor,
        mask: Tensor,
        current_quality: Tensor,
        reference_quality: Tensor,
        count: Tensor,
        age: Tensor,
    ) -> Tensor:
        if self.variant in {ModelVariant.B3, ModelVariant.B4, ModelVariant.B5}:
            value = torch.ones_like(current_quality)
        elif self.variant is ModelVariant.G2:
            value = current_quality * reference_quality
        else:
            gates = self.recent_gates if branch == "recent" else self.historical_gates
            value = gates[region](
                current, proposal, current_quality, reference_quality, count, age
            )
        return value * mask.to(value.dtype)

    @staticmethod
    def _metadata(
        current_valid: Tensor,
        current_quality: Tensor,
        recent_mask: Tensor,
        recent_quality: Tensor,
        recent_count: Tensor,
        recent_age: Tensor,
        historical_mask: Tensor,
        historical_quality: Tensor,
        historical_count: Tensor,
        historical_age: Tensor,
    ) -> Tensor:
        fields = torch.stack(
            [
                current_valid.to(torch.float32),
                current_quality,
                recent_mask.to(torch.float32),
                recent_quality,
                torch.log1p(recent_count),
                torch.log1p(recent_age),
                historical_mask.to(torch.float32),
                historical_quality,
                torch.log1p(historical_count),
                torch.log1p(historical_age),
            ],
            dim=-1,
        )
        return fields.flatten(start_dim=-2)

    def forward(self, batch: dict[str, Any]) -> VisualReferenceOutput:
        current_valid = batch["visual_valid"].bool()
        active = torch.zeros(len(REGIONS), dtype=torch.bool, device=current_valid.device)
        if self.variant.uses_visual and "visual" in self.config.enabled_modalities:
            active[list(self.variant.active_regions)] = True
        current_valid = current_valid & active.view(*([1] * (current_valid.ndim - 1)), -1)
        current_quality = batch["visual_quality"].float() * current_valid
        current = self._project(batch["visual"].float(), current_valid)
        (
            recent,
            recent_quality,
            recent_count,
            recent_age,
            recent_mask,
            historical,
            historical_quality,
            historical_count,
            historical_age,
            historical_mask,
            historical_members,
            historical_member_mask,
        ) = self._reference_summary(batch, current_valid)

        metadata = self._metadata(
            current_valid,
            current_quality,
            recent_mask,
            recent_quality,
            recent_count,
            recent_age,
            historical_mask,
            historical_quality,
            historical_count,
            historical_age,
        )
        if self.variant is ModelVariant.B0:
            metadata = torch.zeros_like(metadata)

        updates: list[Tensor] = []
        recent_gates: list[Tensor] = []
        historical_gates: list[Tensor] = []
        for region in range(len(REGIONS)):
            appearance = self.appearance[region](current[..., region, :])
            if self.variant is ModelVariant.C1:
                appearance = appearance + self.capacity_control[region](current[..., region, :])

            recent_proposal = self.recent_proposals[region](
                current[..., region, :], recent[..., region, :]
            ) * recent_mask[..., region, None]
            historical_reference = historical[..., region, :]
            historical_reference_quality = historical_quality[..., region]
            if self.variant is ModelVariant.N4:
                historical_reference = recent[..., region, :]
                historical_reference_quality = recent_quality[..., region]
            historical_proposal = self.historical_proposals[region](
                current[..., region, :], historical_reference
            ) * historical_mask[..., region, None]

            recent_gate = torch.zeros_like(current_quality[..., region])
            historical_gate = torch.zeros_like(current_quality[..., region])
            if self.variant.uses_recent and self.variant is not ModelVariant.C3:
                recent_gate = self._gate(
                    "recent",
                    region,
                    current[..., region, :],
                    recent_proposal,
                    recent_mask[..., region],
                    current_quality[..., region],
                    recent_quality[..., region],
                    recent_count[..., region],
                    recent_age[..., region],
                )
                appearance = appearance + recent_gate.unsqueeze(-1) * recent_proposal
            if self.variant.uses_historical and self.variant is not ModelVariant.C3:
                historical_gate = self._gate(
                    "historical",
                    region,
                    current[..., region, :],
                    historical_proposal,
                    historical_mask[..., region],
                    current_quality[..., region],
                    historical_reference_quality,
                    historical_count[..., region],
                    historical_age[..., region],
                )
                appearance = appearance + historical_gate.unsqueeze(-1) * historical_proposal
            if self.variant is ModelVariant.C3:
                candidates = torch.cat(
                    [recent[..., region, None, :], historical_members[..., region, :, :]], dim=-2
                )
                candidate_mask = torch.cat(
                    [recent_mask[..., region, None], historical_member_mask[..., region, :]], dim=-1
                )
                attended = self.history_attention[region](
                    current[..., region, :], candidates, candidate_mask
                )
                appearance = appearance + attended * current_valid[..., region, None]
            appearance = appearance * current_valid[..., region, None]
            updates.append(appearance)
            recent_gates.append(recent_gate)
            historical_gates.append(historical_gate)

        regional = torch.stack(updates, dim=-2)
        scores = (regional * self.region_attention).sum(dim=-1)
        has_region = current_valid.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~current_valid, float("-inf"))
        masked_scores = torch.where(has_region, masked_scores, torch.zeros_like(masked_scores))
        attention = torch.softmax(masked_scores, dim=-1) * current_valid.to(scores.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        visual = (attention.unsqueeze(-1) * regional).sum(dim=-2)
        visual = visual * has_region.to(visual.dtype)
        return VisualReferenceOutput(
            visual=visual,
            metadata=metadata,
            regional=regional,
            attention=attention,
            recent_gate=torch.stack(recent_gates, dim=-1),
            historical_gate=torch.stack(historical_gates, dim=-1),
            recent_mask=recent_mask,
            historical_mask=historical_mask,
        )
