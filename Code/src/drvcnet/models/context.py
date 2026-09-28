"""Causal utterance-token fusion and dialogue context encoding."""

from __future__ import annotations

from torch import Tensor, nn

from drvcnet.config import ModelConfig


class MaskedProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.projection(values) * mask.unsqueeze(-1).to(values.dtype)


class CausalContextEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.position_embedding = nn.Embedding(config.context_window, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.context_heads,
            dim_feedforward=config.context_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.context_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, tokens: Tensor, valid_tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if length > self.config.context_window:
            raise ValueError("Sequence exceeds configured causal context window")
        positions = self.position_embedding.weight[:length].unsqueeze(0)
        values = tokens + positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=tokens.device, dtype=tokens.dtype
        )
        return self.encoder(
            values,
            mask=causal_mask,
            src_key_padding_mask=~valid_tokens.bool(),
            is_causal=True,
        )
