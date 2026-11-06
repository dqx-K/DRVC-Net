"""Utterance-token fusion and configurable dialogue context encoding."""

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
        if config.context_encoder == "transformer":
            self.position_embedding: nn.Embedding | None = nn.Embedding(
                config.context_window, config.hidden_dim
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.context_heads,
                dim_feedforward=config.context_ff_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.encoder: nn.Module = nn.TransformerEncoder(
                layer,
                num_layers=config.context_layers,
                norm=nn.LayerNorm(config.hidden_dim),
            )
        else:
            self.position_embedding = None
            self.encoder = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.context_layers,
                batch_first=True,
                dropout=config.dropout if config.context_layers > 1 else 0.0,
                bidirectional=False,
            )

    def forward(self, tokens: Tensor, valid_tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if length > self.config.context_window:
            raise ValueError("Sequence exceeds configured causal context window")
        if self.config.context_encoder == "gru":
            output, _state = self.encoder(tokens)
            return output * valid_tokens.unsqueeze(-1).to(output.dtype)

        if self.position_embedding is None:
            raise AssertionError("Transformer position embeddings are unavailable")
        positions = self.position_embedding.weight[:length].unsqueeze(0)
        values = tokens + positions
        if self.config.context_direction == "bidirectional":
            return self.encoder(values, src_key_padding_mask=~valid_tokens.bool())
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=tokens.device, dtype=tokens.dtype
        )
        return self.encoder(
            values,
            mask=causal_mask,
            src_key_padding_mask=~valid_tokens.bool(),
            is_causal=True,
        )
