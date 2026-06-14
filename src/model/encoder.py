"""
Transformer Encoder.

Stack of N identical encoder layers, each consisting of:
1. Multi-Head Self-Attention (with residual connection + LayerNorm)
2. Position-wise Feed-Forward Network (with residual connection + LayerNorm)
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer.

    Architecture:
        x → Self-Attention → Add & Norm → Feed-Forward → Add & Norm → output

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process input through one encoder layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            src_mask: Source padding mask [batch_size, 1, 1, seq_len].

        Returns:
            Output tensor [batch_size, seq_len, d_model].
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class Encoder(nn.Module):
    """Transformer Encoder: stack of N EncoderLayers.

    Args:
        n_layers: Number of encoder layers.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process input through all encoder layers.

        Args:
            x: Input embeddings [batch_size, seq_len, d_model].
            src_mask: Source padding mask.

        Returns:
            Encoder output [batch_size, seq_len, d_model].
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)
