"""
Transformer Decoder.

Stack of N identical decoder layers, each consisting of:
1. Masked Multi-Head Self-Attention (causal, with residual + LayerNorm)
2. Multi-Head Cross-Attention over encoder output (with residual + LayerNorm)
3. Position-wise Feed-Forward Network (with residual + LayerNorm)
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer.

    Architecture:
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn (over encoder output) → Add & Norm
          → Feed-Forward → Add & Norm → output

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

        # Masked self-attention (causal)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-attention (attends to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process input through one decoder layer.

        Args:
            x: Decoder input [batch_size, tgt_len, d_model].
            encoder_output: Encoder output [batch_size, src_len, d_model].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].
            tgt_mask: Target causal + padding mask [batch_size, 1, tgt_len, tgt_len].

        Returns:
            Output tensor [batch_size, tgt_len, d_model].
        """
        # 1. Masked self-attention
        self_attn_output = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 2. Cross-attention (query from decoder, key/value from encoder)
        cross_attn_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Decoder(nn.Module):
    """Transformer Decoder: stack of N DecoderLayers.

    Args:
        n_layers: Number of decoder layers.
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
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process input through all decoder layers.

        Args:
            x: Target embeddings [batch_size, tgt_len, d_model].
            encoder_output: Encoder output [batch_size, src_len, d_model].
            src_mask: Source padding mask.
            tgt_mask: Target causal + padding mask.

        Returns:
            Decoder output [batch_size, tgt_len, d_model].
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
