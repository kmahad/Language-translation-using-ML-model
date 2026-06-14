"""
Sinusoidal Positional Encoding.

Implements the positional encoding from "Attention Is All You Need"
(Vaswani et al., 2017) using sine and cosine functions of different
frequencies to inject position information.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The encoding is pre-computed for up to max_len positions and
    registered as a buffer (not a learned parameter).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] — batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
