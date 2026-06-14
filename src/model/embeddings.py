"""
Token Embeddings with positional encoding.

Combines learned token embeddings (scaled by √d_model) with
sinusoidal positional encodings.
"""

import math
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """Token embedding + positional encoding.

    Follows the Transformer convention of scaling embeddings by √d_model
    before adding positional encoding. This scaling balances the magnitude
    of the embeddings with the positional encoding.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimension.
        max_len: Maximum sequence length for positional encoding.
        dropout: Dropout rate.
        padding_idx: Index of the padding token (default 0).
    """
    d_model: int
    token_embedding: nn.Embedding
    positional_encoding: PositionalEncoding
    scale: float

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens and add positional encoding.

        Args:
            x: Token ID tensor of shape [batch_size, seq_len].

        Returns:
            Embedded tensor of shape [batch_size, seq_len, d_model].
        """
        # Scale embeddings by √d_model
        embedded = self.token_embedding(x) * self.scale
        # Add positional encoding
        return self.positional_encoding(embedded)
