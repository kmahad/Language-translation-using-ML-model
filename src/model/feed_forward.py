"""
Position-wise Feed-Forward Network.

Two linear transformations with a ReLU activation in between:
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

This is applied identically to each position in the sequence.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Args:
        d_model: Input and output dimension.
        d_ff: Inner (hidden) dimension (typically 4× d_model).
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor of same shape.
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
