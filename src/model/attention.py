"""
Multi-Head Attention mechanism.

Implements Scaled Dot-Product Attention and Multi-Head Attention
from "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Splits queries, keys, and values into multiple heads, computes
    scaled dot-product attention independently per head, then
    concatenates and projects the results.

    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate for attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Store attention weights for visualization (optional)
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: [batch_size, query_len, d_model]
            key:   [batch_size, key_len, d_model]
            value: [batch_size, key_len, d_model]
            mask:  Boolean mask [batch_size, 1, query_len, key_len] or
                   [batch_size, 1, 1, key_len]. True = attend, False = mask.

        Returns:
            Output tensor of shape [batch_size, query_len, d_model].
        """
        batch_size = query.size(0)

        # 1. Linear projections: [B, seq_len, d_model] → [B, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads: [B, seq_len, d_model] → [B, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        # scores: [B, n_heads, query_len, key_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))

        # Attention weights: [B, n_heads, query_len, key_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store for visualization
        self.attention_weights = attn_weights.detach()

        # 4. Weighted sum: [B, n_heads, query_len, d_k]
        context = torch.matmul(attn_weights, V)

        # 5. Concatenate heads: [B, query_len, d_model]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 6. Final linear projection
        output = self.W_o(context)

        return output
