"""
Label-Smoothed Cross-Entropy Loss.

Replaces hard one-hot targets with a smoothed distribution,
which acts as a regularizer and improves generalization.
Ignores padding tokens in loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Instead of using hard targets (0 or 1), distributes a small amount
    of probability (smoothing) uniformly across all classes:
        target_dist[correct_class] = 1 - smoothing
        target_dist[other_classes] = smoothing / (num_classes - 1)

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing).
        padding_idx: Token index to ignore in loss computation.
        reduction: Loss reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        padding_idx: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size].
            targets: Target token IDs [batch_size, seq_len].

        Returns:
            Scalar loss value.
        """
        vocab_size = logits.size(-1)

        # Flatten: [B * seq_len, vocab_size] and [B * seq_len]
        logits = logits.contiguous().view(-1, vocab_size)
        targets = targets.contiguous().view(-1)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        # Fill with uniform probability
        smooth_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
        # Set the correct class probability
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        # Zero out padding index
        smooth_targets[:, self.padding_idx] = 0

        # Create mask for non-padding positions
        non_pad_mask = targets != self.padding_idx
        smooth_targets[~non_pad_mask] = 0

        # Compute KL divergence loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            # Average only over non-padding tokens
            non_pad_count = non_pad_mask.sum()
            if non_pad_count > 0:
                return loss.sum() / non_pad_count
            return loss.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
