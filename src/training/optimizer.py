"""
Optimizer and Learning Rate Scheduler.

Implements the Noam learning rate schedule from "Attention Is All You Need":
    lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

This creates a warmup phase followed by inverse-square-root decay.
"""

import torch
import torch.optim as optim


class NoamScheduler:
    """Noam learning rate scheduler.

    Linearly increases the learning rate during warmup, then decreases it
    proportionally to the inverse square root of the step number.

    Args:
        optimizer: PyTorch optimizer.
        d_model: Model dimension (used for scaling).
        warmup_steps: Number of warmup steps.
        factor: Scaling factor (default 1.0).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0

    def step(self) -> float:
        """Update the learning rate and step the optimizer scheduler.

        Call this after each training step (not each epoch).

        Returns:
            Current learning rate.
        """
        self._step += 1
        rate = self._get_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate
        return rate

    def _get_rate(self) -> float:
        """Compute the learning rate for the current step."""
        step = self._step
        warmup = self.warmup_steps
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    @property
    def current_lr(self) -> float:
        """Get the current learning rate."""
        return self._rate

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "step": self._step,
            "rate": self._rate,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from a checkpoint."""
        self._step = state_dict["step"]
        self._rate = state_dict["rate"]


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9,
) -> optim.Adam:
    """Create an Adam optimizer with Transformer-specific defaults.

    Args:
        model: The model whose parameters to optimize.
        learning_rate: Initial learning rate (overridden by scheduler).
        betas: Adam beta parameters.
        eps: Adam epsilon for numerical stability.

    Returns:
        Configured Adam optimizer.
    """
    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
