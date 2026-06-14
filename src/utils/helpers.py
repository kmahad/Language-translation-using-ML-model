"""
General utility functions for the translation project.

Provides seed setting, device detection, and parameter counting.
"""

import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


_device = None

def get_device() -> torch.device:
    """Detect the best available device (CUDA > CPU)."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        _device = torch.device("cpu")
        print("No GPU detected — using CPU (training will be slow)")
    return _device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """Format a large number with commas for readability."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def time_since(start_time: float) -> str:
    """Format elapsed time as a human-readable string."""
    import time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
