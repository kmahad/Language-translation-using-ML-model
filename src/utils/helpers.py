"""
General utility functions for the translation project.

Provides seed setting, device detection, and parameter counting.
"""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> str:
    """Return device (no-op, CPU only for SMT)."""
    return "cpu"


def count_parameters(model) -> int:
    """Count the total number of translation rules in the SMT phrase table."""
    total = 0
    if hasattr(model, "phrase_table") and hasattr(model.phrase_table, "phrase_probs"):
        for src_phrase, candidates in model.phrase_table.phrase_probs.items():
            total += len(candidates)
    return total


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
