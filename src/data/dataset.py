"""
PyTorch Dataset and DataLoader for translation pairs.

Handles tokenized source/target pairs with dynamic padding
and proper masking for the Transformer.
"""

from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer, PAD_ID, BOS_ID, EOS_ID


class TranslationDataset(Dataset):
    """Dataset of tokenized source-target translation pairs.

    Each item is a pair of token ID lists (src_ids, tgt_ids), truncated
    to max_seq_len.
    """

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_seq_len: int = 128,
    ):
        """Initialize the dataset.

        Args:
            src_sentences: List of source language sentences.
            tgt_sentences: List of target language sentences.
            src_tokenizer: Trained source tokenizer.
            tgt_tokenizer: Trained target tokenizer.
            max_seq_len: Maximum sequence length (including BOS/EOS).
        """
        assert len(src_sentences) == len(tgt_sentences), \
            "Source and target must have equal length"

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Get a single tokenized pair.

        Returns:
            Tuple of (src_ids, tgt_ids), both truncated to max_seq_len.
        """
        src_ids = self.src_tokenizer.encode(
            self.src_sentences[idx], add_bos=True, add_eos=True
        )
        tgt_ids = self.tgt_tokenizer.encode(
            self.tgt_sentences[idx], add_bos=True, add_eos=True
        )

        # Truncate to max_seq_len
        src_ids = src_ids[:self.max_seq_len]
        tgt_ids = tgt_ids[:self.max_seq_len]

        return src_ids, tgt_ids


def collate_fn(batch: List[Tuple[List[int], List[int]]]) -> dict:
    """Custom collate function for dynamic padding.

    Pads all sequences in a batch to the length of the longest sequence
    in that batch (not the global max_seq_len), for efficiency.

    Args:
        batch: List of (src_ids, tgt_ids) tuples.

    Returns:
        Dictionary containing:
            - src: Padded source tensor [batch_size, src_len]
            - tgt_input: Target input (shifted right, starts with BOS) [batch_size, tgt_len-1]
            - tgt_output: Target labels (ends with EOS) [batch_size, tgt_len-1]
            - src_mask: Source padding mask [batch_size, 1, 1, src_len]
            - tgt_mask: Combined causal + padding mask [batch_size, 1, tgt_len-1, tgt_len-1]
    """
    src_batch, tgt_batch = zip(*batch)

    # Determine max lengths in this batch
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    # Pad sequences
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []

    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        # Pad source
        src_padded.append(src_ids + [PAD_ID] * (src_max_len - len(src_ids)))

        # Target input: everything except the last token (teacher forcing)
        # e.g., [BOS, w1, w2, w3] → input is [BOS, w1, w2]
        tgt_in = tgt_ids[:-1]
        tgt_in_padded = tgt_in + [PAD_ID] * (tgt_max_len - 1 - len(tgt_in))
        tgt_input_padded.append(tgt_in_padded)

        # Target output: everything except the first token (labels)
        # e.g., [BOS, w1, w2, EOS] → output is [w1, w2, EOS]
        tgt_out = tgt_ids[1:]
        tgt_out_padded = tgt_out + [PAD_ID] * (tgt_max_len - 1 - len(tgt_out))
        tgt_output_padded.append(tgt_out_padded)

    # Convert to tensors
    src = torch.tensor(src_padded, dtype=torch.long)
    tgt_input = torch.tensor(tgt_input_padded, dtype=torch.long)
    tgt_output = torch.tensor(tgt_output_padded, dtype=torch.long)

    # Create masks
    src_mask = create_padding_mask(src)                          # [B, 1, 1, src_len]
    tgt_padding_mask = create_padding_mask(tgt_input)            # [B, 1, 1, tgt_len]
    tgt_causal_mask = create_causal_mask(tgt_input.size(1))      # [1, 1, tgt_len, tgt_len]
    tgt_mask = tgt_padding_mask & tgt_causal_mask                # [B, 1, tgt_len, tgt_len]

    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """Create a padding mask (True where NOT padded).

    Args:
        seq: Token ID tensor of shape [batch_size, seq_len].

    Returns:
        Boolean mask of shape [batch_size, 1, 1, seq_len].
        True = attend, False = ignore (pad).
    """
    return (seq != PAD_ID).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]


def create_causal_mask(size: int) -> torch.Tensor:
    """Create a causal (look-ahead) mask for the decoder.

    Prevents attention to future tokens.

    Args:
        size: Sequence length.

    Returns:
        Boolean lower-triangular mask of shape [1, 1, size, size].
        True = attend, False = mask out.
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]


def create_dataloaders(
    splits: dict,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    max_seq_len: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict:
    """Create DataLoaders for train, val, and test splits.

    Args:
        splits: Dictionary from preprocessing with 'train', 'val', 'test' keys.
        src_tokenizer: Trained source tokenizer.
        tgt_tokenizer: Trained target tokenizer.
        max_seq_len: Maximum sequence length.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary mapping split names to DataLoader instances.
    """
    loaders = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue

        dataset = TranslationDataset(
            src_sentences=splits[split_name]["src"],
            tgt_sentences=splits[split_name]["tgt"],
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_seq_len=max_seq_len,
        )

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )

    return loaders
