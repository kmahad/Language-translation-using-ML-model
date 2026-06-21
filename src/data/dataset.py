"""
Pure Python Translation Dataset and DataLoader for Statistical Machine Translation.

Handles tokenized source/target pairs without any PyTorch dependencies.
"""

from typing import List, Tuple, Dict, Any
import random


class TranslationDataset:
    """Dataset of tokenized source-target translation pairs (lists of pieces)."""

    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], src_tokenizer, tgt_tokenizer, max_seq_len: int = 128):
        """Initialize dataset.

        Args:
            src_sentences: List of raw source sentences.
            tgt_sentences: List of raw target sentences.
            src_tokenizer: Tokenizer for source language.
            tgt_tokenizer: Tokenizer for target language.
            max_seq_len: Max length for truncation.
        """
        assert len(src_sentences) == len(tgt_sentences), "Source and target sentences must match in count."
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        # For SMT, we work with lists of token strings (pieces)
        src_tokens = self.src_tokenizer.encode_as_pieces(self.src_sentences[idx])
        tgt_tokens = self.tgt_tokenizer.encode_as_pieces(self.tgt_sentences[idx])

        # Truncate
        if len(src_tokens) > self.max_seq_len:
            src_tokens = src_tokens[:self.max_seq_len]
        if len(tgt_tokens) > self.max_seq_len:
            tgt_tokens = tgt_tokens[:self.max_seq_len]

        return src_tokens, tgt_tokens


class DataLoader:
    """A simple batch iterator for TranslationDataset."""

    def __init__(self, dataset: TranslationDataset, batch_size: int, shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        n_batches = len(self)
        for i in range(n_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            batch_src = []
            batch_tgt = []
            for idx in batch_indices:
                src, tgt = self.dataset[idx]
                batch_src.append(src)
                batch_tgt.append(tgt)
            yield {
                "src": batch_src,
                "tgt": batch_tgt
            }


def create_dataloaders(
    splits: dict,
    src_tokenizer,
    tgt_tokenizer,
    max_seq_len: int = 128,
    batch_size: int = 64,
    num_workers: int = 0, # Ignored in pure Python loader
) -> dict:
    """Create DataLoaders for train, val, and test splits.

    Args:
        splits: Dict from load_and_split_data.
        src_tokenizer: Source language tokenizer.
        tgt_tokenizer: Target language tokenizer.
        max_seq_len: Max sequence length.
        batch_size: Batch size.
        num_workers: Unused, kept for compatibility.

    Returns:
        Dict mapping split name -> DataLoader.
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
            drop_last=(split_name == "train"),
        )

    return loaders
