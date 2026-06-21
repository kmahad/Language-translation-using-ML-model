"""
Data preprocessing for parallel translation corpora.

Handles CSV/TSV loading, text cleaning, and train/val/test splitting.
"""

import unicodedata
from pathlib import Path
from typing import Tuple, List

import pandas as pd


def clean_text(text: str) -> str:
    """Clean and normalize a text string.

    - Unicode NFKC normalization
    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())  # collapse whitespace
    return text.strip()


def load_parallel_corpus(
    data_file: str,
    src_column: str = "source",
    tgt_column: str = "target",
    separator: str = ",",
    max_samples: int = None,
) -> Tuple[List[str], List[str]]:
    """Load a parallel corpus from a CSV or TSV file.

    Args:
        data_file: Path to the CSV/TSV file.
        src_column: Column name for source language sentences.
        tgt_column: Column name for target language sentences.
        separator: Column separator ("," for CSV, "\\t" for TSV).
        max_samples: Maximum number of samples to load.

    Returns:
        Tuple of (source_sentences, target_sentences).
    """
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Handle escaped tab character from YAML
    if separator == "\\t":
        separator = "\t"

    df = pd.read_csv(path, sep=separator, encoding="utf-8", on_bad_lines="skip", nrows=max_samples)

    if src_column not in df.columns:
        raise ValueError(
            f"Source column '{src_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if tgt_column not in df.columns:
        raise ValueError(
            f"Target column '{tgt_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean text
    src_sentences = [clean_text(s) for s in df[src_column].astype(str).tolist()]
    tgt_sentences = [clean_text(s) for s in df[tgt_column].astype(str).tolist()]

    # Filter out empty pairs
    pairs = [(s, t) for s, t in zip(src_sentences, tgt_sentences) if s and t]
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]

    return src_sentences, tgt_sentences


def split_data(
    src_sentences: List[str],
    tgt_sentences: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split parallel data into train/val/test sets.

    Args:
        src_sentences: List of source sentences.
        tgt_sentences: List of target sentences.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each mapping to
        a dict with 'src' and 'tgt' sentence lists.
    """
    import random

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    assert len(src_sentences) == len(tgt_sentences), \
        "Source and target must have the same number of sentences"

    n = len(src_sentences)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": {
            "src": [src_sentences[i] for i in indices[:train_end]],
            "tgt": [tgt_sentences[i] for i in indices[:train_end]],
        },
        "val": {
            "src": [src_sentences[i] for i in indices[train_end:val_end]],
            "tgt": [tgt_sentences[i] for i in indices[train_end:val_end]],
        },
        "test": {
            "src": [src_sentences[i] for i in indices[val_end:]],
            "tgt": [tgt_sentences[i] for i in indices[val_end:]],
        },
    }

    return splits


def save_text_files(
    splits: dict,
    output_dir: str,
    src_lang: str = "en",
    tgt_lang: str = "fr",
) -> dict:
    """Save split data as plain text files (one sentence per line).

    These text files are used to train SentencePiece tokenizers.

    Returns:
        Dictionary mapping split names to file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, data in splits.items():
        src_path = out / f"{split_name}.{src_lang}"
        tgt_path = out / f"{split_name}.{tgt_lang}"

        with open(src_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data["src"]))
        with open(tgt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data["tgt"]))

        paths[split_name] = {"src": str(src_path), "tgt": str(tgt_path)}

    return paths


def load_and_split_data(config) -> Tuple[dict, dict]:
    """End-to-end data loading and splitting using a TranslationConfig.

    Args:
        config: TranslationConfig instance.

    Returns:
        Tuple of (splits_dict, file_paths_dict).
    """
    print(f"Loading data from: {config.data.data_file}")
    src, tgt = load_parallel_corpus(
        data_file=config.data.data_file,
        src_column=config.data.src_column,
        tgt_column=config.data.tgt_column,
        separator=config.data.separator,
        max_samples=getattr(config.data, "max_samples", None),
    )
    print(f"Loaded {len(src)} sentence pairs")

    splits = split_data(
        src, tgt,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.split_seed,
    )

    for name, data in splits.items():
        print(f"  {name}: {len(data['src'])} pairs")

    paths = save_text_files(
        splits,
        output_dir=config.tokenizer.model_dir,
        src_lang=config.data.src_lang,
        tgt_lang=config.data.tgt_lang,
    )

    return splits, paths
