"""
Data Preparation Script.

End-to-end pipeline:
1. Load CSV/TSV parallel corpus
2. Clean and split into train/val/test
3. Train SentencePiece tokenizers
4. Verify everything works

Usage:
    python scripts/prepare_data.py --config config/default.yaml
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reconfigure stdout/stderr to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import get_config_from_args
from src.data.preprocessing import load_and_split_data
from src.data.tokenizer import train_tokenizers


def main():
    # Load configuration
    config = get_config_from_args()

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"Data file: {config.data.data_file}")
    print(f"Languages: {config.data.src_lang} -> {config.data.tgt_lang}")
    print(f"Vocab size: {config.tokenizer.vocab_size}")
    print(f"Tokenizer type: {config.tokenizer.model_type}")
    print("=" * 60)

    # Step 1: Load and split data
    print("\n[1/3] Loading and splitting data...")
    splits, file_paths = load_and_split_data(config)

    # Step 2: Train tokenizers
    print("\n[2/3] Training tokenizers...")
    tokenizers = train_tokenizers(config, file_paths)

    # Step 3: Verify
    print("\n[3/3] Verification...")
    src_tok = tokenizers["src"]
    tgt_tok = tokenizers["tgt"]

    # Test with a sample sentence
    sample_src = splits["train"]["src"][0]
    sample_tgt = splits["train"]["tgt"][0]

    print(f"\n  Sample source: {sample_src}")
    src_ids = src_tok.encode(sample_src)
    src_pieces = src_tok.encode_as_pieces(sample_src)
    print(f"  Source tokens: {src_pieces[:15]}{'...' if len(src_pieces) > 15 else ''}")
    print(f"  Source IDs: {src_ids[:15]}{'...' if len(src_ids) > 15 else ''}")
    print(f"  Decoded back: {src_tok.decode(src_ids)}")

    print(f"\n  Sample target: {sample_tgt}")
    tgt_ids = tgt_tok.encode(sample_tgt)
    tgt_pieces = tgt_tok.encode_as_pieces(sample_tgt)
    print(f"  Target tokens: {tgt_pieces[:15]}{'...' if len(tgt_pieces) > 15 else ''}")
    print(f"  Target IDs: {tgt_ids[:15]}{'...' if len(tgt_ids) > 15 else ''}")
    print(f"  Decoded back: {tgt_tok.decode(tgt_ids)}")

    print(f"\n  Source vocab size: {src_tok.vocab_size}")
    print(f"  Target vocab size: {tgt_tok.vocab_size}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  Processed files saved to: {config.tokenizer.model_dir}")
    print("  You can now run: python scripts/train.py --config config/default.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
