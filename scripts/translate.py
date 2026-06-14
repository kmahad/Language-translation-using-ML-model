"""
Interactive Translation CLI.

Type a sentence in the source language and get a translation.

Usage:
    python scripts/translate.py --config config/default.yaml
    python scripts/translate.py --config config/default.yaml --beam_size 1  # greedy
    python scripts/translate.py --config config/default.yaml --checkpoint checkpoints/best.pt
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

import torch

from src.config import get_config_from_args
from src.data.tokenizer import load_tokenizers
from src.model import Transformer
from src.evaluation import Evaluator
from src.utils.helpers import set_seed, get_device, count_parameters, format_number


def main():
    # Load configuration
    config = get_config_from_args()

    # Set seed
    set_seed(config.training.seed)

    # Device
    device = get_device()

    # Load tokenizers
    print("Loading tokenizers...")
    tokenizers = load_tokenizers(config)
    src_vocab_size = tokenizers["src"].vocab_size
    tgt_vocab_size = tokenizers["tgt"].vocab_size

    # Build model
    print("Building model...")
    model = Transformer.from_config(config, src_vocab_size, tgt_vocab_size)
    model = model.to(device)
    print(f"Model: {format_number(count_parameters(model))} parameters")

    # Load checkpoint
    checkpoint_path = getattr(config, "_checkpoint", None) or \
                      os.path.join(config.training.checkpoint_dir, "best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Train a model first: python scripts/train.py --config config/default.yaml")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create evaluator (used for its translate_sentence method)
    evaluator = Evaluator(
        model=model,
        src_tokenizer=tokenizers["src"],
        tgt_tokenizer=tokenizers["tgt"],
        device=device,
        beam_size=config.inference.beam_size,
        max_decode_len=config.inference.max_decode_len,
        length_penalty=config.inference.length_penalty,
    )

    # Interactive loop
    decoding = "Beam Search" if config.inference.beam_size > 1 else "Greedy"
    print("\n" + "=" * 60)
    print(f"INTERACTIVE TRANSLATOR")
    print(f"  {config.data.src_lang} -> {config.data.tgt_lang}")
    print(f"  Decoding: {decoding} (beam_size={config.inference.beam_size})")
    print(f"  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        try:
            # Get input
            print()
            source = input(f"[{config.data.src_lang}] > ").strip()

            if not source:
                continue

            if source.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Translate
            translation = evaluator.translate_sentence(source)
            print(f"[{config.data.tgt_lang}] > {translation}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
