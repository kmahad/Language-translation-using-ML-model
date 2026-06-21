"""
Testing / Evaluation Script.

Evaluates a trained model on the test set and reports BLEU scores.

Usage:
    python scripts/test.py --config config/default.yaml
    python scripts/test.py --config config/default.yaml --checkpoint checkpoints/best.pt
    python scripts/test.py --config config/default.yaml --beam_size 1  # greedy
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
from src.data.tokenizer import load_tokenizers
from src.model import SMTModel
from src.evaluation import Evaluator
from src.utils.helpers import set_seed, get_device, count_parameters, format_number


def main():
    # Load configuration
    config = get_config_from_args()

    # Set seed
    set_seed(config.training.seed)

    # Device
    device = get_device()

    print("=" * 60)
    print("SMT TRANSLATION - EVALUATION")
    print("=" * 60)

    # Load data (we only need the test split)
    print("Loading data...")
    splits, _ = load_and_split_data(config)
    test_src = splits["test"]["src"]
    test_tgt = splits["test"]["tgt"]
    if getattr(config.training, "smoke_test", False):
        print("Smoke test mode: limiting evaluation to 10 sentences.")
        test_src = test_src[:10]
        test_tgt = test_tgt[:10]
    print(f"Test set: {len(test_src)} sentence pairs")

    # Load tokenizers
    print("Loading tokenizers...")
    tokenizers = load_tokenizers(config)
    src_vocab_size = tokenizers["src"].vocab_size
    tgt_vocab_size = tokenizers["tgt"].vocab_size

    # Build model
    print("Building model...")
    model = SMTModel(
        max_phrase_len=config.model.max_phrase_len,
        lm_order=config.model.lm_order,
        alignment_iterations=config.model.alignment_iterations,
    )

    # Load checkpoint
    checkpoint_path = getattr(config, "_checkpoint", None) or \
                      os.path.join(config.training.checkpoint_dir, "best.json")

    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Train a model first: python scripts/train.py --config config/default.yaml")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model.load(checkpoint_path)
    print(f"Loaded model rules: {format_number(count_parameters(model))}")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        src_tokenizer=tokenizers["src"],
        tgt_tokenizer=tokenizers["tgt"],
        device=device,
        beam_size=config.inference.beam_size,
        max_decode_len=config.inference.max_decode_len,
    )

    # Evaluate
    output_file = os.path.join(config.logging.log_dir, "translations.tsv")
    results = evaluator.evaluate(
        src_sentences=test_src,
        ref_sentences=test_tgt,
        output_file=output_file,
        num_samples=15,
    )

    print(f"\nFinal BLEU Score: {results['bleu']['bleu']:.2f}")


if __name__ == "__main__":
    main()
