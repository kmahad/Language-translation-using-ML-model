"""
Training Script.

Trains the Transformer translation model.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --epochs 50 --batch_size 32
    python scripts/train.py --config config/default.yaml --checkpoint checkpoints/best.pt
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
from src.data.dataset import create_dataloaders
from src.config import get_config_from_args
from src.data.preprocessing import load_and_split_data
from src.data.tokenizer import load_tokenizers
from src.data.dataset import create_dataloaders
from src.model import SMTModel
from src.training import Trainer
from src.utils.helpers import set_seed, get_device, count_parameters, format_number
from src.utils.logging_utils import setup_logger


def main():
    # Load configuration
    config = get_config_from_args()

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Setup logging
    logger = setup_logger(
        name="translator",
        log_dir=config.logging.log_dir,
        level=config.logging.level,
    )

    # Determine device
    device = get_device()

    logger.info("=" * 60)
    logger.info("SMT TRANSLATION - TRAINING")
    logger.info("=" * 60)
    logger.info(f"Languages: {config.data.src_lang} -> {config.data.tgt_lang}")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("\nLoading data...")
    splits, file_paths = load_and_split_data(config)

    # Load tokenizers (must be trained first with prepare_data.py)
    logger.info("Loading tokenizers...")
    tokenizers = load_tokenizers(config)
    src_vocab_size = tokenizers["src"].vocab_size
    tgt_vocab_size = tokenizers["tgt"].vocab_size
    logger.info(f"Source vocab: {src_vocab_size}, Target vocab: {tgt_vocab_size}")

    # Create data loaders
    logger.info("Creating data loaders...")
    loaders = create_dataloaders(
        splits=splits,
        src_tokenizer=tokenizers["src"],
        tgt_tokenizer=tokenizers["tgt"],
        max_seq_len=config.data.max_seq_len,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    # Build model
    logger.info("Building model...")
    model = SMTModel(
        max_phrase_len=config.model.max_phrase_len,
        lm_order=config.model.lm_order,
        alignment_iterations=config.model.alignment_iterations,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        logger=logger,
    )

    # Resume from checkpoint if specified
    checkpoint_path = getattr(config, "_checkpoint", None)
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    # Train
    logger.info("\nStarting training...")
    results = trainer.train()

    # Model parameter stats (number of translation rules)
    num_params = count_parameters(model)
    logger.info(f"Model phrase rules: {format_number(num_params)} ({num_params:,})")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation BLEU: {results['best_val_bleu']:.2f}")
    logger.info(f"Total time: {results['total_time']}")
    logger.info(f"Checkpoint saved to: {config.training.checkpoint_dir}/best.json")
    logger.info("Next: python scripts/test.py --config config/default.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
