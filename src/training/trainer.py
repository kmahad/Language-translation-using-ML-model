"""
Trainer for SMT Model.

Trains alignments, extracts phrase table, trains language model, and tunes
log-linear weights on the validation dataset using Coordinate Ascent.
"""

import time
from pathlib import Path
from typing import List, Dict, Any

from ..evaluation.metrics import compute_bleu
from ..utils.helpers import time_since


class Trainer:
    """Trainer for SMT model.

    Coordinates alignment training, phrase table extraction, language model training,
    and validation weight tuning.
    """

    def __init__(self, model, config, train_loader, val_loader, logger=None):
        """Initialize trainer.

        Args:
            model: SMTModel instance.
            config: TranslationConfig instance.
            train_loader: DataLoader for train data.
            val_loader: DataLoader for validation data.
            logger: Logger instance.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_bleu = -1.0
        self.training_history = []

    def train(self) -> dict:
        """Run the training and weight-tuning process.

        Returns:
            Dictionary with training history summary.
        """
        self._log("=" * 60)
        self._log("STARTING SMT MODEL TRAINING")
        self._log("=" * 60)

        start_time = time.time()

        # 1. Collect tokenized sentences for training
        self._log("Collecting tokenized parallel training sentences...")
        src_train_tokens = []
        tgt_train_tokens = []
        for i in range(len(self.train_loader.dataset)):
            src_t, tgt_t = self.train_loader.dataset[i]
            src_train_tokens.append(src_t)
            tgt_train_tokens.append(tgt_t)

        # 2. Train alignment models and extract phrase table
        self._log("\n--- Training Alignment & Building Phrase Table ---")
        self.model.train_alignment_and_phrases(
            src_sentences=src_train_tokens,
            tgt_sentences=tgt_train_tokens,
            logger=self.logger
        )

        # 3. Train language model
        self._log("\n--- Training N-gram Language Model ---")
        self.model.train_language_model(
            tgt_sentences=tgt_train_tokens,
            logger=self.logger
        )

        # 4. Tune weights on a subset of the validation set
        self._log("\n--- Tuning Log-Linear Weights on Validation Set ---")
        # Prepare validation sentences
        val_subset_size = 50
        if getattr(self.config.training, "smoke_test", False):
            val_subset_size = 5

        # Get raw sentences for validation evaluation
        src_val_sentences = self.val_loader.dataset.src_sentences[:val_subset_size]
        tgt_val_sentences = self.val_loader.dataset.tgt_sentences[:val_subset_size]

        # Log initial validation BLEU
        initial_bleu = self._evaluate_bleu(src_val_sentences, tgt_val_sentences)
        self._log(f"Initial BLEU score on validation subset: {initial_bleu:.4f}")

        # Tune weights using Coordinate Ascent
        best_bleu = self._tune_weights(src_val_sentences, tgt_val_sentences, initial_bleu)

        # Save the final best model checkpoint
        checkpoint_path = self.checkpoint_dir / "best.json"
        self.model.save(str(checkpoint_path))
        self._log(f"\nSaved best model checkpoint to: {checkpoint_path}")

        total_time = time_since(start_time)
        self._log(f"\nTraining Complete! Total elapsed time: {total_time}")
        self._log(f"Best Validation BLEU: {best_bleu:.4f}")

        return {
            "best_val_loss": 0.0,  # for compatibility
            "best_val_bleu": best_bleu,
            "total_time": total_time,
            "history": self.training_history,
        }

    def _evaluate_bleu(self, src_sentences: List[str], tgt_sentences: List[str]) -> float:
        """Decode and evaluate BLEU on validation subset."""
        hypotheses = []
        for src_s in src_sentences:
            hyp = self.model.translate(
                src_text=src_s,
                src_tokenizer=self.train_loader.dataset.src_tokenizer,
                tgt_tokenizer=self.train_loader.dataset.tgt_tokenizer,
                beam_size=self.config.inference.beam_size,
                max_decode_len=self.config.inference.max_decode_len,
            )
            hypotheses.append(hyp)

        res = compute_bleu(hypotheses, tgt_sentences)
        return res["bleu"]

    def _tune_weights(self, src_val: List[str], tgt_val: List[str], initial_bleu: float) -> float:
        """Coordinate ascent to tune log-linear feature weights."""
        best_bleu = initial_bleu
        best_weights = self.model.weights.copy()

        # Candidate multipliers/scales for weights
        candidate_values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        epochs = self.config.training.epochs
        if getattr(self.config.training, "smoke_test", False):
            epochs = 1

        for epoch in range(1, epochs + 1):
            self._log(f"Tuning Epoch {epoch}/{epochs}")
            for weight_name in self.model.weights.keys():
                best_val_for_weight = self.model.weights[weight_name]
                improved = False

                for val in candidate_values:
                    self.model.weights[weight_name] = val
                    bleu = self._evaluate_bleu(src_val, tgt_val)

                    if bleu > best_bleu:
                        best_bleu = bleu
                        best_weights = self.model.weights.copy()
                        best_val_for_weight = val
                        improved = True
                        self._log(
                            f"  * Improved: {weight_name}={val} -> BLEU = {best_bleu:.2f} "
                            f"(weights: {best_weights})"
                        )

                # Restore/update best value
                self.model.weights[weight_name] = best_val_for_weight

            self.training_history.append({
                "epoch": epoch,
                "val_bleu": best_bleu,
                "weights": best_weights.copy(),
            })

        self.model.weights = best_weights
        self._log(f"Tuning finished. Best weights: {self.model.weights} | Best BLEU: {best_bleu:.2f}")
        return best_bleu

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint to resume training or decode."""
        self._log(f"Loading checkpoint: {checkpoint_path}")
        self.model.load(checkpoint_path)

    def _log(self, msg: str) -> None:
        """Log messages."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
