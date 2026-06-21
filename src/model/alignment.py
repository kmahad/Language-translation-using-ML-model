"""
IBM Model 1 Word Alignment using the EM Algorithm.

Learns word-level translation probabilities from a parallel corpus
using the Expectation-Maximization (EM) algorithm. This is a classical
machine learning approach — no neural networks or deep learning.

References:
    Brown et al. (1993), "The Mathematics of Statistical Machine Translation"
"""

import math
from collections import defaultdict
from typing import List, Tuple, Dict

from tqdm import tqdm


class IBMModel1:
    """IBM Model 1 word alignment using EM.

    Learns P(target_word | source_word) translation probabilities
    by iteratively:
      E-step: Compute expected counts using current probabilities
      M-step: Re-estimate probabilities from expected counts

    Attributes:
        translation_probs: Dict mapping (tgt_word, src_word) -> probability.
        iterations: Number of EM iterations to run.
    """

    def __init__(self, iterations: int = 10):
        """Initialize the aligner.

        Args:
            iterations: Number of EM iterations.
        """
        self.iterations = iterations
        self.translation_probs: Dict[Tuple[str, str], float] = {}
        self._src_vocab: set = set()
        self._tgt_vocab: set = set()

    def train(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        logger=None,
    ) -> None:
        """Train IBM Model 1 on tokenized parallel sentences.

        Args:
            src_sentences: List of tokenized source sentences (lists of tokens).
            tgt_sentences: List of tokenized target sentences (lists of tokens).
            logger: Optional logger for progress messages.
        """
        assert len(src_sentences) == len(tgt_sentences), \
            "Source and target must have the same number of sentences"

        def _log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        # Build vocabularies
        self._src_vocab = set()
        self._tgt_vocab = set()
        for src_tokens in src_sentences:
            self._src_vocab.update(src_tokens)
        for tgt_tokens in tgt_sentences:
            self._tgt_vocab.update(tgt_tokens)

        # Add NULL token for unaligned target words
        self._src_vocab.add("NULL")

        src_vocab_size = len(self._src_vocab)
        _log(f"  Source vocab: {src_vocab_size} tokens")
        _log(f"  Target vocab: {len(self._tgt_vocab)} tokens")

        # Initialize: uniform translation probabilities
        # P(t|s) = 1 / |tgt_vocab| for all t, s
        initial_prob = 1.0 / len(self._tgt_vocab)
        t_probs: Dict[Tuple[str, str], float] = defaultdict(lambda: initial_prob)

        # EM iterations
        for iteration in range(1, self.iterations + 1):
            # E-step: accumulate expected counts
            counts: Dict[Tuple[str, str], float] = defaultdict(float)
            totals: Dict[str, float] = defaultdict(float)

            for src_tokens, tgt_tokens in tqdm(
                zip(src_sentences, tgt_sentences),
                total=len(src_sentences),
                desc=f"  EM Iteration {iteration}/{self.iterations}",
                leave=False,
            ):
                # Add NULL to source
                src_with_null = ["NULL"] + list(src_tokens)

                for t in tgt_tokens:
                    # Compute normalization: sum of t(t|s) for all s
                    z = sum(t_probs[(t, s)] for s in src_with_null)
                    if z == 0:
                        continue

                    for s in src_with_null:
                        # Expected count
                        c = t_probs[(t, s)] / z
                        counts[(t, s)] += c
                        totals[s] += c

            # M-step: re-estimate probabilities
            new_t_probs: Dict[Tuple[str, str], float] = defaultdict(float)
            for (t, s), count in counts.items():
                if totals[s] > 0:
                    new_t_probs[(t, s)] = count / totals[s]

            t_probs = defaultdict(lambda: 1e-12, new_t_probs)

            # Compute log-likelihood for convergence monitoring
            log_likelihood = 0.0
            for src_tokens, tgt_tokens in zip(src_sentences, tgt_sentences):
                src_with_null = ["NULL"] + list(src_tokens)
                for t in tgt_tokens:
                    p = sum(t_probs[(t, s)] for s in src_with_null)
                    if p > 0:
                        log_likelihood += math.log(p)

            _log(f"  EM Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")

        # Store final probabilities (filter out very small values to save memory)
        self.translation_probs = {
            k: v for k, v in t_probs.items() if v > 1e-8
        }

    def get_alignment(
        self, src_tokens: List[str], tgt_tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Get the Viterbi (best) word alignment for a sentence pair.

        For each target word, finds the source word with highest
        translation probability.

        Args:
            src_tokens: Source sentence tokens.
            tgt_tokens: Target sentence tokens.

        Returns:
            List of (src_index, tgt_index) alignment pairs.
            src_index = -1 indicates alignment to NULL.
        """
        src_with_null = ["NULL"] + list(src_tokens)
        alignments = []

        for j, t in enumerate(tgt_tokens):
            best_i = 0
            best_prob = 0.0

            for i, s in enumerate(src_with_null):
                prob = self.translation_probs.get((t, s), 1e-12)
                if prob > best_prob:
                    best_prob = prob
                    best_i = i

            # Convert: 0 = NULL -> -1, otherwise subtract 1 for original index
            src_idx = best_i - 1
            alignments.append((src_idx, j))

        return alignments

    def get_translation_prob(self, tgt_word: str, src_word: str) -> float:
        """Get P(tgt_word | src_word).

        Args:
            tgt_word: Target language word.
            src_word: Source language word.

        Returns:
            Translation probability.
        """
        return self.translation_probs.get((tgt_word, src_word), 1e-12)

    def to_dict(self) -> dict:
        """Serialize alignment model to a dictionary for saving."""
        # Convert tuple keys to strings for JSON serialization
        probs = {}
        for (t, s), p in self.translation_probs.items():
            key = f"{t}|||{s}"
            probs[key] = p

        return {
            "iterations": self.iterations,
            "translation_probs": probs,
            "src_vocab_size": len(self._src_vocab),
            "tgt_vocab_size": len(self._tgt_vocab),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IBMModel1":
        """Deserialize alignment model from a dictionary."""
        model = cls(iterations=data.get("iterations", 10))

        probs = {}
        for key, p in data.get("translation_probs", {}).items():
            parts = key.split("|||")
            if len(parts) == 2:
                probs[(parts[0], parts[1])] = p
        model.translation_probs = probs

        return model
