"""
N-gram Language Model with Laplace (Add-k) smoothing.

Trained on target language sentences to evaluate fluency of candidate translations.
"""

import math
from collections import defaultdict
from typing import List, Tuple, Dict, Any


class LanguageModel:
    """N-gram language model with Laplace (Add-k) smoothing.

    Attributes:
        order: The order of the n-gram model (e.g., 3 for trigram).
        k: Smoothing constant.
        counts: Dict mapping n-gram tuple to its occurrence count.
        context_counts: Dict mapping context tuple to its occurrence count.
        vocab: Set of unique target tokens seen during training.
    """

    def __init__(self, order: int = 3, k: float = 0.01):
        """Initialize the language model.

        Args:
            order: The n-gram order (e.g. 3 = trigram).
            k: The add-k smoothing parameter.
        """
        self.order = order
        self.k = k
        self.counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab: set = set()

    def train(self, sentences: List[List[str]]) -> None:
        """Train the language model on tokenized target sentences.

        Args:
            sentences: List of tokenized sentences.
        """
        self.counts.clear()
        self.context_counts.clear()
        self.vocab.clear()

        # Build vocabulary and count n-grams
        for tokens in sentences:
            self.vocab.update(tokens)

            # Pad sentence with BOS (<s>) and EOS (</s>)
            # Prepend (order - 1) BOS tokens and append EOS token
            padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]

            for n in range(1, self.order + 1):
                for i in range(len(padded) - n + 1):
                    ngram = tuple(padded[i:i+n])
                    self.counts[ngram] += 1
                    if n > 1:
                        context = ngram[:-1]
                        self.context_counts[context] += 1

        # Add special tokens to vocab if not present
        self.vocab.update(["<s>", "</s>", "<unk>", "<pad>"])

    def get_prob(self, context: Tuple[str, ...], token: str) -> float:
        """Get the smoothed probability of a token given its context.

        Args:
            context: Context tokens as a tuple.
            token: The token to predict.

        Returns:
            P(token | context)
        """
        # Truncate context to match order - 1
        if len(context) >= self.order:
            context = context[-(self.order - 1):]

        ngram = context + (token,)
        ngram_count = self.counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        vocab_size = len(self.vocab)
        # Laplace (add-k) smoothing:
        prob = (ngram_count + self.k) / (context_count + self.k * vocab_size)
        return prob

    def score_sentence(self, tokens: List[str]) -> float:
        """Compute the log-probability of a sentence.

        Args:
            tokens: List of tokenized words.

        Returns:
            Sum of log-probabilities of the tokens in the sentence.
        """
        padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
        log_prob = 0.0

        for i in range(self.order - 1, len(padded)):
            token = padded[i]
            context = tuple(padded[i - (self.order - 1):i])
            prob = self.get_prob(context, token)
            log_prob += math.log(prob)

        return log_prob

    def to_dict(self) -> dict:
        """Serialize model parameters to a dictionary."""
        # Convert tuple keys to strings for JSON serialization
        counts_str = {" ".join(k): v for k, v in self.counts.items()}
        context_counts_str = {" ".join(k): v for k, v in self.context_counts.items()}

        return {
            "order": self.order,
            "k": self.k,
            "vocab": list(self.vocab),
            "counts": counts_str,
            "context_counts": context_counts_str,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LanguageModel":
        """Deserialize language model from a dictionary."""
        lm = cls(order=data.get("order", 3), k=data.get("k", 0.01))
        lm.vocab = set(data.get("vocab", []))

        # Restore counts from space-separated strings
        for k_str, v in data.get("counts", {}).items():
            k = tuple(k_str.split()) if k_str else ()
            lm.counts[k] = v

        for k_str, v in data.get("context_counts", {}).items():
            k = tuple(k_str.split()) if k_str else ()
            lm.context_counts[k] = v

        return lm
