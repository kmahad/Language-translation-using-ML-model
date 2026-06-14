"""
Translation quality metrics.

Computes BLEU score (the standard metric for machine translation)
using the sacrebleu library for consistent, reproducible scoring.
"""

from typing import List

import sacrebleu


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """Compute corpus-level BLEU score.

    Uses sacrebleu for standardized, reproducible BLEU computation
    with built-in tokenization.

    Args:
        hypotheses: List of model-generated translations.
        references: List of reference (ground truth) translations.

    Returns:
        Dictionary with:
            - bleu: Overall BLEU score (0-100).
            - precisions: 1-gram through 4-gram precisions.
            - brevity_penalty: Brevity penalty factor.
            - signature: sacrebleu signature string for reproducibility.
    """
    # Use sacrebleu BLEU class to get signature cleanly
    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(hypotheses, [references])

    return {
        "bleu": bleu.score,
        "precisions": bleu.precisions,
        "brevity_penalty": bleu.bp,
        "signature": str(metric.get_signature()),
    }


def compute_sentence_bleu(
    hypothesis: str,
    reference: str,
) -> float:
    """Compute sentence-level BLEU score.

    Args:
        hypothesis: Single model-generated translation.
        reference: Single reference translation.

    Returns:
        BLEU score (0-100).
    """
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu.score
