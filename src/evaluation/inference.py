"""
Monotonic Phrase-Based Beam Search Decoder for SMT.

Decodes a source sentence to a target sentence by translating source phrases
monotonically from left-to-right using the SMT log-linear model.
"""

import math
from typing import List, Tuple, Dict, Any


def safe_log(val: float) -> float:
    """Compute log value safely to avoid domain errors."""
    return math.log(max(val, 1e-15))


def get_lm_score(lm: Any, context_tokens: List[str], new_tokens: List[str]) -> Tuple[float, List[str]]:
    """Compute LM log-probability and new context for a sequence of target tokens."""
    curr_context = list(context_tokens)
    score = 0.0
    for token in new_tokens:
        prob = lm.get_prob(tuple(curr_context), token)
        score += safe_log(prob)
        curr_context.append(token)
        if len(curr_context) >= lm.order:
            curr_context = curr_context[-(lm.order - 1):]
    return score, curr_context


class Hypothesis:
    """A search hypothesis representing a partial translation."""

    def __init__(self, src_index: int, tgt_tokens: List[str], score: float, lm_context: List[str]):
        self.src_index = src_index
        self.tgt_tokens = tgt_tokens
        self.score = score
        self.lm_context = lm_context


def beam_search_decode(
    src_pieces: List[str],
    model: Any,
    beam_size: int = 4,
    max_decode_len: int = 64,
) -> List[str]:
    """Perform monotonic phrase-based stack decoding.

    Translates the source sentence from left to right.

    Args:
        src_pieces: List of source sentence piece tokens.
        model: Trained SMTModel.
        beam_size: Number of hypotheses to keep per stack.
        max_decode_len: Max target sequence length limit.

    Returns:
        List of target sentence piece tokens.
    """
    I = len(src_pieces)
    if I == 0:
        return []

    # Initialize stacks: stacks[i] holds hypotheses that have translated i source tokens
    stacks: Dict[int, List[Hypothesis]] = {i: [] for i in range(I + 1)}

    # Initial hypothesis
    initial_lm_context = ["<s>"] * (model.language_model.order - 1)
    stacks[0].append(Hypothesis(
        src_index=0,
        tgt_tokens=[],
        score=0.0,
        lm_context=initial_lm_context
    ))

    weights = model.weights

    # Iterate through all stacks
    for i in range(I):
        if not stacks[i]:
            continue

        # Prune stack i to only keep top beam_size hypotheses
        stacks[i].sort(key=lambda x: x.score, reverse=True)
        stacks[i] = stacks[i][:beam_size]

        for hyp in stacks[i]:
            # Try to extract phrases of length L starting at src_index i
            for L in range(1, min(model.phrase_table.max_phrase_len, I - i) + 1):
                src_phrase = tuple(src_pieces[i : i + L])
                next_src_index = i + L

                # Retrieve candidate translations from phrase table
                candidates = model.phrase_table.get_translations(src_phrase)

                # Fallback / Out-of-Vocabulary:
                # If length L = 1 and we have no translation, we treat it as pass-through (OOV)
                if L == 1 and not candidates:
                    # Pass-through: translate as itself
                    oov_word = src_pieces[i]
                    tgt_phrase = (oov_word,)
                    
                    lm_score, next_lm_context = get_lm_score(
                        model.language_model,
                        hyp.lm_context,
                        list(tgt_phrase)
                    )
                    
                    # Compute log-linear score for OOV option
                    oov_score = (
                        weights.get("p_tgt_src", 1.0) * safe_log(1e-12) +
                        weights.get("p_src_tgt", 1.0) * safe_log(1e-12) +
                        weights.get("lex_tgt_src", 0.5) * safe_log(1e-12) +
                        weights.get("lex_src_tgt", 0.5) * safe_log(1e-12) +
                        weights.get("lm", 1.0) * lm_score +
                        weights.get("phrase_penalty", -0.5) * 1.0 +
                        weights.get("word_penalty", -0.5) * 1.0
                    )
                    
                    new_hyp = Hypothesis(
                        src_index=next_src_index,
                        tgt_tokens=hyp.tgt_tokens + list(tgt_phrase),
                        score=hyp.score + oov_score,
                        lm_context=next_lm_context
                    )
                    stacks[next_src_index].append(new_hyp)
                    continue

                for tgt_phrase, scores in candidates:
                    # Check maximum length constraint
                    if len(hyp.tgt_tokens) + len(tgt_phrase) > max_decode_len:
                        continue

                    # LM score for this phrase
                    lm_score, next_lm_context = get_lm_score(
                        model.language_model,
                        hyp.lm_context,
                        list(tgt_phrase)
                    )

                    # Compute features
                    p_tgt_src = scores.get("p_tgt_src", 1e-12)
                    p_src_tgt = scores.get("p_src_tgt", 1e-12)
                    lex_tgt_src = scores.get("lex_tgt_src", 1e-12)
                    lex_src_tgt = scores.get("lex_src_tgt", 1e-12)

                    step_score = (
                        weights.get("p_tgt_src", 1.0) * safe_log(p_tgt_src) +
                        weights.get("p_src_tgt", 1.0) * safe_log(p_src_tgt) +
                        weights.get("lex_tgt_src", 0.5) * safe_log(lex_tgt_src) +
                        weights.get("lex_src_tgt", 0.5) * safe_log(lex_src_tgt) +
                        weights.get("lm", 1.0) * lm_score +
                        weights.get("phrase_penalty", -0.5) * 1.0 +
                        weights.get("word_penalty", -0.5) * len(tgt_phrase)
                    )

                    new_hyp = Hypothesis(
                        src_index=next_src_index,
                        tgt_tokens=hyp.tgt_tokens + list(tgt_phrase),
                        score=hyp.score + step_score,
                        lm_context=next_lm_context
                    )
                    stacks[next_src_index].append(new_hyp)

    # Sort final stack and get the best hypothesis
    final_stack = stacks[I]
    if not final_stack:
        # Fallback: if we somehow failed to reach the end, return the best partial translation
        all_hyps = []
        for i in range(I + 1):
            all_hyps.extend(stacks[i])
        if not all_hyps:
            return []
        all_hyps.sort(key=lambda x: x.score, reverse=True)
        return all_hyps[0].tgt_tokens

    final_stack.sort(key=lambda x: x.score, reverse=True)
    return final_stack[0].tgt_tokens


def greedy_decode(
    model: Any,
    src_pieces: List[str],
    max_len: int = 64,
) -> List[str]:
    """Greedy decoding: a special case of beam search with beam_size = 1."""
    return beam_search_decode(
        src_pieces=src_pieces,
        model=model,
        beam_size=1,
        max_decode_len=max_len,
    )
