"""
Phrase Table for Statistical Machine Translation.

Extracts phrase pairs from word-aligned parallel corpora and computes translation probabilities.
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any


class PhraseTable:
    """Phrase Table storing translation probabilities for phrase pairs.

    Attributes:
        max_phrase_len: Maximum length of extracted source/target phrases.
        phrase_probs: Dict mapping source_phrase -> list of (target_phrase, scores_dict).
    """

    def __init__(self, max_phrase_len: int = 5):
        """Initialize the phrase table."""
        self.max_phrase_len = max_phrase_len
        # Maps source_phrase tuple -> list of (target_phrase tuple, {feature_name: value})
        self.phrase_probs: Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], Dict[str, float]]]] = defaultdict(list)

    def extract_phrases(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        alignments: List[List[Tuple[int, int]]],
    ) -> None:
        """Extract phrase pairs consistent with word alignments from parallel corpus.

        Consistent phrase pairs are extracted using the standard SMT heuristic.
        """
        phrase_counts = defaultdict(int)
        src_counts = defaultdict(int)
        tgt_counts = defaultdict(int)

        for src, tgt, alignment in zip(src_sentences, tgt_sentences, alignments):
            # alignment is a list of (src_idx, tgt_idx) pairs.
            # Filter out NULL alignments (-1)
            aligned = {(i, j) for (i, j) in alignment if i >= 0 and j >= 0}
            aligned_src = {i for (i, j) in aligned}

            len_src = len(src)
            len_tgt = len(tgt)

            # Loop over all target spans [j1, j2]
            for j1 in range(len_tgt):
                for j2 in range(j1, min(len_tgt, j1 + self.max_phrase_len)):
                    # Find all source indices aligned to target words in the span [j1, j2]
                    src_indices = [i for (i, j) in aligned if j1 <= j <= j2]
                    if not src_indices:
                        continue
                    
                    i1 = min(src_indices)
                    i2 = max(src_indices)

                    # Check consistency:
                    # 1. No source word in [i1, i2] is aligned to a target word outside [j1, j2]
                    # 2. No target word in [j1, j2] is aligned to a source word outside [i1, i2]
                    consistent = True
                    for (i, j) in aligned:
                        if (i1 <= i <= i2) and not (j1 <= j <= j2):
                            consistent = False
                            break
                        if (j1 <= j <= j2) and not (i1 <= i <= i2):
                            consistent = False
                            break

                    if not consistent:
                        continue

                    # Extract phrase and its extensions with unaligned source words
                    curr_i1 = i1
                    while curr_i1 >= 0:
                        if curr_i1 < i1 and curr_i1 in aligned_src:
                            break
                        
                        curr_i2 = i2
                        while curr_i2 < len_src:
                            if curr_i2 > i2 and curr_i2 in aligned_src:
                                break
                            
                            # Check phrase length constraint
                            if (curr_i2 - curr_i1 + 1) <= self.max_phrase_len:
                                src_phrase = tuple(src[curr_i1 : curr_i2 + 1])
                                tgt_phrase = tuple(tgt[j1 : j2 + 1])
                                
                                phrase_counts[(src_phrase, tgt_phrase)] += 1
                                src_counts[src_phrase] += 1
                                tgt_counts[tgt_phrase] += 1
                            
                            curr_i2 += 1
                        curr_i1 -= 1

        # Compute translation probabilities:
        # P(tgt | src) = count(src, tgt) / count(src)
        # P(src | tgt) = count(src, tgt) / count(tgt)
        probs_temp = defaultdict(list)
        for (src_phrase, tgt_phrase), count in phrase_counts.items():
            p_tgt_src = count / src_counts[src_phrase]
            p_src_tgt = count / tgt_counts[tgt_phrase]
            
            probs_temp[src_phrase].append((tgt_phrase, p_tgt_src, p_src_tgt))

        # Filter the phrase table: keep only top translation candidates to speed up decoding
        # Typically we keep top 20 candidates per source phrase
        self.phrase_probs.clear()
        for src_phrase, candidates in probs_temp.items():
            # Sort by P(tgt | src) descending
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:20]
            for tgt_phrase, p_tgt_src, p_src_tgt in sorted_candidates:
                self.phrase_probs[src_phrase].append((
                    tgt_phrase,
                    {
                        "p_tgt_src": p_tgt_src,
                        "p_src_tgt": p_src_tgt,
                    }
                ))

    def compute_lexical_weights(
        self,
        aligner_src_tgt: Any,
        aligner_tgt_src: Any,
    ) -> None:
        """Compute lexical weights for all phrase pairs in the phrase table.

        Args:
            aligner_src_tgt: IBMModel1 mapping src -> tgt (provides P(t | s))
            aligner_tgt_src: IBMModel1 mapping tgt -> src (provides P(s | t))
        """
        for src_phrase, candidates in self.phrase_probs.items():
            for idx, (tgt_phrase, scores) in enumerate(candidates):
                # Forward lexical weight: lex(tgt | src) = product_j max_i P(t_j | s_i)
                lex_tgt_src = 1.0
                for t in tgt_phrase:
                    max_p = aligner_src_tgt.get_translation_prob(t, "NULL")
                    for s in src_phrase:
                        max_p = max(max_p, aligner_src_tgt.get_translation_prob(t, s))
                    lex_tgt_src *= max_p

                # Backward lexical weight: lex(src | tgt) = product_i max_j P(s_i | t_j)
                lex_src_tgt = 1.0
                for s in src_phrase:
                    max_p = aligner_tgt_src.get_translation_prob(s, "NULL")
                    for t in tgt_phrase:
                        max_p = max(max_p, aligner_tgt_src.get_translation_prob(s, t))
                    lex_src_tgt *= max_p

                scores["lex_tgt_src"] = lex_tgt_src
                scores["lex_src_tgt"] = lex_src_tgt

    def get_translations(self, src_phrase: Tuple[str, ...]) -> List[Tuple[Tuple[str, ...], Dict[str, float]]]:
        """Get possible translation candidates for a source phrase."""
        return self.phrase_probs.get(src_phrase, [])

    def to_dict(self) -> dict:
        """Serialize phrase table to a dictionary."""
        serialized = {}
        for src_phrase, candidates in self.phrase_probs.items():
            src_str = " ".join(src_phrase)
            serialized_candidates = []
            for tgt_phrase, scores in candidates:
                tgt_str = " ".join(tgt_phrase)
                serialized_candidates.append([tgt_str, scores])
            serialized[src_str] = serialized_candidates

        return {
            "max_phrase_len": self.max_phrase_len,
            "phrase_probs": serialized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhraseTable":
        """Deserialize phrase table from a dictionary."""
        table = cls(max_phrase_len=data.get("max_phrase_len", 5))
        
        phrase_probs = defaultdict(list)
        for src_str, candidates in data.get("phrase_probs", {}).items():
            src_phrase = tuple(src_str.split()) if src_str else ()
            for tgt_str, scores in candidates:
                tgt_phrase = tuple(tgt_str.split()) if tgt_str else ()
                phrase_probs[src_phrase].append((tgt_phrase, scores))
        
        table.phrase_probs = phrase_probs
        return table
