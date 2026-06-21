"""
Unified Statistical Machine Translation (SMT) Model.

Ties together word alignments, phrase tables, language models, and a log-linear model.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

from .alignment import IBMModel1
from .phrase_table import PhraseTable
from .language_model import LanguageModel


class SMTModel:
    """Statistical Machine Translation model with a log-linear combination of features.

    Features:
        - log P(tgt_phrase | src_phrase)
        - log P(src_phrase | tgt_phrase)
        - log lex(tgt_phrase | src_phrase)
        - log lex(src_phrase | tgt_phrase)
        - log P_LM(target_tokens)
        - phrase penalty
        - word penalty
    """

    def __init__(self, max_phrase_len: int = 5, lm_order: int = 3, alignment_iterations: int = 10):
        """Initialize SMTModel components."""
        self.aligner_src_tgt = IBMModel1(iterations=alignment_iterations)
        self.aligner_tgt_src = IBMModel1(iterations=alignment_iterations)
        self.phrase_table = PhraseTable(max_phrase_len=max_phrase_len)
        self.language_model = LanguageModel(order=lm_order)

        # Log-linear weights
        self.weights = {
            "p_tgt_src": 1.0,
            "p_src_tgt": 1.0,
            "lex_tgt_src": 0.5,
            "lex_src_tgt": 0.5,
            "lm": 1.0,
            "phrase_penalty": -0.5,
            "word_penalty": -0.5,
        }

    def train_alignment_and_phrases(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        logger=None,
    ) -> None:
        """Step 1 & 2: Train word alignments (both directions) and extract phrase table."""
        def _log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        _log("Training aligner (source -> target)...")
        self.aligner_src_tgt.train(src_sentences, tgt_sentences, logger)

        _log("Training aligner (target -> source)...")
        self.aligner_tgt_src.train(tgt_sentences, src_sentences, logger)

        _log("Getting Viterbi alignments...")
        alignments = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            # Use aligner_src_tgt to get alignments
            alignments.append(self.aligner_src_tgt.get_alignment(src, tgt))

        _log("Extracting phrase table...")
        self.phrase_table.extract_phrases(src_sentences, tgt_sentences, alignments)

        _log("Computing phrase lexical weights...")
        self.phrase_table.compute_lexical_weights(self.aligner_src_tgt, self.aligner_tgt_src)
        _log(f"Phrase table size: {len(self.phrase_table.phrase_probs)} source phrases")

    def train_language_model(self, tgt_sentences: List[List[str]], logger=None) -> None:
        """Step 3: Train target language model."""
        if logger:
            logger.info("Training language model...")
        else:
            print("Training language model...")
        self.language_model.train(tgt_sentences)
        if logger:
            logger.info(f"Language model trained. Vocab size: {len(self.language_model.vocab)}")

    def translate(
        self,
        src_text: str,
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        beam_size: int = 4,
        max_decode_len: int = 64,
    ) -> str:
        """Translate source text using beam search decoding.

        Deferred to src.evaluation.inference to avoid circular imports.
        """
        from ..evaluation.inference import beam_search_decode
        
        # Tokenize source sentence to piece strings
        src_pieces = src_tokenizer.encode_as_pieces(src_text)
        
        # Perform beam search
        tgt_pieces = beam_search_decode(
            src_pieces=src_pieces,
            model=self,
            beam_size=beam_size,
            max_decode_len=max_decode_len,
        )
        
        # Convert pieces back to string
        tgt_ids = [tgt_tokenizer.piece_to_id(p) for p in tgt_pieces]
        return tgt_tokenizer.decode(tgt_ids)

    def save(self, filepath: str) -> None:
        """Save the entire SMT model to a JSON checkpoint."""
        data = {
            "aligner_src_tgt": self.aligner_src_tgt.to_dict(),
            "aligner_tgt_src": self.aligner_tgt_src.to_dict(),
            "phrase_table": self.phrase_table.to_dict(),
            "language_model": self.language_model.to_dict(),
            "weights": self.weights,
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """Load the entire SMT model from a JSON checkpoint."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.aligner_src_tgt = IBMModel1.from_dict(data["aligner_src_tgt"])
        self.aligner_tgt_src = IBMModel1.from_dict(data["aligner_tgt_src"])
        self.phrase_table = PhraseTable.from_dict(data["phrase_table"])
        self.language_model = LanguageModel.from_dict(data["language_model"])
        self.weights = data.get("weights", self.weights)
