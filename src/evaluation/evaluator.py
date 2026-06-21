"""
Full test-set evaluator for SMT model.

Runs inference on all test examples, computes BLEU scores,
saves translations side-by-side, and prints sample outputs.
"""

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .metrics import compute_bleu, compute_sentence_bleu
from ..data.tokenizer import Tokenizer


class Evaluator:
    """Evaluate a trained SMT translation model on a test set.

    Args:
        model: Trained SMTModel.
        src_tokenizer: Source language tokenizer.
        tgt_tokenizer: Target language tokenizer.
        device: Unused (kept for compatibility).
        beam_size: Beam size for decoding.
        max_decode_len: Maximum decoding length.
    """

    def __init__(
        self,
        model,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        device=None,
        beam_size: int = 4,
        max_decode_len: int = 64,
        length_penalty: float = 0.6,      # Unused, kept for signature match
        repetition_penalty: float = 1.0,  # Unused, kept for signature match
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.beam_size = beam_size
        self.max_decode_len = max_decode_len

    def translate_sentence(self, sentence: str) -> str:
        """Translate a single sentence.

        Args:
            sentence: Source language sentence.

        Returns:
            Translated sentence in the target language.
        """
        return self.model.translate(
            src_text=sentence,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            beam_size=self.beam_size,
            max_decode_len=self.max_decode_len,
        )

    def evaluate(
        self,
        src_sentences: List[str],
        ref_sentences: List[str],
        output_file: Optional[str] = None,
        num_samples: int = 10,
    ) -> dict:
        """Evaluate the model on a test set.

        Args:
            src_sentences: Source sentences.
            ref_sentences: Reference (ground truth) translations.
            output_file: Optional path to save translations.
            num_samples: Number of sample translations to print.

        Returns:
            Dictionary with BLEU scores and sample translations.
        """
        print(f"\nEvaluating on {len(src_sentences)} sentences...")
        print(f"Decoding method: Beam Search (k={self.beam_size})")

        hypotheses = []

        for sentence in tqdm(src_sentences, desc="Translating"):
            translation = self.translate_sentence(sentence)
            hypotheses.append(translation)

        # Compute BLEU
        bleu_results = compute_bleu(hypotheses, ref_sentences)

        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"BLEU Score: {bleu_results['bleu']:.2f}")
        print(f"Precisions: {', '.join(f'{p:.1f}' for p in bleu_results['precisions'])}")
        print(f"Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
        print(f"{'='*60}")

        # Print sample translations
        print(f"\nSample Translations ({num_samples}):")
        print(f"{'-'*60}")
        samples = []
        for i in range(min(num_samples, len(src_sentences))):
            sent_bleu = compute_sentence_bleu(hypotheses[i], ref_sentences[i])
            sample = {
                "source": src_sentences[i],
                "reference": ref_sentences[i],
                "hypothesis": hypotheses[i],
                "sentence_bleu": sent_bleu,
            }
            samples.append(sample)

            print(f"\n[{i+1}] (BLEU: {sent_bleu:.1f})")
            print(f"  SRC: {src_sentences[i]}")
            print(f"  REF: {ref_sentences[i]}")
            print(f"  HYP: {hypotheses[i]}")

        # Save translations to file
        if output_file:
            self._save_translations(
                src_sentences, ref_sentences, hypotheses, output_file
            )

        return {
            "bleu": bleu_results,
            "samples": samples,
            "hypotheses": hypotheses,
        }

    def _save_translations(
        self,
        src_sentences: List[str],
        ref_sentences: List[str],
        hypotheses: List[str],
        output_file: str,
    ) -> None:
        """Save translations side-by-side to a file.

        Args:
            src_sentences: Source sentences.
            ref_sentences: Reference translations.
            hypotheses: Model-generated translations.
            output_file: Output file path.
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("SOURCE\tREFERENCE\tHYPOTHESIS\n")
            for src, ref, hyp in zip(src_sentences, ref_sentences, hypotheses):
                f.write(f"{src}\t{ref}\t{hyp}\n")

        print(f"\nTranslations saved to: {output_file}")
