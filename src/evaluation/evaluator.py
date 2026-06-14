"""
Full test-set evaluator.

Runs inference on all test examples, computes BLEU scores,
saves translations side-by-side, and prints sample outputs.
"""

from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from .inference import greedy_decode, beam_search_decode
from .metrics import compute_bleu, compute_sentence_bleu
from ..data.tokenizer import Tokenizer
from ..data.dataset import create_padding_mask
from ..model import Transformer


class Evaluator:
    """Evaluate a trained translation model on a test set.

    Args:
        model: Trained Transformer model.
        src_tokenizer: Source language tokenizer.
        tgt_tokenizer: Target language tokenizer.
        device: Device to run on.
        beam_size: Beam size for beam search (1 = greedy).
        max_decode_len: Maximum decoding length.
        length_penalty: Length penalty for beam search.
    """

    def __init__(
        self,
        model: Transformer,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        beam_size: int = 4,
        max_decode_len: int = 128,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.beam_size = beam_size
        self.max_decode_len = max_decode_len
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty

    def translate_sentence(self, sentence: str) -> str:
        """Translate a single sentence.

        Args:
            sentence: Source language sentence.

        Returns:
            Translated sentence in the target language.
        """
        self.model.eval()

        # Tokenize source
        src_ids = self.src_tokenizer.encode(sentence, add_bos=True, add_eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = create_padding_mask(src_tensor).to(self.device)

        # Cap decode length by model's target positional encoding size to prevent RuntimeError
        # Safely unwrap model if wrapped (e.g. DataParallel) and check for positional encoding
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "tgt_embedding") and hasattr(raw_model.tgt_embedding, "positional_encoding"):
            pe_tensor = getattr(raw_model.tgt_embedding.positional_encoding, "pe", None)
            if pe_tensor is not None:
                max_pe_len = pe_tensor.size(1)
            else:
                max_pe_len = self.max_decode_len
        else:
            max_pe_len = self.max_decode_len
        max_len = min(self.max_decode_len, max_pe_len)

        # Decode
        if self.beam_size > 1:
            output_ids = beam_search_decode(
                self.model, src_tensor, src_mask,
                beam_size=self.beam_size,
                max_len=max_len,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                device=self.device,
            )
        else:
            output_ids = greedy_decode(
                self.model, src_tensor, src_mask,
                max_len=max_len,
                repetition_penalty=self.repetition_penalty,
                device=self.device,
            )

        # Decode token IDs back to text
        translation = self.tgt_tokenizer.decode(output_ids[0].tolist())
        return translation

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
        print(f"Decoding method: {'Beam Search (k=' + str(self.beam_size) + ')' if self.beam_size > 1 else 'Greedy'}")

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
