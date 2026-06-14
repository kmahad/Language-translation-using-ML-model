"""
SentencePiece tokenizer wrapper for the translation project.

Trains BPE or Unigram tokenizers and provides encode/decode interfaces.
"""

import os
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm


# Special token IDs (must match SentencePiece training config)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Special token strings
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Tokenizer:
    """Wrapper around SentencePiece for training and using BPE/Unigram tokenizers.

    Attributes:
        sp: The SentencePiece processor instance.
        vocab_size: Size of the vocabulary.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize tokenizer, optionally loading an existing model.

        Args:
            model_path: Path to a trained .model file. If None, call train() first.
        """
        self.sp = spm.SentencePieceProcessor()
        self._vocab_size = 0

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        input_files: List[str],
        model_prefix: str,
        vocab_size: int = 16000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> str:
        """Train a SentencePiece tokenizer on the given text files.

        Args:
            input_files: List of text file paths (one sentence per line).
            model_prefix: Output path prefix for .model and .vocab files.
            vocab_size: Target vocabulary size.
            model_type: "bpe" or "unigram".
            character_coverage: Character coverage (1.0 for non-Latin scripts).

        Returns:
            Path to the trained .model file.
        """
        # Ensure output directory exists
        Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

        input_str = ",".join(input_files)

        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            # Special tokens configuration
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_piece=PAD_TOKEN,
            unk_piece=UNK_TOKEN,
            bos_piece=BOS_TOKEN,
            eos_piece=EOS_TOKEN,
            # Additional settings
            normalization_rule_name="nmt_nfkc",
            shuffle_input_sentence=True,
            train_extremely_large_corpus=False,
        )

        model_path = f"{model_prefix}.model"
        self.load(model_path)
        print(f"Tokenizer trained: {model_path} (vocab_size={self._vocab_size})")
        return model_path

    def load(self, model_path: str) -> None:
        """Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp.load(model_path)
        self._vocab_size = self.sp.get_piece_size()

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text to a list of token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to prepend <bos> token.
            add_eos: Whether to append <eos> token.

        Returns:
            List of integer token IDs.
        """
        ids = self.sp.encode_as_ids(text)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text.

        Automatically filters out special tokens.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        # Filter out special tokens
        filtered = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode_ids(filtered)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (for debugging/visualization).

        Args:
            text: Input text string.

        Returns:
            List of subword token strings.
        """
        return self.sp.encode_as_pieces(text)

    def id_to_piece(self, id: int) -> str:
        """Convert a single token ID to its string representation."""
        return self.sp.id_to_piece(id)

    def piece_to_id(self, piece: str) -> int:
        """Convert a token string to its ID."""
        return self.sp.piece_to_id(piece)


def train_tokenizers(config, file_paths: dict) -> dict:
    """Train source and target tokenizers using config and data file paths.

    Args:
        config: TranslationConfig instance.
        file_paths: Dict from preprocessing with 'train' split paths.

    Returns:
        Dict with 'src' and 'tgt' keys mapping to trained Tokenizer instances.
    """
    model_dir = config.tokenizer.model_dir
    src_lang = config.data.src_lang
    tgt_lang = config.data.tgt_lang

    if config.tokenizer.shared_vocab:
        # Train a single shared tokenizer on both languages
        print("Training shared tokenizer...")
        tokenizer = Tokenizer()
        input_files = [file_paths["train"]["src"], file_paths["train"]["tgt"]]
        tokenizer.train(
            input_files=input_files,
            model_prefix=os.path.join(model_dir, f"sp_shared"),
            vocab_size=config.tokenizer.vocab_size,
            model_type=config.tokenizer.model_type,
            character_coverage=config.tokenizer.character_coverage,
        )
        return {"src": tokenizer, "tgt": tokenizer}
    else:
        # Train separate tokenizers for source and target
        print(f"Training source tokenizer ({src_lang})...")
        src_tokenizer = Tokenizer()
        src_tokenizer.train(
            input_files=[file_paths["train"]["src"]],
            model_prefix=os.path.join(model_dir, f"sp_{src_lang}"),
            vocab_size=config.tokenizer.vocab_size,
            model_type=config.tokenizer.model_type,
            character_coverage=config.tokenizer.character_coverage,
        )

        print(f"Training target tokenizer ({tgt_lang})...")
        tgt_tokenizer = Tokenizer()
        tgt_tokenizer.train(
            input_files=[file_paths["train"]["tgt"]],
            model_prefix=os.path.join(model_dir, f"sp_{tgt_lang}"),
            vocab_size=config.tokenizer.vocab_size,
            model_type=config.tokenizer.model_type,
            character_coverage=config.tokenizer.character_coverage,
        )

        return {"src": src_tokenizer, "tgt": tgt_tokenizer}


def load_tokenizers(config) -> dict:
    """Load previously trained tokenizers from disk.

    Args:
        config: TranslationConfig instance.

    Returns:
        Dict with 'src' and 'tgt' Tokenizer instances.
    """
    model_dir = config.tokenizer.model_dir
    src_lang = config.data.src_lang
    tgt_lang = config.data.tgt_lang

    if config.tokenizer.shared_vocab:
        model_path = os.path.join(model_dir, "sp_shared.model")
        tokenizer = Tokenizer(model_path)
        return {"src": tokenizer, "tgt": tokenizer}
    else:
        src_path = os.path.join(model_dir, f"sp_{src_lang}.model")
        tgt_path = os.path.join(model_dir, f"sp_{tgt_lang}.model")
        return {
            "src": Tokenizer(src_path),
            "tgt": Tokenizer(tgt_path),
        }
