"""
Configuration loader for the translation project.

Loads hyperparameters from a YAML file into a structured dataclass,
with support for CLI overrides.
"""

import yaml
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    data_file: str = "data/raw/corpus.csv"
    src_column: str = "source"
    tgt_column: str = "target"
    src_lang: str = "en"
    tgt_lang: str = "fr"
    separator: str = ","
    max_seq_len: int = 128
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    max_samples: Optional[int] = None


@dataclass
class TokenizerConfig:
    """SentencePiece tokenizer configuration."""
    vocab_size: int = 16000
    model_type: str = "bpe"
    character_coverage: float = 1.0
    shared_vocab: bool = False
    model_dir: str = "data/processed"


@dataclass
class ModelConfig:
    """Statistical Machine Translation model configuration."""
    max_phrase_len: int = 5
    lm_order: int = 3
    alignment_iterations: int = 10


@dataclass
class TrainingConfig:
    """Training / weight-tuning configuration."""
    epochs: int = 10
    batch_size: int = 64
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    early_stopping_patience: int = 5
    checkpoint_every: int = 5
    smoke_test: bool = False



@dataclass
class InferenceConfig:
    """Inference / decoding configuration."""
    beam_size: int = 4
    max_decode_len: int = 64


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    level: str = "INFO"
    sample_every: int = 500


@dataclass
class TranslationConfig:
    """Top-level configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TranslationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        config = cls()

        if raw is None:
            return config

        # Map each section to its dataclass
        section_map = {
            "data": (config.data, DataConfig),
            "tokenizer": (config.tokenizer, TokenizerConfig),
            "model": (config.model, ModelConfig),
            "training": (config.training, TrainingConfig),
            "inference": (config.inference, InferenceConfig),
            "logging": (config.logging, LoggingConfig),
        }

        for section_name, (section_obj, _section_cls) in section_map.items():
            section_data = raw.get(section_name, {})
            if section_data:
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

        return config

    def apply_overrides(self, overrides: dict) -> None:
        """Apply flat key overrides like {'epochs': 50, 'beam_size': 4}.

        Searches all sub-configs for matching attribute names.
        """
        for key, value in overrides.items():
            found = False
            for section in [self.data, self.tokenizer, self.model,
                            self.training, self.inference, self.logging]:
                if hasattr(section, key):
                    # Cast to the expected type
                    expected_type = type(getattr(section, key))
                    setattr(section, key, expected_type(value))
                    found = True
                    break
            if not found:
                print(f"Warning: Unknown config override '{key}' — ignored.")


def load_config(config_path: Optional[str] = None) -> TranslationConfig:
    """Load config from YAML, falling back to defaults if no file given."""
    if config_path and Path(config_path).exists():
        return TranslationConfig.from_yaml(config_path)
    return TranslationConfig()


def get_config_from_args() -> TranslationConfig:
    """Parse CLI arguments and return a TranslationConfig.

    Usage:
        python scripts/train.py --config config/default.yaml --epochs 10
    """
    parser = argparse.ArgumentParser(description="Translation Config")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file")

    # Allow flat overrides for any config key
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_phrase_len", type=int, default=None)
    parser.add_argument("--lm_order", type=int, default=None)
    parser.add_argument("--alignment_iterations", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--src_lang", type=str, default=None)
    parser.add_argument("--tgt_lang", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming/testing")
    parser.add_argument("--smoke_test", action="store_true", default=None,
                        help="Run a quick smoke test with minimal steps")

    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config", "checkpoint")}
    if overrides:
        config.apply_overrides(overrides)

    # Store checkpoint path as an extra attribute
    config._checkpoint = getattr(args, "checkpoint", None)

    return config
