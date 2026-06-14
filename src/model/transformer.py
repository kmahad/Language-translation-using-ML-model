"""
Full Transformer Model for Sequence-to-Sequence Translation.

Assembles all components:
    Source Embedding → Encoder → Target Embedding → Decoder → Output Projection

Supports optional weight tying between target embedding and output
projection layer.
"""

import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """Complete Transformer encoder-decoder for translation.

    Args:
        src_vocab_size: Source language vocabulary size.
        tgt_vocab_size: Target language vocabulary size.
        d_model: Model/embedding dimension (default: 512).
        n_heads: Number of attention heads (default: 8).
        n_encoder_layers: Number of encoder layers (default: 6).
        n_decoder_layers: Number of decoder layers (default: 6).
        d_ff: Feed-forward inner dimension (default: 2048).
        dropout: Dropout rate (default: 0.1).
        max_seq_len: Maximum sequence length (default: 128).
        weight_tying: Whether to tie target embedding and output weights.
        padding_idx: Padding token index (default: 0).
    """
    d_model: int
    padding_idx: int
    src_embedding: TransformerEmbedding
    tgt_embedding: TransformerEmbedding
    encoder: Encoder
    decoder: Decoder
    output_projection: nn.Linear

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        weight_tying: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx

        # Source and target embeddings (separate for different vocabularies)
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_seq_len, dropout, padding_idx
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_seq_len, dropout, padding_idx
        )

        # Encoder and Decoder stacks
        self.encoder = Encoder(
            n_encoder_layers, d_model, n_heads, d_ff, dropout
        )
        self.decoder = Decoder(
            n_decoder_layers, d_model, n_heads, d_ff, dropout
        )

        # Output projection: d_model → target vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: share weights between target embedding and output projection
        if weight_tying:
            self.output_projection.weight = self.tgt_embedding.token_embedding.weight

        # Initialize parameters with Xavier uniform
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform distribution.

        Embedding weights are initialized with N(0, d_model^-0.5) following
        the original Transformer paper, which ensures output logits have
        unit variance when weight tying is used.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1 and "embedding" not in name:
                nn.init.xavier_uniform_(param)

        # Scale embedding weights properly (critical for weight tying)
        nn.init.normal_(self.src_embedding.token_embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.tgt_embedding.token_embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        # Re-zero padding vector
        with torch.no_grad():
            self.src_embedding.token_embedding.weight[self.padding_idx].zero_()
            self.tgt_embedding.token_embedding.weight[self.padding_idx].zero_()

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for training (teacher forcing).

        Args:
            src: Source token IDs [batch_size, src_len].
            tgt_input: Target input IDs (shifted right) [batch_size, tgt_len].
            src_mask: Source padding mask [batch_size, 1, 1, src_len].
            tgt_mask: Target mask [batch_size, 1, tgt_len, tgt_len].

        Returns:
            Logits over target vocabulary [batch_size, tgt_len, tgt_vocab_size].
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt_input, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode source sequence (used during inference).

        Args:
            src: Source token IDs [batch_size, src_len].
            src_mask: Source padding mask.

        Returns:
            Encoder output [batch_size, src_len, d_model].
        """
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode target sequence given encoder output (used during inference).

        Args:
            tgt: Target token IDs [batch_size, tgt_len].
            encoder_output: Encoder output [batch_size, src_len, d_model].
            src_mask: Source padding mask.
            tgt_mask: Target mask.

        Returns:
            Decoder output [batch_size, tgt_len, d_model].
        """
        tgt_embedded = self.tgt_embedding(tgt)
        return self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

    @classmethod
    def from_config(cls, config, src_vocab_size: int, tgt_vocab_size: int) -> "Transformer":
        """Create a Transformer from a TranslationConfig.

        Args:
            config: TranslationConfig instance.
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.

        Returns:
            Configured Transformer instance.
        """
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            max_seq_len=config.data.max_seq_len,
            weight_tying=config.model.weight_tying,
        )
