"""
Inference: Greedy decoding and Beam Search for translation.

These methods generate translations token-by-token at inference time,
since the full target sequence is unknown.
"""

import torch
import torch.nn.functional as F

from ..data.tokenizer import BOS_ID, EOS_ID, PAD_ID
from ..data.dataset import create_padding_mask, create_causal_mask


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int = 128,
    repetition_penalty: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Greedy decoding: pick the highest-probability token at each step.

    Args:
        model: Trained Transformer model.
        src: Source token IDs [1, src_len] (single sentence).
        src_mask: Source padding mask [1, 1, 1, src_len].
        max_len: Maximum number of tokens to generate.
        repetition_penalty: Repetition penalty (1.0 = no penalty, >1.0 penalizes already generated tokens).
        device: Device to run on.

    Returns:
        Generated token IDs [1, generated_len] (including BOS, up to EOS).
    """
    if device is None:
        device = src.device

    model.eval()

    # Encode source once
    encoder_output = model.encode(src, src_mask)

    # Start with BOS token
    tgt_ids = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        # Create target mask (causal + padding)
        tgt_mask = create_causal_mask(tgt_ids.size(1)).to(device)

        # Decode
        decoder_output = model.decode(tgt_ids, encoder_output, src_mask, tgt_mask)

        # Get logits for the last position
        logits = model.output_projection(decoder_output[:, -1, :])  # [1, vocab_size]

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token in set(tgt_ids[0].tolist()):
                if token not in (PAD_ID, BOS_ID, EOS_ID):
                    val = logits[0, token].item()
                    if val > 0:
                        logits[0, token] /= repetition_penalty
                    else:
                        logits[0, token] *= repetition_penalty

        # Pick the highest probability token
        next_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]

        # Append to generated sequence
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

        # Stop if EOS is generated
        if next_token.item() == EOS_ID:
            break

    return tgt_ids


@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Beam search decoding with length normalization.

    Maintains `beam_size` hypotheses at each step and selects the
    best complete sequence based on normalized log-probability.

    Args:
        model: Trained Transformer model.
        src: Source token IDs [1, src_len].
        src_mask: Source padding mask [1, 1, 1, src_len].
        beam_size: Number of beams to maintain.
        max_len: Maximum number of tokens to generate.
        length_penalty: Alpha for length normalization (0 = no penalty).
        repetition_penalty: Repetition penalty (1.0 = no penalty, >1.0 penalizes already generated tokens).
        device: Device to run on.

    Returns:
        Best generated token IDs [1, generated_len].
    """
    if device is None:
        device = src.device

    model.eval()

    # Encode source once
    encoder_output = model.encode(src, src_mask)  # [1, src_len, d_model]

    # Expand for beam search: [beam_size, src_len, d_model]
    encoder_output = encoder_output.repeat(beam_size, 1, 1)
    src_mask_expanded = src_mask.repeat(beam_size, 1, 1, 1)

    # Initialize beams: each beam starts with [BOS]
    # beams: list of (sequence, cumulative_log_prob)
    beams = [(torch.tensor([[BOS_ID]], dtype=torch.long, device=device), 0.0)]
    completed = []

    for step in range(max_len - 1):
        all_candidates = []

        for seq, score in beams:
            # If this beam already ended with EOS, move to completed
            if seq[0, -1].item() == EOS_ID:
                completed.append((seq, score))
                continue

            # Create target mask
            tgt_mask = create_causal_mask(seq.size(1)).to(device)

            # Decode (use first beam's encoder output since they're all the same)
            decoder_output = model.decode(
                seq,
                encoder_output[:1],  # [1, src_len, d_model]
                src_mask,
                tgt_mask,
            )

            # Get log probabilities for the last position
            logits = model.output_projection(decoder_output[:, -1, :])  # [1, vocab_size]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in set(seq[0].tolist()):
                    if token not in (PAD_ID, BOS_ID, EOS_ID):
                        val = logits[0, token].item()
                        if val > 0:
                            logits[0, token] /= repetition_penalty
                        else:
                            logits[0, token] *= repetition_penalty

            log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]

            # Get top-k candidates
            top_log_probs, top_ids = log_probs.topk(beam_size, dim=-1)  # [1, beam_size]

            for i in range(beam_size):
                token_id = top_ids[0, i].unsqueeze(0).unsqueeze(0)  # [1, 1]
                token_score = top_log_probs[0, i].item()

                new_seq = torch.cat([seq, token_id], dim=1)
                new_score = score + token_score

                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        # Select top beam_size candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_size]

        # Check if all beams are complete
        if all(seq[0, -1].item() == EOS_ID for seq, _ in beams):
            completed.extend(beams)
            break

    # Add remaining beams to completed
    completed.extend(beams)

    # Apply length normalization and select best
    def normalize_score(seq, score):
        length = seq.size(1) - 1  # exclude BOS
        if length_penalty > 0 and length > 0:
            return score / (length ** length_penalty)
        return score

    completed.sort(key=lambda x: normalize_score(x[0], x[1]), reverse=True)

    return completed[0][0] if completed else beams[0][0]
