# Transformer Architecture Documentation

This document explains the Transformer architecture implemented in this project, based on the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Overview

The Transformer is a sequence-to-sequence model that relies entirely on attention mechanisms, dispensing with recurrence and convolutions. It processes all positions in parallel, making it highly efficient to train on modern GPUs.

```
Source Sentence → [Encoder] → Memory → [Decoder] → Target Sentence
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER MODEL                         │
│                                                              │
│  ┌─────────────────┐           ┌─────────────────────┐      │
│  │    ENCODER       │           │     DECODER          │      │
│  │                  │           │                      │      │
│  │  ┌────────────┐  │           │  ┌────────────────┐  │      │
│  │  │ Encoder    │  │           │  │ Decoder Layer  │  │      │
│  │  │ Layer ×N   │  │  Memory   │  │ ×N             │  │      │
│  │  │            │  │ ────────► │  │                │  │      │
│  │  │ Self-Attn  │  │           │  │ Masked Self-   │  │      │
│  │  │ + FFN      │  │           │  │ Attn + Cross-  │  │      │
│  │  │            │  │           │  │ Attn + FFN     │  │      │
│  │  └────────────┘  │           │  └────────────────┘  │      │
│  │                  │           │                      │      │
│  │  Positional Enc  │           │  Positional Enc      │      │
│  │  + Embedding     │           │  + Embedding         │      │
│  └─────────────────┘           └─────────────────────┘      │
│         ▲                              ▲        │            │
│         │                              │        ▼            │
│    Source Tokens                  Target Tokens  Output       │
│                                  (shifted)      Projection   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Token Embeddings (`embeddings.py`)

Converts token IDs into dense vectors of dimension `d_model`.

```python
embedding(x) = Embedding(x) × √d_model
```

The scaling by `√d_model` ensures the embeddings have a similar magnitude to the positional encodings.

**File:** `src/model/embeddings.py`

### 2. Positional Encoding (`positional_encoding.py`)

Since the Transformer has no recurrence, it needs explicit position information. We use fixed sinusoidal encodings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties:
- Each position gets a unique encoding
- Relative positions can be learned via linear combinations
- Generalizes to longer sequences than seen during training
- Not learned — registered as a buffer

**File:** `src/model/positional_encoding.py`

### 3. Multi-Head Attention (`attention.py`)

The core mechanism of the Transformer. Computes how much each token should "attend to" every other token.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What information do I provide?
- **Scaling by √d_k**: Prevents softmax saturation in high dimensions

**Multi-Head:** Instead of one attention function, we run `n_heads` attention functions in parallel with different learned projections:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_o
where headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᵏ, V·Wᵢᵛ)
```

This allows the model to attend to information from different representation subspaces.

**Masking:**
- **Padding mask**: Prevents attention to `<pad>` tokens
- **Causal mask**: Prevents decoder from seeing future tokens (lower triangular matrix)

**File:** `src/model/attention.py`

### 4. Feed-Forward Network (`feed_forward.py`)

Applied independently to each position:

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

The inner dimension `d_ff` (default 2048) is typically 4× the model dimension, providing additional representational capacity.

**File:** `src/model/feed_forward.py`

### 5. Encoder (`encoder.py`)

Stack of N identical layers (default N=6). Each layer:

```
x → Multi-Head Self-Attention → Add & LayerNorm → FFN → Add & LayerNorm → output
```

- **Self-Attention**: Each position attends to all positions in the input
- **Residual connections**: Help gradient flow through deep networks
- **LayerNorm**: Stabilizes training by normalizing activations

**File:** `src/model/encoder.py`

### 6. Decoder (`decoder.py`)

Stack of N identical layers (default N=6). Each layer has three sub-layers:

```
x → Masked Self-Attention → Add & Norm
  → Cross-Attention (over encoder output) → Add & Norm
  → FFN → Add & Norm → output
```

- **Masked Self-Attention**: Attends only to previous positions (causal masking)
- **Cross-Attention**: Queries come from decoder, keys/values from encoder output
- This is how the decoder "reads" the source sentence

**File:** `src/model/decoder.py`

### 7. Full Transformer (`transformer.py`)

Assembles all components:

1. Source tokens → Source Embedding → Encoder → Memory
2. Target tokens → Target Embedding → Decoder (+ Memory) → Output
3. Output → Linear Projection → Target Vocabulary Logits

**Weight Tying**: Optionally shares weights between the target embedding layer and the output projection, reducing parameters and often improving quality.

**File:** `src/model/transformer.py`

## Hyperparameter Choices

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `d_model` | 512 | Standard base model size; good balance of capacity and speed |
| `n_heads` | 8 | d_model/n_heads = 64 per head; standard choice |
| `d_ff` | 2048 | 4× d_model; provides additional capacity |
| `n_layers` | 6+6 | Standard depth; deeper models need more data |
| `dropout` | 0.1 | Standard regularization for medium datasets |
| `label_smoothing` | 0.1 | Prevents overconfident predictions |
| `warmup_steps` | 4000 | Noam schedule; stabilizes early training |

## Parameter Count

With default settings:
- Embedding layers: ~16.4M (2 × vocab_size × d_model)
- Encoder: ~12.6M
- Decoder: ~16.8M
- Output projection: shared with target embedding
- **Total: ~30M parameters** (with weight tying)

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer." EMNLP.
3. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL.
