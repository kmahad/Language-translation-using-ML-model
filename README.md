# Neural Machine Translation — From-Scratch Transformer

A complete, configurable Transformer-based translation system built entirely from scratch in PyTorch. Train on any language pair using your own parallel corpus (CSV/TSV).

## Features

- **Full Transformer from scratch** — every component (Multi-Head Attention, Encoder, Decoder, Positional Encoding) implemented by hand
- **SentencePiece BPE tokenization** — language-agnostic, works with any script
- **Configurable language pairs** — just change the YAML config for any source ↔ target
- **Complete pipeline** — data prep → train → evaluate → interactive translation
- **BLEU evaluation** — standardized scoring with sacrebleu
- **Beam search & greedy decoding** — configurable inference strategies
- **Checkpointing & early stopping** — save best model, resume training
- **Comprehensive documentation** — architecture guide, training guide, API reference

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create a CSV file at `data/raw/corpus.csv` with two columns:

```csv
source,target
Hello,Bonjour
How are you?,Comment allez-vous?
Thank you very much,Merci beaucoup
...
```

### 3. Prepare Data & Train Tokenizers

```bash
python scripts/prepare_data.py --config config/default.yaml
```

### 4. Train the Model

```bash
python scripts/train.py --config config/default.yaml
```

Override settings on the fly:
```bash
python scripts/train.py --config config/default.yaml --epochs 50 --batch_size 32
```

Resume from a checkpoint:
```bash
python scripts/train.py --config config/default.yaml --checkpoint checkpoints/best.pt
```

### 5. Evaluate on Test Set

```bash
python scripts/test.py --config config/default.yaml
```

### 6. Interactive Translation

```bash
python scripts/translate.py --config config/default.yaml
```

## Project Structure

```
translator/
├── config/default.yaml          # All hyperparameters
├── data/
│   ├── raw/                     # Your CSV/TSV datasets
│   └── processed/               # Tokenized data (auto-generated)
├── src/
│   ├── config.py                # YAML config loader
│   ├── data/                    # Data pipeline
│   │   ├── preprocessing.py     # CSV loading, cleaning, splitting
│   │   ├── tokenizer.py         # SentencePiece wrapper
│   │   └── dataset.py           # PyTorch Dataset + DataLoader
│   ├── model/                   # Transformer (from scratch)
│   │   ├── attention.py         # Multi-Head Attention
│   │   ├── feed_forward.py      # Position-wise FFN
│   │   ├── positional_encoding.py
│   │   ├── embeddings.py        # Token + Positional Embeddings
│   │   ├── encoder.py           # Encoder stack
│   │   ├── decoder.py           # Decoder stack
│   │   └── transformer.py       # Full model assembly
│   ├── training/                # Training pipeline
│   │   ├── loss.py              # Label-smoothed CrossEntropy
│   │   ├── optimizer.py         # Adam + Noam LR scheduler
│   │   └── trainer.py           # Training loop
│   ├── evaluation/              # Evaluation pipeline
│   │   ├── inference.py         # Greedy + Beam search
│   │   ├── metrics.py           # BLEU scores
│   │   └── evaluator.py         # Full test-set evaluation
│   └── utils/                   # Helpers
├── scripts/                     # CLI entry points
│   ├── prepare_data.py
│   ├── train.py
│   ├── test.py
│   └── translate.py
├── checkpoints/                 # Saved models (auto-generated)
├── logs/                        # Training logs (auto-generated)
└── docs/                        # Documentation
```

## Configuration

All settings are in [`config/default.yaml`](config/default.yaml). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `data.src_lang` | `en` | Source language code |
| `data.tgt_lang` | `fr` | Target language code |
| `model.d_model` | `512` | Embedding/hidden dimension |
| `model.n_heads` | `8` | Number of attention heads |
| `model.n_encoder_layers` | `6` | Encoder layers |
| `model.n_decoder_layers` | `6` | Decoder layers |
| `training.batch_size` | `64` | Batch size |
| `training.epochs` | `30` | Training epochs |
| `tokenizer.vocab_size` | `16000` | Vocabulary size |
| `inference.beam_size` | `4` | Beam search width |

## Dataset Format

Your dataset should be a CSV or TSV file with parallel sentences:

**CSV (default):**
```csv
source,target
Hello world,Bonjour le monde
Good morning,Bonjour
```

**TSV:**
```
source	target
Hello world	Bonjour le monde
```

For TSV, set `separator: "\t"` in the config.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, CPU works but is ~10-50× slower)
- 10K–100K sentence pairs for meaningful results

## License

MIT
