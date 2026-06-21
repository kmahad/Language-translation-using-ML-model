# Statistical Machine Translation (SMT) — From-Scratch Phrase-Based SMT

A complete, configurable Phrase-Based Statistical Machine Translation (SMT) system built entirely from scratch in Python. Train on any language pair using your own parallel corpus (CSV/TSV).

## Features

- **IBM Model 1 Word Alignment** — implemented from scratch with Expectation-Maximization (EM) training in both source-to-target and target-to-source directions.
- **Phrase Extraction & Lexical Weighting** — heuristic phrase extraction from alignments and lexical weight estimation for extracted phrase pairs.
- **N-gram Language Model** — target language n-gram language model with Kneser-Ney or simple smoothing.
- **Log-Linear Model & Tuning** — combines translation features (phrase translation probabilities, lexical weights, language model score, phrase penalty, word penalty) via a log-linear model, with iterative weight tuning on a validation set to maximize BLEU.
- **SentencePiece BPE tokenization** — language-agnostic subword tokenization, works with any script.
- **Beam Search Decoding** — beam search decoder to efficiently search the phrase-translation space.
- **BLEU evaluation** — standardized scoring with sacrebleu.
- **Checkpointing** — saves trained alignments, phrase tables, language models, and weights to a JSON checkpoint.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Provide a CSV file (e.g. `../en-fr.csv`) with parallel sentences:

```csv
en,fr
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

Train the word aligner, extract the phrase table, train the language model, and tune log-linear weights:

```bash
python scripts/train.py --config config/default.yaml
```

Override settings on the fly:
```bash
python scripts/train.py --config config/default.yaml --epochs 10 --max_samples 15000
```

### 5. Evaluate on Test Set

Evaluate the trained model on the test split:

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
│   │   └── dataset.py           # PyTorch/Python Dataset helpers
│   ├── model/                   # SMT components (from scratch)
│   │   ├── alignment.py         # IBM Model 1 Alignment (EM)
│   │   ├── phrase_table.py      # Phrase Extraction & Lexical Probabilities
│   │   ├── language_model.py    # N-gram target language model
│   │   └── smt_model.py         # SMT Model class tying everything together
│   ├── training/                # Training/Tuning pipeline
│   │   └── trainer.py           # Log-linear weight tuning loop
│   ├── evaluation/              # Evaluation pipeline
│   │   ├── inference.py         # Beam search decoder
│   │   ├── metrics.py           # BLEU scores
│   │   └── evaluator.py         # Full test-set evaluation
│   └── utils/                   # Helpers
├── scripts/                     # CLI entry points
│   ├── prepare_data.py
│   ├── train.py
│   ├── test.py
│   └── translate.py
├── checkpoints/                 # Saved models (auto-generated)
├── logs/                        # Logs & translations (auto-generated)
└── docs/                        # Documentation
```

## Configuration

All settings are in [`config/default.yaml`](config/default.yaml). Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `data.data_file` | `../en-fr.csv` | Path to parallel corpus |
| `data.src_lang` | `en` | Source language code |
| `data.tgt_lang` | `fr` | Target language code |
| `data.max_samples` | `15000` | Max sample limit for efficiency |
| `model.max_phrase_len` | `5` | Maximum phrase length for extraction |
| `model.lm_order` | `3` | N-gram language model order (e.g. trigram) |
| `model.alignment_iterations` | `4` | IBM Model 1 EM iterations |
| `training.epochs` | `10` | Tuning epochs |
| `tokenizer.vocab_size` | `16000` | Vocabulary size |
| `inference.beam_size` | `4` | Beam search width |

## Requirements

- Python 3.8+
- sacrebleu
- pyyaml
- sentencepiece
- numpy
- pandas
- tqdm

## License

MIT
