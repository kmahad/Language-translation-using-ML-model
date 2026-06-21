# SMT Training & Tuning Guide

Step-by-step guide for preparing parallel data, training tokenizers, building the phrase table, training the language model, and tuning log-linear weights.

## Dataset Requirements

### Format
Your dataset must be a **parallel corpus** — a CSV or TSV file where each row contains a source sentence and its matching translation.

**Supported formats:**
- **CSV** (comma-separated): Default format
- **TSV** (tab-separated): Set `separator: "\t"` in config

### Columns
The file must have header columns matching your config:
```yaml
data:
  src_column: "en"  # Column name for source language
  tgt_column: "fr"  # Column name for target language
```

### Size & Performance Recommendations

| Dataset Size | Recommended Max Samples | Training Time (CPU) |
|-------------|-------------------------|---------------------|
| < 10K pairs | All samples             | < 2 minutes         |
| 10K–50K pairs | 15,000 (default)        | ~15-25 minutes      |
| 50K+ pairs   | 15,000 - 30,000         | ~30-60 minutes      |

*Note: SMT training runs entirely on the CPU and requires computing alignments and extracting millions of phrase rules. Setting `data.max_samples` (e.g. `15000`) protects against memory exhaustion and long runs on large datasets.*

---

## Step-by-Step Workflow

### Step 1: Configure Your Project

Edit `config/default.yaml` to point to your parallel corpus and define the column names:

```yaml
data:
  data_file: "../en-fr.csv"
  src_column: "en"
  tgt_column: "fr"
  src_lang: "en"
  tgt_lang: "fr"
  separator: ","
  max_samples: 15000
```

### Step 2: Prepare Data & Train Tokenizers

```bash
python scripts/prepare_data.py --config config/default.yaml
```

This script will:
1. Load your dataset and check column names.
2. Clean the text (lowercasing, unicode normalization, whitespace cleanup).
3. Split into 80% train / 10% validation / 10% test.
4. Train SentencePiece BPE tokenizers for both languages.
5. Save the tokenized corpus.

### Step 3: Run the Training Pipeline

```bash
python scripts/train.py --config config/default.yaml
```

This runs the SMT pipeline:
1. **Word Alignment**: IBM Model 1 EM training (bi-directional: source $\to$ target and target $\to$ source).
2. **Phrase Extraction**: Extract consistent phrase pairs from Viterbi alignments.
3. **Lexical Weighting**: Calculate lexical translation weights for extracted phrases.
4. **Language Model**: Train a trigram language model on the target language training sentences.
5. **Log-Linear weight tuning**: Iteratively tune the log-linear weights on the validation subset to maximize BLEU score.
6. **Save Checkpoint**: Saves the entire model checkpoint to `checkpoints/best.json`.

### Step 4: Run Evaluation on Test Set

Evaluate the model on the test split:

```bash
python scripts/test.py --config config/default.yaml
```

This will run the decoder on the test sentences, output translations to `logs/translations.tsv`, and report the final corpus BLEU score using `sacrebleu`.

### Step 5: Interactive Translation

Translate custom text interactively:

```bash
python scripts/translate.py --config config/default.yaml
```

---

## Hyperparameter Tuning

### Phrase Length (`model.max_phrase_len`)
- Default is `5`.
- Larger phrase lengths capture longer context but increase phrase table size exponentially, requiring more memory and decoding time.

### Alignment Iterations (`model.alignment_iterations`)
- Default is `4`.
- Controls the number of EM training iterations for IBM Model 1. Typically 4-10 iterations are sufficient for model convergence.

### Language Model Order (`model.lm_order`)
- Default is `3` (Trigram).
- High-order n-gram language models capture longer target fluency context but can suffer from sparsity.

### Beam Size (`inference.beam_size`)
- Default is `4`.
- Controls the width of the beam search. Larger beam sizes find higher scoring translations but slow down inference. Greedy decoding corresponds to a beam size of `1`.
