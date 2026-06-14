# Training Guide

Step-by-step guide for preparing data, configuring the model, training, and troubleshooting.

## Dataset Requirements

### Format
Your dataset must be a **parallel corpus** — a file where each row contains a source sentence and its translation.

**Supported formats:**
- **CSV** (comma-separated): Default format
- **TSV** (tab-separated): Set `separator: "\t"` in config

### Columns
The file must have header columns matching your config:
```yaml
data:
  src_column: "source"  # Column name for source language
  tgt_column: "target"  # Column name for target language
```

### Example Dataset

```csv
source,target
The cat sat on the mat.,Le chat s'est assis sur le tapis.
I love machine learning.,J'adore l'apprentissage automatique.
```

### Size Recommendations

| Dataset Size | Expected Quality | Training Time (GPU) |
|-------------|------------------|---------------------|
| < 10K pairs | Low — tokenizer may underfit | Minutes |
| 10K–50K pairs | Moderate — can learn patterns | 1-3 hours |
| 50K–100K pairs | Good — reasonable translations | 3-8 hours |
| 100K+ pairs | Best results | 8+ hours |

## Step-by-Step Workflow

### Step 1: Configure Your Project

Edit `config/default.yaml`:

```yaml
data:
  data_file: "data/raw/corpus.csv"
  src_column: "source"
  tgt_column: "target"
  src_lang: "en"
  tgt_lang: "fr"
  separator: ","
```

### Step 2: Prepare Data

```bash
python scripts/prepare_data.py --config config/default.yaml
```

This will:
1. Load your CSV/TSV file
2. Clean text (Unicode normalization, whitespace cleanup)
3. Split into 80% train / 10% validation / 10% test
4. Train SentencePiece tokenizers for both languages
5. Verify tokenization with a sample roundtrip

### Step 3: Train

```bash
python scripts/train.py --config config/default.yaml
```

You'll see output like:
```
Epoch 1/30 [Train]: 100%|██████████| 156/156 [01:23<00:00, loss=6.8432, lr=1.2e-05]
Epoch 1/30 [Val]:   100%|██████████| 20/20 [00:05<00:00]
Epoch 1/30 | Train Loss: 6.8432 | Val Loss: 6.5210 | LR: 1.23e-05 | Time: 1m 28s
  ✓ New best validation loss: 6.5210
```

### Step 4: Test

```bash
python scripts/test.py --config config/default.yaml
```

### Step 5: Interactive Translation

```bash
python scripts/translate.py --config config/default.yaml
```

## Hyperparameter Tuning

### For Small Datasets (< 30K pairs)

Reduce model capacity to prevent overfitting:

```yaml
model:
  d_model: 256
  n_heads: 4
  n_encoder_layers: 3
  n_decoder_layers: 3
  d_ff: 1024
  dropout: 0.3

training:
  batch_size: 32
  label_smoothing: 0.1

tokenizer:
  vocab_size: 8000
```

### For Large Datasets (> 100K pairs)

Use the full model:

```yaml
model:
  d_model: 512
  n_heads: 8
  n_encoder_layers: 6
  n_decoder_layers: 6
  d_ff: 2048
  dropout: 0.1

training:
  batch_size: 64
  warmup_steps: 4000
```

### Learning Rate

The Noam scheduler automatically handles learning rate warmup and decay. Key settings:
- **warmup_steps**: Higher values = slower warmup, more stable training
- **learning_rate**: This is only used as the initial rate before the scheduler takes over

### Batch Size

- Larger batches → smoother gradients, faster convergence
- Limited by GPU memory
- If you get OOM errors, reduce `batch_size` or `max_seq_len`

## Common Issues

### 1. `FileNotFoundError: Data file not found`
→ Make sure your CSV/TSV is at the path specified in `data.data_file`

### 2. `Source column 'X' not found`
→ Check that your CSV header matches `data.src_column` and `data.tgt_column`

### 3. `CUDA out of memory`
→ Reduce `training.batch_size` (try 32 or 16)
→ Reduce `data.max_seq_len` (try 64)
→ Reduce model size (fewer layers, smaller d_model)

### 4. Training loss not decreasing
→ Check that your data is properly formatted (no empty rows)
→ Increase `training.warmup_steps`
→ Try a smaller learning rate
→ Verify tokenizers are working (run `prepare_data.py` again)

### 5. BLEU score is 0
→ The model hasn't converged yet — train for more epochs
→ Dataset may be too small or noisy
→ Try greedy decoding first: `--beam_size 1`

### 6. Training is very slow
→ Make sure you're using a GPU (`torch.cuda.is_available()` should be True)
→ Reduce `data.max_seq_len`
→ Increase `training.num_workers` for data loading
