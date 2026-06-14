# API Reference

Module-level API documentation for the translation project.

## Data Pipeline

### `src.data.preprocessing`

#### `load_parallel_corpus(data_file, src_column, tgt_column, separator)`
Load a parallel corpus from CSV/TSV.
- **Returns:** `Tuple[List[str], List[str]]` — source and target sentences

#### `split_data(src_sentences, tgt_sentences, train_ratio, val_ratio, test_ratio, seed)`
Split data into train/val/test sets.
- **Returns:** `dict` with keys `'train'`, `'val'`, `'test'`

#### `load_and_split_data(config)`
End-to-end loading and splitting using a config object.
- **Returns:** `Tuple[dict, dict]` — splits and file paths

---

### `src.data.tokenizer`

#### `Tokenizer(model_path=None)`
SentencePiece tokenizer wrapper.
- `train(input_files, model_prefix, vocab_size, model_type, character_coverage)` → `str`
- `load(model_path)` → `None`
- `encode(text, add_bos=True, add_eos=True)` → `List[int]`
- `decode(ids)` → `str`
- `encode_as_pieces(text)` → `List[str]`
- `vocab_size` → `int` (property)

#### `train_tokenizers(config, file_paths)` → `dict`
Train source and target tokenizers.

#### `load_tokenizers(config)` → `dict`
Load previously trained tokenizers.

**Special Token IDs:**
- `PAD_ID = 0`
- `UNK_ID = 1`
- `BOS_ID = 2`
- `EOS_ID = 3`

---

### `src.data.dataset`

#### `TranslationDataset(src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_seq_len)`
PyTorch Dataset for tokenized translation pairs.

#### `create_dataloaders(splits, src_tokenizer, tgt_tokenizer, max_seq_len, batch_size, num_workers)` → `dict`
Create DataLoaders for all splits.

#### Batch Format
Each batch from the DataLoader is a dict:
```python
{
    "src":        Tensor[B, src_len],      # Padded source IDs
    "tgt_input":  Tensor[B, tgt_len-1],    # Target input (BOS → second-to-last)
    "tgt_output": Tensor[B, tgt_len-1],    # Target labels (second → EOS)
    "src_mask":   Tensor[B, 1, 1, src_len], # Source padding mask
    "tgt_mask":   Tensor[B, 1, tgt_len, tgt_len], # Causal + padding mask
}
```

---

## Model

### `src.model.transformer.Transformer`

```python
Transformer(
    src_vocab_size, tgt_vocab_size,
    d_model=512, n_heads=8,
    n_encoder_layers=6, n_decoder_layers=6,
    d_ff=2048, dropout=0.1,
    max_seq_len=128, weight_tying=True,
)
```

- `forward(src, tgt_input, src_mask, tgt_mask)` → `Tensor[B, tgt_len, tgt_vocab]`
- `encode(src, src_mask)` → `Tensor[B, src_len, d_model]`
- `decode(tgt, encoder_output, src_mask, tgt_mask)` → `Tensor[B, tgt_len, d_model]`
- `from_config(config, src_vocab_size, tgt_vocab_size)` → `Transformer` (classmethod)

### `src.model.attention.MultiHeadAttention`
```python
MultiHeadAttention(d_model, n_heads, dropout=0.1)
forward(query, key, value, mask=None) → Tensor
```

### `src.model.encoder.Encoder`
```python
Encoder(n_layers, d_model, n_heads, d_ff, dropout=0.1)
forward(x, src_mask=None) → Tensor
```

### `src.model.decoder.Decoder`
```python
Decoder(n_layers, d_model, n_heads, d_ff, dropout=0.1)
forward(x, encoder_output, src_mask=None, tgt_mask=None) → Tensor
```

---

## Training

### `src.training.trainer.Trainer`

```python
Trainer(model, config, train_loader, val_loader, logger=None)
```

- `train()` → `dict` — Run full training loop, returns history
- `load_checkpoint(path)` — Resume from saved checkpoint

### `src.training.loss.LabelSmoothingLoss`
```python
LabelSmoothingLoss(smoothing=0.1, padding_idx=0)
forward(logits, targets) → Tensor
```

### `src.training.optimizer`
- `get_optimizer(model, learning_rate, betas, eps)` → `Adam`
- `NoamScheduler(optimizer, d_model, warmup_steps)` — call `.step()` after each batch

---

## Evaluation

### `src.evaluation.evaluator.Evaluator`

```python
Evaluator(model, src_tokenizer, tgt_tokenizer, device, beam_size=4, max_decode_len=128)
```

- `translate_sentence(sentence)` → `str`
- `evaluate(src_sentences, ref_sentences, output_file=None, num_samples=10)` → `dict`

### `src.evaluation.inference`
- `greedy_decode(model, src, src_mask, max_len=128)` → `Tensor`
- `beam_search_decode(model, src, src_mask, beam_size=4, max_len=128)` → `Tensor`

### `src.evaluation.metrics`
- `compute_bleu(hypotheses, references)` → `dict` with `bleu`, `precisions`, `brevity_penalty`
- `compute_sentence_bleu(hypothesis, reference)` → `float`

---

## Configuration

### `src.config.TranslationConfig`

```python
config = TranslationConfig.from_yaml("config/default.yaml")
config.apply_overrides({"epochs": 50})
```

Sub-configs: `config.data`, `config.tokenizer`, `config.model`, `config.training`, `config.inference`, `config.logging`
