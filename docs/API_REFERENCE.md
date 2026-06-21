# SMT API Reference

Module-level API documentation for the SMT translation project.

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

---

### `src.data.dataset`

#### `TranslationDataset(src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_seq_len)`
Python Dataset of tokenized translation pairs (lists of pieces).

#### `DataLoader(dataset, batch_size, shuffle=False, drop_last=False)`
Simple batch iterator for the `TranslationDataset`.

#### `create_dataloaders(splits, src_tokenizer, tgt_tokenizer, max_seq_len, batch_size, num_workers)` → `dict`
Create DataLoaders for all splits.

---

## Model Components

### `src.model.alignment.IBMModel1`
```python
IBMModel1(iterations=10)
```
- `train(src_sentences, tgt_sentences, logger=None)` → `None` — Trains lexical translation probabilities using EM.
- `get_alignment(src_tokens, tgt_tokens)` → `List[Tuple[int, int]]` — Returns the Viterbi word alignment for a sentence pair.

### `src.model.phrase_table.PhraseTable`
```python
PhraseTable(max_phrase_len=5)
```
- `extract_phrases(src_sentences, tgt_sentences, alignments)` → `None` — Heuristically extracts phrases consistent with the word alignments.
- `compute_lexical_weights(aligner_src_tgt, aligner_tgt_src)` → `None` — Calculates directional lexical phrase translation weights.

### `src.model.language_model.LanguageModel`
```python
LanguageModel(order=3)
```
- `train(sentences)` → `None` — Trains an N-gram language model on the target language sentences.
- `get_ngram_prob(ngram)` → `float` — Returns the smoothed probability of a given subword sequence.

### `src.model.smt_model.SMTModel`
```python
SMTModel(max_phrase_len=5, lm_order=3, alignment_iterations=10)
```
- `train_alignment_and_phrases(src_sentences, tgt_sentences, logger=None)` → `None`
- `train_language_model(tgt_sentences, logger=None)` → `None`
- `translate(src_text, src_tokenizer, tgt_tokenizer, beam_size=4, max_decode_len=64)` → `str` — Decodes source text into target string using beam search.
- `save(filepath)` → `None`
- `load(filepath)` → `None`

---

## Training & Tuning

### `src.training.trainer.Trainer`
```python
Trainer(model, config, train_loader, val_loader, logger=None)
```
- `train()` → `dict` — Runs SMT alignment training, phrase table extraction, language model training, and log-linear weight coordinate ascent tuning. Returns tuning history.
- `load_checkpoint(checkpoint_path)` → `None` — Loads a trained model JSON checkpoint.

---

## Evaluation

### `src.evaluation.evaluator.Evaluator`
```python
Evaluator(model, src_tokenizer, tgt_tokenizer, device, beam_size=4, max_decode_len=64)
```
- `translate_sentence(sentence)` → `str`
- `evaluate(src_sentences, ref_sentences, output_file=None, num_samples=15)` → `dict` — Computes corpus-level BLEU on the test set.

### `src.evaluation.inference`
- `beam_search_decode(src_pieces, model, beam_size=4, max_decode_len=64)` → `List[str]` — Searches the space of translation hypotheses to find the highest-scoring target translation pieces.

### `src.evaluation.metrics`
- `compute_bleu(hypotheses, references)` → `dict` — Computes corpus BLEU score using `sacrebleu`.
- `compute_sentence_bleu(hypothesis, reference)` → `float`
