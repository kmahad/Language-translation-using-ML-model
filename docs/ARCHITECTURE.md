# Statistical Machine Translation (SMT) Architecture Documentation

This document explains the Statistical Machine Translation (SMT) architecture implemented in this project. The system is built from scratch as a Phrase-Based SMT model with a log-linear feature combination.

## Overview

The SMT system translates a source sentence by breaking it into phrases, translating those phrases using a phrase translation table, reordering them, and selecting the highest-scoring translation hypothesis according to a log-linear model.

```
Source Sentence → [Tokenizer] → [Decoder / Beam Search] → Target Sentence
                                       ▲
                            [Phrase Table / LM / Weights]
```

## Architecture Components

The architecture consists of the following pipeline components:

1. **Word Alignment (IBM Model 1)** (`src/model/alignment.py`)
2. **Phrase Table Extraction & Lexical Weighting** (`src/model/phrase_table.py`)
3. **N-gram Language Model** (`src/model/language_model.py`)
4. **Log-Linear Model / Weights** (`src/model/smt_model.py`)
5. **Beam Search Decoder** (`src/evaluation/inference.py`)

---

## 1. Word Alignment (IBM Model 1)

Before extracting phrase pairs, we must determine word-level alignments between parallel sentences. We implement **IBM Model 1** trained using the **Expectation-Maximization (EM)** algorithm.

* **Model assumption**: The translation probability of a target sentence given a source sentence is the sum over all possible alignments of the product of translation probabilities of individual words:
  $$P(\mathbf{f} | \mathbf{e}) = \frac{\epsilon}{(I+1)^J} \sum_{a_1=0}^I \cdots \sum_{a_J=0}^I \prod_{j=1}^J t(f_j | e_{a_j})$$
* **EM Training**:
  1. **Expectation (E-step)**: Compute fractional counts of alignments based on current lexical translation probabilities $t(f|e)$.
  2. **Maximization (M-step)**: Re-estimate $t(f|e)$ by normalizing the fractional counts.
* **Bi-directional Alignment**: We run IBM Model 1 in both directions ($src \to tgt$ and $tgt \to src$) and combine/use them to extract high-quality alignments.

**File:** [alignment.py](file:///d:/ML/translator/src/model/alignment.py)

---

## 2. Phrase Table Extraction & Lexical Weighting

Once word alignments are obtained, we extract phrase pairs that are consistent with the alignments.

* **Consistency Criterion**: A source phrase $S$ and target phrase $T$ are consistent with alignment $A$ if:
  1. At least one word in $S$ is aligned to a word in $T$.
  2. No word in $S$ is aligned to a word outside $T$.
  3. No word in $T$ is aligned to a word outside $S$.
* **Probability Estimation**:
  * $P(T|S) = \frac{count(S, T)}{count(S)}$
  * $P(S|T) = \frac{count(S, T)}{count(T)}$
* **Lexical Weights**: Because counts can be sparse, we compute lexical translation weights $lex(T|S)$ and $lex(S|T)$ using word translation probabilities $t(f|e)$ to smooth the phrase-level probabilities:
  $$lex(T|S) = \prod_{i=1}^{|T|} \frac{1}{|A(t_i)|} \sum_{s \in A(t_i)} t(t_i | s)$$

**File:** [phrase_table.py](file:///d:/ML/translator/src/model/phrase_table.py)

---

## 3. N-gram Language Model

The Language Model (LM) measures the fluency of the generated target text. We implement an $N$-gram Language Model (default $N=3$, Trigram) over target subwords.

* **Probability Computation**:
  $$P_{LM}(W) = \prod_{i=1}^{M} P(w_i | w_{i-N+1} \dots w_{i-1})$$
* **Smoothing**: We implement Laplace (add-1) or simple add-$\alpha$ smoothing to handle out-of-vocabulary n-grams and zero counts.

**File:** [language_model.py](file:///d:/ML/translator/src/model/language_model.py)

---

## 4. Log-Linear Model & Weight Tuning

The translation hypotheses are scored using a log-linear model combining several features:

$$Score(S, T) = \sum_{i} \lambda_i h_i(S, T)$$

### Feature Functions ($h_i$)
1. **$p(T|S)$**: Log phrase translation probability ($tgt$ given $src$).
2. **$p(S|T)$**: Log phrase translation probability ($src$ given $tgt$).
3. **$lex(T|S)$**: Log lexical translation weight ($tgt$ given $src$).
4. **$lex(S|T)$**: Log lexical translation weight ($src$ given $tgt$).
5. **$LM$**: Log language model probability of the target sequence.
6. **Phrase Penalty**: Constant cost per phrase to control the preference for longer or shorter phrases.
7. **Word Penalty**: Constant cost per word to control the total length of the target sentence.

### Tuning
We run validation-set tuning (coordinate descent or random search) to find the weights $\lambda_i$ that maximize the BLEU score on the validation subset.

**File:** [smt_model.py](file:///d:/ML/translator/src/model/smt_model.py)

---

## 5. Beam Search Decoder

The search for the best translation is performed by a phrase-based beam search decoder.

* **Hypothesis Generation**: The source sentence is covered step-by-step. For each uncovered span, we look up translation options in the phrase table.
* **Beam Search**: We maintain a queue of hypotheses sorted by log-linear score. At each step, we expand the best hypotheses by translating a new source span and prune the beam to keep only the top `beam_size` candidates.
* **Decoding Parameters**: Controlled by `beam_size` and `max_decode_len` in the configuration.

**File:** [inference.py](file:///d:/ML/translator/src/evaluation/inference.py)

---

## Configuration Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.max_phrase_len` | 5 | Maximum phrase length for phrase table extraction |
| `model.lm_order` | 3 | Order of target language model (e.g. 3 for Trigram) |
| `model.alignment_iterations` | 4 | Number of EM iterations for IBM Model 1 |
| `training.epochs` | 10 | Epochs of weight tuning on validation set |
| `inference.beam_size` | 4 | Beam width for decoding |

## References

1. Koehn, P., et al. (2003). "Statistical Phrase-Based Translation." NAACL.
2. Brown, P. F., et al. (1993). "The Mathematics of Statistical Machine Translation: Parameter Estimation." Computational Linguistics (IBM Models 1–5).
3. Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL.
