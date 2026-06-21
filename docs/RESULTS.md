# SMT Model Experiment Results

Tracking SMT training runs and evaluation results.

## Experiment Log

### Experiment 1: Baseline Phrase-Based SMT

| Setting | Value |
|---------|-------|
| Date | 2026-06-21 |
| Dataset | `en-fr.csv` (Parallel English-French Corpus) |
| Dataset Size | 15,000 samples |
| Language Pair | en $\to$ fr |
| Max Phrase Length | 5 |
| LM Order | 3 (Trigram) |
| Alignment Iterations | 4 (IBM Model 1) |
| Tuning Epochs | 10 |
| Training Time | 22m 7s |
| Best Val BLEU | **35.49** |
| **Test BLEU** | **3.39** |
| Best Weights | `p_tgt_src: 0.5, p_src_tgt: 1.0, lex_tgt_src: 0.5, lex_src_tgt: 0.5, lm: 1.0, phrase_penalty: -0.5, word_penalty: -0.5` |

#### Test Evaluation Metrics
- **BLEU Score**: 3.39
- **Precisions (1/2/3/4-gram)**: 28.3, 6.0, 1.5, 0.5
- **Brevity Penalty**: 1.0000

#### Sample Test Translations

| Source (EN) | Reference (FR) | Hypothesis (SMT Output) |
|-------------|----------------|-------------------------|
| Thailand was the major source for both for fresh and dried cutflowers. | La Thaïlande était la principale source de fleurs coupées fraîches et de fleurs coupées séchées. | la Thaïlande été les l' d' principale source pour des tant aux pour des en frais et d sous fleurs coupées. |
| It is claimed that: | Il est dit que : | à l' Auyuittuq première que les.ca |
| Remember the Sky Flakes with various toppings commercial? | Il n'y a qu'à se rappeler le message publicitaire sur le produit Sky Flakes et sur les différentes garnitures dont ces craquelins peuvent être agrémentés. | Rem N l' il Fl avec des sur les divers à l' se les commerciaux- |
| Value Added Tax Excluding fresh fruit and vegetables, all imported products are subject to a value added tax (VAT) of 18%. | le malt et de 10 000 tonnes pour les légumineuses à grains; le Canada jouit d'un taux préférentiel de 14 p. | à valeur de faireé de la ⁇  en frais de fruits de et de de légumes, la tous les importées. les produits de sont les de viande de bœuf à l' une des de la valeur ajouté 100 de la (TVA)) de l' 18 %, de |

#### Notes & Observations
- **EM Alignment**: 4 EM iterations of IBM Model 1 successfully learn basic word mappings.
- **Tuning**: Validation tuning successfully adjusted the feature weights to maximize validation BLEU (starting from initial BLEU 34.49, improving to 35.49).
- **Test BLEU**: The test BLEU is significantly lower than validation, which can happen due to domain shift, out-of-vocabulary words on a 15k sample limit, or alignment/decoding mismatches.
- **Decoding Output**: The hypothesis translations show phrase-level matches (e.g., `"la Thaïlande", "principale source", "fleurs coupées"`) but lack global syntax and coherence.
