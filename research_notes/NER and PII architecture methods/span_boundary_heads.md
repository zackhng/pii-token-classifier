# Span-based, boundary-aware and structured-decoding heads for NER (2019-2026)

Scope: heads that sit on top of a transformer encoder. The goal is to find what could beat or complement our current best, XLM-R base + GlobalPointer (RoPE, multi-label CE). On our structural dev sets that setup gained +0.013 strict F1 over the softmax BIO head, with +0.058 on ADDRESS. Our setting is flat, 8 types, 12 languages plus code-switching, 384-token windows with 96 overlap, strict span F1, and CPU latency matters.

Method note: the numbers below come from the papers' own result tables. I extracted the text from the arXiv/ACL PDFs and read the tables directly. "Flat" means CoNLL03, OntoNotes 5, and the Chinese MSRA, Resume, Weibo and OntoNotes 4 sets. "Nested" means ACE04, ACE05, GENIA, KBP17 and NNE. The papers use different encoders (BERT-base, BERT-large, RoBERTa-base/large), so compare numbers across papers with care.

---

## Q1. What are the main head families, what gains do they report, and how do they compare to GlobalPointer?

### Takeaway
On flat English benchmarks every modern span or structured head lands in a narrow band: CoNLL03 about 92.8-94.0 and OntoNotes 5 about 90.5-91.9. Within each paper, the gains over a strong tagger or biaffine baseline are usually +0.1 to +0.7 F1. The clearest results that carry over to our setting are three:
- **Boundary smoothing** gives consistent small gains on both flat and Chinese data and costs nothing at inference.
- **Deep span representations (DSpERT)** give large gains on long spans.
- **Combining a span head with a sequence tagger** (the two make different errors) helps.

Query-based, diffusion and generative heads (PIQN, DiffusionNER, BartNER) are built for nested NER. On flat benchmarks they do not beat biaffine or W2NER.

### Cited Findings

**GlobalPointer / Efficient GlobalPointer (our current best head)**
- **What it is.** Each type α has start and end projections, q_i,α = W_q,α·h_i and k_j,α = W_k,α·h_j. The span score is s_α(i,j) = q_i,αᵀ R_{j−i} k_j,α, where RoPE injects relative position. The loss is a class-imbalance-aware "multi-label softmax CE". — [Global Pointer: Novel Efficient Span-based Approach for NER, Su et al., arXiv 2022](https://arxiv.org/abs/2208.03054)
- **Results vs BERT-CRF (BERT-base, mean of 5 runs, macro-F1).**

  | Dataset | BERT-CRF | GlobalPointer |
  |---|---|---|
  | People's Daily | 95.46 | 95.51 |
  | CLUENER | 78.70 | 79.44 |
  | CMeEE | 64.39 | 65.98 |
  | CoNLL04 (entities) | 85.46 | 88.57 |
  | GENIA | 73.02 | 74.64 |

  The authors call the gain biggest on "more challenging" datasets. The People's Daily gain (+0.05) is noise. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Ablation on CoNLL04.** Removing RoPE drops F1 from 88.57 to 77.14. Replacing the custom loss with BCE drops it to 88.48 (−0.09, within noise). — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Long entities (CoNLL04, vs PFN).** For entities of 6 or more tokens, F1 is 86.49 vs 79.49 (about +7), but only 7 sentences fall in that bucket. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Efficient GlobalPointer.** This variant factorises the score into a shared "is this an entity" part plus a cheap per-type classifier: s_α(i,j) = q_iᵀk_j + w_αᵀ[q_i; k_i; q_j; k_j]. Each added type then costs 4d parameters instead of 2vd (with v=768 and d=64, about 384× fewer per type). — [Su et al. 2022](https://arxiv.org/abs/2208.03054)

  | Dataset | GlobalPointer | Efficient GlobalPointer |
  |---|---|---|
  | People's Daily | 95.51 | 95.36 |
  | CLUENER (10 types) | 79.44 | 80.04 |
  | CMeEE (9 types) | 65.98 | 66.54 |

  The authors attribute the gain to less overfitting when there are many types. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Speed vs BERT-CRF.** Training is 1.22-1.56× faster and inference 1.0-1.13× faster on GPU. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Caveat.** The GlobalPointer paper is an arXiv preprint with no peer-reviewed venue listed. It compares only against BERT-CRF and PFN, not against biaffine, W2NER or boundary smoothing. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)

**Biaffine NER (Yu, Bohnet & Poesio, ACL 2020)**
- **What it is.** Separate start and end FFNNs feed a biaffine classifier that scores every (start, end) pair over C types. The original encoder stacks BERT + fastText + char-CNN under a BiLSTM. For flat NER, decoding ranks all typed spans by score and greedily keeps each span that does not clash with a higher-ranked one. — [Named Entity Recognition as Dependency Parsing, ACL 2020](https://aclanthology.org/2020.acl-main.577/)
- **Results.** CoNLL03 93.5 (trained on train+dev), OntoNotes 91.3, ACE04 86.7, ACE05 85.4. — reported in the tables of [Boundary Smoothing, Zhu & Li, ACL 2022](https://aclanthology.org/2022.acl-long.490/). Re-implemented by the W2NER authors under their setting, it scores 92.52 on CoNLL03 and 89.89 on OntoNotes. — [W2NER, Li et al., AAAI 2022](https://arxiv.org/abs/2112.10070)
- **Relation to GlobalPointer.** Biaffine adds a bilinear type tensor and a bias term. GlobalPointer's per-type q·k with RoPE is essentially a low-rank, position-aware biaffine. No paper I found compares the two directly on the same encoder (see Gaps).

**Boundary smoothing (Zhu & Li, ACL 2022)**
- **What it is.** Label smoothing for span heads. A fraction ε of each gold span's probability mass is spread over spans whose start and/or end lies within distance D of the gold boundary. The baseline is RoBERTa-base (English) or BERT-base-wwm (Chinese) + BiLSTM + biaffine. — [Boundary Smoothing for NER, ACL 2022](https://aclanthology.org/2022.acl-long.490/) / [arXiv 2204.12031](https://arxiv.org/abs/2204.12031)
- **Test F1, baseline → +BS.**

  | Dataset | Type | Baseline | +BS | Gain |
  |---|---|---|---|---|
  | CoNLL03 | flat | 93.48 | 93.65 | +0.17 |
  | OntoNotes 5 | flat | 91.21 | 91.74 | +0.53 |
  | ACE04 | nested | 87.54 | 87.98 | +0.44 |
  | ACE05 | nested | 86.56 | 87.15 | +0.59 |
  | OntoNotes 4 (zh) | flat | 82.03 | 82.83 | +0.80 |
  | MSRA (zh) | flat | 95.80 | 96.26 | +0.46 |
  | Weibo (zh) | flat | 71.41 | 72.66 | +1.25 |
  | Resume (zh) | flat | 96.34 | 96.66 | +0.32 |

  In 5 of 8 datasets BS raises precision with a slight recall drop. — [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)
- **Hyperparameters.** The paper grid-searched ε ∈ {0.1, 0.2, 0.3} and D ∈ {1, 2}; most settings beat the baseline. The best setting depends on the dataset: CoNLL03 ε=0.3, D=1; ACE05 ε=0.1, D=2; Resume ε=0.2, D=1. Plain label smoothing over types (α = 0.1-0.3) does not help and sometimes hurts, e.g. ACE05 86.56 → 85.65 at α=0.3. — [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)
- **Robustness across encoders.** The gain holds:

  | Encoder | Dataset | Gain from BS |
  |---|---|---|
  | BERT-base | CoNLL03 | +0.21 |
  | BERT-large | CoNLL03 | +0.16 |
  | RoBERTa-large | CoNLL03 | +0.11 |
  | (all encoders) | ACE05 | +0.4 to +0.6 |
  | MacBERT base/large | Resume | +0.29 to +0.34 |
  | No BiLSTM | CoNLL03 / ACE05 / Resume | +0.17 / +0.36 / +0.32 |

  The authors say the gain is "roughly comparable to the gains by switching the pretrained language model from the base size to the large size." Removing the BiLSTM costs 0.1-0.5 F1. — [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)
- **Noise.** The CoNLL03 gains (+0.1 to +0.2) are small, comparable to the seed std reported elsewhere on CoNLL03 (±0.06 to ±0.32 in [DSpERT](https://arxiv.org/abs/2210.04182) Table 3). The OntoNotes 5, ACE and Weibo gains are larger. BS also improves calibration (lower ECE). — [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)

**W2NER (Li et al., AAAI 2022)**
- **What it is.** NER becomes word-word relation classification on a 2D grid with two relation types: NNW (next-neighboring-word) and THW-type (tail-head-word). The grid is built from BERT + BiLSTM, refined with multi-granularity dilated 2D convolutions (dilation 1, 2, 3), and scored by a "co-predictor" that combines an MLP with biaffine. It handles flat, nested and discontinuous entities. — [Unified NER as Word-Word Relation Classification, AAAI 2022](https://arxiv.org/abs/2112.10070)
- **Results (5-run averages).**

  | Dataset | W2NER | Comparison |
  |---|---|---|
  | CoNLL03 | 93.07 | BartNER re-implemented 93.05; biaffine re-implemented 92.52 |
  | OntoNotes 5 | 90.50 | BartNER 90.27; biaffine 89.89 |
  | OntoNotes 4 (zh) | 83.08 | best tagging baseline 82.81 |
  | MSRA (zh) | 96.10 | best tagging baseline 96.09 |
  | Resume (zh) | 96.65 | best tagging baseline 96.11 |
  | Weibo (zh) | 72.32 | best tagging baseline 70.50 |

  — [Li et al. 2022](https://arxiv.org/abs/2112.10070)
- **Ablation on CoNLL03.** Removing all dilated convolutions costs −0.76. Removing the MLP from the co-predictor costs −1.20. Removing biaffine costs −0.05 and removing NNW −0.42; both matter more on discontinuous data (CADEC −4.2 without NNW). — [Li et al. 2022](https://arxiv.org/abs/2112.10070)
- **Speed.** On CADEC (RTX 3090), W2NER (112.3M parameters) trains at 116.1 sent/s and infers at 365.7 sent/s. That is about 3× a span-based baseline and 5× a transition-based one. There is no comparison with a plain softmax tagger. — [Li et al. 2022](https://arxiv.org/abs/2112.10070)
- **Against boundary smoothing.** W2NER's flat English numbers (93.07 / 90.50) are below boundary-smoothed biaffine (93.65 / 91.74) and DSpERT. — [DSpERT, Findings ACL 2023](https://arxiv.org/abs/2210.04182), Table 2

**SpanNER (Fu et al., ACL 2021)**
- **What it is.** The span representation is the boundary-token representations plus a span-length embedding. Flat decoding is heuristic: among overlapping spans, keep the one with the highest probability. — [SpanNER: Named Entity Re-/Recognition as Span Prediction, ACL 2021](https://arxiv.org/abs/2106.00641)
- **Decoding and length features.**

  | Variant | CoNLL03 F1 |
  |---|---|
  | Generic | 91.57 |
  | + heuristic decoding | 91.89 |
  | + length embedding | 92.22 |
  | + both | 92.28 |

  "+both" is significantly better than generic (p < 0.01) on all 11 datasets, e.g. OntoNotes BC 84.58 → 87.54. — [Fu et al. 2021](https://arxiv.org/abs/2106.00641)
- **Span and tagger heads make different errors.** Taggers (SeqLab) beat the generic span model on long entities with low label consistency and on long sentences. Span models do better on OOV-heavy sentences and medium-length entities. With decoding plus length features, span models "do slightly better in long sentences and long entities". When SpanNER is used to combine systems, base systems "with SPANNER gain mainly from long entities". — [Fu et al. 2021](https://arxiv.org/abs/2106.00641)

**PL-Marker (packed levitated markers; Ye et al., ACL 2022)**
- **What it is.** Each candidate span gets start and end "levitated" marker tokens that share position IDs with the span boundaries. Markers for neighbouring spans are packed into one encoder pass (group size K), so the span representation is computed inside the encoder rather than pooled afterwards. — [Packed Levitated Marker for Entity and Relation Extraction, ACL 2022](https://arxiv.org/abs/2109.06067)
- **Flat results (large encoders).**

  | Model | CoNLL03 | OntoNotes 5 | Few-NERD |
  |---|---|---|---|
  | PL-Marker | 94.0 | 91.9 | 70.9 |
  | SeqTagger (same encoder) | 93.6 | 91.2 | 69.0 |
  | T-Concat (span endpoints) | 93.0 | 91.7 | 70.6 |

  — [Ye et al. 2022](https://arxiv.org/abs/2109.06067)
- **Cost.** On an A100 with batch size 32, PL-Marker (K=128) runs at 54.8 sent/s on CoNLL03 vs 138.7 for SeqTagger, about 60% slower. Maximum span length was 8 (CoNLL03) and 16 (Few-NERD). A two-stage variant, where a fast base-size T-Concat model filters candidate spans before PL-Marker, matches PL-Marker's F1 (94.0 CoNLL03, 70.9 Few-NERD) at 83.3 and 79.8 sent/s. — [Ye et al. 2022](https://arxiv.org/abs/2109.06067)

**PIQN (Parallel Instance Query Network; Shen et al., ACL 2022)**
- **What it is.** Learnable instance queries are processed jointly with the tokens; each query predicts one entity (left boundary, right boundary, type). Training uses bipartite matching against gold entities. — [Parallel Instance Query Network for NER, ACL 2022](https://arxiv.org/abs/2203.10545)
- **Results.** Nested gains are +0.37 to +1.23 (e.g. ACE04 88.14, GENIA 81.77). Flat results are not better: CoNLL03 92.87 and OntoNotes 90.96 with BERT-large, both below biaffine with boundary smoothing. Chinese MSRA is 93.48. — [Shen et al. 2022](https://arxiv.org/abs/2203.10545)

**DiffusionNER (Shen et al., ACL 2023)**
- **What it is.** Boundary-denoising diffusion: noisy (start, end) pairs are iteratively refined into entity boundaries. — [DiffusionNER: Boundary Diffusion for NER, ACL 2023](https://arxiv.org/abs/2305.13298)
- **Results.** Nested ACE04 88.39 and GENIA 81.53 (about +0.4 to +0.8). Flat CoNLL03 is 92.78, below W2NER's 93.07; OntoNotes is 90.66 (+0.16 over W2NER); MSRA 94.91 vs 94.97 for W2NER as reproduced in the same paper. Flat gains are within noise. At 10 denoising steps it is 3-18× faster than generative baselines, but it is not compared with a tagger. — [Shen et al. 2023](https://arxiv.org/abs/2305.13298)

**CNN-Nested-NER (Yan et al., "An Embarrassingly Easy but Strong Baseline for Nested NER", ACL 2023 short)**
- **What it is.** A plain 2D CNN over the span score matrix on top of a biaffine/multi-head scorer, which models span-span interactions. — [Yan et al., arXiv 2208.04534](https://arxiv.org/abs/2208.04534)
- **Results (nested only, RoBERTa-base).** ACE04 87.31 vs 87.08 for boundary smoothing re-run by the authors, and 86.48 without the CNN. ACE05 87.42 vs 87.20. On flat entities inside ACE, the CNN improves precision (ACE04 86.3 → 86.9; ACE05 85.2 → 86.2). Its main effect is on nested-entity recall (ACE05 86.2 → 89.0). It has not been evaluated on flat benchmarks. — [Yan et al. 2022/2023](https://arxiv.org/abs/2208.04534)

**DSpERT (Deep Span Representations; Zhu & Li, Findings ACL 2023)**
- **What it is.** Alongside the token Transformer runs a span Transformer. Span representations for every size k ≤ K act as queries that attend over token states at each layer. Span representations are therefore built bottom-up through L̃ layers instead of being pooled from the top layer. The span Transformer is initialised from the PLM. The setup uses RoBERTa-base plus BiLSTM, with boundary smoothing (ε=0.1) applied. — [Deep Span Representations for NER, Findings ACL 2023](https://arxiv.org/abs/2210.04182)
- **Results.** Flat CoNLL03 is 93.70 at best and 93.64±0.06 averaged; OntoNotes 5 is 91.76 at best and 91.06±0.26 averaged. Nested ACE04 reaches 88.31 and GENIA 81.90 (best of 5). — [Zhu & Li 2023](https://arxiv.org/abs/2210.04182)
- **Depth ablation (5-run means).**

  | Configuration | ACE04 | GENIA | CoNLL03 |
  |---|---|---|---|
  | Best shallow pooling | 84.38 | 79.44 | 93.21 |
  | L̃=2 | 87.87 | – | 93.30 |
  | L̃=12 | 88.05 | 81.46 | 93.64 |

  Most of the nested gain arrives by L̃=2; the flat gain grows with depth. — [Zhu & Li 2023](https://arxiv.org/abs/2210.04182)
- **Long spans.** On ACE04, DSpERT beats its shallow counterpart by 2-12 F1 on spans shorter than 10 tokens and by more than 30 F1 on spans longer than 10. Similar patterns hold on GENIA and CoNLL03. — [Zhu & Li 2023](https://arxiv.org/abs/2210.04182)

**Two-stage boundary detection then classification / Locate-and-Label (Shen et al., ACL 2021)**
- **What it is.** A filter proposes seed spans, a regressor adjusts their boundaries, and a classifier then labels them. The method is modelled on two-stage object detectors. — [Locate and Label: A Two-stage Identifier for Nested NER, ACL 2021](https://arxiv.org/abs/2105.06804)
- **Results.** Flat CoNLL03 is 92.94, below W2NER and BS. Weibo is 69.16, MSRA 91.46 (MSRA run by later authors). — tables in [Zhu & Li 2022](https://arxiv.org/abs/2204.12031) and [DiffusionNER](https://arxiv.org/abs/2305.13298)

**Pointer / generative (BartNER; Yan et al., ACL 2021)**
- **What it is.** A BART-large seq2seq model with a pointer mechanism outputs entity start and end indices plus a type. — [A Unified Generative Framework for Various NER Subtasks, ACL 2021](https://arxiv.org/abs/2106.01223)
- **Results.** CoNLL03 is 93.24 (train+dev) and OntoNotes 90.38. — [DSpERT Table 2](https://arxiv.org/abs/2210.04182)
- **Cost.** The model has 408.4M parameters and infers at 19.2 sent/s vs 365.7 for W2NER. — [W2NER Table 8](https://arxiv.org/abs/2112.10070)

**Semi-Markov CRF / Filtered Semi-Markov CRF (Zaratiana et al., Findings EMNLP 2023)**
- **What it is.** A segment-level CRF; FSemiCRF first filters candidate segments, then runs global decoding over a graph of non-overlapping segments. — [Filtered Semi-Markov CRF, arXiv 2311.18028](https://arxiv.org/abs/2311.18028)
- **Results (bert-large, same setup).**

  | Model | CoNLL03 | OntoNotes 5 | Arabic ACE |
  |---|---|---|---|
  | CRF | 92.75 | 89.57 | 83.61 |
  | Semi-CRF | 91.42 | 89.34 | 83.60 |
  | FSemiCRF | 93.89 | 91.00 | 84.46 |

  The vanilla Semi-CRF "often lags behind, either matching or slightly underperforming the CRF". — [Zaratiana et al. 2023](https://arxiv.org/abs/2311.18028)
- **Cost.** Decoding takes 0.2 ms/sample vs 2.7 for CRF (CoNLL03) and 4.4 for CRF vs 27.5 for Semi-CRF (OntoNotes, 18 types). Overall inference is 1.8-6.3× faster than CRF or Semi-CRF. The same group's earlier global span selection model (maximum weighted independent set) reports CoNLL03 93.81. — [Zaratiana et al. 2023](https://arxiv.org/abs/2311.18028)

**Label-aware / query heads**
- **MRC-NER.** Each type is a natural-language query; start/end pointers plus a span-match classifier extract entities, one pass per type. Results: CoNLL03 93.04, OntoNotes 91.11, MSRA 95.75, OntoNotes 4 82.11. — [A Unified MRC Framework for NER, ACL 2020](https://arxiv.org/abs/1910.11476); numbers as tabulated in [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)
- **BINDER (bi-encoder, contrastive span-type matching, ICLR 2023).** CoNLL03 93.33. Speed in words/s on the same machine:

  | Model | Train | Inference |
  |---|---|---|
  | MRC-NER | 147 | 1,110 |
  | Biaffine-NER | 1,548 | 3,634 |
  | BERT-CRF | 2,273 | 8,596 |
  | BINDER | 2,571 | – |

  BINDER is 17× faster to train and 8× faster at inference than MRC-NER, and 1.7× / 2.4× faster than biaffine. — [Optimizing Bi-Encoder for NER via Contrastive Learning, ICLR 2023](https://arxiv.org/abs/2208.14565)
- **PIQN vs MRC.** PIQN improves on type-specific MRC queries by 0.5-3.5 F1 on nested data and runs 30.46× faster on NNE (114 types), because MRC "requires one inference for each type". — [Shen et al. 2022](https://arxiv.org/abs/2203.10545)

**Multilingual evidence (XLM-R)**
- On MultiCoNER 2 (SemEval-2023, multilingual, fine-grained), the CLaC team's best XLM-R span-prediction system scored "slightly higher" than its best sequence-labeling system on test. The abstract gives no figures. Using XLM-R large helped substantially, and both paradigms improved by using XLM-R's `<s>`/`</s>` special tokens. — [CLaC at SemEval-2023 Task 2: Comparing Span-Prediction and Sequence-Labeling approaches for NER, SemEval 2023](https://arxiv.org/abs/2305.03845)

### Inferences
- GlobalPointer is in the same family as biaffine: a per-type pairwise start-end scorer. The +0.013 we measured is in line with the +0.2 to +1.6 F1 that GlobalPointer reports over BERT-CRF. Swapping GlobalPointer for biaffine or W2NER is unlikely to give more than a few tenths of a point on flat data. W2NER's flat numbers do not beat biaffine + BS.
- The more promising additions are orthogonal to how the head scores spans:
  - **Loss and regularisation:** boundary smoothing.
  - **Span representation depth:** DSpERT, PL-Marker.
  - **Decoding:** non-overlap / global span selection, semi-CRF filtering.
  - **Ensembling with the BIO head:** the SpanNER evidence that the two make different errors.
- PIQN, DiffusionNER, CNN-Nested-NER and Locate-and-Label are motivated by nested NER. On flat data they are at or below biaffine + BS. They are low priority for our 8 flat types.

### Gaps
- I found no paper that puts GlobalPointer side by side with biaffine, W2NER or BS on the same encoder and data. GlobalPointer is compared only with BERT-CRF and PFN.
- No head comparison exists on PII-type entities (ADDRESS, ACCOUNT, TIN) or on Thai, Tagalog, Malay or Indonesian.
- I could not confirm whether the boundary smoothing main table is a single run or a multi-seed mean from the parts I read.
- The CLaC multilingual span-vs-tagger F1 gap is not stated in the abstract. I did not read the full paper.

---

## Q2. Which methods help flat NER (not just nested), boundary precision and long entities?

### Takeaway
For flat NER the helpful levers are:
- **Boundary smoothing:** consistent on 4 flat English and Chinese datasets, mostly by raising precision.
- **Deep or in-encoder span representations (DSpERT, PL-Marker):** about +0.2 to +0.7 on flat sets, with large long-span gains in DSpERT.
- **Span-length features plus non-overlap decoding (SpanNER):** about +0.7 on CoNLL03 over a generic span model.
- **Global segment decoding (FSemiCRF):** +1.1 over CRF on CoNLL03 with bert-large.

For long entities, the most direct evidence is DSpERT's long-span gains and GlobalPointer's own long-entity bucket. For boundary precision specifically, the evidence points to boundary smoothing and to RoPE / relative position.

### Cited Findings
- **Boundary smoothing improves flat NER in every dataset tested:** CoNLL03 +0.17, OntoNotes +0.53, OntoNotes 4 +0.80, MSRA +0.46, Weibo +1.25, Resume +0.32. It "significantly increases the precision rate with a slight drop in the recall" in 5/8 datasets. — [Zhu & Li, ACL 2022](https://arxiv.org/abs/2204.12031)
- **DSpERT is effective on flat data.** Flat CoNLL03 is 93.64±0.06 (5-run mean) vs 92.99-93.21 for shallow pooling. On long spans, DSpERT exceeds shallow aggregation by more than 30 F1 for ACE04 spans longer than 10 tokens, with "similar patterns" on CoNLL03. — [Zhu & Li, Findings ACL 2023](https://arxiv.org/abs/2210.04182)
- **PL-Marker on flat data** improves on SeqTagger by +0.4 (CoNLL03), +0.7 (OntoNotes 5) and +1.9 (Few-NERD, 66 types). The gain grows with the number of types. — [Ye et al., ACL 2022](https://arxiv.org/abs/2109.06067)
- **Length embedding and non-overlap decoding** each help a span model: CoNLL03 91.57 → 92.22 and 91.89 respectively, 92.28 together. Both help on long entities. — [Fu et al., ACL 2021](https://arxiv.org/abs/2106.00641)
- **Span heads gain on long entities, taggers on short ones.** In system combination, base taggers benefit most on short entities and span-based systems on long entities. — [Fu et al., ACL 2021](https://arxiv.org/abs/2106.00641)
- **GlobalPointer's long-entity bucket.** For entities of 6 or more tokens, F1 is +7 over PFN on CoNLL04, from a small sample (7 sentences). Relative position via RoPE is the key component: without it F1 falls by 11.43. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **W2NER on flat data.** It improves Weibo (72.32 vs 70.50) and Resume (96.65 vs 96.11) over Chinese tagging baselines. Its CoNLL03 and MSRA gains are at most 0.02-0.03, which is noise. — [Li et al., AAAI 2022](https://arxiv.org/abs/2112.10070)
- **Global segment decoding on flat data.** FSemiCRF beats token CRF on CoNLL03 (93.89 vs 92.75) and OntoNotes (91.00 vs 89.57) with the same bert-large encoder. Vanilla Semi-CRF does not. — [Zaratiana et al. 2023](https://arxiv.org/abs/2311.18028)
- **Nested-only evidence.** CNN-Nested-NER, PIQN and DiffusionNER report their main gains on nested data. Their flat gains are within noise or negative. — [Yan et al. 2022](https://arxiv.org/abs/2208.04534); [Shen et al. 2022](https://arxiv.org/abs/2203.10545); [Shen et al. 2023](https://arxiv.org/abs/2305.13298)

### Inferences
- Our ADDRESS (+0.058) and BUSINESS gains from GlobalPointer match the literature's pattern that span heads help long entities. Further long-entity gains are most likely from:
  - **Deeper span representations.** DSpERT-lite with L̃=2 captures most of the nested gain at lower cost.
  - **Decoding that enforces non-overlap.**
- Boundary smoothing should help strict F1 on PII types whose boundaries are fuzzy: titles in PERSON, trailing punctuation or postcodes in ADDRESS, and "Ltd./Sdn Bhd" in BUSINESS. This is our inference. The paper shows the effect on generic NER only.

### Gaps
- No per-type or per-length breakdown of boundary smoothing on flat data was found. Its long-entity effect is unknown.
- No study measures cross-window (sliding-window) boundary errors for span heads vs taggers.

---

## Q3. What is the inference cost relative to a softmax head, and how practical is each on CPU?

### Takeaway
Simple pairwise heads (GlobalPointer, Efficient GlobalPointer, biaffine with a small hidden size) add negligible compute compared with an XLM-R base forward pass at n=384. Their decoding is cheap thresholding or greedy selection. The expensive heads are:
- **Grid-CNN heads (W2NER, CNN-Nested):** convolution over n² cells.
- **In-encoder span representations:** PL-Marker is about 60% slower on GPU; DSpERT trains about 5× slower and adds O(K²Td) attention.
- **Generative and diffusion heads.**

Token CRF Viterbi decoding is sequential and can dominate decode time. Filtered semi-CRF decoding is faster.

### Cited Findings
- **GlobalPointer vs BERT-CRF on GPU.** Inference is 1.0-1.13× and training 1.22-1.56× as fast. Per-type parameters are 2vd for GlobalPointer and 4d for Efficient GlobalPointer. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Biaffine vs BERT-CRF.** Biaffine-NER infers at 3,634 words/s vs 8,596 for BERT-CRF and trains at 1,548 vs 2,273 (same machine). — [BINDER, ICLR 2023](https://arxiv.org/abs/2208.14565)
- **W2NER.** It infers at 365.7 sent/s on CADEC vs 19.2 for BartNER (RTX 3090). — [Li et al. 2022](https://arxiv.org/abs/2112.10070)
- **PL-Marker.** It infers at 54.8 sent/s vs 138.7 for SeqTagger on CoNLL03 (A100, large encoder) and slows further as group size grows (22.9 sent/s at K=512). The two-stage filter variant runs at 83.3 sent/s with equal F1. — [Ye et al. 2022](https://arxiv.org/abs/2109.06067)
- **DSpERT.** It adds O(K²Td) attention and O(KTd²) FFN complexity, with training "about five times the time for a shallow model". Fewer span layers reduce the cost. Entities longer than K are "theoretically irretrievable"; K was 10 for CoNLL03 and 25 for ACE. — [Zhu & Li 2023](https://arxiv.org/abs/2210.04182)
- **CRF vs Semi-CRF vs FSemiCRF decode time (ms/sample).**

  | Dataset | CRF | Semi-CRF | FSemiCRF |
  |---|---|---|---|
  | CoNLL03 | 2.7 | 3.7 | 0.2 |
  | OntoNotes | 4.4 | 27.5 | 0.2 |

  Overall FSemiCRF is up to 6× faster than CRF and 2× faster than Semi-CRF (base encoder). CRF trains fastest. — [Zaratiana et al. 2023](https://arxiv.org/abs/2311.18028)
- **DiffusionNER.** At 10 denoising steps it is 3-18× faster than generative NER, but each step reruns the decoder. — [Shen et al. 2023](https://arxiv.org/abs/2305.13298)
- **MRC-NER.** It needs one forward pass per type, which makes it 8× slower at inference than BINDER. — [BINDER, ICLR 2023](https://arxiv.org/abs/2208.14565)

### Inferences
These are back-of-envelope estimates, not measured.
- **XLM-R base forward pass.** With about 86M non-embedding parameters, the forward pass at n=384 costs about 2·86M·384 ≈ 66 GFLOPs.
- **GlobalPointer head (8 types, d=64).** The projections cost about 384·768·(2·64·8) ≈ 0.3 GFLOP-equivalent MACs. The pairwise scores cost 384²·8·64 ≈ 75M MACs. Together that is under 1% of the encoder.
- **Biaffine with a 150-dimensional hidden layer.** Also around 1% of the encoder.
- **W2NER-style dilated 2D convolutions.** These run over about 147k cells; with tens of channels and 3×3 kernels they could reach several to tens of GFLOPs. That is a real CPU cost, and it should be benchmarked before adoption.
- **GlobalPointer memory.** The n² × types logits (384² × 8 ≈ 1.2M floats per window) are fine on CPU.
- **Width limit.** A width limit of W tokens cuts scoring to n·W cells. For PII, about 64 tokens should cover ADDRESS; this needs checking against our length distribution.
- **What CPU latency really depends on.** The softmax head is the cheapest, but GlobalPointer's extra cost on CPU is dominated by the encoder, not the head. CPU latency for these heads depends mostly on decoding and window stitching, not FLOPs.

### Gaps
- None of the papers report CPU latency. All speed numbers are GPU.
- No measured head-only latency for GlobalPointer vs softmax on CPU was found. We should measure it ourselves; the repo already has latency harness outputs in `results/*latency*`.

---

## Q4. How can the tricks be combined (boundary smoothing on span heads, span-length embeddings, width limits, constrained decoding)?

### Takeaway
The literature already stacks these:
- DSpERT uses boundary smoothing (ε=0.1), a BiLSTM and a max span size K.
- SpanNER shows length embedding and non-overlap decoding add up (+0.71 on CoNLL03).
- PL-Marker shows a cheap span-proposal filter in front of an expensive classifier keeps accuracy at 1.5-3× the speed.

Boundary smoothing needs soft targets, so it fits the BCE / softmax losses of biaffine-style heads directly. Applying it to GlobalPointer's multi-label CE requires a soft-label version of that loss.

### Cited Findings
- **Boundary smoothing on span heads.** BS is "easy to implement and integrate into any span-based neural NER models, with almost no side effects". The biaffine baseline uses softmax over (types + O) per span with soft targets. — [Zhu & Li 2022](https://arxiv.org/abs/2204.12031)
- **DSpERT stacks the tricks:** deep spans, BS with ε=0.1, a BiLSTM before the classifier, and a dataset-specific maximum span size. — [Zhu & Li 2023](https://arxiv.org/abs/2210.04182)
- **SpanNER combination:** length embedding plus heuristic non-overlap decoding gives +0.71 on CoNLL03 and is significant on 11/11 datasets. — [Fu et al. 2021](https://arxiv.org/abs/2106.00641)
- **Biaffine's flat decoding:** rank spans by score and accept a span only if it does not clash with a higher-ranked one. — [Yu et al., ACL 2020](https://aclanthology.org/2020.acl-main.577/)
- **CNN-Nested-NER's decoding** symmetrises scores, P̂_ij = (P_ij + P_ji)/2. It then prunes spans scoring 0.5 or less and greedily selects by maximum entity score, following Yu et al. — [Yan et al. 2022](https://arxiv.org/abs/2208.04534)
- **Global vs greedy selection.** FSemiCRF's global loss plus segment-graph decoding beats greedy span selection (maximum weighted independent set) on OntoNotes (91.00 vs 90.71) and Arabic ACE (84.46 vs 84.49, a tie). Removing the global loss costs 0.4-0.8. — [Zaratiana et al. 2023](https://arxiv.org/abs/2311.18028)
- **Two-stage cascade.** A fast span filter followed by PL-Marker matches full PL-Marker F1 at 1.5-3.1× the speed. — [Ye et al. 2022](https://arxiv.org/abs/2109.06067)
- **Efficient GlobalPointer's shared extraction term** works like an "entity vs non-entity" boundary detector plus a type classifier, i.e. a one-stage version of boundary detection followed by typing. — [Su et al. 2022](https://arxiv.org/abs/2208.03054)
- **Hybrid tagger + span loss on multilingual data.** A MultiCoNER 2022 system kept a CRF token head on XLM-R and added a span-classification loss. It ranked 6th (multilingual) and 8th (code-mixed). This comes from the search-result summary; I did not read the full paper, so treat it as weak evidence. — search result via [CLaC SemEval-2023 listing](https://arxiv.org/abs/2305.03845) (related work, not primary)

### Inferences
- Standard GlobalPointer decoding in common implementations keeps every (i, j, type) with a score above 0, which can yield overlapping or nested predictions in a flat task. Adding biaffine/SpanNER-style greedy non-overlap decoding is a free inference-time change. It should raise precision on flat strict F1. This is our inference about typical GlobalPointer code; check `src/predict.py`.
- For sliding windows, span scores from overlapping windows can be averaged per (i, j, type) before greedy decoding. That is more natural than BIO label voting and may reduce cross-window boundary splits. This is our inference.

### Gaps
- I found no published soft-label variant of GlobalPointer's multi-label CE loss with experiments. Su's blog (kexue.fm) was inaccessible (HTTP 403).
- There are no published results for BS + GlobalPointer.

---

## Q5. Proposed single-variable experiments, in priority order

Each experiment changes exactly one variable against XLM-R base + GlobalPointer ("GP") or XLM-R base + softmax BIO ("SM"). Evaluate on the structural dev sets, ml_real, en_test and the per-type and per-length slices, with at least 3 seeds. CoNLL-scale seed std is about ±0.06 to ±0.3, per [DSpERT](https://arxiv.org/abs/2210.04182).

### Takeaway
The cheapest high-value experiments are:
1. Non-overlap greedy decoding on GP outputs (inference only).
2. Boundary smoothing on GP.
3. Efficient GlobalPointer.
4. Span-score averaging across window overlaps.
5. A width or length prior.

Next come the medium-cost representation changes: DSpERT-lite (L̃=2) and a GP + BIO ensemble. W2NER, PL-Marker, FSemiCRF and biaffine are lower priority. Skip PIQN, DiffusionNER, MRC and generative heads.

### Cited Findings (basis for each experiment)

| # | Experiment (one variable) | Baseline | Evidence | Expected cost |
|---|---|---|---|---|
| E1 | **Non-overlap greedy decoding**: rank GP spans by score and drop any span that overlaps a higher-scoring one. Inference-time only. | GP | +0.32 CoNLL03 from heuristic decoding in [SpanNER](https://arxiv.org/abs/2106.00641); flat decoding in [Biaffine](https://aclanthology.org/2020.acl-main.577/) | ~0 |
| E2 | **Boundary smoothing** ε ∈ {0.1, 0.2}, D=1 on GP targets. Requires GP with a soft-label-capable loss; run GP+BCE hard targets as the control first, since BCE was −0.09 vs GP's loss in [GlobalPointer](https://arxiv.org/abs/2208.03054). | GP (BCE) | +0.17 to +1.25 flat in [BS](https://arxiv.org/abs/2204.12031) | ~0 inference |
| E3 | **Efficient GlobalPointer** (shared q·k extraction + per-type linear) instead of per-type q/k | GP | +0.6 CLUENER, +0.56 CMeEE, −0.15 People's Daily in [GlobalPointer](https://arxiv.org/abs/2208.03054) | fewer params |
| E4 | **Cross-window span-logit averaging**: average GP scores for (i, j, type) across the 96-token overlap before decoding, instead of keeping one window's predictions | GP | none published (our inference) | ~0 |
| E5 | **Span-width prior**: add a learned per-type bias by length bucket to the GP score; or alternatively a hard width cap W (e.g. 64) | GP | +0.65 CoNLL03 from length embedding in [SpanNER](https://arxiv.org/abs/2106.00641) | ~0 |
| E6 | **DSpERT-lite**: replace GP's top-layer q/k with span reps from a 2-layer span Transformer (K≈32-64) initialised from the last 2 XLM-R layers | GP | depth 2 recovers most nested gain; large long-span gains in [DSpERT](https://arxiv.org/abs/2210.04182) | moderate inference, ~5× training at full depth |
| E7 | **GP + BIO ensemble** (union/intersection or SpanNER-style rescoring of both heads' spans) on one shared encoder trained with both losses | SM and GP | complementary errors in [SpanNER](https://arxiv.org/abs/2106.00641) | small |
| E8 | **Biaffine head** (Yu 2020, no BiLSTM) instead of GP | GP | biaffine ≈ GP family; BS was designed on biaffine, see [BS](https://arxiv.org/abs/2204.12031) | small |
| E9 | **W2NER grid head** instead of GP | GP | Weibo +1.8, Resume +0.5 over tagging baselines; flat EN below biaffine+BS in [W2NER](https://arxiv.org/abs/2112.10070) | high CPU, n² CNN |
| E10 | **FSemiCRF** instead of CRF / softmax | SM | +1.1 over CRF on CoNLL03 in [FSemiCRF](https://arxiv.org/abs/2311.18028) | low decode cost |
| E11 | **Two-stage**: GP proposals → PL-Marker span re-classifier on top-k candidates | GP | +0.4 to +1.9 over tagger; two-stage keeps F1 at 1.5-3× speed in [PL-Marker](https://arxiv.org/abs/2109.06067) | high (second encoder pass) |

Experiment notes:
- **E1:** Report precision change on ADDRESS, BUSINESS and PERSON.
- **E2:** Use a single global ε at first. The BS paper found the best (ε, D) depends on the dataset.
- **E3:** Few types (8), so the effect is likely small. It may still help low-resource languages by sharing the extraction scorer across types.
- **E4:** Directly targets errors from long entities cut at window edges.
- **E5:** Check how this interacts with RoPE, which already encodes j − i.
- **E6:** Train with ε=0.1 BS as in the paper only in a follow-up, so that this experiment changes one variable. Target the ADDRESS and BUSINESS slices.
- **E7:** For a strict one-variable test, change only the decode rule on fixed trained models.
- **E8:** Only worth it as a BS carrier or to test whether bilinear type interaction beats per-type q·k.
- **E10:** A cheaper alternative to GP for the "structured decoding" question.
- **Do not prioritise:**
  - **PIQN and DiffusionNER:** flat gains within noise, and both target nested NER ([PIQN](https://arxiv.org/abs/2203.10545), [DiffusionNER](https://arxiv.org/abs/2305.13298)).
  - **CNN-Nested-NER:** nested-only evaluation ([Yan et al.](https://arxiv.org/abs/2208.04534)).
  - **MRC-NER:** 8× slower inference ([BINDER](https://arxiv.org/abs/2208.14565)).
  - **BartNER:** 408M parameters, 19 sent/s ([W2NER](https://arxiv.org/abs/2112.10070)).
  - **Label-aware bi-encoders (BINDER):** fine, but with 8 fixed types the label-semantics benefit is mainly for zero-shot or new types.

### Inferences
- **Recommended order.** Run E1 and E4 first: no retraining, they directly address flat strict-F1 precision and window boundaries. Then E2, E3 and E5 (cheap retrains). Then E6 or E7 if ADDRESS and BUSINESS remain the main error sources.
- **Expected size of gains.** E2 is the best-supported training-side change, with consistent flat gains across English and Chinese and both base and large encoders. Gains are likely a few tenths of strict F1. That is comparable to our measured +0.013 only on noisier slices such as Weibo-like social text, so multi-seed evaluation is required.

### Gaps
- **No PII-specific or multilingual evidence:** I found no published evaluation of any of these heads on PII entity types, or on th, tl, ms, id or hi.
- **Code-switching:** no evidence on how these heads behave on code-switched text.
- **Not verified:** GLiNER (span-label matching, NAACL 2024, [arXiv 2311.08526](https://arxiv.org/abs/2311.08526)) and 2024-2026 follow-ups to DSpERT and W2NER were not checked in this pass. Their numbers are not reported here.
