# PII Detection: Datasets, Systems, Winning Architectures and Tricks (2019-2026), plus Financial NER

Context for fit notes: our v5.0 baseline is an XLM-R base token classifier with a GlobalPointer span head. It covers 12 languages plus English+X code-switching, with 8 labels (PERSON, ADDRESS, ACCOUNT, BUSINESS, DOB, PHONE, EMAIL, TIN). Inference uses sliding windows, with defaults max_length=384 and stride=96 (`src/predict.py` line 53; `src/tokenize_bio.py` `window_encode`). A grep of `src/` found no class weighting, O-threshold, checksum validator or ensembling code in the current pipeline.

Source-type legend: [peer-reviewed] = journal or ACL-venue paper; [arXiv] = preprint, not peer-reviewed; [blog/card] = vendor blog, model card or dataset card; [competition] = Kaggle write-up or discussion, or a third-party summary of one.

## 1. PII datasets and benchmarks, and the model results reported on them

### Takeaway
Public PII corpora are mostly synthetic and LLM-generated (AI4Privacy, Gretel, Nemotron-PII, SPY). Models score well in-distribution, but cross-dataset transfer is poor:
- NVIDIA GLiNER-PII reaches 0.87 strict F1 on its own Nemotron data, 0.70 on Argilla and 0.64 on AI4Privacy.
- On the 10-source PIIBench consolidation, every off-the-shelf system scores below 0.14 span F1.
- A plain fine-tuned DeBERTa reaches about 0.65 on PIIBench.

For our purposes, a held-out per-source evaluation matters more than any in-distribution number.

### Cited Findings
**AI4Privacy open-pii-masking-500k** [blog/card]
- About 580k rows (464k train / 116k validation), CC-BY-4.0. The sample rows show English, French, German, Spanish, Italian, Dutch, Hindi and Telugu, covering 11 regions in Latin, Devanagari and Telugu scripts.
- Labels include GIVENNAME, SURNAME, STREET, BUILDINGNUM, ZIPCODE, EMAIL, TELEPHONENUM, IDCARDNUM, PASSPORTNUM, SOCIALNUM and TAXNUM. Spans are given as `privacy_mask` plus mBERT BIO tokens.
- The fetched page did not show the generation method, model results, or how it relates to pii-masking-300k/400k.
- Some sample rows contain noisy mixed-language text, for example a Chinese character inside German/Dutch text.
- Source: [HF dataset card](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)

**AI4Privacy and code-switching** [arXiv]
- The REDACT paper notes that AI4Privacy spans 23 languages but "uses templated generation that limits behavioral diversity".
- REDACT itself has 13,427 records, 324,078 entity annotations, 51 types and 25 languages in 9 scripts. Its generation axes include format, difficulty, density, code-switching and adjacency.
- Source: [REDACT, arXiv 2606.19881](https://arxiv.org/pdf/2606.19881). I saw only the search snippet, not the full text.

**Gretel gretel-pii-masking-en-v1** [blog/card]
- 60k rows (50k / 5k / 5k), English only, Apache-2.0.
- 47 industry domains, including finance, banking, financial services, insurance and digital payments.
- About 45 types, including account_number, bank_routing_number, swift_bic, credit_card_number, tax_id, ssn and date_of_birth. IBAN is not listed.
- Generated with Gretel Navigator on a mistral-nemo-2407 backend, with automated validation.
- The card reports no metrics. It only links fine-tuned GLiNER small/base/large models.
- Source: [HF dataset card](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1)

**NVIDIA Nemotron-PII** [blog/card]
- 100K fully synthetic records (50k train / 50k test) built with NeMo Data Designer.
- Covers 55+ PII/PHI types and mixes structured and unstructured formats (forms, logs, emails, free text).
- Grounded in Nemotron-Personas, with US and international formats.
- Source: [NVIDIA HF blog, "Nemotron PII"](https://huggingface.co/blog/nvidia/nemotron-pii)

**NVIDIA GLiNER-PII** [blog/card]
- About 570M parameters.
- The model card names urchade/gliner_large-v2.1 as the base model. The NVIDIA build/NIM pages name knowledgator/gliner-bi-large-v1.0. The sources conflict on this.
- Strict F1: 0.87 on Nemotron-PII, 0.70 on Argilla PII, 0.64 on AI4Privacy. Evaluated at threshold 0.3; the usage example uses 0.5. Tagged English.
- Sources: [HF model card nvidia/gliner-PII](https://huggingface.co/nvidia/gliner-PII); [NVIDIA build model card](https://build.nvidia.com/nvidia/gliner-pii/modelcard)

**PIIBench (Jha, arXiv 2604.15776)** [arXiv; single independent author]
- Consolidates 10 public datasets: synthetic PII corpora, multilingual NER benchmarks and financial-domain annotated text.
- 2,369,883 sequences and 3.35M mentions. 80+ source labels are mapped to 48 canonical types in BIO, with stratified 80/10/10 splits.
- Eight published systems score below 0.14 span F1: Presidio, spaCy, BERT-base NER, XLM-R NER, two SpanMarker models, Piiranha DeBERTa and XtremeDistil FiNER. The best is Presidio at 0.1385, which still has zero recall on most types.
- Source: [arXiv 2604.15776](https://arxiv.org/abs/2604.15776)

**DeBERTa fine-tuned on PIIBench (arXiv 2605.25816)** [arXiv]
- Results (entity-level support-weighted F1):

| Model | test_5k F1 | Full 100k test F1 |
|---|---|---|
| Plain direct token-classification fine-tune | 0.6476 | 0.6455 |
| Source-conditioned hierarchical model (SC+H) | 0.5899 | 0.5894 |
| SC+H plus 3-phase curriculum | 0.2772 | not reported |
| Best published comparator | 0.1723 | not reported |

- Validation had initially favoured SC+H, but the plain model won on test.
- The plain model won 54 of 82 fine-grained types and all 10 coarse groups.
- The authors attribute the gains to diverse task data plus "a simple weighted cross-entropy objective", not to architecture.
- Source: [Fine-Tuning Over Architectural Complexity, arXiv 2605.25816](https://www.alphaxiv.org/abs/2605.25816)

**PII-Bench (Shen et al., Fudan, arXiv 2502.18545)** [arXiv]
- 2,842 test samples, 7 PII types with 55 subcategories. Each sample has a query, a context and the query-relevant PII.
- The task is query-aware masking of prompts to LLMs. Models do "adequately in basic PII detection" but struggle to judge query relevance, especially with multiple subjects.
- This is a different task framing from ours.
- Source: [PII-Bench, arXiv 2502.18545](https://arxiv.org/html/2502.18545v2)

**SPY (Savkin, Ionov, Konovalov)** [peer-reviewed: NAACL 2025 Student Research Workshop, pp. 236-246]
- Faker generates fake entities, and Llama-3-70B writes text around them. No external data is needed, and the method is domain-adaptable.
- Released data: 4,491 medical consultations and 4,197 questions.
- Finding: dedicated NER models are limited on PII-specific contexts, because PII is not the same as named entities.
- Sources: [ACL Anthology 2025.naacl-srw.23](https://preview.aclanthology.org/setup/2025.naacl-srw.23); [code and data on GitHub](https://github.com/LogicZMaksimka/SPY_Dataset)

**TAB, Text Anonymization Benchmark (Pilán et al.)** [peer-reviewed: Computational Linguistics 48(4), 2022, pp. 1053-1101]
- 1,268 English ECHR court cases, annotated with semantic category, identifier type (DIRECT / QUASI), confidential attributes and coreference.
- Baselines: spaCy RoBERTa-OntoNotes NER, Presidio, and a fine-tuned Longformer (IOB, single MASK label).
- Fine-tuned Longformer, window 4096, label weight (10,1), test set: R_di+qi 0.919, ER_di 1.000, ER_qi 0.916, P 0.836, WP 0.850.
- RoBERTa was 2-4 F1 points below Longformer in early experiments.
- Source: [ACL Anthology 2022.cl-4.19](https://aclanthology.org/2022.cl-4.19); [arXiv 2202.00443](https://arxiv.org/abs/2202.00443v2)

**TAB transfer to Wikipedia biographies** [peer-reviewed: same paper]

| Model | R | ER_di | ER_qi | P | WP |
|---|---|---|---|---|---|
| TAB-fine-tuned Longformer | 0.952 | 0.999 | 0.923 | 0.708 | 0.765 |
| OntoNotes RoBERTa | 0.845 | 0.810 | 0.801 | 0.770 | 0.836 |

- Source: [arXiv 2202.00443](https://arxiv.org/abs/2202.00443v2)

**Kaggle "PII Data Detection" (Learning Agency Lab, 2024)** [competition]
- Student essays, BIO labels. Scored with micro F5 at spaCy-English-token level.
- A third-party write-up notes that because F5 weights recall heavily, precision around 0.2 can still give F5 around 0.9.
- The organizers recommended external data because some labels were extremely rare.
- Source: [Zenn retrospective, Japanese](https://zenn.dev/sinchir0/articles/396967387196dc)

**Piiranha v1** [blog/card]
- mdeberta-v3-base, 256-token context, 17 PII types, 6 European languages.
- Self-reported: 98.27% PII token catch rate, 99.44% accuracy.
- The HF card lists the licence as cc-by-nc-nd-4.0, while MarkTechPost reports MIT. These sources conflict.
- An independent German/English/Dutch test gave F1 0.677.
- Sources: [HF card](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information); [MarkTechPost](https://www.marktechpost.com/2024/09/14/piiranha-v1-released-a-280m-small-encoder-open-model-for-pii-detection-with-98-27-token-detection-accuracy-supporting-6-languages-and-17-pii-types-released-under-mit-license/)

**"Unmasking the Reality of PII Masking Models" (arXiv 2504.12308)** [arXiv]
- Evaluates Piiranha (300k+ downloads) and StarPII (580k+ downloads) on about 17K semi-synthetic sentences with 16 PII types from India, the UK and the US.
- Six dimensions: basic recognition, contextual disambiguation, noisy data, novel entities, cross-lingual, adversarial.
- Per a search summary of its table, StarPII missed 54% vs Piiranha 22% in basic entity recognition. I did not verify this in the full text.
- The authors say context words, non-diverse training data and out-of-distribution tests substantially degrade performance.
- Source: [arXiv 2504.12308](https://arxiv.org/abs/2504.12308)

**OpenAI Privacy Filter (OPF) evaluation (arXiv 2608.02616)** [arXiv]
- OPF is a 1.5B autoregressive LM converted into a bidirectional detector with 33 types. It is evaluated on 32 benchmarks, 14 languages and 5 domains, including SPY, AI4Privacy and Kiji.
- Zero-shot OPF scores 0.464 F1 on SPY and 0.855 on AI4Privacy, beating Presidio and XLM-R-large-NER zero-shot.
- Fine-tuned XLM-R overtakes zero-shot OPF with about 500 labelled English synthetic examples, about 100 non-English Kiji examples, and about 1,000 synthetic medical examples.
- OPF collapses on non-Latin scripts (Arabic 0.04, Cyrillic 0.03).
- Per-type F1: email 0.78, phone 0.76, person 0.40, address 0.49.
- Recall-biased: precision 0.31-0.54, recall 0.70-0.85.
- Source: [arXiv 2608.02616](https://arxiv.org/abs/2608.02616)

### Inferences
- GLiNER-PII's 0.64-0.70 on foreign PII sets, and the sub-0.14 off-the-shelf scores on PIIBench, mean synthetic-trained detectors transfer badly across label schemas and generators. The leave-one-source-out results we already produce (AI4Privacy / Nemotron / Gretel / Kiii slices) are the right lens. Label-schema mismatch probably explains much of the PIIBench zero-shot collapse, so not all of it is model weakness.
- The OPF paper's "XLM-R fine-tuned beats a 1.5B zero-shot model with 100-1,000 examples" supports keeping a fine-tuned XLM-R encoder over swapping in an LLM detector, especially for ar/th/hi scripts.
- Address (0.49) and person (0.40) are the hard classes for general models. Email and phone are easy. Our effort should go to PERSON, ADDRESS, BUSINESS and the numeric classes (ACCOUNT, TIN).

### Gaps
- I did not find published model results for pii-masking-200k/300k/400k on dataset cards. The 500k card text I could see had no metrics or generation description.
- I could not confirm StarPII's architecture from its card. It is commonly described as an encoder for code PII, but I did not verify that in this session.
- The Kaggle final leaderboard scores (private LB) were not retrievable: the Kaggle pages render only titles to the fetcher.
- I found no public benchmark on wealth-management or private-banking documents, nor any public PII benchmark covering th, vi, ms, id or tl specifically. REDACT's 25-language list was not checked.

## 2. Architectures and tricks that won

### Takeaway
The repeated winners are:
- A strong encoder (DeBERTa-v3-large in English).
- Down-weighting or thresholding the O class to buy recall.
- Long context windows (1-4k tokens).
- Diverse external or synthetic data.
- Ensembles of architecture-diverse heads with Optuna-tuned weights and thresholds.
- Cheap rule-based post-processing.

In industry, Presidio-style regex plus checksum plus context-word validators sit beside NER. LLMs are competitive mainly when fine-tuned, or for semantic disambiguation, at much higher cost.

### Cited Findings
**Kaggle 1st place solution** [competition]
- Five DeBERTa-v3-large models with diverse heads: multi-sample dropout; a BiLSTM with special initialization to avoid NaN loss; knowledge distillation from the best multi-sample-dropout teacher.
- Augmentation: name swaps (exp073) and random insertion of names (exp076).
- Trained on 8x A100 40GB.
- Sources: [GitHub mirror of the 1st-place repo](https://github.com/ympaik87/pii-detection-1st-place); [Kaggle discussion 497374, title only retrievable](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)

**Top-5 Kaggle solutions, per a third-party summary** [competition; not primary]
Source for all items below: [Zenn retrospective](https://zenn.dev/sinchir0/articles/396967387196dc)
- 1st place:
  - Kept no-PII samples, with O-class loss weight 0.05.
  - LSTM head plus knowledge distillation.
  - 10 models with max_length 1600-2048.
  - External data from nbroad and mpware.
  - Weighted softmax voting with a vote threshold. Weights and thresholds were tuned per model with Optuna on CV.
  - Per-label thresholds, plus rules such as a casing check on NAME_STUDENT.
- 2nd place:
  - Averaged sub-token probabilities per word (`is_split_into_words`).
  - Dropped B-/I- prefixes, leaving 7 classes.
  - 6 models with max_length 512/1024/2048 and stride 32.
  - nbroad data with weight 0.5.
  - Post-processing deleted single-character or non-title-case names.
- 3rd place: one DeBERTa pretrained on mpware data, then fine-tuned. Post-processing used an O threshold and converted I-start spans to B.
- 4th place:
  - Focal loss with BiLSTM and GRU heads.
  - A [SPACE] token for whitespace.
  - Trained at max_length 1280; inference at 4000 with stride 1024.
  - Names paraphrased with the Dipper paraphraser for augmentation.
  - Unicode normalization; rules removing names that contain digits.
- 5th place:
  - Lower O weight, LSTM, partly frozen layers.
  - 12 models with inference windows up to 4096 and stride 384.
  - Voting ensemble.
  - Regex and article-based rules (e.g. a non-name following "The"); AMP for speed.
- Common to all top 5: DeBERTa-v3-large plus nbroad and mpware external data.

**TAB: class weighting as an explicit recall/precision dial** [peer-reviewed]
Longformer results by window size and (mask, O) loss weight:

| Window | Weight | R | P |
|---|---|---|---|
| 4096 | (1,1) | 0.860 | 0.925 |
| 4096 | (5,1) | 0.916 | 0.843 |
| 4096 | (10,1) | 0.935 | 0.795 |
| 512 | (10,1) | 0.937 | 0.767 |

- The authors note the larger window mainly gave higher precision.
- Source: [arXiv 2202.00443](https://arxiv.org/abs/2202.00443v2)

**Presidio analyzer design** [docs, not peer-reviewed]
- Runs per-entity recognizers that use "regex, Named Entity Recognition and other types of logic". A `PatternRecognizer` combines regex, context words, and validation and invalidation logic.
- The default `LemmaContextAwareEnhancer` boosts scores using lemmatized context words.
- Recognizers are tagged by `country_code`. NLP engines include spaCy, stanza and Hugging Face, and GLiNER can be plugged in as an external recognizer.
- Source: [Presidio analyzer docs](https://presidio.dataprivacystack.org/analyzer/)

**Presidio IBAN recognizer** [code]
- A generic regex starts at base score 0.5.
- `validate_result` applies the mod-97 check (rearrange, map A=10..Z=35, compare 98 - n mod 97 with the check digits), then a per-country length and structure regex.
- A valid match gets MAX_SCORE and an invalid one MIN_SCORE (dropped). An uppercase-only match stays at 0.5.
- If a longer candidate fails, it retries shorter capture groups, so a trailing word like "2" in "...3000 2 days" is dropped.
- Context words: iban, bank, transaction.
- Source: [presidio iban_recognizer.py](https://raw.githubusercontent.com/microsoft/presidio/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/generic/iban_recognizer.py)

**Robustness of the three detector families** [arXiv]
- Compares spaCy NER, Presidio (rule hybrid) and Qwen2.5-3B (LLM) on 7 categories of natural distribution shift. All three degrade substantially.
- The failure modes differ:
  - spaCy: unfamiliar surface forms and boundary errors.
  - Presidio: non-standard formats.
  - LLM: type confusion and unstable generation.
- "No single architecture is reliable across all PII categories."
- Source: [Mind the Gap, arXiv 2609.03464](https://arxiv.org/pdf/2609.03464)

**LLM-based detection vs fine-tuned models** [arXiv]
- Fine-tuned GPT-4o-mini on educational data reached recall 0.9589 on CRAPII. Precision tripled and compute cost was about one tenth of Azure AI Language. The comparison was against Presidio and Azure, not against a fine-tuned encoder. Source: [arXiv 2501.09765](https://www.arxiv.org/pdf/2501.09765)
- Fine-tuned T5-small and Mistral-7B-Instruct on AI4Privacy-derived data were "comparable to frontier LLMs". Mistral had higher F1 and recall but "significantly higher generation latency". Source: [arXiv 2512.18608](https://arxiv.org/html/2512.18608v1)
- RECAP (regex plus context LLM, with a 3-phase disambiguation and filtering pipeline) covers 13 low-resource locales, including Hindi, Vietnamese and two Chinese locales. It reports +82% weighted F1 over fine-tuned NER and +17% over zero-shot LLMs. Absolute numbers and cost were not in the abstract. Source: [arXiv 2510.07551](https://arxiv.org/abs/2510.07551)
- Domain-aware LLM prompting cut numeric false positives on math tutoring dialogue: Presidio 0.379 F1, math-aware prompting 0.802, segment-aware prompting 0.821. Source: [MathEd-PII, arXiv 2602.16571](https://arxiv.org/abs/2602.16571)
- On PIIBench, a plain DeBERTa fine-tune with weighted CE beat hierarchical and curriculum variants. Source: [arXiv 2605.25816](https://www.alphaxiv.org/abs/2605.25816)

### Inferences
- The transferable recipe for us is about loss, decoding, context and ensembling. The backbone is less important. DeBERTa-v3 is English-only and mDeBERTa-v3 is the multilingual option; whether it beats XLM-R base on our scripts is untested here.
- The O-weight trick does not map directly onto GlobalPointer. GlobalPointer scores spans with a multi-label loss, not token softmax with an O class. The equivalent knobs are:
  - a per-label span-score decision threshold (default 0);
  - the positive/negative balance in the multilabel categorical cross-entropy (e.g. a margin or positive weight).
  
  In BIO mode, the O-weight and O-threshold apply directly.
- RECAP's large gain over fine-tuned NER was measured against a weak baseline (probably generic NER). The relative claim should not be read as "LLM beats a well-trained in-domain encoder".

### Gaps
- I found no peer-reviewed head-to-head of fine-tuned multilingual encoder vs prompted frontier LLM on PII with per-document cost figures.
- The exact private-LB scores and O-threshold values of the Kaggle winners were not retrievable.
- I found no published study of checksum validators combined with neural NER measuring the F1 delta on financial identifiers. Presidio's design is documentation, not an ablation.

## 3. Multilingual PII detection, financial NER and numeric look-alikes

### Takeaway
- Multilingual PII evidence says fine-tuned XLM-R is strong on non-Latin scripts, where decoder-derived detectors collapse.
- Code-switching and format diversity are recognized gaps in existing synthetic corpora.
- Financial NER (FiNER) shows that numeric tokens are labelled by context, not form, and that subword-fragmented numbers hurt BERT.
- Number-shape pseudo-tokens fix much of this, which is directly relevant to ACCOUNT vs amount vs postcode confusion.

### Cited Findings
**FiNER-139** [peer-reviewed: Loukas et al., "FiNER: Financial Numeric Entity Recognition for XBRL Tagging", ACL 2022, pp. 4419-4431]
- About 1.1M sentences and 139 XBRL entity types.
- Most tagged tokens are numeric, and the correct tag "depends mostly on context, not the token itself".
- Subword fragmentation of numbers hurts BERT; word-level BiLSTMs do better.
- Replacing numbers with pseudo-tokens encoding shape or magnitude ([XX.X] style) fixes this. SEC-BERT-NUM and SEC-BERT-SHAPE perform best. A secondary summary gives 82.1 micro-F1 for sec-bert-shape.
- Sources: [ACL Anthology 2022.acl-long.303](https://aclanthology.org/2022.acl-long.303); [arXiv 2203.06482](https://arxiv.org/abs/2203.06482v2); [alphaxiv overview (secondary)](https://alphaxiv.org/overview/2203.06482v2)

**Numeric look-alikes in another domain** [arXiv]
- Math expressions resemble dates and IDs, so generic detectors over-redact. The false positives cluster in number-dense regions.
- Domain-aware context fixes much of it: F1 rises from 0.379 to 0.82.
- Source: [MathEd-PII, arXiv 2602.16571](https://arxiv.org/abs/2602.16571)

**Multilingual evidence from the OPF study** [arXiv]
- XLM-R leads OPF on all 13 Indic and non-Latin languages in multilingual NER.
- OPF reaches only 0.04 on Arabic.
- Source: [arXiv 2608.02616](https://arxiv.org/pdf/2608.02616)

**Other multilingual work** [arXiv]
- REDACT: 25 languages and 9 scripts, with code-switching as an explicit generation axis. Source: [arXiv 2606.19881](https://arxiv.org/pdf/2606.19881)
- RECAP: hybrid regex plus LLM for 13 low-resource locales, including hi, vi and zh. Source: [arXiv 2510.07551](https://arxiv.org/abs/2510.07551)
- "Unmasking": includes a cross-lingual dimension and finds degradation outside the training distribution. Source: [arXiv 2504.12308](https://arxiv.org/abs/2504.12308)

### Inferences
- Our ACCOUNT, TIN and PHONE confusions with amounts, postcodes and dates are FiNER-type errors, where context decides the label. Three levers follow:
  - Larger windows, as in TAB and Kaggle, so the cue word (e.g. "A/C No.", "custodian account", "NRIC/TIN") stays in context.
  - A number-shape feature or input normalization.
  - Post-hoc validators. For example, a Luhn-valid 16-digit number is a card number, not our ACCOUNT. A mod-97-valid string is an IBAN, so ACCOUNT. Postcode-shaped 5-6 digit numbers next to an address are not ACCOUNT.
- Given our memory rule that account formats must be verified per jurisdiction, validators should only encode formats with a published checksum or structure: IBAN mod-97, ABA routing checksum, and national TIN check digits where documented. They should not invent patterns.

### Gaps
- I found no peer-reviewed multilingual PII paper covering th, ms, id, tl, ja or ko in financial documents. Kiii is Korean financial, but I found no paper reporting on it.
- I found no study that applies FiNER-style number-shape pseudo-tokens to multilingual PII detection.

## 4. Evaluation practices and privacy-risk-weighted metrics, plus one-variable experiments against v5.0

### Takeaway
- PII evaluation is moving from strict span F1 toward recall-weighted, risk-weighted measures: Kaggle used token-level F5, and TAB uses entity-level recall split by DIRECT/QUASI identifiers plus information-weighted precision.
- Reported numbers are often token-level ("catch rate"), which flatters models relative to strict span F1.
- We should report strict span F1, a span-recall-weighted F-beta, and a leak rate per label.

### Cited Findings
- Kaggle scoring was micro F5 at spaCy-token level, so precision around 0.2 can still give F5 around 0.9. [competition] [Zenn](https://zenn.dev/sinchir0/articles/396967387196dc)
- TAB metrics [peer-reviewed]:
  - Entity-level recall: an entity counts as protected only if all its mentions are masked. It is split into ER_di (direct identifiers) and ER_qi (quasi-identifiers).
  - Token-level precision, with weighted precision WP weighting by information content.
  - Source: [arXiv 2202.00443](https://arxiv.org/abs/2202.00443v2)
- Metric choice diverges in practice:
  - GLiNER-PII reports strict F1 [card]. [nvidia/gliner-PII](https://huggingface.co/nvidia/gliner-PII)
  - Piiranha reports token "catch rate" [card]. [Piiranha card](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)
  - The PIIBench DeBERTa paper uses support-weighted entity F1 [arXiv]. [arXiv 2605.25816](https://www.alphaxiv.org/abs/2605.25816)
  - RECAP uses nervaluate weighted F1 [arXiv]. [arXiv 2510.07551](https://arxiv.org/abs/2510.07551)
- Aggregate scores hide category-specific deployment failures. [arXiv] [arXiv 2609.03464](https://arxiv.org/pdf/2609.03464)

### Inferences: one-variable experiments relative to the v5.0 baseline
Each experiment changes one thing and keeps the v5 data, seed and eval suite. Expected effects are my estimates from the cited analogues, not measured results.

1. **Decision-threshold sweep, post-hoc**
   - Change: the GlobalPointer span threshold per label (default 0), tuned on the structured dev set and wm_dev for F2/F5.
   - Analogue: Kaggle per-label thresholds; GLiNER evaluated at 0.3.
   - Expected effect: +recall on ACCOUNT, TIN and ADDRESS, with some precision loss. This is the cheapest lever.
   - Inference cost: zero.
2. **Positive-class loss weight**
   - Change: upweight positives in the GlobalPointer multilabel CE, or in BIO mode set the O weight to 0.1-0.2.
   - Analogue: Kaggle 1st place O weight 0.05. TAB (5,1) and (10,1) raised recall 0.860 to 0.916/0.935 while precision fell 0.925 to 0.843/0.795.
   - Expected effect: a similar recall-for-precision trade.
   - Inference cost: zero.
3. **Window length**
   - Change: max_length 384 to 1024, with stride kept proportional. XLM-R base has 512 positions, so this needs a long-context multilingual encoder. With XLM-R itself, test 512 (the maximum) vs 384 as the single variable.
   - Analogue: TAB Longformer, where 4096 vs 512 gave higher precision; Kaggle 1st/4th/5th used 1.6-4k at inference.
   - Expected effect: fewer context-dependent numeric confusions.
   - Inference cost: roughly linear in tokens for 384 to 512. A long-context encoder costs 2-3x.
4. **Checksum and structure validator post-filter**
   - Change: add one post-processing step. Keep ACCOUNT spans that are IBAN-shaped only if they pass mod-97 plus the country length check, as Presidio does. Relabel Luhn-valid 13-19 digit spans away from ACCOUNT, since cards are not in our schema. Keep EMAIL only if it matches RFC-lite syntax.
   - Expected effect: +precision on ACCOUNT. Recall is unaffected, because the filter only removes or relabels.
   - Inference cost: microseconds.
5. **Number-shape normalization (FiNER-style)**
   - Change: a training and inference input transform mapping digit runs to shape tokens, or adding a digit-shape embedding. Train once with it on.
   - Expected effect: better ACCOUNT vs amount vs postcode discrimination. This is a risk for languages with mixed native digits.
   - Inference cost: negligible.
6. **Word-level probability averaging**
   - Change: average subtoken scores per whitespace or script word before decoding (Kaggle 2nd place). Only meaningful for space-delimited languages; skip zh/ja/th.
   - Expected effect: fewer fragmented spans in en/vi/ms/id/tl.
   - Inference cost: negligible.
7. **Ensemble of two heads (BIO softmax + GlobalPointer)**
   - Change: average or vote the outputs of the two heads, with weights tuned on dev.
   - Analogue: diverse-head ensembles in the Kaggle 1st/4th/5th solutions.
   - Expected effect: +1-2 F1 typical of ensembles. Unmeasured for us.
   - Inference cost: about 2x encoder time if separate encoders. Much less if it is a shared encoder with two heads.
8. **Name-swap and name-insertion augmentation**
   - Change: replace PERSON surface forms in v5 training text with names from other locales (Kaggle 1st place exp073/076).
   - Expected effect: better PERSON robustness in code-switched text.
   - Inference cost: zero.
9. **Small LLM verifier on low-confidence spans only**
   - Change: send only spans whose score falls in the uncertainty band to an LLM classifier.
   - Analogue: RECAP and MathEd-PII context disambiguation.
   - Expected effect: +precision on numeric look-alikes.
   - Inference cost: LLM latency on a small fraction of spans. It is an operational dependency and should be evaluated last.
10. **Evaluation addition, no model change**
    - Change: report per-label span recall, F2/F5 and a document-level "leak rate", meaning the fraction of documents with at least one missed PII span. Optionally weight misses by risk (TIN/ACCOUNT/DOB above BUSINESS), in the spirit of TAB's DIRECT/QUASI split.

### Gaps
- I found no published, standardized privacy-risk weighting scheme across PII types. TAB's DIRECT/QUASI split is the closest peer-reviewed one, so per-label risk weights would be our own policy choice.
- I found no evidence for how O-weight or threshold tuning behaves with GlobalPointer heads specifically. All the cited evidence is from BIO token classification.
