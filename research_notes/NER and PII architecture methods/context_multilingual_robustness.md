# NER methods: document context, multilingual / code-switched modelling, and robustness to entity-context bias (2019-2026)

Scope note: numbers below were read from the paper PDFs or abstracts linked next to them (all fetched 2026-10-10). "Our setup" means the XLM-R v5.0 baseline: 12 languages plus English+X code-switching, 8 PII types, wealth-management documents, independent 384-token windows with 96-token overlap. Every experiment proposed below changes exactly one variable from v5.0.

## Q1. Document-level / cross-window context (FLERT, CMV, memory/label consistency, retrieval-augmented NER, long-context vs sliding windows)

### Takeaway
The cheapest and best-supported gain is giving each token surrounding context from the same document and never across document boundaries. With a fine-tuned transformer this adds about 1 F1 point on average, and Contextual Majority Voting across overlapping windows adds more. External retrieval (search engine or Wikipedia) adds a further 0.3-2.6 F1, but needs a retrieval call per sentence. Cooperative learning keeps part of that gain with no retrieval at inference, which suits PII, where documents cannot be sent to a search engine.

### Cited Findings
**FLERT (Schweter & Akbik, arXiv 2020/2021)**
- Method: each sentence gets 64 subtokens of left and right context. Self-attention runs over the whole input, but only the sentence's own tokens are predicted. — [FLERT: Document-Level Features for NER, arXiv 2011.06993](https://arxiv.org/abs/2011.06993)
- Test-set F1, fine-tuned Transformer-Linear (XLM-R large), without vs with document features: EN 92.79 -> 93.64, DE 86.60 -> 86.99, DE06 90.04 -> 91.55, NL 93.50 -> 94.87, ES 89.94 -> 90.14. The paper reports document features beat sentence-level features by +1.15 pp on average, and fine-tuning beats feature-based LSTM-CRF by about 2 pp. — [FLERT, arXiv 2011.06993](https://arxiv.org/abs/2011.06993)
- Enforcing document boundaries (no context taken from a neighbouring document) raised F1 on nearly all sets: EN 93.75, DE 87.35, NL 95.21. The authors had expected transformers to learn boundaries on their own; they did not, so the paper recommends enforcing boundaries. — [FLERT](https://arxiv.org/abs/2011.06993)
- Context window search over {48, 64, 96, 128} subtokens changed results only marginally. Average dev F1 was 92.67 / 92.77 / 92.70 / 92.56, with 64 best and 128 slightly worse. — [FLERT, Table 4](https://arxiv.org/abs/2011.06993)
- Per-type effect: ORG and PER gain most from document context (for example PER +1.19 EN, +1.50 DE06), while LOC/MISC are mixed and sometimes negative. — [FLERT, Table 5](https://arxiv.org/abs/2011.06993)
- In the feature-based setting the gain is larger: LSTM-CRF all-layer-mean EN dev 94.37 -> 96.09. — [FLERT, Table 2](https://arxiv.org/abs/2011.06993)

**Cross-sentence context + Contextual Majority Voting (Luoma & Pyysalo, COLING 2020)**
- Neighbouring sentences are packed into BERT's input. CMV then combines the labels a sentence receives in each of the windows it appears in: label first, then majority vote. — [Exploring Cross-sentence Contexts for NER with BERT, COLING 2020](https://aclanthology.org/2020.coling-main.78/)
- Test F1, Single sentence -> First (following-sentence context) -> CMV: English 91.70 -> 93.44 -> 93.42; Dutch 91.53 -> 93.21 -> 93.19; Finnish 91.42 -> 93.39 -> 93.66; German 83.64 -> 85.21 -> 85.63. Spanish has no document boundaries: 87.66 -> 87.06 -> 87.89. Multilingual mBERT on Spanish: 87.50 -> 87.40 -> 87.95. — [Luoma & Pyysalo 2020, Table 2 (PDF)](https://aclanthology.org/2020.coling-main.78.pdf)
- Placing the target sentence mid-window helped for English, but the pattern was not stable across languages. Performance generally improved when the target was moved away from either edge of the window. — [Luoma & Pyysalo 2020](https://aclanthology.org/2020.coling-main.78.pdf)

**Document-level memory / label consistency**
- Gui et al. (IJCAI 2020): a key-value memory stores draft labels from a base tagger. A multi-channel Transformer then refines those labels using co-occurrence across the document, and Bayesian uncertainty flags drafts that are likely wrong. The paper reports significant gains over state of the art on three benchmarks; exact numbers were not retrieved. — [Leveraging Document-Level Label Consistency for NER, IJCAI 2020](https://ijcai.org/proceedings/2020/550)
- Hire-NER (Luo et al., AAAI 2020): a key-value memory of document-level representations per unique word, combined with BERT. It reported state of the art on CoNLL-03 and OntoNotes 5.0 at the time. — [Hierarchical Contextualized Representation for NER, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6363/6219)
- ScdNER (EMNLP 2023) argues that word-level global memory adds noise when the same word carries different labels in different places. It switches to span-level global features, gated by a first-stage entity-probability classifier. — [ScdNER, EMNLP 2023](https://preview.aclanthology.org/update-py-deps/2023.emnlp-main.970); see also [Exploiting Global Contextual Information for Document-level NER, arXiv 2106.00887](https://arxiv.org/abs/2106.00887v1)

**Retrieval-augmented NER (Wang et al., ACL 2021, CL-L2 / CL-KL)**
- Method: the sentence is used as a Google Search query (internal engine for e-commerce), returning up to 20 results. They are re-ranked with BERTScore, the top 6 are concatenated, and the input is truncated at 510 subtokens. Average input length grows from about 19 to about 139 tokens on WNUT. Cooperative learning (CL) adds an L2 or KL loss that pulls the no-context view toward the with-context view. — [Improving NER by External Context Retrieving and Cooperative Learning, ACL 2021 (arXiv 2105.03654)](https://arxiv.org/abs/2105.03654)
- F1 when evaluated **without context** at inference (no retrieval cost), baseline -> CL-KL: WNUT-16 56.04 -> 58.14, WNUT-17 57.86 -> 59.33, CoNLL-03 93.03 -> 93.21, CoNLL++ 94.20 -> 94.55, BC5CDR 90.52 -> 90.73, NCBI 88.65 -> 89.24, E-commerce 81.47 -> 82.31. — [Wang et al. 2021, Table 2](https://arxiv.org/abs/2105.03654)
- With context at inference, CL-KL reaches WNUT-17 60.45, CoNLL-03 93.56, E-commerce 83.99. — [Wang et al. 2021](https://arxiv.org/abs/2105.03654)
- Context quality matters. On WNUT-17, retrieved context scores 60.20 against 57.86 without context. Context taken from the dataset itself scores 57.21, generated context 57.71, randomly retrieved context 57.53, and random data context 47.69, which is harmful. — [Wang et al. 2021, Table 6](https://arxiv.org/abs/2105.03654)

**Retrieval in multilingual shared tasks**
- DAMO-NLP's knowledge-based system for MultiCoNER 2022 retrieves related context from a multilingual Wikipedia KB and appends it to the sentence. It won 10 of 13 tracks. — [DAMO-NLP at SemEval-2022 Task 11, arXiv 2203.00545](https://arxiv.org/abs/2203.00545)
- U-RaNER (MultiCoNER II, 2023) adds Wikidata entity knowledge and an infusion approach to widen the contextual scope. It won 9 of 13 tracks. The authors' error analysis found that previous top systems were bottlenecked by insufficient knowledge. — [DAMO-NLP at SemEval-2023 Task 2, arXiv 2305.03688](https://arxiv.org/abs/2305.03688)

### Inferences
- Our 384/96 windowing already gives most tokens more than FLERT's 64-subtoken context. The open questions are how predictions in the overlap are merged and whether tokens near a window edge get poor context. CMV and FLERT both suggest the best policy is to take each token's label from the window where it sits most centrally, or to vote or average logits across windows. This costs no extra compute.
- FLERT's boundary finding transfers directly. When a long file contains several logical documents (email thread replies, concatenated KYC forms, markdown tables), windows should not mix them.
- Web retrieval is not acceptable for PII text. The transferable part is CL-KL: train with an extra "rich context" view (for example a document header, form field labels, or the preceding table header row) and serve the plain view at zero extra inference cost.
- Document-level label consistency fits our domain well, because the same account number or person name repeats across a KYC/SOW document. A cheap version is post-hoc span propagation: if a string is tagged ACCOUNT with high confidence in one window, re-tag its identical occurrences elsewhere. This is a heuristic inferred from Gui et al., not a published result.

### Proposed one-variable experiments (vs XLM-R v5.0)
1. **Overlap merge policy only**: change logit merging in the 96-token overlap from the current policy to (a) "most-central window wins" or (b) mean of logits (CMV-style). Training is unchanged. Inference cost is about 0.
2. **Document boundary enforcement only**: segment emails, threads and concatenated forms on boundary markers before windowing. Cost: more but shorter windows, roughly neutral.
3. **Overlap size only**: 96 -> 128 tokens. Stride goes from 288 to 256, about 1.125x more windows.
4. **CL-KL with a header-context view only**: during training, also feed each window with a prepended document-type or section header and add a KL loss between the two views. Serve the plain view. Inference cost is 0; training cost is about 2x.
5. **Document-level span propagation only** (post-processing): propagate high-confidence ACCOUNT/PERSON/EMAIL strings to identical unlabeled occurrences in the same document. Inference cost is negligible (string matching).

### Gaps
- I found no paper directly comparing long-context encoders (Longformer, BigBird, or XLM-R variants with longer positions) with sliding windows for NER, with costs. Not established here.
- Exact numbers for Gui et al. 2020, Hire-NER and ScdNER were not retrieved (the abstracts give none).
- No paper found on document context in PII detection on financial documents specifically.

## Q2. Multilingual and code-switched NER (LinCE/CALCS, MultiCoNER architectures, translate-train, consistency training, language-adversarial training, script handling)

### Takeaway
External knowledge plus transformer ensembles won MultiCoNER 2022 and 2023, and noisy input cost every system about 10% F1. For cross-lingual transfer with XLM-R, consistency training on translated unlabeled text (ConNER) beats translate-train and knowledge distillation, adding +2.61 F1 on average and +6.6 on Arabic. For code-mixed data, mention replacement combined with shuffling within segments, plus adversarial perturbation, gave about +1.2-2.3 macro-F1 in one MultiCoNER code-mixed system.

### Cited Findings
**MultiCoNER (SemEval 2022 / 2023)**
- MultiCoNER II: 13 tracks across 12 languages (monolingual, multilingual and noisy settings), 47 teams, 34 system papers. Methods that fuse external knowledge into transformers performed best, and all top teams used KBs such as Wikipedia or gazetteers. Ensembles helped. Noisy (typo-perturbed) test data caused an average drop of about 10%. Winners beat the baseline by about 35% F1 on average. — [SemEval-2023 Task 2: MultiCoNER 2 overview, arXiv 2305.06586](https://arxiv.org/abs/2305.06586)
- 2022 winners: DAMO-NLP with Wikipedia-retrieved context won 10 of 13 tracks — [arXiv 2203.00545](https://arxiv.org/abs/2203.00545). 2023 winners: U-RaNER with Wikidata won 9 of 13 tracks — [arXiv 2305.03688](https://arxiv.org/abs/2305.03688)
- Code-mixed track example (Qtrade AI, MultiCoNER 2022, 4th in Code-Mixed, 3rd in Multilingual). Dev macro-F1 on Mix-Code: baseline 75.93. Multi-sentence mention-replacement augmentation ("mulsent-uni") reached 77.19 (+1.26). PGD adversarial training alone gave 75.96, roughly no effect on code-mixed but +1.06 on Chinese. Augmentation plus PGD together reached 78.21 (+2.28). Augmentations used were mention replacement (same-type entity swap) and shuffle-within-segments (non-entity words shuffled, entity order kept). — [Qtrade AI at SemEval-2022 Task 11](https://aclweb.org/anthology/2022.semeval-1.222.pdf) *(PDF read via the anthology; URL pattern assumed from the task proceedings, so verify the paper ID)*

**Code-switching benchmarks**
- LinCE (LREC 2020) covers 10 corpora, four language pairs (Spanish-English, Nepali-English, Hindi-English, MSA-Egyptian Arabic) and four tasks including NER, with LSTM, ELMo and mBERT baselines. — [LinCE, arXiv 2005.04322](https://arxiv.org/abs/2005.04322)
- Winata et al. found that pretrained multilingual models "do not necessarily guarantee high-quality representations on code-switching". Meta-embeddings reached similar NER/POS results with far fewer parameters. — [Are Multilingual Models Effective in Code-Switching?, arXiv 2103.13309](https://arxiv.org/abs/2103.13309)

**Consistency training / translate-train (ConNER, EMNLP 2022)**
- Method: translation-based consistency on unlabeled target-language text (bi-directional KL between an entity span and its translation), plus dropout-based (R-Drop-like) consistency on labeled source data. Backbone is XLM-R large. — [ConNER: Consistency Training for Cross-lingual NER, EMNLP 2022 (arXiv 2211.09394)](https://arxiv.org/abs/2211.09394)
- Zero-shot EN -> De/Es/Nl average F1: vanilla 77.68; ConNER 80.29 (+2.61); translate-train BWET 77.93, MulDA 77.64, MTL-WS 76.76; KD UniTrans 79.01, AdvPicker 78.97; xTune 78.85; MLM-replace consistency 77.39 (hurts on De). — [ConNER, Table 1](https://arxiv.org/abs/2211.09394)
- Distant languages (WikiAnn): Zh 33.10 -> 39.17, Ar 53.00 -> 59.62, Hi 73.00 -> 74.49. — [ConNER, Table 3](https://arxiv.org/abs/2211.09394)
- Ablation (avg F1): translation consistency on unlabeled data only 79.77; dropout consistency on labeled data only 78.99; consistency on translated *labeled* data 77.23, below vanilla. Dropout consistency on unlabeled target data collapses to 73.44. — [ConNER, Table 4](https://arxiv.org/abs/2211.09394)
- Low-resource (5% / 10% / 25% of English training data): +4.32 / +2.20 / +1.77 F1. — [ConNER, Table 2](https://arxiv.org/abs/2211.09394)

**Language-adversarial training**
- Keung et al. (EMNLP-IJCNLP 2019): language-adversarial training on mBERT pushes the embeddings of English documents and their translations to align, improving zero-resource cross-lingual NER on CoNLL 2002/2003. Exact gains were not retrieved. — [Adversarial Learning with Contextual Embeddings for Zero-resource Cross-lingual Classification and NER, EMNLP 2019](https://aclanthology.org/D19-1138/)

### Inferences
- Our setting is supervised in all 12 languages, not zero-shot, so ConNER's large zero-shot gains are an upper bound. The dropout-consistency component (labeled data, +1.3 avg in ablation) is the piece that applies directly. The translation-consistency component maps to our code-switching need: KL between an English+X sentence and its monolingual variant.
- MultiCoNER's lesson (KB retrieval) concerns named entities that live in Wikipedia. PERSON/ACCOUNT/TIN/PHONE in client documents are not in any KB, so retrieval-based winners transfer poorly. The transferable lessons are noise robustness (a 10% drop on typos) and mention-replacement augmentation.
- The ConNER finding that consistency on translated labeled data hurts is a warning for us: label projection noise in translate-train (for example zh/ja/th tokenisation) can cost more than it gains.

### Proposed one-variable experiments
1. **R-Drop / dropout consistency on labeled data only**: two forward passes with a symmetric KL loss (alpha about 0.5). Inference cost 0; training about 2x forward.
2. **Code-switch consistency only**: for each English+X training window, build its monolingual-X (or EN) counterpart with the same entities and add a token-level KL on the aligned entity tokens. Inference cost 0.
3. **Typo/noise augmentation only**: character-level noise on context tokens, never on entity tokens, mimicking the MultiCoNER noisy subset; evaluate on a noised copy of ml_real. Inference cost 0.
4. **Adversarial embedding perturbation (PGD/FGM) only**. Inference cost 0; training about 2-4x per step.

### Gaps
- I could not retrieve per-pair LinCE NER F1 for XLM-R (leaderboard tables were garbled), and found no LinCE/CALCS results for our language pairs (EN+zh/ja/ko/th/vi/ms/id/tl/ar/hi). LinCE NER covers only es-en, hi-en and msa-ea.
- No source found on script/romanisation handling for NER (for example romanised Hindi, or Arabizi in English+X text). Not established here.
- Exact language-adversarial NER gains (Keung et al.) not retrieved.
- The MultiCoNER 2022 overview (code-mixed track statistics) was not fetched.

## Q3. Robustness to unseen contexts and entity-context bias (Ma et al. 2023, Lin et al. 2020, counterfactual augmentation, MINER, context-only training)

### Takeaway
Pretrained NER models lean on entity surface forms because, in standard datasets, entities carry more usable information than their context. The best-measured fix is data-side: replace redundant, easy entities with hard (low-PVI) ones, or swap easy-context instances toward high-PVI contexts. At a 40% replacement rate this lifts OOV F1 by up to +7.5 and cross-category F1 by +11 at a cost of about 0.6-1.5 in-distribution F1. The data fix stacks with model-side methods such as MINER.

### Cited Findings
**Lin et al. 2020 (EMNLP)**
- Randomisation tests that separately erase name regularity, mention coverage and context diversity found that name regularity is critical for generalising to unseen mentions. High mention coverage may hurt generalisation, and context patterns need relatively little data to learn with pretrained encoders. The authors also released an open-type NER dataset with weak name regularity and low mention coverage. — [A Rigorous Study on NER: Can Fine-tuning Pretrained Model Lead to the Promised Land?, EMNLP 2020 (arXiv 2004.12126)](https://arxiv.org/abs/2004.12126)

**Ma et al. 2023 (EMNLP), "Towards Building More Robust NER Datasets"**
- Difficulty is measured with pointwise V-information (PVI), using separate context-only and entity-only probe models. The CEIM score is PVI_E minus PVI_C, and low CEIM means the context is easier than the entity. In all 6 datasets studied, entity V-information exceeds context V-information. The largest group of instances has near-zero CEIM, where entity and context are equally easy, and these do not teach the model to use context. — [Ma et al., EMNLP 2023, pp. 4616-4630](https://aclanthology.org/2023.emnlp-main.281)
- Adding 1k instances to BERT on CoNLL03 (Test / OOV / CrossCategory): Base 87.15 / 64.28 / 40.74. Adding high-CEIM gives OOV -0.51; adding near-zero CEIM gives OOV -0.23; adding low-CEIM gives OOV +3.68 and Cate +4.19. — [Ma et al. 2023, Table 3](https://aclanthology.org/2023.emnlp-main.281.pdf)
- Raising the low-CEIM proportion first improves and then hurts OOD performance, while i.i.d. Test drops steadily. The paper suggests a sweet spot of about 10-30% low-CEIM for CoNLL03 and 20-40% for ACE05. — [Ma et al. 2023, Sec 4.2](https://aclanthology.org/2023.emnlp-main.281.pdf)
- Entity-side fixes, BERT-base on CoNLL03 (Test / OOV / Cate). Base 91.18 / 70.89 / 44.48. HighC2Low 40% (replace entities only in instances with high-PVI context) gives 90.55 / 77.11 / 48.92. Redundant2Low 40% (replace redundant easy entities without adding data) gives 90.42 / 77.34 / 48.07. Baselines for comparison: DataAug (Dai & Adel) 90.52 / 73.45 / 46.39; MINER 90.98 / 76.89 / 49.97; LPFT 91.14 / 73.89 / 48.32. — [Ma et al. 2023, Table 4](https://aclanthology.org/2023.emnlp-main.281.pdf)
- Context-side fix (Low2High 40%: delete low-PVI-context instances and replace them with high-PVI contexts), BERT CoNLL03: 89.72 / 78.41 / 55.70, i.e. Cate +11.2 and Test -1.46. On RoBERTa-large: 90.92 / 79.58 / 61.93. — [Ma et al. 2023, Table 5](https://aclanthology.org/2023.emnlp-main.281.pdf)
- The approaches stack. MINER + HighC2Low 40% on BERT CoNLL03 gives 90.71 / 79.85 / 53.42, and on RoBERTa-large 91.70 / 80.33 / 61.16. Datasets built with BERT's PVI also help RoBERTa-large (BERT.HighC2Low 40%: 91.74 / 79.18 / 55.15). — [Ma et al. 2023, Tables 4-5](https://aclanthology.org/2023.emnlp-main.281.pdf)

**Counterfactual data augmentation (Zeng et al., EMNLP 2020)**
- Each sentence is split into entity and context parts, analysed with a structural causal model. Counterfactuals are generated by swapping an entity for another of the same type, and a discriminator decides which to keep. This improved generalisation with limited data on three datasets and removed part of the spurious context-label correlation. Code: github.com/xijiz/cfgen. — [Counterfactual Generator: A Weakly-Supervised Method for NER, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.590)

**MINER (Wang et al., ACL 2022)**
- Two mutual-information objectives: generalising-information maximisation, and superfluous-information minimisation to discourage memorising entity names. They improve OOV entity prediction. — [MINER: Improving OOV NER from an Information Theoretic Perspective, ACL 2022](https://aclanthology.org/2022.acl-long.383) ([arXiv 2204.04391](https://arxiv.org/abs/2204.04391v2))
- Independently measured by Ma et al.: BERT CoNLL03 OOV 70.89 -> 76.89, Cate 44.48 -> 49.97, Test -0.20. On ACE05 MINER gave little gain (OOV 79.80 -> 79.23). — [Ma et al. 2023, Table 4](https://aclanthology.org/2023.emnlp-main.281.pdf)

### Inferences
- Our v5 synthetic frames already raise context information (a Low2High-style intervention). The next untested lever from Ma et al. is the entity side: Redundant2Low, which keeps data size fixed while replacing redundant easy entities with hard look-alikes. In PII terms, "hard entities" are accounts written in amount-like or postcode-like shapes, and Korean account numbers formatted like TIN/RRN strings. For those, only context ("계좌번호", "account no.", "주민등록번호") can disambiguate. This targets our Korean ACCOUNT vs TIN and account vs amount/postcode errors directly.
- The converse also matters for look-alikes: hard negatives, meaning amounts and postcodes in account-like shapes labelled O, in contexts that clearly are not accounts. These force context use in the other direction. This is an application of Ma et al.'s logic, not a published experiment.
- Expect a trade-off: every method above costs 0.3-1.5 i.i.d. F1 at the replacement rates that maximise OOD gains. Run a rate sweep (for example 20% and 40%) rather than one point.
- Computing PVI needs two probe models (context-only with entities masked, entity-only). These are cheap XLM-R-base fine-tunes and can be reused to score our synthetic frames and select which instances to modify.

### Proposed one-variable experiments
1. **Redundant2Low only** (data): replace X% (20 / 40) of the most redundant ACCOUNT/TIN/PHONE surface forms with verified-format but context-dependent look-alikes; total size unchanged. Inference cost 0.
2. **Hard-negative look-alikes only** (data): add amount/postcode/reference strings in account shapes labelled O. Inference cost 0.
3. **Context-only auxiliary objective only** (training): with probability p, mask entity tokens ([MASK] or random same-type format) and still require the correct label from context. Inference cost 0. This is close to Zeng et al.'s entity swap and to Ma et al.'s context-PVI probe.
4. **MINER loss only** (training): add MINER's two MI objectives to the v5.0 recipe. Inference cost 0; training cost moderate (extra variational heads).
5. **Evaluation only**: build OOV and cross-category test slices for our data, following Ma et al.'s OOV/CrossCategory protocol, so experiments 1-4 can be measured. Currently known-account-format and ml_real slices exist, but no entity-swap slice.

### Gaps
- No ConNER-style or Ma-style results exist on non-English, PII, or numeric-identifier entity types. All robustness numbers above are on CoNLL03/ACE05 (PER/ORG/LOC/MISC), so how well they transfer to ACCOUNT/TIN look-alikes is unverified.
- Zeng et al. 2020 exact numbers were not retrieved.
- No published debiasing study was found specifically on Korean, or on account-vs-TIN confusion.
