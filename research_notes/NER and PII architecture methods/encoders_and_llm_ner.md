# Encoder choices for NER token classification: modern encoders, decoder LLMs converted to encoders, generative / universal NER (2019-2026)

Context: multilingual PII NER (12 languages + English+X code-switching), 8 types, 384-token windows, strict span F1, CPU latency matters. Baseline is XLM-R base with a softmax BIO head; a GlobalPointer head helped. Queued swaps: mmBERT-small, mmBERT-base, EuroBERT-210m, and Qwen3-Embedding-0.6B (causal attention, run as Qwen3ForTokenClassification).

Throughout, **supervised** means fine-tuned on in-domain labelled data. **Zero-shot** means no target-dataset training. **Cross-lingual zero-shot** (as in XTREME/PAN-X) means fine-tuned on English and evaluated on other languages; that is not our setting.

## Q1. Modern multilingual encoders: NER results, language coverage, tokenizer, speed

### Takeaway
No modern encoder clearly beats XLM-R on NER. mmBERT base wins XTREME classification and QA but trails XLM-R base by about 3 F1 on WikiANN; its authors blame the tokenizer's inconsistent prefix-whitespace handling. EuroBERT also lags XLM-R on NER and covers only 6 of our 12 languages (en, zh, ja, vi, ar, hi). It misses ko, th, ms, id and tl. Expect mmBERT and EuroBERT to give speed and context-length gains rather than accuracy gains. The swaps that literature suggests would raise accuracy are RemBERT, XLM-R large and mDeBERTa-v3.

### Cited Findings
- **mmBERT** (Marone et al., "mmBERT: A Modern Multilingual Encoder with Annealed Language Learning", arXiv 2025):
  - **Size and tokenizer:** base is 307M total and about 110M non-embedding parameters; small is 140M total and 42M non-embedding. It uses the Gemma 2 tokenizer (256k vocab).
  - **Pre-training:** 1833 languages, introduced in stages (60 → 110 → 1833). Context grows from 1024 to 8192 tokens.
  - Source: [arXiv 2509.06888](https://arxiv.org/html/2509.06888)
- **mmBERT on XTREME** (same source; the WikiANN column is cross-lingual zero-shot):

  | Model | WikiANN | UDPOS | XTREME avg |
  |---|---|---|---|
  | mmBERT small | 54.3 | 70.6 | 68.6 |
  | mmBERT base | 58.2 | 74.0 | 72.8 |
  | XLM-R base | 61.4 | 74.3 | 70.4 |
  | mGTE base | 60.7 | 74.3 | 71.1 |
  | Multilingual MiniLM | 59.2 | 74.2 | 67.1 |

  - mmBERT base beats XLM-R base on XNLI (77.1 vs 74.6), XQuAD (77.6 vs 73.4) and TyDiQA (74.5 vs 70.5).
  - The authors say mmBERT "lags on POS" and attribute the token-classification weakness to "the lack of a consistent prefix-whitespace token" in pre-training. They say ModernBERT shares this problem.
  - Source: [arXiv 2509.06888](https://arxiv.org/html/2509.06888)
- **mmBERT speed:**
  - Base is reported as ">2x faster" on variable-length inputs and about 4x faster on long contexts than earlier multilingual encoders. Small is about 2x faster again.
  - XLM-R and mMiniLM are capped at 512 tokens.
  - The text gives no absolute throughput figures; they appear only in Figure 3. There is no speed comparison with mDeBERTa.
  - Source: [arXiv 2509.06888](https://arxiv.org/html/2509.06888)
- **EuroBERT** (Boizard et al., "EuroBERT: Scaling Multilingual Encoders for European Languages", arXiv 2025):
  - **Sizes and tokenizer:** 210M, 610M and 2.1B parameters. It uses the Llama 3 tokenizer (128k vocab) and supports 8192 tokens.
  - **Languages (15):** en, fr, de, es, zh, it, ru, pl, pt, ja, vi, nl, ar, tr, hi.
  - **NER:** the NER row of Table 1 reads 94.76 (210M), 95.94 (610M), 95.26 (2.1B) and 96.32 for XLM-RoBERTa at 3.5B (XL). The extracted table layout was ambiguous, so treat these numbers as approximate.
  - The authors state that EuroBERT "lags behind XLM-RoBERTa" on NER. They attribute this partly to tokenizer fertility: XLM-R's 250k vocabulary splits entities into fewer pieces.
  - Source: [arXiv 2503.05500](https://arxiv.org/html/2503.05500)
- **RemBERT** (Chung et al., "Rethinking embedding coupling in pre-trained language models", ICLR 2021): on the XTREME leaderboard comparison, NER F1 is 70.1 for RemBERT vs 65.4 for XLM-R. That is English fine-tuning with cross-lingual zero-shot evaluation, and the column is presumably PAN-X. Task-average is 74.4 vs 70.1. — [arXiv 2010.12821](https://arxiv.org/pdf/2010.12821)
- **mDeBERTa-v3:** a Slovak NER study (WikiGoldSK, BSNLP2017, WikiANN) found mDeBERTa-v3 "very similar or better" than XLM-R on all datasets. — ["WikiGoldSK", arXiv 2304.04026](https://arxiv.org/pdf/2304.04026)
- **XLM-R large vs base:** on English WikiAnn, XLM-R base scored 82.7 and large 84.0. — ["T-NER", EACL 2021 demo, arXiv 2209.12616](https://arxiv.org/pdf/2209.12616)
- **Glot500-m:** it beats XLM-R large on several low-resource languages but trails on high-resource ones (en 83.3 vs 84.5 in its per-language table). The search snippet did not show which task that table covers. — ["Glot500", ACL 2023, arXiv 2305.12182](https://arxiv.org/pdf/2305.12182)
- **XLM-R base on 12 languages:** a SemEval-2023 MultiCoNER II system fine-tuned XLM-R base on all 12 languages. — [LLM-RM at SemEval-2023 Task 2](https://preview.aclanthology.org/setup/2023.semeval-1.62)
- **XLM-R base supervised baseline on MultiCoNER:** averages 54.9 F1 in GLiNER's Table 3. — [GLiNER, arXiv 2311.08526](https://arxiv.org/html/2311.08526)

### Inferences
- **Language coverage** for our 12 languages:
  - XLM-R, mDeBERTa-v3 (CC100, same 100 languages) and mmBERT cover all of them, with tl only weakly in CC100.
  - RemBERT (110 languages) and Glot500 cover all of them.
  - EuroBERT misses ko, th, ms, id and tl. It would be expected to underperform on 5 of 12 languages unless fine-tuning data alone carries them.
- **Tokenizer behaviour** (inference):
  - Gemma 2 (mmBERT) and Llama 3 (EuroBERT) split digits into single-digit tokens. That makes ACCOUNT, PHONE and TIN strings longer in tokens but more regular.
  - The XLM-R sentencepiece tokenizer merges digit runs unpredictably.
  - The whitespace-prefix issue the mmBERT authors flag may matter for word-initial B- tags. We use subword-level labels, so this is less clear-cut.
  - Neither is verified for our data: run a fertility and boundary-alignment check per language.
- **Speed:** mmBERT-small, with 42M non-embedding parameters vs about 85M for XLM-R base, plus FlashAttention and unpadding, is the most promising CPU-latency candidate. The reported speedups are GPU-centric, so CPU gains must be measured.
- **Expected accuracy ordering** from the literature (supervised, in-language): XLM-R large ≈ mDeBERTa-v3-base ≥ XLM-R base ≥ mmBERT base > mmBERT small. These gaps come from cross-lingual zero-shot benchmarks and usually shrink with in-language supervised data.
- **One-variable experiments** (each relative to the XLM-R-base + GlobalPointer baseline):
  1. **mmBERT-small:** swap only the backbone; keep the head, LR schedule and 384 window. Report strict span F1 per language plus CPU latency per 100k characters. Hypothesis: about −1 to −3 F1 at about 2x speed.
  2. **mmBERT-base:** same swap; compare per-format ACCOUNT, PHONE and TIN F1 to test the tokenizer digit effect.
  3. **EuroBERT-210m:** same swap, but report F1 separately for covered (en, zh, ja, vi, ar, hi) and uncovered (ko, th, ms, id, tl) languages.
  4. **mDeBERTa-v3-base:** same swap. It has about the same parameter count and the same CC100 coverage, and is the strongest "free" accuracy candidate. It is slower on CPU than XLM-R because of disentangled attention (inference, not measured here).
  5. **XLM-R large:** same swap, to find the accuracy ceiling at about 3x cost. Useful as a distillation teacher (see Q4).
  6. **RemBERT:** optional; large (about 575M), so it is mainly a teacher candidate.

### Gaps
- I found no paper reporting **ModernBERT or NeoBERT** NER numbers. Both are English-only, so they are not relevant for 12 languages; their reports were not fetched.
- I found no clean **paper-level PAN-X comparison** of mDeBERTa-v3, XLM-R large, mT5 encoder and Glot500. The XTREME-R PDF fetch returned binary content.
- I found no **MasakhaNER or MultiCoNER** numbers for mmBERT or EuroBERT.
- **EuroBERT's NER dataset** behind its Table 1 was not identified. The extraction was ambiguous.
- I found no **CPU latency benchmarks** for any of these encoders.

## Q2. Decoder LLMs for token classification: causal mask, unmasking, LLM2Vec, and our Qwen3 setup

### Takeaway
A causal decoder fine-tuned for NER with its mask intact underperforms badly. On CoNLL03 test, Llama2-7B scores 0.744 F1 causal vs 0.908 unmasked; RoBERTa-large scores 0.900. Removing the mask from some or all layers during fine-tuning fixes this at 7B. However, at 68M parameters unmasking gave no gain and encoders won. Our Qwen3-Embedding-0.6B run with causal attention is expected to underperform. The literature fix is to remove the causal mask, ideally only in the upper layers, and optionally add MNTP adaptation first (LLM2Vec).

### Cited Findings
- **Dukić & Šnajder, "Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling"** (Findings of ACL 2024). Fine-tuning used QLoRA (r=64, 4-bit, query and value matrices), 5 epochs, averaged over 5 seeds. CoNLL03 NER micro-F1:

  | Model / setting | Valid | Test |
  |---|---|---|
  | Llama2-7B, causal (0000) | 0.788 | 0.744 |
  | Llama2-7B, all layers unmasked (1111) | 0.936 | 0.908 |
  | Llama2-7B, best layer groups (0110) | 0.947 | 0.920 |
  | Mistral-7B, causal | 0.789 | 0.756 |
  | Mistral-7B, all layers unmasked | 0.953 | 0.927 |
  | RoBERTa-large | 0.924 | 0.900 |
  | RoBERTa-base | 0.897 | 0.883 |
  | Mistral-7B instruction-tuned (generative) | 0.897 | 0.887 |

  - Unmasking even one layer group "significantly boosts" F1, but unmasking all layers is rarely optimal. The best layers are task-dependent and usually closer to the output. NER gains the most.
  - **At 68M parameters, mask removal gave no gains, and encoders matched or beat small decoders.**
  - Code is at github.com/dd1497/llm-unmasking.
  - Source: [arXiv 2401.14556](https://arxiv.org/html/2401.14556v3); [ACL Anthology](https://preview.aclanthology.org/credits/2024.findings-acl.843)
- **Li et al., "Label Supervised LLaMA Finetuning"** (arXiv 2023): this paper introduced LS-LLaMA and LS-unLLaMA.
  - **Method:** a label head on final-layer LLaMA-2-7B representations, trained with LoRA. Removing the causal mask gives LS-unLLaMA.
  - **Claim:** the abstract claims state-of-the-art NER. The exact F1 was not retrieved.
  - Source: [arXiv 2310.01208](https://arxiv.org/abs/2310.01208)
- **BehnamGhader et al., "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"** (COLM 2024; arXiv v2, Aug 2024).
  - **Method:** enable bidirectional attention, then apply masked next-token prediction (MNTP) with LoRA, then optionally unsupervised SimCSE.
  - **Word-level evaluation:** linear probes on **frozen** last-layer features (CoNLL-2003; probe scores, not end-to-end fine-tuning). NER scores:

  | Model | Uni | Bi | Bi+MNTP | Bi+MNTP+SimCSE |
  |---|---|---|---|---|
  | S-LLaMA-1.3B | 96.09 | 92.17 | 96.59 | 95.90 |
  | LLaMA-2-7B | 96.59 | 92.31 | 97.16 | 96.05 |
  | Mistral-7B | 96.52 | 97.14 | 97.18 | 96.87 |

  - Baselines: DeBERTa-v3-large 94.97, BERT-large 90.09.
  - **Key point: naive mask removal without adaptation hurts** (LLaMA and S-LLaMA drop about 4 points). MNTP recovers it and adds gains. Mistral tolerates naive removal.
  - SimCSE, the sentence-level contrastive stage, slightly hurts word-level tasks.
  - Source: [arXiv 2404.05961](https://arxiv.org/html/2404.05961)
- **Qwen3 Embedding** (tech report, arXiv 2025): sizes 0.6B, 4B and 8B, built on Qwen3 foundation models. The abstract page did not state the attention or pooling details. — [arXiv 2506.05176](https://arxiv.org/abs/2506.05176)

### Inferences
- **Our Qwen3-Embedding-0.6B causal run** will probably underperform XLM-R base, for three reasons:
  1. Dukić & Šnajder show a 15-17 F1 gap between causal and unmasked fine-tuning at 7B.
  2. 0.6B is closer to the "small decoder" regime, where encoders match or beat decoders.
  3. Embedding models are contrastively trained for sentence pooling. LLM2Vec shows that a SimCSE-style stage slightly hurts word-level probes.

  A specific failure mode to expect: a B- tag on the first token of a name can't see the following tokens. "Lee" in "Lee Hsien Loong" vs "Lee Road" is ambiguous left-to-right. ADDRESS and BUSINESS spans, whose type is resolved by later tokens, should suffer most.
- **Literature-suggested fixes, in order of cost** (one variable each):
  - **(a)** Same Qwen3-0.6B, with `is_causal=False` (attention mask made fully bidirectional) in all layers, fine-tuned end to end. Compare to the causal run.
  - **(b)** Unmask only the top half of layers (Dukić & Šnajder's best configs, e.g. 0110 or 0111).
  - **(c)** Run LLM2Vec-style MNTP adaptation on unlabelled multilingual financial text before (a).
  - **(d)** Use base Qwen3-0.6B instead of the Embedding variant, to remove the contrastive-training confound.
  - Implementation note (inference): HF Qwen3 builds the causal mask internally, so overriding `is_causal` or passing a custom 4D mask is needed. Check that FlashAttention/SDPA paths respect it.
- **Cost:** even at 0.6B, Qwen3 is about 2x XLM-R base parameters (278M) with a 151k vocab. CPU latency will be worse than XLM-R base, so it is hard to justify for deployment. Its role is best seen as a teacher or an ablation.

### Gaps
- **BiLLM, Echo embeddings and the "decoder-only LLMs as token classifiers" studies:** not fetched within budget. I have no verified numbers for BiLLM or Echo embeddings on NER. Echo embeddings repeat the input so that tokens in the second copy see the full context; that is a sequence-embedding method, and doubling the length makes it costly for token-level CPU inference.
- **Qwen-family bidirectional NER results:** no multilingual comparison of Qwen-family bidirectional vs XLM-R NER was found.
- **LS-LLaMA's exact NER F1** was not retrieved.

## Q3. Generative and universal NER vs fine-tuned token classifiers (supervised, in-domain)

### Takeaway
GLiNER-style span-matching encoders are the strongest practical universal NER models.
- **Zero-shot:** GLiNER-L at 0.3B beats ChatGPT and UniNER-13B.
- **Supervised in-domain:** they are roughly on par with LLM-based extractors (GLiNER-L 82.9 vs UniNER-7B 84.8 average F1 across 20 datasets).
- **Multilingual zero-shot:** weak. GLiNER-Multi scores 32.9 vs 54.9 for a supervised XLM-R on MultiCoNER.
- **PII:** current GLiNER-PII models score only about 0.37-0.47 span F1 on the SPY benchmark, and their training languages are European.

For our supervised 12-language setup, none is expected to beat a fine-tuned encoder token or span classifier. The GLiNER architecture is effectively a span head over an encoder, conceptually close to GlobalPointer. Its value is as a pretraining or initialisation source, or as a zero-shot labeller.

### Cited Findings
- **GLiNER** (Zaratiana et al., "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer", NAACL 2024; arXiv 2311.08526). Models: S 50M, M 90M, L 0.3B, all on DeBERTa-v3; GLiNER-Multi uses mDeBERTa.

  | Setting | Result |
  |---|---|
  | Zero-shot OOD avg F1 (CrossNER + MIT) | ChatGPT 47.5, InstructUIE-11B 47.2, UniNER-7B 53.7, UniNER-13B 55.6, GLiNER-S 52.7, GLiNER-M 55.4, GLiNER-L 60.9 |
  | Zero-shot, 20 datasets | ChatGPT 36.5, UniNER-7B 45.7, GLiNER-L 47.8 |
  | Multilingual zero-shot, MultiCoNER avg F1 | supervised XLM-R 54.9, ChatGPT 29.9, GLiNER-En 23.6, GLiNER-Multi 32.9 (Korean: GLiNER 28.7 vs ChatGPT 30.0) |
  | Supervised in-domain avg | InstructUIE 81.2, UniNER-7B 84.8, GLiNER-L 82.9 (82.1 without Pile-NER pretraining) |

  - **Speed:** spans are extracted in parallel rather than generated autoregressively, with span length capped at K=12. No measured latency is reported. GLiNER-L trains in about 5 h on one A100.
  - Source: [arXiv 2311.08526](https://arxiv.org/html/2311.08526)
- **UniversalNER** (Zhou et al., "UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition", ICLR 2024):
  - **Method:** ChatGPT-annotated data is distilled into LLaMA-based 7B and 13B students and evaluated on 43 datasets across 9 domains.
  - **Zero-shot:** beats ChatGPT by about 7-9 F1 and Alpaca/Vicuna by about 30 F1.
  - **Supervised:** beats InstructUIE.
  - Source: [arXiv 2308.03279](https://arxiv.org/abs/2308.03279)
- **GNER** (Ding et al., "Rethinking Negative Instances for Generative Named Entity Recognition", Findings of ACL 2024):
  - **Method:** trains generative models to label non-entity tokens ("O") too, as negative instances.
  - **Zero-shot avg F1:** GNER-T5-base (248M) 59.5, large (783M) 63.5, xl (3B) 66.1, xxl (11B) 69.1.
  - **Claim:** beats prior SoTA by about 9 F1 zero-shot, and by 8 (LLaMA) and 11 (T5) points in the paper's comparisons.
  - Source: [arXiv 2402.16602](https://arxiv.org/html/2402.16602v2); [HF model card](https://huggingface.co/dyyyyyyyy/GNER-T5-xxl)
- **GLiNER2-PII** (Zaratiana, Lewis, Hurn-Maloney, arXiv May 2026, under submission):
  - **Model:** 0.3B parameters, adapted from GLiNER2, covering 42 PII types.
  - **Training data:** a synthetic corpus of 4,910 texts in en, fr, es, de, it, pt and nl. **None of our Asian languages are included.**
  - **SPY span-level F1** (exact match; average of Legal and Medical subsets): GLiNER2-PII 0.471, nvidia/gliner-PII 0.391, urchade/gliner_multi_pii-v1 0.384, openai/privacy-filter 0.373, knowledgator/gliner-pii-base-v1.0 0.368.
  - No latency is reported.
  - Source: [arXiv 2605.09973](https://arxiv.org/html/2605.09973)
- **urchade/gliner_multi_pii-v1:** a fine-tune of gliner_multi-v2.1 (mDeBERTa-v3 base, about 278M) on synthetic Mistral-generated PII data. It covers 6 European languages. — [HF post by urchade](https://huggingface.co:443/posts/urchade/899044648100689); [nullpii ONNX derivative card](https://huggingface.co/lBroth/nullpii)
- **InstructUIE (11B):** zero-shot OOD 47.2 and supervised average 81.2, as reported in the GLiNER tables. — [arXiv 2311.08526](https://arxiv.org/html/2311.08526)

### Inferences
- **Supervised comparison:** in supervised in-domain settings, the gap between GLiNER-L (0.3B encoder) and UniNER-7B is under 2 F1. A fine-tuned encoder with a good span head is therefore near the ceiling, and 7B-13B generative models are not worth their latency on CPU. Generative decoding over 100k-character documents would be orders of magnitude slower than 25 s per document (inference, not measured).
- **Off-the-shelf PII models:** none of the GLiNER-PII models were trained on zh, ja, ko, th, vi, hi, ar, ms, id or tl. Their absolute SPY F1 (<0.5 exact span) is far below what supervised models reach, so they are not drop-in replacements.
- **Experiment A, zero-shot reference:** evaluate urchade/gliner_multi_pii-v1, or GLiNER2-PII, zero-shot on our test sets with our 8 label names. This gives a no-training reference point per language; expect it to be well below baseline.
- **Experiment B, GLiNER as a span head:** fine-tune GLiNER-multi (mDeBERTa-v3 backbone) on our training data with our 8 labels, and compare to mDeBERTa-v3 + GlobalPointer. This isolates "GLiNER span head + label-prompt encoding + Pile-NER pretraining" vs our head.
- **GoLLIE** (guideline-following code-style IE, 7B-34B) was not fetched. Like other generative LLM IE models, it is unsuitable for CPU latency.

### Gaps
- **GoLLIE, NuNER-Zero and InstructUIE numbers** were not fetched from primary sources. InstructUIE figures above come via the GLiNER tables.
- **Throughput comparisons:** no paper reports measured throughput for GLiNER vs a BIO token classifier.
- **Asian-language PII results:** none found for any universal-NER model.

## Q4. Distilling LLM annotations into small encoders

### Takeaway
LLM-annotated pretraining data reliably improves small encoders in low-data regimes, though not necessarily in our full-data supervised regime.
- **NuNER (RoBERTa-base, about 125-155M):** pretrained on 1M GPT-3.5-annotated C4 sentences, it matches UniversalNER-7B after fine-tuning with 8-128 shots while being about 56x smaller.
- **What drives the gains:** entity-type diversity and dataset size (up to about 300k sentences); source-text diversity matters less.
- **For us:** a NuNER-style intermediate stage on LLM-annotated multilingual financial text, then supervised fine-tuning, is a plausible route to better mmBERT-small or XLM-R-base models.

### Cited Findings
- **NuNER** (Bogdanov et al., "NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data", arXiv 2024, EMNLP 2024):
  - **Model:** RoBERTa-base text encoder trained contrastively against a concept encoder.
  - **Data:** gpt-3.5-turbo annotated about 1.35M C4 sentences, filtered to 1M with 4.38M entities and about 200k unique concepts.
  - Source: [arXiv 2402.15343](https://arxiv.org/html/2402.15343)
- **NuNER few-shot results** (frozen encoder, macro F1 averaged over 4 datasets):

  | k | RoBERTa | NuNER |
  |---|---|---|
  | 1 | 24.5 | 39.4 |
  | 4 | 44.7 | 59.6 |
  | 16 | 58.1 | 67.8 |
  | 64 | 65.4 | 71.5 |

  - The gap shrinks as labelled data grows.
  - Source: [arXiv 2402.15343](https://arxiv.org/html/2402.15343)
- **NuNER vs UniversalNER** (fine-tuned): k≈8-16 gives UniNER 57.9 vs NuNER 58.8; k≈64-128 gives UniNER 71.0 vs NuNER 70.3. — [arXiv 2402.15343](https://arxiv.org/html/2402.15343)
- **NuNER ablations** (all at k=8):
  - Concept diversity: top-4 concepts 59.35 vs about 80k concepts 63.43.
  - Size: 1k sentences 60.3 vs 1M sentences 64.8, plateauing after 300k.
  - Source text: C4 vs Wikipedia is about equal (63.26 vs 62.76).
  - Encoder size: RoBERTa-large adds a little (64.78 → 65.97).
  - GPT-4 in-context learning still beats NuNER in some settings.
  - Source: [arXiv 2402.15343](https://arxiv.org/html/2402.15343)
- **UniversalNER:** targeted distillation of ChatGPT annotations into 7B/13B students beats the teacher by 7-9 F1 zero-shot. — [arXiv 2308.03279](https://arxiv.org/abs/2308.03279)
- **GLiNER:** Pile-NER pretraining (the UniNER LLM-annotated data) adds about 0.8 F1 in the supervised setting (82.9 vs 82.1), so LLM-annotated pretraining helps little once supervised data is available. — [arXiv 2311.08526](https://arxiv.org/html/2311.08526)
- **Same pattern in a GLiNER model:** GLiNER2-PII is likewise trained on LLM-generated synthetic annotated data (4,910 texts). — [arXiv 2605.09973](https://arxiv.org/html/2605.09973)

### Inferences
- **Expected gain:** with our full supervised training set, NuNER and GLiNER evidence suggests LLM-annotated pretraining adds roughly 0-1 F1 in-domain. It may add more on real, non-synthetic documents and rare formats, which is where our models are weak.
- **The distillation route that best matches our constraints** is teacher → student, not LLM → student:
  1. Train a large teacher: XLM-R large, or a bidirectionally unmasked Qwen3, per Q2.
  2. Pseudo-label unlabelled real wealth-management and code-switched text.
  3. Train mmBERT-small or XLM-R base on gold plus silver data.

  Inference only: none of the fetched papers tests this exact multilingual PII setting.
- **One-variable experiments:**
  - **(a)** Add an intermediate NuNER-style stage: LLM-annotated (8 PII types plus open types) multilingual unlabelled text → fine-tune XLM-R base as before. Compare to direct fine-tuning.
  - **(b)** Keep the student fixed (mmBERT-small) and add teacher-pseudo-labelled real text at a 1:1 ratio with gold. Measure real-slice and code-switch-slice F1 plus latency.

### Gaps
- **Multilingual and PII-specific distillation:** no NuNER-style study was found for multilingual or PII-specific types, or for Asian languages.
- **NuNER-Zero and multilingual NuNER variants:** model cards were not fetched.
- **Supervised large-data gains:** no paper quantifies gains from LLM-annotated pretraining in full-data supervised settings beyond GLiNER's 0.8 F1.
