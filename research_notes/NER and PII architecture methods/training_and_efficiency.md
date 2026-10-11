# Training-time and inference-efficiency methods for transformer NER (2019-2026)

Context: multilingual PII NER, XLM-R base (278M) + GlobalPointer span head, 3 epochs, LR 2e-5, bs 32, single seed. CPU ~25 s / 100k chars (36.8k tokens, 129 windows x 384), GPU ~1.07 s. Small candidate: mmBERT-small (140M).

Verification note: items marked **[fetched]** were confirmed from the source in this session. Items marked **[recalled]** are standard, well-known results cited from the paper's abstract as I know it but not re-fetched this session; the report writer should treat their exact numbers with care.

## Q1. Training methods that raise NER F1 at zero inference cost

### Takeaway
Published NER gains from regularisers (R-Drop, FGM/PGD/FreeLB, boundary smoothing, dice loss) are usually small, about +0.2 to +1.0 F1 points. That is the same size as seed noise (Q2), so a single-seed result cannot confirm any of them. The cheapest strong options for our setup are: train longer (Mosbach 2021, Zhang 2021), boundary smoothing (it is made for span heads like GlobalPointer), PGD/FGM, and checkpoint averaging. DAPT/TAPT on in-domain unlabeled financial text is the most likely source of a larger gain, but it is also the most expensive.

### Cited findings
- **Adversarial training + R-Drop on multilingual NER (closest analogue to our setup) [fetched].** Qtrade AI, SemEval-2022 Task 11 (MultiCoNER), macro-F1 on dev. Baselines were ZH 86.31, Multilingual 85.82, Mix-Code 75.93. FGM gave 86.85 / 86.85 / 75.54. PGD gave 87.37 / 86.33 / 75.96. FreeLB gave 86.81 / 86.14 / 75.88. The settings were FGM eps=0.8; PGD eps=1, alpha=0.1, K=3; FreeLB adv_lr=0.3, mag=0.05, K=3. Adversarial training helped ZH (+0.5 to +1.06) and Multilingual (+0.3 to +1.03). On code-mixed data it was neutral or negative (-0.39 to +0.03). Multilingual-track R-Drop (KL coefficient 0.1) gave +0.22 macro-F1 (87.15). R-Drop combined with private/shared feature extractors gave +0.49 (87.42). All results are single runs. — [Qtrade AI at SemEval-2022 Task 11, SemEval 2022](https://arxiv.org/abs/2204.07459)
- **R-Drop [recalled].** R-Drop minimises the bidirectional KL between two dropout forward passes. Its main gains were reported on MT, summarisation, language modelling and GLUE, not NER. Training costs about 2x the forward/backward compute, and inference cost does not change. — [R-Drop: Regularized Dropout for Neural Networks, NeurIPS 2021](https://arxiv.org/abs/2106.14448)
- **FreeLB / SMART [recalled].** FreeLB adds embedding-space adversarial perturbations with K ascent steps. SMART uses smoothness-inducing adversarial regularisation plus a Bregman proximal-point step. Both report gains on GLUE, not NER benchmarks. Training costs about K+1x, and inference cost does not change. — [FreeLB, ICLR 2020](https://arxiv.org/abs/1909.11764); [SMART, ACL 2020](https://arxiv.org/abs/1911.03437)
- **Boundary smoothing [fetched abstract].** Label smoothing for span-based NER: some probability mass moves from the gold span to spans whose boundaries are within a small distance. The authors report results better than or competitive with prior SOTA on 8 English/Chinese NER benchmarks, plus reduced over-confidence and better calibration. I did not retrieve the exact per-dataset gains. — [Boundary Smoothing for Named Entity Recognition, ACL 2022](https://aclanthology.org/2022.acl-long.490); code: [syuoni/eznlp](https://github.com/syuoni/eznlp)
- **Dice loss for imbalance [recalled].** Self-adjusting dice loss for data-imbalanced NLP. The authors report NER gains on CoNLL03, OntoNotes5, MSRA and OntoNotes4. My recollection is that the gains are under 1 point on English (about +0.3 on CoNLL03, about +1 on OntoNotes5); verify these numbers. — [Dice Loss for Data-imbalanced NLP Tasks, ACL 2020](https://arxiv.org/abs/1911.02855)
- **Re-initialising top layers and longer training [fetched].** Re-initialising the pooler and the top L layers speeds convergence and improves results in few-sample BERT fine-tuning; the best L depends on the task. The paper also finds that common recipes do not train long enough. Once the optimiser (Adam bias correction) and the training length are fixed, layer-wise LR decay (LLRD), Mixout, pre-trained weight decay and intermediate-task training help less. — [Revisiting Few-sample BERT Fine-tuning, ICLR 2021](https://arxiv.org/abs/2006.05987)
- **Stability baseline [fetched].** Fine-tuning instability is an optimisation problem (vanishing gradients early in training), not catastrophic forgetting or small data. The recommended fixes are a small LR, Adam bias correction, and many more iterations instead of a fixed small epoch count. The variance that remains is a generalisation effect: models with the same training loss can differ in test score. — [On the Stability of Fine-tuning BERT, ICLR 2021](https://arxiv.org/abs/2006.04884)
- **LLRD as practitioner recipe [fetched].** The usual recipe sets a top-layer LR and multiplies it by about 0.9 per layer going down. These values are a blog's choices, not values from a paper. — [Towards Data Science: Advanced techniques for fine-tuning transformers](https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e)
- **Weight averaging [recalled].** SWA averages weights along the SGD trajectory and gives flatter optima. Model soups average fine-tuned models from different hyperparameter runs and gain without any inference cost; the paper is about image and text classification. — [SWA, UAI 2018](https://arxiv.org/abs/1803.05407); [Model soups, ICML 2022](https://arxiv.org/abs/2203.05482)
- **DAPT/TAPT [recalled for numbers; paper fetched only partially].** Continued pretraining on domain text (DAPT) and on task unlabeled text (TAPT) improves RoBERTa on classification tasks in 4 domains, and DAPT followed by TAPT works best. The paper evaluates classification, not NER. — [Don't Stop Pretraining, ACL 2020](https://arxiv.org/abs/2004.10964)

### Inferences
- **One-variable experiments against v5.0 (all keep inference cost at 0):**
  1. **Epochs 3 → 6 (or 10 with early stopping on dev).** Expected about +0.2 to +1.0 F1; strongest prior from Mosbach 2021 and Zhang 2021. Training cost 2x.
  2. **Boundary smoothing on GlobalPointer (eps 0.1 to 0.2, D=1).** This is a direct fit because GlobalPointer scores (start, end) span matrices. Expected +0.2 to +0.8 F1 and better calibrated thresholds. Training cost about 1x.
  3. **FGM on word embeddings (eps 0.5 to 1.0).** Expected about +0.3 to +1.0 on monolingual and multilingual slices, and possibly about 0 on code-switch slices, as in the Qtrade Mix-Code result. Training cost 2x. PGD with K=3 costs about 4x.
  4. **R-Drop (alpha 0.1 to 1, applied to span logits).** Expected about +0.2. Training cost 2x.
  5. **Checkpoint averaging of the last N epoch checkpoints, or a soup of seeds.** Expected 0 to +0.5, and it also reduces variance.
  6. **LLRD 0.9 with a higher top LR (for example 5e-5) and a separate, higher LR for the GlobalPointer head.** Expected 0 to +0.5. Span heads initialised from scratch often benefit from a 10 to 100x head LR (a practitioner heuristic; no citation found).
  7. **Re-init the top 1 to 2 layers.** This mostly helps small data. Expect about 0 for a large synthetic set; low priority.
  8. **Focal or dice loss.** GlobalPointer already uses a multilabel-categorical (circle-style) loss that copes with the negative-span imbalance, so expect a small effect. Low priority.
  9. **TAPT: MLM on in-domain unlabeled financial documents before fine-tuning (about 1 to 5 epochs).** Expected gain is uncertain, roughly +0.5 to +2 on real-document slices and about 0 on synthetic slices. This is the costliest option.
- Stacking (for example epochs + FGM + boundary smoothing) is usual in competition systems, but each component has to be validated with at least 3 seeds (see Q2).

### Gaps
- I found no paper that reports boundary-smoothing F1 deltas on a multilingual, PII or GlobalPointer setup; the exact ACL 2022 gains were not retrieved.
- I found no CoNLL-2003 BERT/XLM-R controlled study that reports R-Drop or FGM gains together with seed standard deviation.
- Self-training with unlabeled data and multi-task learning for NER were not covered in this session's searches. No numbers.
- I found no DAPT/TAPT results for NER (Gururangan 2020 covers classification only).

## Q2. Seed variance in NER fine-tuning: how many seeds to trust +0.005 F1

### Takeaway
Changing only the seed moves NER F1 by about 1 point (0.01) for older BiLSTM-CRF systems, and the seed-to-seed differences can be statistically significant. A +0.005 gain on one seed is inside that noise. With a per-run standard deviation of 0.003 to 0.005, confirming a +0.005 gain needs roughly 6 to 16 seeds per arm, or paired designs and pooled test sets.

### Cited findings
- **Seed alone changes F1 for two state-of-the-art NER systems [fetched].** The swing is about 1 F1 point, with differences significant at p < 1e-4. The authors recommend comparing score distributions across many runs. The study covers 50,000 LSTM networks on five tagging tasks. — [Reimers & Gurevych, Reporting Score Distributions Makes a Difference, EMNLP 2017](https://aclanthology.org/D17-1035/)
- **Weight initialisation and data order matter about equally [fetched].** Hundreds of BERT fine-tuning runs on GLUE, varying only the seed: the head's weight initialisation and the training data order each contribute comparable variance. Some initialisations are good across tasks, and many runs on small datasets diverge, so early stopping of bad runs is recommended. Data for 2,100 trials was released. — [Dodge et al., Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping, 2020](https://arxiv.org/abs/2002.06305)
- **Instability is fixable [fetched].** Bias-corrected Adam and longer training remove most failed runs, and Zhang et al. say this makes the many random restarts of Dodge 2020 unnecessary. A generalisation variance remains. — [Mosbach et al., ICLR 2021](https://arxiv.org/abs/2006.04884); [Zhang et al., ICLR 2021](https://arxiv.org/abs/2006.05987)
- **Recommended statistical tooling [recalled].** Use significance testing with an almost-stochastic-order test over multiple seeds. — [Ulmer et al., deep-significance, 2022](https://arxiv.org/abs/2204.06815); [Dror et al., Hitchhiker's Guide to Testing Statistical Significance in NLP, ACL 2018](https://aclanthology.org/P18-1128/)

### Inferences
- Power calculation (two-sided alpha 0.05, 80% power, independent arms): n per arm ≈ 15.7·σ²/Δ². For Δ = 0.005: σ = 0.002 gives n ≈ 3; σ = 0.003 gives n ≈ 6; σ = 0.005 gives n ≈ 16. For an XLM-R base fine-tuned on a large synthetic set, σ is plausibly 0.002 to 0.005. **The cheapest one-variable experiment is to rerun v5.0 with seeds 1 to 5 and measure σ per slice** (en_test, ml_real, mixed and so on). Every later +0.005 claim depends on that number.
- Paired comparisons (same seed and same data order across arms, with paired bootstrap over documents) reduce the variance needed compared with unpaired comparisons.
- Small slices such as real or KIII documents will have much larger σ than large synthetic test sets, so set thresholds per slice.

### Gaps
- I found no published seed σ for XLM-R base on PAN-X/WikiAnn or on a PII dataset in this session. Reimers' numbers are from BiLSTM-CRF systems.

## Q3. Knowledge distillation for NER / token classification

### Takeaway
Multilingual NER distils well. XtremeDistil keeps about 95% of mBERT F1 over 41 languages and reports up to a 51x batch-latency speed-up. Its 28M-parameter student reaches 88.64 average F1 against 91.86 for mBERT. For our setup, distilling the XLM-R base + GlobalPointer teacher (soft span-matrix logits) into mmBERT-small or a pruned XLM-R on large unlabeled or synthetic text is the main route to a small model that keeps F1. Training mmBERT-small directly is a risk: the mmBERT authors say its NER is weak (Q4).

### Cited findings
- **XtremeDistil [fetched].** Multi-stage distillation uses the teacher's internal representations and is agnostic to the teacher architecture. It compresses mBERT by up to 35x in parameters and 51x in batch-inference latency while keeping 95% of F1 on WikiAnn NER over 41 languages (Rahimi et al. 2019 setting). In the table, the 28M-parameter student scores 88.64 average F1 against 91.86 for mBERT (6.4x compression). The 35x and 51x figures are for a different, smaller configuration than the 28M model. The paper also studies how much unlabeled transfer data is needed. — [XtremeDistil, ACL 2020](https://aclanthology.org/2020.acl-main.202)
- **XtremeDistilTransformers [fetched].** A task-agnostic distilled student with 14M parameters matches XtremeDistil on the same 41-language NER setup, within 4% F1 of mBERT, at about 2x more compression. — [XtremeDistilTransformers, 2021](https://arxiv.org/abs/2106.04563)
- **Multilingual MiniLM as a ready student [fetched, from the mmBERT table].** WikiAnn F1: Multilingual MiniLM 59.2, mDistilBERT 54.5, XLM-R base 61.4. — [mmBERT paper, Table 3](https://arxiv.org/html/2509.06888v1). MiniLM architecture [recalled]: 12 layers, hidden size 384, deep self-attention distillation; the multilingual version reuses the XLM-R 250k vocabulary, so the embedding matrix is most of its parameters. — [MiniLM, NeurIPS 2020](https://arxiv.org/abs/2002.10957)
- **TinyBERT [recalled].** Two-stage (general + task) layer-wise distillation of attention and hidden states. The 4-layer model is 7.5x smaller and 9.4x faster than BERT-base, keeping about 96.8% on GLUE. It is evaluated on GLUE/SQuAD, not NER. — [TinyBERT, Findings EMNLP 2020](https://arxiv.org/abs/1909.10351)

### Inferences
- **Experiment D1 (one variable: loss target).** Train mmBERT-small with GlobalPointer on the same v5 data, adding a KL/MSE loss to the XLM-R base teacher's span-logit matrix (which is temperature-softened) alongside the hard labels. Compare against mmBERT-small trained on hard labels only. Expected: closes part of the gap to the teacher (XtremeDistil-style retention suggests 95%+ is achievable), with student latency unchanged.
- **Experiment D2 (one variable: transfer data).** Add teacher-labelled unlabeled real financial text (pseudo-labels or soft spans). XtremeDistil found that the amount of unlabeled data matters.
- Distilling into a 6-layer XLM-R (copying alternate layers of the teacher) keeps the tokenizer, so windowing and post-processing stay identical, and should give about a 2x speed-up on the encoder.

### Gaps
- I found no paper that specifically distils GlobalPointer or biaffine span heads. The method above is extrapolated from token-level KD.
- I found no published XLM-R large → base or base → small distillation numbers for NER in this session.

## Q4. Inference efficiency on CPU while keeping F1

### Takeaway
For our CPU target, the largest low-risk wins are, in order: (1) ONNX Runtime or OpenVINO export plus dynamic int8 of the MatMul/Linear weights (about 2 to 3x, typically under 0.5 to 1 point of F1 loss); (2) fewer tokens processed, via a smaller window overlap/stride and batching windows; (3) a smaller encoder (mmBERT-small, about 2x faster per its authors, but with weaker NER) or distillation (Q3). Vocabulary trimming shrinks memory, not compute. Early exit and token pruning have published results for sequence labelling but are research-grade.

### Cited findings
- **Int8 on CPU for BERT NER [fetched].** Biomedical NER thesis using ONNX Runtime dynamic 8-bit quantisation. With 25% structured pruning plus int8, BioBERT's CPU latency fell from 32.52 ms to 12.02 ms (2.6x) while F1 went from 0.954 to 0.948. 25% structured sparsity alone gave no measurable F1 loss. This is a thesis, not peer-reviewed, and the int8-only F1 effect cannot be separated out. — [Chalmers MSc thesis: Methods for Optimizing BERT Model on Edge Devices](https://odr.chalmers.se/items/46a4159a-1ee3-4960-b0e3-1c13cfa17a03/full)
- **Recipe for ONNX int8 NER [fetched].** quantize_dynamic restricted to MatMul, per-channel, QUInt8 weights, no reduced range. No F1 is reported. — [HF model card luizedu91/ner-bert-lenerbr-onnx-int8](https://huggingface.co/luizedu91/ner-bert-lenerbr-onnx-int8)
- **Static int8 via Optimum/ORT (classification) [fetched].** DistilBERT accuracy went from 92.5% (fp32) to 92.24% (int8), at about 3x lower latency (sequence length 128). — [Phil Schmid, Static Quantization with Hugging Face optimum](https://www.philschmid.de/static-quantization-optimum)
- **OpenVINO / Optimum-Intel NER path [fetched].** OVModelForTokenClassification is a drop-in replacement. NNCF post-training quantisation was calibrated on 100 CoNLL-2003 samples, with QAT as the fallback if accuracy drops. No F1 or latency numbers are published in the notebook. — [OpenVINO NER notebook](https://docs.openvino.ai/2023.3/notebooks/244-named-entity-recognition-with-output.html); [Optimum Intel + OpenVINO blog](https://blog.openvino.ai/blog-posts/accelerate-inference-of-hugging-face-transformer-models-with-optimum-intel-and-openvino)
- **PyTorch dynamic quantisation of BERT [recalled].** On the official tutorial (BERT-base MRPC), the model shrinks from about 438 MB to about 181 MB with an F1 drop of under 1 point. Treat the exact numbers as unverified. — [PyTorch tutorial: Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
- **Q8BERT [recalled].** 8-bit quantisation-aware training loses under 1% on GLUE/SQuAD. — [Q8BERT, 2019](https://arxiv.org/abs/1910.06188)
- **Zero-shot dynamic quantisation for transformers [found, not read].** — [arXiv 2211.09744](https://arxiv.org/abs/2211.09744)
- **Vocabulary trimming [fetched].** In multilingual LMs the embedding matrix can be over 80% of parameters. Trimming to the target languages (frequency on mC4) keeps performance with about 50% of the vocabulary. Post-fine-tuning trimming is the more robust option and needs no retraining. Tool: pip install vocabtrimmer. — [Ushio et al., Efficient Multilingual Language Model Compression through Vocabulary Trimming, Findings EMNLP 2023](https://arxiv.org/abs/2305.15020); [lm-vocab-trimmer](https://github.com/asahi417/lm-vocab-trimmer)
- **mmBERT [fetched].** mmBERT-small has 140M total and 42M non-embedding parameters; mmBERT-base has 307M total and 110M non-embedding. Context length is up to 8192, against 512 for XLM-R. FlashAttention 2 and unpadding come from ModernBERT. Speed: mmBERT-base is more than 2x faster on variable-length input and about 4x faster on long context; mmBERT-small is about 2x faster than base and about 2x faster than similar-size multilingual models (no absolute tokens/s in the text). **WikiAnn NER F1: mmBERT-small 54.3, mmBERT-base 58.2, XLM-R base 61.4, mGTE base 60.7, Multilingual MiniLM 59.2, mDistilBERT 54.5.** The authors note NER/POS do not improve, probably because of tokenizer word-boundary differences. — [mmBERT, arXiv 2509.06888 / ICML 2026](https://arxiv.org/html/2509.06888v1); [HF blog](https://huggingface.co/blog/mmbert)
- **Movement pruning [recalled].** First-order pruning works well at high sparsity during fine-tuning (SQuAD/MNLI). Unstructured sparsity does not speed up dense CPU kernels unless the runtime supports sparsity (for example DeepSparse, or OpenVINO sparse weights on Xeon). — [Movement Pruning, NeurIPS 2020](https://arxiv.org/abs/2005.07683); [OpenVINO sparsity notebook](https://docs.openvino.ai/2023.0/notebooks/116-sparsity-optimization-with-output.html)
- **Head pruning [recalled].** Many attention heads can be removed at test time with little loss. — [Are Sixteen Heads Really Better than One?, NeurIPS 2019](https://arxiv.org/abs/1905.10650)
- **Early exit for sequence labelling [recalled].** SENTEE (sentence-level) and TOKEE (token-level) early exit with window-based uncertainty report a 66 to 75% reduction in inference cost with minimal loss on NER/POS/CWS. — [Li et al., Accelerating BERT Inference for Sequence Labeling via Early-Exit, ACL 2021](https://arxiv.org/abs/2105.13878)

### Inferences
- **Per-window arithmetic.** 129 windows x 384 = 49.5k tokens processed for 36.8k real tokens, so about 35% of compute goes to overlap. A smaller stride overlap (for example 384 with 64 overlap instead of the current setting), or 512-token windows, cuts the windows processed. Attention is quadratic, but at 384 to 512 the FFN dominates, so latency is roughly linear in tokens processed.
- **One-variable experiments against v5.0 (record F1 per slice and CPU s/100k chars):**
  - E1. **Export to ONNX Runtime fp32, no other change.** F1 is identical (to within about 1e-4). CPU latency drops about 1.2 to 1.5x from graph fusion (attention/LayerNorm/GELU fusions).
  - E2. **ORT dynamic int8 (MatMul only, per-channel), applied on top of E1.** Expected F1 change 0 to -0.01 (sources show -0.003 to -0.006 on classification/NER). CPU latency 2 to 3x lower; model 1.1 GB → about 0.3 GB. Keep the GlobalPointer head in fp32 if F1 drops, because span-boundary logits are sensitive.
  - E3. **OpenVINO int8 PTQ (NNCF, 300 calibration windows from real documents).** Similar F1 to E2, often faster on Intel CPUs.
  - E4. **Window overlap / stride sweep at a fixed window of 384.** F1 change about 0 to -0.003 (entities at window edges). Latency falls in proportion to the windows saved.
  - E5. **Window 384 → 512.** Fewer windows and more context per window. F1 0 to +0.003. Latency about 0.85 to 0.95x.
  - E6. **Vocabulary trim to our 8 languages + English (post fine-tuning).** F1 change about 0. Latency about 1x (embedding lookup is cheap). Memory -50% or more (the XLM-R embedding is about 192M of 278M parameters). Do this if RAM or download size matters, not for speed.
  - E7. **PyTorch SDPA attention (attn_implementation="sdpa").** F1 identical. CPU speed-up is small at a length of 384 (FFN-bound).
  - E8. **mmBERT-small in place of XLM-R base (same data and head).** F1 is likely lower: -0.02 to -0.07 by the WikiAnn analogue (54.3 vs 61.4), and that gap is on a harder benchmark. Our in-domain synthetic data may narrow it. Latency about 2x lower, more with unpadding and 8k windows (fewer window seams). Pair it with distillation (D1).
  - E9. **Early exit (TOKEE-style).** Research effort. Possibly about 2x at a small F1 cost. Low priority.
- **Expected combined CPU budget.** E1 + E2 + E4 should plausibly bring about 25 s down to about 8 to 12 s per 100k characters without changing the model. Adding distillation to a 6-layer or small encoder could reach about 4 to 6 s. These are estimates, not measured.

### Gaps
- I found no peer-reviewed int8-only (unpruned) NER F1 delta for XLM-R on CPU. The numbers above come from classification, a thesis, and pruned models.
- I found no study of window size / stride trade-offs for NER F1 in this session.
- I found no token-pruning paper evaluated on token classification; token pruning (for example LTP, PoWER-BERT) targets sequence classification and is hard to apply when every token needs a label.
- mmBERT throughput figures are relative and GPU-oriented; I found no CPU numbers.
- FlashAttention is a GPU kernel. On CPU, SDPA uses a fused path, and I found no published BERT NER benchmark of its CPU gain.
