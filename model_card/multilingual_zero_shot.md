# Multilingual zero-shot test of the English-only model (v3, v2.1)

**Question.** Does the current PII model work outside English? It was fine-tuned (and, for v3,
domain-adapted) on English text only. This is the baseline for the planned multilingual model
(XLM-R + language adapters + shared PII head).

**Answer: no.** English stays at 0.98 F1. Other languages score 0.28–0.76. E-mail addresses and
digit patterns are mostly still found. **Names, companies and addresses written in a non-Latin
script are mostly missed**: recall is 0.00 for Thai and 0.09–0.13 for Arabic. Number types are
mixed up (ACCOUNT / TIN / PHONE) once the cue words are not English. Latin-script Asian languages
are *detected* (≥ 95%) but with wrong boundaries and labels.

| | |
|---|---|
| Models | `outputs/deberta-v3-xsmall-pii-v3` (current), `outputs/deberta-v3-xsmall-pii-v2.1` |
| Languages | en (control), zh-Hans, zh-Hant, ja, ko, hi, ar, th, vi, ms, id, tl |
| Test sets | `ml_synth` 3,600 synthetic wealth-management docs · `ml_real` 10,000 public AI4Privacy docs |
| Scripts | `src/build_ml_eval.py`, `src/ml_templates.py`, `src/eval_multilingual.py` |
| Raw results | `results/{v3,v2.1}_ml_{synth,real}.{txt,json}` |
| Date | 2026-10-06 |

## Test sets

### `ml_synth`: native-language wealth-management documents (300 per language)
Written by hand per language in `src/ml_templates.py`, never used for training. Spans are exact
because the values are inserted into the templates.
- **Documents:** KYC / account-opening forms, statement headers (multi-line address), RM e-mails,
  transfer instructions, suitability notes and signature blocks. Field cues are in the language
  (姓名 / 出生日期 / 账号, 氏名 / 生年月日 / 口座番号, 성명 / 생년월일 / 계좌번호, नाम / जन्म तिथि, الاسم /
  تاريخ الميلاد, ชื่อ-นามสกุล / วันเกิด, Họ và tên / Ngày sinh, Nama / Tarikh Lahir, NPWP, …).
- **Values in local form:** native-script names, companies with local suffixes (有限公司, 株式会社,
  주식회사, प्रा. लि., ذ.م.م, บริษัท … จำกัด, Công ty TNHH, Sdn Bhd, PT … Tbk), native addresses,
  local phone / account / tax-ID formats (PAN, NPWP, MST, Thai 13-digit, My Number, 統一編號,
  UAE TRN …). Dates appear in local calendars and digits: 昭和60年, 民國74年, Thai Buddhist era
  พ.ศ. 2528, and Arabic-Indic / Devanagari digits.
- **Look-alikes that must stay O:** amounts, ordinary dates and SWIFT / branch codes (3,000+).
- **Control:** the same templates in English. v3 scores **0.98** on them, so the template format
  is not what makes the other languages hard.
- zh-Hant reuses the zh-Hans text through OpenCC (`s2twp`, Taiwan phrasing), with Taiwan / Hong
  Kong addresses, phones, IDs and amounts.

### `ml_real`: public data (1,000 per language)
- AI4Privacy OpenPII-1.5M **validation** split (en, zh, ja, ko, vi, ms, id, tl) and
  open-pii-masking-500k validation split (hi). None of it was trained on: training used English
  rows only.
- Same label map and cleanup as the English training data (`configs/label_map.yaml`). Dates
  become DOB when a birth cue in that language is next to them (出生, 生年月日, …生まれ, 생년월일, जन्म,
  ngày sinh, lahir, kapanganakan, الميلاد, เกิด).
- zh-Hant = the same zh docs converted with OpenCC `s2t`. No doc changed length, so the offsets
  are valid.
- Caveats: the texts are generic, not finance, and there are no company names (no BUSINESS).
  **In the Hindi rows every entity value is romanised**, so `ml_real/hi` tests Latin-script
  entities in Devanagari text. Native Devanagari entities appear only in `ml_synth`. There is no
  public Arabic or Thai PII data.

Every gold span carries a script class: **nonlatin** (contains a CJK / kana / hangul /
Devanagari / Arabic / Thai letter or a non-ASCII digit) or **latin** (romanised names, e-mails,
ASCII numbers).

## Two decodings: the word-grouping artifact

`predict.py` gives each word the label of its first sub-token (the OpenMed recipe used in
training). Words come from `tokenize_bio.word_starts`, which splits only at spaces and
punctuation. In Chinese, Japanese and Thai, and in Korean where particles attach to the word, a
whole unspaced run is **one word**:

```
电话13812345678   ->  电 话 13 812 345 678  = one "word"  ->  label of 电 (O) for all of it
```

Even a correct PHONE prediction on the digits is thrown away. Every document is therefore decoded
twice from the same forward pass:
- **default**: first sub-token decides the word (the shipped behaviour)
- **alltok**: every token keeps its own prediction

The difference between the two measures the artifact, separately from what the model knows. It
is in our decoding code, so any future model trained with `label_all_tokens: false` has the same
problem in these languages.

## Results (v3, strict F1 = exact boundaries + label)

### Overall

| Set / decoding | EN | ZH-Hans | ZH-Hant | JA | KO | HI | AR | TH | VI | MS | ID | TL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synth · default | **0.982** | 0.461 | 0.427 | 0.304 | 0.540 | 0.357 | 0.445 | 0.278 | 0.554 | 0.415 | 0.422 | 0.685 |
| synth · alltok | 0.975 | 0.540 | 0.518 | 0.360 | 0.512 | 0.332 | 0.380 | 0.268 | 0.477 | 0.380 | 0.401 | 0.662 |
| real · default | **0.983** | 0.362 | 0.368 | 0.290 | 0.333 | 0.724 | – | – | 0.637 | 0.763 | 0.666 | 0.753 |
| real · alltok | 0.977 | 0.504 | 0.510 | 0.497 | 0.604 | 0.673 | – | – | 0.573 | 0.726 | 0.635 | 0.711 |

Lenient F1 (any overlap + label) for the non-English languages is 0.29 (th) to 0.88, and the
share of gold spans touched by *any* prediction is 0.29 (th) to 0.98 (tl). Much of the strict loss is
boundaries and labels, not missed entities. Thai is the exception: the model finds almost nothing.

### Recall by script of the entity (synth, default)

| | ZH-Hans | ZH-Hant | JA | KO | HI | AR | TH |
|---|---|---|---|---|---|---|---|
| latin (e-mails, numbers) | 0.44 | 0.39 | 0.34 | 0.60 | 0.66 | 0.81 | 0.42 |
| **nonlatin** (names, companies, addresses, native dates) | 0.30 | 0.29 | 0.22 | 0.49 | 0.26 | **0.13** | **0.00** |

### Per entity (synth, default)

| | PERSON | BUSINESS | ADDRESS | DOB | ACCOUNT | PHONE | EMAIL | TIN |
|---|---|---|---|---|---|---|---|---|
| en | 0.990 | 0.987 | 0.961 | 1.000 | 0.955 | 1.000 | 1.000 | 0.961 |
| zh-Hans | 0.570 | 0.366 | 0.284 | 0.333 | 0.519 | 0.455 | 0.597 | 0.030 |
| zh-Hant | 0.517 | 0.199 | 0.541 | 0.373 | 0.274 | 0.438 | 0.651 | 0.027 |
| ja | 0.383 | 0.274 | 0.000 | 0.222 | 0.000 | 0.561 | 0.657 | 0.009 |
| ko | 0.627 | 0.487 | 0.291 | 0.203 | 0.201 | 0.746 | 0.828 | 0.567 |
| hi | 0.227 | 0.017 | 0.185 | 0.279 | 0.877 | 0.596 | 0.921 | 0.000 |
| ar | 0.180 | 0.194 | 0.098 | 0.228 | 0.776 | 0.826 | 1.000 | 0.000 |
| th | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.610 | 0.948 | 0.000 |
| vi | 0.500 | 0.278 | 0.477 | 0.448 | 0.689 | 0.791 | 1.000 | 0.189 |
| ms | 0.235 | 0.544 | 0.375 | 0.511 | 0.289 | 0.750 | 1.000 | 0.010 |
| id | 0.307 | 0.420 | 0.227 | 0.607 | 0.453 | 0.734 | 0.995 | 0.000 |
| tl | 0.487 | 0.846 | 0.511 | 0.637 | 0.949 | 0.956 | 1.000 | 0.673 |

### Look-alikes tagged as PII (amounts, ordinary dates, codes)
Default decoding. Synth: English 3.1%; ja / ko / th 0.2–1.7%; zh-Hans / zh-Hant / hi / vi / ms
4–6%; ar / id / tl 9–15%. Real: English 1.1%; others 2.9–14.7% (hi, vi, tl highest).

## What goes wrong

Label-confusion percentages come from the alltok predictions (`results/v3_ml_synth.txt`, last section).

1. **Native-script names, companies and addresses.** Thai: 0% of PERSON / BUSINESS / ADDRESS
   found. Arabic: 54% of names and 76% of companies missed. Hindi companies are read as PERSON
   (37%) or ADDRESS (24%). Chinese misses about a third of names and half of companies.
2. **Number types depend on English cues.** The digits are still found, but their type is a
   guess: ja ACCOUNT → TIN 63%, ar TIN → ACCOUNT 100%, ms TIN → ACCOUNT 89%, id TIN → PHONE 64%,
   zh TIN → ACCOUNT 81%, ko ACCOUNT → PHONE 41%. An NPWP `90.689.044.4-230.540` is split into
   three ACCOUNT pieces.
3. **Dates of birth.** Recognised only 4% (th) to 36% (vi) of the time in CJK / Thai / Arabic / Vietnamese.
   Non-English birth cues (出生日期, 생년월일, วันเกิด) are not linked to DOB, and dates written in
   local calendars or digits (民國74年, พ.ศ. 2528, ١٥/٠٨/١٩٨٥) are missed or read as PHONE / ADDRESS.
4. **Latin-script Asian languages: found, not delimited.** ms / id / tl reach 95–98% detection
   but 0.42–0.69 strict F1. Ordinary words next to entities get tagged: in
   `Pelanggan Wong Wei Ming … Beliau bekerja di Cahaya Timur Trading Sdn Bhd dan menetap di …`
   the model returns PERSON `Pelanggan Wong Wei Ming`, PERSON `Beliau bekerja` ("he works"),
   ADDRESS `Cahaya Timur Trading Sdn Bhd` and ADDRESS `menetap` ("resides"). 20–54% of ms / id
   names and companies are labelled ADDRESS.
5. **The word-grouping artifact.** On the real set, alltok raises ko 0.33 → 0.60, ja 0.29 → 0.50
   and zh 0.36 → 0.50, e.g. zh PHONE 0.61 → 0.92 and ko ACCOUNT 0.41 → 0.92. Digits and e-mails
   written right next to CJK characters, and Korean particles (`동촌로16길 257입니다`), are merged
   into the neighbouring word. On space-separated languages alltok is slightly *worse* (it splits
   words the model labels as one). On synthetic Korean, Hindi and Arabic it does not help either,
   because those documents put spaces around values. The fix belongs in `word_starts` (split per
   CJK / Thai character, or label all tokens). It changes how training labels are built, so it is
   for the next model, not this test.

## v2.1 vs v3 (does the extra English pretraining in v3 help elsewhere?)

| Strict F1, default | ALL synth | ALL real | th synth | ar synth | hi synth |
|---|---|---|---|---|---|
| v2.1 | 0.510 | 0.574 | 0.238 | **0.523** | **0.392** |
| v3 | 0.496 | **0.598** | **0.278** | 0.445 | 0.357 |

Mixed and within a few points. Adapting on 178M English tokens neither helped nor broke other
languages in a consistent way.

## Examples (v3)

```
ar  طلب نورة بن سعيد النعيمي (الرقم الضريبي 100915023461212) استرداد AED 2,765,700 من المحفظة 6287781646398.
    gold  PERSON نورة بن سعيد النعيمي · TIN 100915023461212 · ACCOUNT 6287781646398
    pred  PERSON طلب ("requested") · ACCOUNT 100915023461212 · ACCOUNT 6287781646398

th  กรุณาโอนเงิน ฿1,071,800.00 จากบัญชี 3049267585 ของ ศักดิ์ชัย ศรีวงศ์ ไปยัง บริษัท เอเชียแคปปิตอล จำกัด (มหาชน) …
    gold  ACCOUNT 3049267585 · PERSON ศักดิ์ชัย ศรีวงศ์ · BUSINESS บริษัท เอเชียแคปปิตอล จำกัด (มหาชน)
    pred  (nothing)

ja  佐藤 美咲様（マイナンバー4997 0585 0534）より、ポートフォリオ5393500から¥2,696,200の解約…
    gold  PERSON 佐藤 美咲 · TIN 4997 0585 0534 · ACCOUNT 5393500
    pred  PERSON 美咲様 · PHONE 0585 0534

hi  कृपया 12-08-2025 को अनीता मिश्रा के खाते 23877040742903 से श्री गणेश कैपिटल लिमिटेड (SWIFT …) को INR 49,69,000 …
    gold  PERSON अनीता मिश्रा · ACCOUNT 23877040742903 · BUSINESS श्री गणेश कैपिटल लिमिटेड
    pred  PERSON अनीता मिश्रा · ACCOUNT 23877040742903 · PERSON गणेश कैपिटल लि · PERSON ड

ko  … 모임 장소는 제주시 신제주에 위치한 동촌로16길 257입니다.          (real)
    gold     ADDRESS 동촌로16길 257
    default  ADDRESS 동촌로16길 257입니다   (particle merged into the word)
    alltok   ADDRESS 동촌로16길 257
```

## Limitations

- The synthetic templates were written by a non-native author (an LLM) and have not been reviewed
  by native speakers. Phrasing may be stilted, though the field cues and formats follow common
  local usage. Six prose templates per language plus generated forms give limited variety.
- AI4Privacy non-English rows are machine-generated with some label noise (e.g. a Korean PERSON
  span ending mid-word, `상현 경구 아`). They are generic rather than financial.
- zh-Hant is converted from zh-Hans, not native Traditional Chinese text (real set), and is
  OpenCC-converted template text (synth).
- Real-set Hindi entities are all romanised. Arabic and Thai have synthetic data only.
- Khmer, Burmese and Lao are not covered.

## Reproduce

```bash
pip install opencc-python-reimplemented
cd src
python download.py --only ai4privacy500k      # Hindi (data/raw/ai4privacy500k/validation.jsonl, 142 MB)
python build_ml_eval.py                       # -> data/processed/ml_synth.jsonl, ml_real.jsonl
python eval_multilingual.py --split ml_synth --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --out ../results/v3_ml_synth
python eval_multilingual.py --split ml_real  --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --out ../results/v3_ml_real
pytest ../tests/test_ml_eval.py
```

About 2.5 min (synth) and 7 min (real) per model on an RTX 5060 Ti, both decodings included.
