"""Char-span <-> BIO token label conversion with sliding windows.

Follows OpenMed's recipe of labelling only the first sub-token of each word
(`label_all_tokens=False`); continuation sub-tokens get -100 in training and inherit
their word's label at inference. OpenMed's BERT tokenizers split punctuation into
separate words, while DeBERTa's SentencePiece only splits on spaces, so a "word" here
also breaks at every letter/punctuation boundary (BERT BasicTokenizer semantics).
"""
from common import IGNORE

IGNORE_ID = -100


def _trim_offset(text: str, start: int, end: int) -> tuple[int, int]:
    # DeBERTa's SentencePiece tokens can carry the leading space ("▁word") in their offsets.
    while start < end and text[start].isspace():
        start += 1
    return start, end


def _is_word_char(c: str) -> bool:
    return c.isalnum()


def script_of(c: str) -> str:
    """Coarse script class for word boundaries in unspaced text."""
    if c.isdigit():
        return "digit"
    o = ord(c)
    if 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F:
        return "han"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x0E00 <= o <= 0x0E7F:
        return "thai"
    return "other"


# scripts written without spaces between words: every token is its own word
UNSPACED = {"han", "kana", "thai"}


def word_starts(text: str, offsets, script_boundaries: bool = False) -> list[bool]:
    """True where a (non-special) token begins a new word.

    `script_boundaries` (multilingual models): also start a word at every token of an unspaced
    script (CJK, kana, Thai) and wherever the script class changes ("电话|13812345678",
    "257|입니다"). Off reproduces v1-v3 exactly."""
    starts, prev_end = [], None
    for s, e in offsets:
        s, e = _trim_offset(text, s, e)
        if s >= e:
            starts.append(False)
            continue
        new = (prev_end is None or s > prev_end                     # whitespace gap
               or not _is_word_char(text[s])                         # punctuation token
               or not _is_word_char(text[prev_end - 1]))             # after punctuation
        if script_boundaries and not new:
            cur, prev = script_of(text[s]), script_of(text[prev_end - 1])
            new = cur in UNSPACED or cur != prev
        starts.append(new)
        prev_end = e
    return starts


def _partial_mask(tok_text: str, partial: str | None) -> bool:
    """True if a token outside every gold span is unlabelled (-100) for a partially annotated
    source: "all" (only its spans are known), "digits" (no number labels: dates, accounts ...),
    "letters" (no name labels)."""
    if partial == "all":
        return True
    if partial == "digits":
        return any(c.isdigit() for c in tok_text)
    if partial == "letters":
        return any(c.isalpha() for c in tok_text)
    return False


def token_labels(text: str, offsets, spans: list[dict], label2id: dict, special_mask=None,
                 label_all_tokens: bool = False, script_boundaries: bool = False,
                 partial: str | None = None) -> list[int]:
    """BIO label id per token. Special tokens, IGNORE spans and (optionally)
    word-continuation tokens -> -100. A span labelled "O" is a known non-PII look-alike: always O,
    even in a partially annotated source (`partial`, see _partial_mask)."""
    offsets = [(0, 0) if (special_mask is not None and special_mask[i]) else o
               for i, o in enumerate(offsets)]
    starts = word_starts(text, offsets, script_boundaries)
    labels, prev_span, k = [], None, 0
    for i, (s, e) in enumerate(offsets):
        s, e = _trim_offset(text, s, e)
        if s >= e:
            # special token: closes any span. Whitespace-only token (a line-break / tab marker
            # or a lone "▁"): no label, and the span continues - so "12 Main St⏎London" stays
            # one B-I-I-I address instead of two.
            if offsets[i][0] == offsets[i][1]:
                prev_span = None
            labels.append(IGNORE_ID)
            continue
        while k < len(spans) and spans[k]["end"] <= s:
            k += 1
        hit = k if k < len(spans) and s < spans[k]["end"] and e > spans[k]["start"] else None
        if hit is None:
            lab_id = IGNORE_ID if _partial_mask(text[s:e], partial) else label2id["O"]
        elif spans[hit]["label"] == "O":
            lab_id = label2id["O"]
        elif spans[hit]["label"] == IGNORE:
            lab_id = IGNORE_ID
        else:
            prefix = "I" if prev_span == hit else "B"
            lab_id = label2id[f"{prefix}-{spans[hit]['label']}"]
        prev_span = hit
        if not label_all_tokens and not starts[i]:
            lab_id = IGNORE_ID
        labels.append(lab_id)
    return labels


# DeBERTa-v3's SentencePiece normalises "\n" and "\t" to plain spaces, so the model cannot see
# line or table-cell breaks (multi-line addresses, form fields). They are shown to it as these
# in-vocabulary marker tokens instead; offsets are mapped back to the original text.
VISIBLE_BREAKS = {"\n": " ¶ ", "\t": " | ", "\r": " "}


def visible_breaks(text: str) -> tuple[str, list[int]]:
    """(text with breaks replaced by markers, original index of every new character)."""
    out, back = [], []
    for i, c in enumerate(text):
        rep = VISIBLE_BREAKS.get(c, c)
        out.append(rep)
        back.extend([i] * len(rep))
    return "".join(out), back


def window_encode(tokenizer, text: str, max_length: int, stride: int,
                  show_breaks: bool = True) -> list[dict]:
    """Tokenize the whole text once, then cut overlapping windows of `max_length`
    (incl. [CLS]/[SEP]); consecutive windows share `stride` tokens. Offsets refer to `text`.
    `show_breaks`: line/tab markers (v2.1+ models); False reproduces v1/v2 tokenization.

    Done by hand because `return_overflowing_tokens` in transformers 5.x drops most of
    long documents for DeBERTa-v3's tokenizer.
    """
    shown, back = visible_breaks(text) if show_breaks else (text, list(range(len(text))))
    enc = tokenizer(shown, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = [(back[s], back[e - 1] + 1) if e > s else (back[s] if s < len(back) else len(text),) * 2
            for s, e in enc["offset_mapping"]]
    body = max_length - 2
    step = max(1, body - stride)
    windows, start = [], 0
    while True:
        chunk = slice(start, start + body)
        windows.append({
            "input_ids": [tokenizer.cls_token_id] + ids[chunk] + [tokenizer.sep_token_id],
            "offset_mapping": [(0, 0)] + [tuple(o) for o in offs[chunk]] + [(0, 0)],
            "special_tokens_mask": [1] + [0] * len(ids[chunk]) + [1],
        })
        if start + body >= len(ids):
            break
        start += step
    return windows


def make_tokenize_fn(tokenizer, label2id: dict, max_length: int, stride: int,
                     label_all_tokens: bool = False, show_breaks: bool = True,
                     script_boundaries: bool = False):
    """Batched map fn: {text, spans} -> windowed input_ids/attention_mask/labels."""
    id2name = {v: k for k, v in label2id.items()}

    def fn(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        partials = batch.get("partial") or [None] * len(batch["text"])
        for text, spans, partial in zip(batch["text"], batch["spans"], partials):
            spans = sorted(spans, key=lambda s: s["start"])
            for win in window_encode(tokenizer, text, max_length, stride, show_breaks):
                labs = token_labels(text, win["offset_mapping"], spans, label2id,
                                    win["special_tokens_mask"], label_all_tokens, script_boundaries,
                                    partial or None)
                # a window that starts mid-span must open with B-, not I-
                for j, l in enumerate(labs):
                    if l == IGNORE_ID:
                        continue
                    if id2name[l].startswith("I-"):
                        labs[j] = label2id["B-" + id2name[l][2:]]
                    break
                out["input_ids"].append(win["input_ids"])
                out["attention_mask"].append([1] * len(win["input_ids"]))
                out["labels"].append(labs)
        return out

    return fn


def fill_continuations(names: list, starts: list[bool]) -> list:
    """Continuation sub-tokens take their word's label (B-X -> I-X)."""
    out, head = [], None
    for name, st in zip(names, starts):
        if st or head is None:
            head = name
            out.append(name)
        else:
            out.append(None if head is None else ("O" if head == "O" else "I-" + head.split("-", 1)[1]))
    return out


def labels_to_spans(offsets, label_names: list, text: str) -> list[dict]:
    """Decode per-token BIO names (None = skip) back to char spans."""
    spans, cur = [], None
    for (s, e), name in zip(offsets, label_names):
        if name is None or s == e:
            continue
        s, e = _trim_offset(text, s, e)
        if s >= e:
            continue
        if name == "O":
            if cur:
                spans.append(cur)
            cur = None
            continue
        tag, ent = name.split("-", 1)
        if tag == "I" and cur and cur["label"] == ent:
            cur["end"] = e
        else:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "label": ent}
    if cur:
        spans.append(cur)
    return spans
