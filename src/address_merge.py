"""Merge fragmented ADDRESS spans (v5, plan 4f).

AI4Privacy labels street / building number / postcode as separate fields, so after label mapping an
address like  389 号 临江大道 ，邮编 710000  is three ADDRESS spans with the joiners left O. That
teaches split addresses. merge_address_spans() joins ADDRESS spans whose gap is short and holds only
punctuation, digits or address words (a spaced " / " separates two addresses and is never merged).

A postcode still alone afterwards becomes IGNORE (loss-masked), not O: in these sources it usually
sits next to address text the source left unlabelled ("张家堡街道，邮编 510140"), so O would be wrong
as often as right. The clean lesson "a standalone postcode is O" (user rule) comes from v5 frames.
"""
import re

from common import IGNORE

MAX_GAP = 25
# words that only join address parts (all languages); matched case-insensitively against the gap
JOINERS = [
    "号", "號", "街", "路", "道", "巷", "弄", "栋", "棟", "楼", "樓", "层", "層", "室", "座", "邮编", "郵編", "邮政编码",
    "郵遞區號", "号館の", "番地", "丁目", "番", "〒", "の", "郵便番号", "우편번호", "번길", "번지", "동", "호",
    "số", "đường", "phường", "quận", "mã bưu điện", "jalan", "jln", "lorong", "poskod", "kode pos", "blk", "block",
    "zip", "postal code", "postcode", "p.o. box", "street", "road", "unit", "floor", "flr", "level", "no", "no.",
    "ซอย", "ถนน", "แขวง", "เขต", "รหัสไปรษณีย์", "शहर", "पिन", "पिन कोड", "ص.ب", "الرمز البريدي", "شارع",
    "barangay", "brgy", "kalye", "zip code", "trên", "tại", "ở", "and", "in", "at", "에", "이며", "는", "은",
]
_JOIN_RX = re.compile("|".join(sorted((re.escape(j) for j in JOINERS), key=len, reverse=True)), re.I)
_FILLER = re.compile(r"^[\s\d,，、.。:：;；#\-–/()（）\[\]'\"「」]*$")
POSTCODE = re.compile(r"^(?:\d{3}-\d{4}|\d{4,6}|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2})$")


def _gap_ok(gap: str) -> bool:
    if len(gap) > MAX_GAP or "\n\n" in gap or " / " in gap:
        return False
    return bool(_FILLER.match(_JOIN_RX.sub("", gap)))


def merge_address_spans(text: str, spans: list[dict]) -> tuple[list[dict], dict]:
    """Returns (new spans, stats). Non-ADDRESS spans are kept as they are; a gap that overlaps any
    other span is never merged across."""
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    others = [s for s in spans if s["label"] != "ADDRESS"]
    addr = [dict(s) for s in spans if s["label"] == "ADDRESS"]
    stats = {"merged": 0, "postcode_ignored": 0}
    out = []
    for s in addr:
        if out:
            p = out[-1]
            gap = text[p["end"]:s["start"]]
            blocked = any(o["start"] < s["start"] and o["end"] > p["end"] for o in others)
            if s["start"] >= p["end"] and not blocked and _gap_ok(gap):
                p["end"] = max(p["end"], s["end"])
                p["_n"] = p.get("_n", 1) + 1
                stats["merged"] += 1
                continue
        out.append(s)
    for s in out:
        if s.pop("_n", 1) == 1 and POSTCODE.match(text[s["start"]:s["end"]].strip()):
            s["label"] = IGNORE                   # lone postcode: context unreliable in the source
            stats["postcode_ignored"] += 1
    return sorted(others + out, key=lambda s: (s["start"], s["end"])), stats
