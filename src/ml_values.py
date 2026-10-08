"""v5 value generators for frame filling (gen_frames.py / build_ml_train.py) and ml_struct.

Accounts come only from configs/account_formats.yaml: each institution's sourced layouts (length,
fixed prefix, separators as printed), random digits. `verified` entries are sampled at full
weight, `partial` at PARTIAL_WEIGHT, `unverified` never. HELD_OUT institutions appear only in the
ml_struct test set, so it can measure generalisation to account shapes never trained on.

account_phrase() renders a cue + one or more accounts in the cue styles seen in real documents
("account 123456-1", "account - …", "account: …", "a/c no. …", "accounts …, … and …",
"(acct …)"), in the document language or code-switched with an English cue. It returns the text
and its spans (ACCOUNT per number, BUSINESS for the institution name when included).
"""
import random
import re
import string
from functools import lru_cache

from common import load_yaml

PARTIAL_WEIGHT = 0.4
# test-only institutions (ml_struct): one or two per market, never the ones the user named
# (Pershing, BNY, IBKR, Leonteq, Standard Chartered stay in training)
HELD_OUT = {"hk_dahsing", "hk_spd", "kr_daegu", "kr_kbank", "my_hongleong", "id_bni", "th_gsb",
            "in_hdfc", "ph_pnb", "vn_vietin", "sg_posb", "jp_yucho", "ch_ubs_depot"}

# markets per document language: (jurisdiction, weight); global hubs added for every language
MARKETS = {
    "en": [("SG", 3), ("HK", 2), ("UK", 1), ("US", 2), ("MY", 1), ("IN", 1), ("AE", 1), ("CH", 1), ("AU", 0.5)],
    "zh-Hans": [("CN", 3), ("SG", 2), ("HK", 2), ("MY", 1)],
    "zh-Hant": [("HK", 3), ("TW", 3)],
    "ja": [("JP", 5)], "ko": [("KR", 5)], "hi": [("IN", 5)], "ar": [("AE", 4), ("CH", 1)],
    "th": [("TH", 5)], "vi": [("VN", 5)], "ms": [("MY", 4), ("SG", 1)], "id": [("ID", 5)],
    "tl": [("PH", 5)],
}
GLOBAL_MARKETS = [("GLOBAL", 1.5), ("US", 0.5), ("CH", 0.5)]

# account cue words (ordinary vocabulary); "pl" = plural / list form where the language has one
CUES = {
    "en": {"one": ["account", "Account", "account no.", "Account No.", "account number", "a/c", "A/C No.", "acct", "Acct #",
                   "custody account", "brokerage account", "trading account", "securities account", "portfolio no."],
           "pl": ["accounts", "Accounts", "account nos.", "a/cs", "accts"]},
    "zh-Hans": {"one": ["账户", "账号", "帐号", "银行账号", "账户号码", "证券账户", "托管账户"], "pl": ["账户", "账号"]},
    "zh-Hant": {"one": ["帳戶", "帳號", "銀行帳號", "帳戶號碼", "證券帳戶", "託管帳戶", "戶口", "戶口號碼", "銀行戶口"], "pl": ["帳戶", "戶口"]},
    "ja": {"one": ["口座番号", "口座", "口座No.", "証券口座", "預金口座番号"], "pl": ["口座番号", "口座"]},
    "ko": {"one": ["계좌번호", "계좌", "계좌 번호", "증권계좌", "입금계좌"], "pl": ["계좌번호", "계좌"]},
    "hi": {"one": ["खाता संख्या", "खाता नं.", "खाता", "डीमैट खाता"], "pl": ["खाते", "खाता संख्याएँ"]},
    "ar": {"one": ["رقم الحساب", "الحساب", "حساب رقم", "رقم الآيبان"], "pl": ["الحسابات", "أرقام الحسابات"]},
    "th": {"one": ["เลขที่บัญชี", "บัญชี", "บัญชีเลขที่", "เลขบัญชี"], "pl": ["บัญชี", "เลขที่บัญชี"]},
    "vi": {"one": ["số tài khoản", "STK", "tài khoản", "TK số"], "pl": ["các tài khoản", "số tài khoản"]},
    "ms": {"one": ["nombor akaun", "akaun", "no. akaun", "akaun bank"], "pl": ["akaun-akaun", "nombor akaun"]},
    "id": {"one": ["nomor rekening", "rekening", "no. rek.", "no rekening", "rekening efek"], "pl": ["rekening-rekening", "nomor rekening"]},
    "tl": {"one": ["account number", "numero ng account", "account", "acct. no."], "pl": ["mga account", "account numbers"]},
}
# cue -> value joiners; "：" variants for CJK
JOIN = [" ", " - ", ": ", " – ", " #", " no. ", ":", " : "]
JOIN_CJK = {"zh-Hans": ["：", ": ", "", "：", "为", "是"], "zh-Hant": ["：", ": ", "", "：", "為", "是"],
            "ja": ["：", ": ", " ", "：", "は"]}
AND = {"en": " and ", "zh-Hans": "和", "zh-Hant": "和", "ja": "と", "ko": " 및 ", "hi": " और ", "ar": " و",
       "th": " และ ", "vi": " và ", "ms": " dan ", "id": " dan ", "tl": " at "}
COMMA = {"zh-Hans": "、", "zh-Hant": "、", "ja": "、", "ar": "، "}
CJK = {"zh-Hans", "zh-Hant", "ja"}


@lru_cache(maxsize=None)
def registry() -> list[dict]:
    return [i for i in load_yaml("account_formats.yaml")["institutions"]
            if i["status"] != "unverified" and i["patterns"]]


def render_pattern(pattern: str, rng) -> str:
    """'#' digit, 'A' capital letter, 'X' capital or digit, [XYZ] one of those, else literal."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "[":
            j = pattern.index("]", i)
            out.append(rng.choice(pattern[i + 1:j]))
            i = j + 1
            continue
        out.append(rng.choice(string.digits) if c == "#" else rng.choice(string.ascii_uppercase) if c == "A"
                   else rng.choice(string.ascii_uppercase + string.digits) if c == "X" else c)
        i += 1
    return "".join(out)


def pattern_regex(pattern: str) -> re.Pattern:
    out = []
    for tok in re.findall(r"\[[^\]]+\]|.", pattern):          # a [XYZ] choice set, or one character
        out.append("[" + re.escape(tok[1:-1]) + "]" if len(tok) > 1 else r"\d" if tok == "#" else "[A-Z]" if tok == "A"
                   else "[A-Z0-9]" if tok == "X" else re.escape(tok))
    return re.compile("^" + "".join(out) + "$")


def institutions(lang: str, split: str = "train") -> tuple[list[dict], list[float]]:
    """Candidate institutions for a document language and their sampling weights."""
    mk = dict(MARKETS[lang])
    for j, w in GLOBAL_MARKETS:
        mk[j] = mk.get(j, 0) + w
    pool, wts = [], []
    for inst in registry():
        if (inst["id"] in HELD_OUT) != (split == "test") or inst["jurisdiction"] not in mk:
            continue
        pool.append(inst)
        wts.append(mk[inst["jurisdiction"]] * (1.0 if inst["status"] == "verified" else PARTIAL_WEIGHT))
    return pool, wts


def account_value(inst: dict, rng) -> str:
    v = render_pattern(rng.choice(inst["patterns"]), rng)
    if inst.get("plain") and rng.random() < 0.35:
        v = re.sub(r"[-\s.]", "", v)
    return v


def account_phrase(lang: str, rng, n: int = 1, split: str = "train", english_cue: bool = False,
                   with_name: float = 0.4, bracket: float = 0.15) -> tuple[str, list[dict]]:
    """Cue + n accounts from one institution. Returns (text, spans with start / end / label)."""
    pool, wts = institutions(lang, split)
    inst = rng.choices(pool, wts)[0]
    vals = [account_value(inst, rng) for _ in range(n)]
    cue_lang = "en" if english_cue else lang
    cue = rng.choice(CUES[cue_lang]["pl" if n > 1 else "one"])
    join = rng.choice(JOIN_CJK[cue_lang] if cue_lang in CJK and rng.random() < 0.7 else JOIN)
    if cue.rstrip().endswith(("#", ":", "：")) or cue.rstrip().endswith("No.") and join == " no. ":
        join = rng.choice([" ", ""]) if cue.endswith(("#", ":", "：")) else " "
    parts, spans = [], []

    def add(s, label=None):
        start = sum(map(len, parts))
        parts.append(s)
        if label:
            spans.append({"start": start, "end": start + len(s), "label": label})

    name = rng.choice(inst["names"]) if rng.random() < with_name else None
    if name and rng.random() < 0.5:                     # "<bank> account 123" / "DBS 账户 123"
        add(name, "BUSINESS")
        add("" if lang in CJK and not english_cue else " ")
        name = None
    add(cue)
    add(join if n == 1 or rng.random() < 0.5 else " ")
    in_br = n > 1 and rng.random() < 0.3
    if in_br:
        add("(")
    comma = COMMA.get(lang, ", ") if not english_cue else ", "
    for k, v in enumerate(vals):
        if k:
            add(AND[cue_lang] if k == n - 1 and rng.random() < 0.6 else comma if rng.random() < 0.8 else " / ")
        add(v, "ACCOUNT")
    if in_br:
        add(")")
    if name:                                            # "account 123 (DBS)" / "account 123 with DBS"
        add(rng.choice([" (", " at ", " with ", " - ", " @ "]) if cue_lang == "en" else " (")
        add(name, "BUSINESS")
        if parts[-2].strip() == "(":
            add(")")
    text = "".join(parts)
    if n == 1 and rng.random() < bracket:               # "(account 123456-1)"
        o, c = rng.choice([("(", ")"), ("（", "）"), ("[", "]")])
        text = o + text + c
        spans = [{**s, "start": s["start"] + len(o), "end": s["end"] + len(o)} for s in spans]
    return text, spans


if __name__ == "__main__":
    r = random.Random(0)
    for lang in MARKETS:
        for n in (1, 1, 3):
            t, sp = account_phrase(lang, r, n=n, english_cue=r.random() < 0.3)
            print(f"{lang:8s} {t!r:70s} {[t[s['start']:s['end']] + '/' + s['label'] for s in sp]}")
