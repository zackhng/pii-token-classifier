"""v5 sentence frames: hand-written templates with typed slots, filled with exact spans.

Frames are written by Claude (user decision 2026-10-08) in frames/{train,dev,test}/<name>.txt (versioned; data/ is git-ignored):

    #lang: en                 matrix language of the file (row "lang")
    #cs: zh-Hans              optional: code-switched file, English + this language
    ## register: email        sets the register for the frames below
    ## slice: acct_list        test frames only: the ml_struct slice the frames below measure
    Hi {PERSON}, please credit {AMOUNT} to {ACCT} by Friday.\\nThanks, {PERSON2}

One frame per line, "\\n" = line break. Slots: {TYPE} or {TYPE@lang} (value from that language's
pools, e.g. {PERSON@zh-Hans} inside an English sentence). Entity slots (labelled):
    PERSON PERSON2 BUSINESS TICKER BANK ADDRESS ADDRESS_ML DOB PHONE EMAIL TIN
    ACCOUNT (bare number; the frame writes its own cue)  ACCOUNTS (2-3 numbers, listed)
    ACCT (cue + number from ml_values.account_phrase, own cue style)
    ALIAS@zh-Hans|zh-Hant|ja|ko (one person: native name + its romanisation, two PERSON spans)
Look-alike slots (explicit O spans): AMOUNT DATE POSTCODE REF CODE ISIN
A frame with a look-alike in the same position as an entity in another frame is a matched
contrast (plan 4c); the "context decides" share is tracked with difficulty.py (plan 4e).

Train frames take values from real spans in ml_train (public sources, fragments merged), the
account registry and tickers.yaml (train split); test frames (ml_struct) take values from the
test-only ml_templates pools and held-out institutions / tickers, so no value is shared.
"""
import json
import random
import re
import string
from collections import defaultdict
from functools import lru_cache

from address_merge import merge_address_spans
from common import ROOT, load_yaml
from ml_values import CJK, account_phrase, account_value, institutions

SLOT = re.compile(r"\{([A-Z_0-9]+)(?:@([A-Za-z-]+))?\}")
ENTITY = {"PERSON": "PERSON", "PERSON2": "PERSON", "BUSINESS": "BUSINESS", "TICKER": "BUSINESS", "BANK": "BUSINESS",
          "ADDRESS": "ADDRESS", "ADDRESS_ML": "ADDRESS", "DOB": "DOB", "PHONE": "PHONE", "EMAIL": "EMAIL",
          "TIN": "TIN", "ACCOUNT": "ACCOUNT", "ACCOUNTS": "ACCOUNT", "ACCT": "ACCOUNT", "ALIAS": "PERSON"}
LOOKALIKE = {"AMOUNT", "DATE", "POSTCODE", "REF", "CODE", "ISIN"}
LANGS = ["en", "zh-Hans", "zh-Hant", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"]
FRAMES_DIR = ROOT / "frames"
CACHE_DIR = ROOT / "data/frames"

# market of a value written in a language (tickers, postcodes); en mixes the hub markets
MARKET = {"en": ["SG", "US", "HK", "UK", "IN", "MY"], "zh-Hans": ["CN", "SG", "HK"], "zh-Hant": ["HK", "TW"],
          "ja": ["JP"], "ko": ["KR"], "hi": ["IN"], "ar": ["AE"], "th": ["TH"], "vi": ["VN"], "ms": ["MY"],
          "id": ["ID"], "tl": ["PH"]}
# postal code shapes per market (national formats); AE / HK have no postcodes
POSTCODE = {"SG": ["######"], "MY": ["#####"], "ID": ["#####"], "TH": ["#####"], "KR": ["#####"], "US": ["#####", "#####-####"],
            "CN": ["######"], "IN": ["######", "### ###"], "VN": ["######"], "JP": ["###-####", "〒###-####"],
            "TW": ["###", "#####"], "PH": ["####"], "UK": ["A# #AA", "AA## #AA", "AA# #AA"]}
CURRENCY = {"en": ["SGD {n}", "USD {d}", "US${d}", "HKD {n}", "£{n}", "S${n}", "{n} SGD", "INR {n}", "RM {n}"],
            "zh-Hans": ["人民币{n}元", "¥{n}", "{n}元", "{w}万元", "USD {d}"], "zh-Hant": ["港幣{n}元", "HK${n}", "新台幣{n}元", "NT${n}"],
            "ja": ["{n}円", "¥{n}", "金{n}円"], "ko": ["{n}원", "₩{n}", "{w}만 원"], "hi": ["₹{n}", "रु. {n}", "INR {n}"],
            "ar": ["{n} درهم", "AED {n}", "{n} ريال"], "th": ["{n} บาท", "฿{n}", "THB {n}"], "vi": ["{n} VNĐ", "{n} đồng", "{n}đ"],
            "ms": ["RM{n}", "RM {d}", "MYR {n}"], "id": ["Rp {n}", "Rp{n},-", "IDR {n}"], "tl": ["₱{n}", "PHP {d}", "P{n}"]}


def _render(shape: str, rng) -> str:
    return "".join(rng.choice(string.digits) if c == "#" else rng.choice(string.ascii_uppercase) if c == "A" else c
                   for c in shape)


# ---------------------------------------------------------------- look-alikes (O)
def amount(lang, rng):
    n = rng.choice([rng.randint(100, 9_999), rng.randint(10_000, 999_999), rng.randint(1_000_000, 99_999_999)])
    sep = "." if lang in ("id", "vi") else ","
    fmt = rng.choice(CURRENCY[lang])
    return fmt.format(n=f"{n:,}".replace(",", sep), d=f"{n:,}.{rng.randint(0, 99):02d}", w=max(1, n // 10_000))


def postcode(lang, rng):
    m = rng.choice([x for x in MARKET[lang] if x in POSTCODE] or ["SG"])
    return _render(rng.choice(POSTCODE[m]), rng)


def ref(lang, rng):
    """Reference / transaction numbers with the same lengths as accounts (plan 4c)."""
    inst = rng.choice(institutions(lang)[0])
    core = re.sub(r"\D", "", account_value(inst, rng)) or "".join(rng.choices(string.digits, k=10))
    return rng.choice(["TT{c}", "REF{c}", "REF-{c}", "{c}", "TXN{c}", "FT{c}", "INV-{c}", "CN{c}"]).format(c=core)


def isin(lang, rng):
    cc = rng.choice(["US", "SG", "HK", "JP", "KR", "IN", "CH", "LU", "GB", "XS"])
    return cc + "".join(rng.choices(string.ascii_uppercase + string.digits, k=9)) + rng.choice(string.digits)


def code(lang, rng):
    """SWIFT / BIC (frames put {CODE} after a SWIFT cue): 4 bank letters + country + location (+ branch)."""
    from ml_lookalikes import COUNTRY, code_value
    if lang == "en":
        cc = rng.choice(["SG", "HK", "GB", "US", "CH", "MY", "IN"])
        return _render("AAAA", rng) + cc + _render(rng.choice(["AA", "A#", "AAAAA", "AA###"]), rng)
    return code_value("swift", lang if lang in COUNTRY else "zh-Hans", rng)


# ---------------------------------------------------------------- value pools
# PERSON values from real-name sources first (NER corpora / curated generators); AI4Privacy's
# synthetic names are kept only where they read as real local names (ms, id, tl, vi)
PERSON_SOURCES = {"en": ["english"], "zh-Hans": ["cluener"], "zh-Hant": ["cluener"], "ja": ["stockmark_ja"],
                  "ko": ["kiii", "klue_ner"], "hi": ["naamapadam_hi", "gen_hi"], "ar": ["gen_ar", "anercorp", "wikiann_ar"],
                  "th": ["gen_th", "thainer"], "vi": ["ai4p_vi", "wikiann_vi"], "ms": ["ai4p_ms", "wikiann_ms"],
                  "id": ["ai4p_id", "nergrit"], "tl": ["tlunified", "ai4p_tl"]}

# ---------------------------------------------------------------- dates (DOB and look-alike dates)
MONTHS = {
    "en": "January February March April May June July August September October November December".split(),
    "hi": "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split(),
    "ar": "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split(),
    "th": "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม".split(),
    "th_abbr": "ม.ค. ก.พ. มี.ค. เม.ย. พ.ค. มิ.ย. ก.ค. ส.ค. ก.ย. ต.ค. พ.ย. ธ.ค.".split(),
    "ms": "Januari Februari Mac April Mei Jun Julai Ogos September Oktober November Disember".split(),
    "id": "Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember".split(),
    "tl": "Enero Pebrero Marso Abril Mayo Hunyo Hulyo Agosto Setyembre Oktubre Nobyembre Disyembre".split(),
}
AR_DIGITS, HI_DIGITS = "٠١٢٣٤٥٦٧٨٩", "०१२३४५६७८९"


def _ja_era(y):
    return f"令和{y - 2018}" if y >= 2019 else f"平成{y - 1988}" if y >= 1989 else f"昭和{y - 1925}"


def date_value(lang: str, rng, lo: int, hi: int) -> str:
    """A date in a format used in that language's documents; DOB (lo, hi = birth years) and
    look-alike dates (recent years) share the formats, so only context separates them."""
    y, m, d = rng.randint(lo, hi), rng.randint(1, 12), rng.randint(1, 28)
    en = MONTHS["en"][m - 1]
    fmts = {
        "en": [f"{d} {en} {y}", f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{en} {d}, {y}", f"{d:02d}-{en[:3]}-{y}", f"{d}.{m}.{y}"],
        "zh-Hans": [f"{y}年{m}月{d}日", f"{y}-{m:02d}-{d:02d}", f"{y}/{m:02d}/{d:02d}", f"{y}.{m:02d}.{d:02d}"],
        "zh-Hant": [f"{y}年{m}月{d}日", f"{y}/{m:02d}/{d:02d}", f"民國{y - 1911}年{m}月{d}日", f"{y}-{m:02d}-{d:02d}"],
        "ja": [f"{y}年{m}月{d}日", f"{_ja_era(y)}年{m}月{d}日", f"{y}/{m:02d}/{d:02d}", f"{y}.{m}.{d}"],
        "ko": [f"{y}년 {m}월 {d}일", f"{y}.{m:02d}.{d:02d}", f"{y}. {m}. {d}.", f"{y}-{m:02d}-{d:02d}"],
        "hi": [f"{d:02d}/{m:02d}/{y}", f"{d:02d}-{m:02d}-{y}", f"{d} {MONTHS['hi'][m - 1]} {y}",
               f"{d:02d}/{m:02d}/{y}".translate(str.maketrans("0123456789", HI_DIGITS))],
        "ar": [f"{d:02d}/{m:02d}/{y}", f"{d} {MONTHS['ar'][m - 1]} {y}", f"{y}-{m:02d}-{d:02d}",
               f"{d:02d}/{m:02d}/{y}".translate(str.maketrans("0123456789", AR_DIGITS))],
        "th": [f"{d} {MONTHS['th'][m - 1]} {y + 543}", f"{d:02d}/{m:02d}/{y + 543}", f"{d} {MONTHS['th_abbr'][m - 1]} {y + 543}",
               f"{d:02d}/{m:02d}/{y}"],
        "vi": [f"{d:02d}/{m:02d}/{y}", f"ngày {d} tháng {m} năm {y}", f"{d:02d}-{m:02d}-{y}"],
        "ms": [f"{d:02d}/{m:02d}/{y}", f"{d} {MONTHS['ms'][m - 1]} {y}", f"{d:02d}-{m:02d}-{y}"],
        "id": [f"{d:02d}/{m:02d}/{y}", f"{d} {MONTHS['id'][m - 1]} {y}", f"{d:02d}-{m:02d}-{y}"],
        "tl": [f"{d:02d}/{m:02d}/{y}", f"{MONTHS['tl'][m - 1]} {d}, {y}", f"{en} {d}, {y}"],
    }
    return rng.choice(fmts[lang])


# PERSON names must be written in the language's own script (AI4Privacy mixes scripts and spaces
# CJK names token by token: "进芬 渭贞 都"); hi / tl / ms / id / vi names are Latin in the sources
NAME_SCRIPT = {"zh-Hans": r"[\u4e00-\u9fff·]{2,5}", "zh-Hant": r"[\u4e00-\u9fff·]{2,5}",
               "ja": r"[\u3040-\u30ff\u4e00-\u9fff・]{2,8}", "ko": r"[가-힣]{2,5}", "th": r"[\u0e00-\u0e7f ]{3,40}",
               "ar": r"[\u0600-\u06ff ]{3,40}", "hi": r"[\u0900-\u097f ]{3,40}"}
LATIN_NAME = r"[A-Za-zÀ-ÿĀ-žƠ-ưẠ-ỹ'.\- ]{3,40}"
# one Vietnamese syllable: optional initial, vowel cluster, optional final (rejects "Bertotti", "Stêphanô")
_VI_V = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
VI_SYLLABLE = re.compile(rf"(?i)^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnpqrstvx])?[{_VI_V}]{{1,3}}(ch|ng|nh|[cmnpt])?$")


def clean_value(v: str, label: str, lang: str) -> str | None:
    """Normalise a span value for reuse in frames, or None when it looks like source noise."""
    if re.search(r"[\[\]{}<>*#_]|\bEntity\b|\bXXX", v):              # placeholders / masked values
        return None
    if label == "PERSON":
        if lang in ("zh-Hans", "zh-Hant", "ja", "ko"):
            v = v.replace(" ", "")
        if lang == "vi" and not (2 <= len(v.split()) <= 4 and all(VI_SYLLABLE.match(t) for t in v.split())):
            return None
        return v if re.fullmatch(NAME_SCRIPT.get(lang, LATIN_NAME), v) else None
    if label == "DOB":
        return re.sub(r"T\d\d:\d\d(:\d\d)?(Z|[+-]\d\d:?\d\d)?$", "", v)   # ISO timestamp -> date
    if label == "BUSINESS" and re.fullmatch(r"[A-Z]{2,5}", v):          # bare letter codes ("OYHE")
        return None
    return v


@lru_cache(maxsize=None)
def train_pools() -> dict:
    """{(lang, label): [values]} from real ml_train spans (fragments merged), filtered to clean values."""
    cache = CACHE_DIR / "pools_train.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {tuple(k.split("|")): v for k, v in raw.items()}
    pools = defaultdict(set)
    ok = {"PERSON": lambda v: 2 <= len(v) <= 40 and not re.search(r"[\d@\n|*]", v),
          "BUSINESS": lambda v: 2 <= len(v) <= 60 and "\n" not in v and "|" not in v,
          "ADDRESS": lambda v: 8 <= len(v) <= 140 and "\n\n" not in v and "|" not in v,
          "DOB": lambda v: 4 <= len(v) <= 30 and re.search(r"\d", v) and "\n" not in v,
          "PHONE": lambda v: 6 <= len(v) <= 25 and "\n" not in v, "EMAIL": lambda v: "@" in v and len(v) <= 50 and " " not in v,
          "TIN": lambda v: 5 <= len(v) <= 25 and "\n" not in v}
    for line in open(ROOT / "data/processed/ml_train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        spans, _ = merge_address_spans(r["text"], r["spans"])
        for s in spans:
            v = clean_value(r["text"][s["start"]:s["end"]].strip(), s["label"], r["lang"])
            if s["label"] == "PERSON" and r["source"] not in PERSON_SOURCES[r["lang"]]:
                continue
            if v and s["label"] in ok and ok[s["label"]](v):
                pools[(r["lang"], s["label"])].add(v)
    for line in open(ROOT / "data/addresses/addresses.jsonl", encoding="utf-8"):
        a = json.loads(line)
        if a.get("split", "train") == "train":
            pools[("en", "ADDRESS_LINES")].add("\n".join(a["lines"]))
    out = {k: sorted(v) for k, v in pools.items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"|".join(k): v for k, v in out.items()}, ensure_ascii=False), encoding="utf-8")
    return out


@lru_cache(maxsize=None)
def tickers(split: str) -> list[dict]:
    cfg = load_yaml("tickers.yaml")
    return [{**c, **{"exchanges": [c["exchange"]] if c.get("exchange") else cfg["markets"][c["market"]]["exchange"],
                     "suffix": c.get("suffix", cfg["markets"][c["market"]]["suffix"])}}
            for c in cfg["companies"] if (c.get("split") == "test") == (split == "test")]


def ticker(lang, rng, split):
    pool = [c for c in tickers(split) if c["market"] in MARKET[lang]] or tickers(split)
    c = rng.choice(pool)
    form = rng.random()
    if form < 0.3:
        return f"{rng.choice(c['exchanges'])}: {c['code']}"
    if form < 0.5 and c["suffix"]:
        return c["code"] + c["suffix"]
    if form < 0.7:
        return f"{rng.choice(c['names'])} ({rng.choice(c['exchanges'])}: {c['code']})"
    if form < 0.85 and not c["code"].isdigit():          # a bare numeric code (1155, 7203) is not a name
        return c["code"]
    return name_for(c["names"], lang, rng)


class Values:
    """Slot values for one split. Test values come from the test-only ml_templates generators."""

    def __init__(self, split: str):
        self.split = split
        if split == "test":
            import ml_templates
            self.gen = {l: L["gen"] for l, L in ml_templates.languages().items()}

    def start_row(self):
        self.row = {"insts": [], "acct": 0, "bank": 0}

    def _inst(self, kind: str, lang: str, rng) -> dict:
        """The k-th ACCOUNT and the k-th BANK of a row share one institution, so the number has
        that institution's format ("account 123-456789-001 at HSBC")."""
        k = self.row[kind]
        self.row[kind] += 1
        if k < len(self.row["insts"]):
            return self.row["insts"][k]
        pool, w = institutions(lang, getattr(self, "acct_split", self.split))
        inst = rng.choices(pool, w)[0]
        self.row["insts"].append(inst)
        return inst

    def _pool(self, lang, label, rng):
        if self.split == "test":
            v = self.gen[lang][label](rng)
            return (", " if lang not in CJK else "").join(x.rstrip(" ,") for x in v) if isinstance(v, list) else v
        p = train_pools().get((lang, label)) or train_pools()[("en", label)]
        return rng.choice(p)

    def value(self, slot: str, lang: str, rng) -> tuple[str, list[dict]]:
        """(text, spans relative to the text). Look-alikes come back as O spans."""
        if slot == "ALIAS":
            return alias(lang, rng, self)
        if slot == "ACCT":
            return account_phrase(lang, rng, split=self.split, english_cue=False)
        if slot in ("ACCOUNT", "ACCOUNTS"):
            inst = self._inst("acct", lang, rng)
            vals = [account_value(inst, rng) for _ in range(1 if slot == "ACCOUNT" else rng.randint(2, 3))]
            joins = ["、"] if lang in CJK else [", ", " / ", "; "]
            return _join(vals, rng.choice(joins), "ACCOUNT")
        if slot == "BANK":
            v = name_for(self._inst("bank", lang, rng)["names"], lang, rng)
        elif slot == "TICKER":
            v = ticker(lang, rng, self.split)
        elif slot == "ADDRESS_ML":
            if self.split == "test":
                v = self.gen[lang]["ADDRESS"](rng)
                v = "\n".join(v) if isinstance(v, list) else v
            else:
                v = rng.choice(train_pools()[("en", "ADDRESS_LINES")]) if lang == "en" else self._pool(lang, "ADDRESS", rng)
        elif slot in LOOKALIKE:
            v = {"AMOUNT": amount, "POSTCODE": postcode, "REF": ref, "ISIN": isin, "CODE": code}.get(slot, None)
            if v:
                v = v(lang, rng)
            else:
                v = date_value(lang, rng, 2016, 2026) if self.split == "train" else self.gen[lang]["DATE"](rng)
            return v, [{"start": 0, "end": len(v), "label": "O", "neg": slot.lower()}]
        else:
            if slot == "DOB" and self.split == "train":
                v = date_value(lang, rng, 1940, 2005)
            else:
                v = self._pool(lang, ENTITY[slot], rng)
            if slot == "ADDRESS":
                v = re.sub(r"\s*\n\s*", "" if lang in CJK else ", ", v)
        return v, [{"start": 0, "end": len(v), "label": ENTITY[slot]}]


LATIN = re.compile(r"^[\x00-\u024f\u1e00-\u1eff\s.,&()'/-]+$")      # Latin incl. Vietnamese diacritics


def name_for(names: list[str], lang: str, rng) -> str:
    """An institution / company name in the script of the document language (fallback: any)."""
    want = [n for n in names if bool(LATIN.match(n)) == (lang in ("en", "ms", "id", "tl", "vi"))]
    return rng.choice(want or names)


# ---------------------------------------------------------------- bilingual aliases (one person, two scripts)
# surname spellings in use: Hanyu Pinyin, and the Hokkien / Teochew / Cantonese forms of SG / MY / HK
ZH_SURNAME = {"陈": ["Chen", "Tan", "Chan"], "陳": ["Chen", "Tan", "Chan"], "林": ["Lin", "Lim", "Lam"], "黄": ["Huang", "Ng", "Wong"],
              "黃": ["Huang", "Ng", "Wong"], "吴": ["Wu", "Goh", "Ng"], "吳": ["Wu", "Goh", "Ng"], "李": ["Li", "Lee"],
              "张": ["Zhang", "Teo", "Cheung"], "張": ["Zhang", "Teo", "Cheung"], "王": ["Wang", "Ong", "Wong"],
              "刘": ["Liu", "Lau"], "劉": ["Liu", "Lau"], "郑": ["Zheng", "Tay", "Cheng"], "鄭": ["Zheng", "Tay", "Cheng"],
              "杨": ["Yang", "Yeo", "Yeung"], "楊": ["Yang", "Yeo", "Yeung"], "何": ["He", "Ho"], "周": ["Zhou", "Chew", "Chow"],
              "梁": ["Liang", "Neo", "Leung"], "许": ["Xu", "Koh", "Hui"], "許": ["Xu", "Koh", "Hui"], "谢": ["Xie", "Chia", "Tse"],
              "謝": ["Xie", "Chia", "Tse"], "郭": ["Guo", "Quek", "Kwok"], "罗": ["Luo", "Loh", "Law"], "羅": ["Luo", "Loh", "Law"]}
ZH_COMPOUND = ("欧阳", "歐陽", "司马", "司馬", "诸葛", "諸葛", "上官", "慕容")
# common Chinese surnames (simplified + traditional); a name must start with one to be romanised -
# the pools also hold transliterated foreign names (法夫尔, 斯圖爾特) whose pinyin would be nonsense
ZH_COMMON = set("王李张張刘劉陈陳杨楊黄黃赵趙吴吳周徐孙孫马馬朱胡郭何高林罗羅郑鄭梁谢謝宋唐许許韩韓冯馮邓鄧曹彭曾肖萧蕭田董袁潘于蒋蔣蔡余杜叶葉程苏蘇魏吕呂丁任沈姚卢盧姜崔钟鍾谭譚陆陸汪范金石廖贾賈夏韦韋付方白邹鄒孟熊秦邱江尹薛闫閆段雷侯龙龍史陶黎贺賀顾顧毛郝龚龔邵万萬钱錢严嚴覃武戴莫孔向汤湯")
KO_SURNAME = {"김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho", "윤": "Yoon",
              "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin", "권": "Kwon", "황": "Hwang"}


def romanise(name: str, lang: str, rng) -> str | None:
    """Romanised form of a native-script name, written the way it appears on passports / KYC forms."""
    if lang in ("zh-Hans", "zh-Hant"):
        from pypinyin import lazy_pinyin
        sur = next((c for c in ZH_COMPOUND if name.startswith(c)), name[0])
        given = name[len(sur):]
        if not 1 <= len(given) <= 2 or (len(sur) == 1 and sur not in ZH_COMMON):
            return None
        s_rom = rng.choice(ZH_SURNAME.get(sur, ["".join(lazy_pinyin(sur)).capitalize()]))
        g = [p.capitalize() for p in lazy_pinyin(given)]
        g_rom = rng.choice([" ".join(g), "".join(g).capitalize(), "-".join(g)])
        return rng.choice([f"{s_rom} {g_rom}", f"{g_rom} {s_rom}", f"{s_rom.upper()} {g_rom}"])
    if lang == "ja":
        import pykakasi
        parts = [p["hepburn"].capitalize() for p in pykakasi.kakasi().convert(name) if p["hepburn"].strip()]
        if len(parts) != 2:                     # only clean family + given segmentations
            return None
        fam, given = parts[0], "".join(parts[1:])
        return rng.choice([f"{given} {fam}", f"{fam.upper()} {given}", f"{fam} {given}"])
    if lang == "ko":
        from korean_romanizer.romanizer import Romanizer
        if len(name) < 2 or name[0] not in KO_SURNAME:
            return None
        syl = [Romanizer(c).romanize().capitalize() for c in name[1:]]
        given = rng.choice(["-".join(syl).capitalize(), "".join(syl).capitalize(), " ".join(syl)])
        sur = KO_SURNAME[name[0]]
        return rng.choice([f"{sur} {given}", f"{given} {sur}", f"{sur.upper()} {given}"])
    return None


ALIAS_BRACKETS = [("{a} ({b})", 0.4), ("{a}（{b}）", 0.1), ("{a} / {b}", 0.15), ("{a} - {b}", 0.05),
                  ("{a}「{b}」", 0.05), ("{a}, also known as {b}", 0.05), ("{a} 【{b}】", 0.05), ("{a} @ {b}", 0.05),
                  ("{a}（英文名：{b}）", 0.05), ("{a} [{b}]", 0.05)]


def alias(lang: str, rng, values) -> tuple[str, list[dict]]:
    """Native name + its romanisation as two PERSON spans, in either order and bracket style."""
    for _ in range(20):
        native = values._pool(lang, "PERSON", rng)
        rom = romanise(native, lang, rng)
        if rom:
            break
    else:
        return values.value("PERSON", lang, rng)
    a, b = (native, rom) if rng.random() < 0.5 else (rom, native)
    fmts, w = zip(*ALIAS_BRACKETS)
    fmt = rng.choices(fmts, w)[0]
    if "英文名" in fmt and b != rom:            # "English name: ..." must hold the romanised form
        a, b = native, rom
    text = fmt.format(a=a, b=b)
    ia = fmt.index("{a}")
    ib = text.index(b, ia + len(a))
    return text, [{"start": ia, "end": ia + len(a), "label": "PERSON"}, {"start": ib, "end": ib + len(b), "label": "PERSON"}]


def _join(vals, sep, label):
    text, spans = "", []
    for k, v in enumerate(vals):
        if k:
            text += sep
        spans.append({"start": len(text), "end": len(text) + len(v), "label": label})
        text += v
    return text, spans


# ---------------------------------------------------------------- frame files
def load_frames(split: str) -> list[dict]:
    out = []
    for path in sorted((FRAMES_DIR / split).glob("*.txt")):
        lang, cs, register, slice_ = None, None, "general", None
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if line.startswith("#lang:"):
                lang = line.split(":", 1)[1].strip()
            elif line.startswith("#cs:"):
                cs = line.split(":", 1)[1].strip()
                cs = None if cs in ("", "none") else cs           # a file may switch blocks
            elif line.startswith("## register:"):
                register = line.split(":", 1)[1].strip()
            elif line.startswith("## slice:"):                 # ml_struct breakdown (test frames)
                slice_ = line.split(":", 1)[1].strip()
            elif not line.startswith("#"):
                assert lang in LANGS, f"{path.name}: missing #lang"
                out.append({"id": f"{path.stem}:{n}", "split": split, "lang": lang, "cs": cs, "register": register,
                            "slice": slice_,
                            "template": line.replace("\\n", "\n")})
    return out


def skeleton(template: str) -> str:
    """Structure key for dedup / train-test disjointness: slots by type, digits -> #, spaces collapsed."""
    s = SLOT.sub(lambda m: "{" + ENTITY.get(m.group(1), m.group(1)) + "}", template)
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", s)).strip().lower()


def fill(frame: dict, rng, values: Values) -> dict:
    text, spans, pos = "", [], 0
    values.start_row()
    for m in SLOT.finditer(frame["template"]):
        text += frame["template"][pos:m.start()]
        slot, lang = m.group(1), m.group(2) or frame["lang"]
        assert slot in ENTITY or slot in LOOKALIKE, f"{frame['id']}: unknown slot {slot}"
        v, sp = values.value(slot, lang, rng)
        spans += [{**s, "start": s["start"] + len(text), "end": s["end"] + len(text)} for s in sp]
        text += v
        pos = m.end()
    text += frame["template"][pos:]
    return {"id": f"frame_{frame['split']}_{frame['id']}",
            "source": "frames" if frame["split"] == "train" else f"ml_struct_{frame['split']}",
            "lang": frame["lang"], "cs": frame["cs"], "register": frame["register"], "slice": frame.get("slice"),
            "kind": "generated",
            "partial": "", "text": text, "spans": spans}


def render(split: str, per_frame: int, seed: int = 42) -> list[dict]:
    """split: train (frames/train, training pools) | dev | test (frames/dev, frames/test; both use the
    test-only pools - ml_struct_dev is used for every choice, ml_struct_test is reported once)."""
    values = Values("train" if split == "train" else "test")
    rng = random.Random(f"frames-{split}-{seed}")
    rows = []
    for f in load_frames(split):
        for k in range(per_frame):
            r = fill(f, rng, values)
            r["id"] += f"-{k}"
            rows.append(r)
    return rows


def export_struct(split: str, per_frame: int, seed: int = 42) -> list[dict]:
    """ml_struct_{dev,test} eval split (build_ml_eval format): entity spans with script class in
    `spans`, look-alikes in `negs` (label = look-alike type), plus `slice` / `cs` for the breakdown."""
    from build_ml_eval import tag_scripts
    out = []
    for r in render(split, per_frame, seed):
        ents = [{k: v for k, v in s.items() if k != "neg"} for s in r["spans"] if s["label"] != "O"]
        negs = [{"start": s["start"], "end": s["end"], "label": s["neg"]} for s in r["spans"] if s["label"] == "O"]
        out.append(tag_scripts({"id": r["id"].replace(f"frame_{split}_", f"struct_{split}_"),
                                "source": f"ml_struct_{split}", "lang": r["lang"],
                                "cs": r["cs"], "slice": r["slice"], "text": r["text"], "spans": ents, "negs": negs}))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--export_struct", choices=["dev", "test"], help="write data/processed/ml_struct_<split>.jsonl")
    a = ap.parse_args()
    if a.export_struct:
        from build_dataset import write_jsonl
        rows = export_struct(a.export_struct, a.n)
        write_jsonl(ROOT / f"data/processed/ml_struct_{a.export_struct}.jsonl", rows)
        from collections import Counter
        print(f"ml_struct_{a.export_struct}: {len(rows)} docs", dict(Counter(r["slice"] for r in rows)))
        raise SystemExit
    rows = render(a.split, a.n)
    print(f"{len(load_frames(a.split))} frames -> {len(rows)} rows")
    for r in random.Random(0).sample(rows, min(a.show, len(rows))):
        print("-" * 80, f"\n[{r['lang']}{' +' + r['cs'] if r['cs'] else ''} / {r['register']}]")
        print(r["text"])
        print("   ", [(r["text"][s["start"]:s["end"]], s["label"]) for s in r["spans"]])
