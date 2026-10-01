"""Train-time text augmentation on raw (source-labelled) spans.

Birth dates and ordinary dates are re-rendered with the same random format mix, so the date
format stops being a shortcut for DOB, and birth-date cues ("date of birth") are swapped for
variants ("DOB", "D.O.B.", ...). Edits keep every span aligned with the new text.
"""
import re
from datetime import date

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_MONTH_IDX = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_IDX.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
_MONTH_IDX["sept"] = 9

# raw labels whose value is a calendar date (birth or not)
DATE_LABELS = {"date_of_birth", "date", "DATE"}

# birth cue followed only by filler ("is", "on", ":", "(DD/MM/YYYY)") up to the date - no digits
# or sentence/field break in between, so "DOB: 1/2/65. Paid 3/4/23" does not make 3/4/23 a DOB
BIRTH_CUE = re.compile(r"\b(birth|born|dob|d\.o\.b)\.?[^.;|\t\n\d]{0,25}$", re.I)
# a birth-date cue directly before a span: (cue)(separator)
_CUE_BEFORE = re.compile(r"\b(date of birth|birth ?date|d\.?o\.?b\.?)(\s*(?:is|:|-)?\s*)$", re.I)
CUE_VARIANTS = ["DOB", "D.O.B.", "DoB", "Date of Birth", "Date of birth", "date of birth",
                "Birth Date", "Birthdate", "Date Of Birth"]

_MON = r"([A-Za-z]{3,9})\.?"
_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:T[\d:.]+)?"), "ymd"),
    (re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"), "ymd"),
    (re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"), "mdy_or_dmy"),
    (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})"), "dmy"),
    (re.compile(_MON + r" (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})"), "Mdy"),
    (re.compile(r"(\d{1,2})(?:st|nd|rd|th)? " + _MON + r",? (\d{4})"), "dMy"),
    (re.compile(r"(\d{1,2})-" + _MON + r"-(\d{4})"), "dMy"),
]


def parse_date(value: str) -> date | None:
    """Parse the common date shapes found in the sources; None if unparseable/invalid."""
    v = value.strip()
    for pat, kind in _PATTERNS:
        m = pat.fullmatch(v)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            if kind == "ymd":
                y, mo, d = int(a), int(b), int(c)
            elif kind == "dmy":
                d, mo, y = int(a), int(b), int(c)
            elif kind == "mdy_or_dmy":   # sources are mostly US; day-first only if unambiguous
                x, z, y = int(a), int(b), int(c)
                mo, d = (z, x) if x > 12 else (x, z)
            else:
                mon, d, y = (a, int(b), int(c)) if kind == "Mdy" else (b, int(a), int(c))
                mo = _MONTH_IDX.get(mon.lower())
                if mo is None:
                    return None
            return date(y, mo, d)
        except ValueError:
            return None
    return None


def _ordinal(n: int) -> str:
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


DATE_FORMATS = [
    lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
    lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}",
    lambda d: f"{d.day}/{d.month}/{d.year}",
    lambda d: f"{d.year}-{d.month:02d}-{d.day:02d}",
    lambda d: f"{d.day:02d}-{d.month:02d}-{d.year}",
    lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
    lambda d: f"{d.year}/{d.month:02d}/{d.day:02d}",
    lambda d: f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
    lambda d: f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}",
    lambda d: f"{d.day} {MONTHS[d.month - 1]} {d.year}",
    lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    lambda d: f"{MONTHS[d.month - 1][:3]} {d.day}, {d.year}",
    lambda d: f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year}",
    lambda d: f"{_ordinal(d.day)} {MONTHS[d.month - 1]} {d.year}",
]


# two-digit years and month/year only - used for a quarter of rendered dates
SHORT_DATE_FORMATS = [
    lambda d: f"{d.day:02d}-{d.month:02d}-{d.year % 100:02d}",
    lambda d: f"{d.day:02d}/{d.month:02d}/{d.year % 100:02d}",
    lambda d: f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}",
    lambda d: f"{d.day:02d}.{d.month:02d}.{d.year % 100:02d}",
    lambda d: f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year % 100:02d}",
    lambda d: f"{d.month:02d}/{d.year % 100:02d}",
    lambda d: f"{d.month:02d}-{d.year % 100:02d}",
    lambda d: f"{d.month:02d}/{d.year}",
    lambda d: f"{MONTHS[d.month - 1][:3]}-{d.year % 100:02d}",
    lambda d: f"{MONTHS[d.month - 1]}/{d.year % 100:02d}",
    lambda d: f"{MONTHS[d.month - 1]} {d.year}",
]
P_SHORT = 0.25


def render_date(d: date, rng) -> str:
    fmts = SHORT_DATE_FORMATS if rng.random() < P_SHORT else DATE_FORMATS
    return rng.choice(fmts)(d)


def has_birth_cue(text: str, start: int, window: int = 40) -> bool:
    """A birth cue (birth/born/DOB) on the same line within `window` chars before `start`."""
    ctx = text[max(0, start - window):start].rsplit("\n", 1)[-1]
    return bool(BIRTH_CUE.search(ctx))


def splice(text: str, spans: list[dict], start: int, end: int, new: str) -> str:
    """Replace text[start:end] with `new`, shifting spans in place. A span equal to the
    replaced range is resized; spans after it move by the length difference."""
    delta = len(new) - (end - start)
    for s in spans:
        if s["start"] >= end:
            s["start"] += delta
            s["end"] += delta
        elif s["start"] == start and s["end"] == end:
            s["end"] = start + len(new)
        elif s["end"] > start and s["end"] >= end:   # span containing the edit (e.g. merged)
            s["end"] += delta
    return text[:start] + new + text[end:]


def augment_doc(text: str, spans: list[dict], rng, p_date: float, p_cue: float):
    """Return (text, spans) with date values re-formatted and birth cues varied.
    `spans` are raw-labelled dicts with int start/end; a new list is returned."""
    spans = [dict(s) for s in spans]
    do_date, do_cue = rng.random() < p_date, rng.random() < p_cue
    if not (do_date or do_cue):
        return text, spans
    edits = []   # (start, end, new)
    for s in spans:
        if s["label"] not in DATE_LABELS:
            continue
        if do_date:
            d = parse_date(text[s["start"]:s["end"]])
            if d is not None:
                edits.append((s["start"], s["end"], render_date(d, rng)))
        if do_cue:
            line_start = text.rfind("\n", 0, s["start"]) + 1
            lo = max(line_start, s["start"] - 40)
            m = _CUE_BEFORE.search(text[lo:s["start"]])
            if m:
                edits.append((lo + m.start(1), lo + m.end(1), rng.choice(CUE_VARIANTS)))
    # apply right-to-left; skip edits overlapping an already applied one
    applied_lo = len(text) + 1
    for start, end, new in sorted(edits, key=lambda e: -e[0]):
        if end > applied_lo:
            continue
        text = splice(text, spans, start, end, new)
        applied_lo = start
    return text, spans
