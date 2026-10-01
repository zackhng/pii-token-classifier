"""Short form-style records that mix target entities with look-alikes we do not detect.

Each record puts e.g. a DOB next to ordinary dates, an account number next to amounts and
codes, a person next to a bank/company, in key-value, table or one-sentence layouts. Target
values are sampled from real spans of a split (train for training, test for the stress set);
look-alike values are generated and recorded as negatives (label O) for eval_confusion.

Two disjoint style pools: "train" and "stress" (different keys, layouts and sentences), so the
stress set measures generalisation rather than template memorisation.
"""
import random
from datetime import date, timedelta

from augment import render_date

# field -> (target label or None for a look-alike, neg group, {pool: key variants})
FIELDS = {
    "person": ("PERSON", None, {
        "train": ["Name", "Full Name", "Customer Name", "Account Holder", "Beneficiary Name",
                  "Applicant", "Cardholder Name", "Client"],
        "stress": ["Holder", "Insured Name", "Payee", "Name of Customer"]}),
    "business": ("BUSINESS", None, {
        "train": ["Bank", "Beneficiary Bank", "Employer", "Company", "Merchant", "Issuing Bank"],
        "stress": ["Bank Name", "Organisation", "Remitting Bank", "Paid To"]}),
    "address": ("ADDRESS", None, {
        "train": ["Address", "Residential Address", "Mailing Address"],
        "stress": ["Home Address", "Correspondence Address"]}),
    "dob": ("DOB", None, {
        "train": ["DOB", "D.O.B.", "Date of Birth", "Birth Date", "Birthdate", "DoB"],
        "stress": ["Date Of Birth (as per NRIC)", "Born", "Birth date", "D.O.B"]}),
    "account": ("ACCOUNT", None, {
        "train": ["Account No.", "A/C No", "Account Number", "IBAN", "Card Number", "Customer ID"],
        "stress": ["Acct #", "A/C", "Credit Card No.", "Policy No."]}),
    "phone": ("PHONE", None, {
        "train": ["Tel", "Phone", "Mobile", "Contact No."],
        "stress": ["Hp", "Telephone", "Mobile No."]}),
    "email": ("EMAIL", None, {"train": ["Email", "E-mail"], "stress": ["Email Address"]}),
    "tin": ("TIN", None, {"train": ["SSN", "Tax ID", "TIN"], "stress": ["Tax Reference No."]}),
    # look-alikes (label O)
    "date": (None, "date", {
        "train": ["Date", "Transaction Date", "Value Date", "Issue Date", "Expiry Date",
                  "Statement Date", "Due Date"],
        "stress": ["Posting Date", "Opening Date", "Maturity Date", "Payment Date"]}),
    "amount": (None, "amount", {
        "train": ["Amount", "Balance", "Available Balance", "Salary", "Credit Limit",
                  "Total Due", "Transfer Amount"],
        "stress": ["Outstanding", "Monthly Income", "Closing Balance", "Debit"]}),
    "code": (None, "code", {
        "train": ["SWIFT", "SWIFT/BIC", "CVV", "PIN", "Routing No.", "Expiry", "Valid Thru"],
        "stress": ["BIC Code", "Security Code", "Branch Code", "Exp."]}),
}
TARGET_FIELDS = [f for f, (lab, _, _) in FIELDS.items() if lab]
NEG_FIELDS = [f for f, (lab, _, _) in FIELDS.items() if not lab]

HEADERS = {"train": ["Customer Details", "Payment Instruction", "Account Summary",
                     "KYC Form", "Remittance Advice", ""],
           "stress": ["Loan Application", "Policy Schedule", "Fund Transfer Request", ""]}

SENTENCES = {
    "train": [
        "Please transfer {amount} to {person} ({account}) at {business} by {date}.",
        "{person}, born {dob}, holds account {account} with a balance of {amount}.",
        "On {date}, {business} credited {amount} to account {account}.",
        "Customer {person} (DOB {dob}) requested a new card; CVV {code} was reissued on {date}.",
    ],
    "stress": [
        "Kindly debit {amount} from A/C {account} held by {person} on {date} via {business}.",
        "Records show {person} was born on {dob} and opened a policy {account} on {date}.",
        "A payment of {amount} from {business} reached {person}'s account {account} ({code}).",
    ],
}
SEPS = {"train": [": ", ":\t", " - ", ": "], "stress": [" = ", " :: ", "\t"]}
CURRENCIES = ["SGD", "USD", "INR", "RM", "MYR", "IDR", "HKD", "$", "S$", "Rp", "₹", "EUR"]


def build_pools(rows) -> dict:
    """Target value pools from gold spans of the given rows."""
    want = {lab: f for f, (lab, _, _) in FIELDS.items() if lab and lab != "DOB"}
    pools = {f: [] for f in want.values()}
    for r in rows:
        for s in r["spans"]:
            f = want.get(s["label"])
            v = r["text"][s["start"]:s["end"]]
            if f and "\n" not in v and len(v) <= 60:
                pools[f].append(v)
    return {f: sorted(set(v)) for f, v in pools.items() if v}


def _rand_date(rng, lo_year, hi_year) -> date:
    start = date(lo_year, 1, 1)
    return start + timedelta(days=rng.randrange((date(hi_year, 12, 31) - start).days))


def _amount(rng) -> str:
    v = rng.choice([rng.uniform(5, 999), rng.uniform(1_000, 99_999), rng.uniform(1e5, 5e6)])
    num = rng.choice([f"{v:,.2f}", f"{v:.2f}", f"{int(v):,}", f"{int(v)}"])
    cur = rng.choice(CURRENCIES)
    return rng.choice([f"{cur} {num}", f"{cur}{num}", f"{num} {cur}", num])


def _code(rng) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    kind = rng.random()
    if kind < 0.4:   # SWIFT/BIC
        return "".join(rng.choice(L) for _ in range(6)) + "".join(
            rng.choice(L + "0123456789") for _ in range(rng.choice([2, 5])))
    if kind < 0.6:   # CVV / PIN
        return "".join(rng.choice("0123456789") for _ in range(rng.choice([3, 4, 6])))
    if kind < 0.8:   # card expiry MM/YY - same shape as a short birth date
        return f"{rng.randint(1, 12):02d}{rng.choice(['/', '-'])}{rng.randint(24, 33)}"
    return "".join(rng.choice("0123456789") for _ in range(9))   # routing number


def _value(field, pools, rng):
    if field == "dob":
        return render_date(_rand_date(rng, 1940, 2005), rng)
    if field == "date":
        return render_date(_rand_date(rng, 2015, 2030), rng)
    if field == "amount":
        return _amount(rng)
    if field == "code":
        return _code(rng)
    return rng.choice(pools[field])


class _Doc:
    def __init__(self):
        self.parts, self.len, self.spans, self.negs = [], 0, [], []

    def add(self, s, field=None):
        if field:
            lab, grp, _ = FIELDS[field]
            span = {"start": self.len, "end": self.len + len(s), "label": lab or grp}
            (self.spans if lab else self.negs).append(span)
        self.parts.append(s)
        self.len += len(s)

    def text(self):
        return "".join(self.parts)


def _pick_fields(rng, pools):
    targets = [f for f in TARGET_FIELDS if f == "dob" or f in pools]
    fields = rng.sample(targets, rng.randint(2, min(5, len(targets))))
    fields += rng.sample(NEG_FIELDS, rng.randint(1, 3))
    # always pair at least one target with its look-alike
    pair = rng.choice([("dob", "date"), ("account", "amount"), ("account", "code"),
                       ("person", "business")])
    fields += [f for f in pair if f not in fields and (f in targets or f in NEG_FIELDS)]
    rng.shuffle(fields)
    return fields


def make_record(pools, rng, style="train") -> dict:
    doc, layout = _Doc(), rng.random()
    header = rng.choice(HEADERS[style])
    if header:
        doc.add(header + "\n")
    if layout < 0.55:            # key: value lines
        sep = rng.choice(SEPS[style])
        for f in _pick_fields(rng, pools):
            doc.add(rng.choice(FIELDS[f][2][style]) + sep)
            doc.add(_value(f, pools, rng), f)
            doc.add("\n")
    elif layout < 0.8:           # table: header row + 1-3 rows
        fields = _pick_fields(rng, pools)[:6]
        delim = rng.choice([" | ", "\t", "  "] if style == "train" else [", ", " ; "])
        doc.add(delim.join(rng.choice(FIELDS[f][2][style]) for f in fields) + "\n")
        for _ in range(rng.randint(1, 3)):
            for i, f in enumerate(fields):
                if i:
                    doc.add(delim)
                doc.add(_value(f, pools, rng), f)
            doc.add("\n")
    else:                        # one sentence
        tpl = rng.choice(SENTENCES[style])
        while "{" in tpl:
            pre, rest = tpl.split("{", 1)
            f, tpl = rest.split("}", 1)
            doc.add(pre)
            doc.add(_value(f, pools, rng), f)
        doc.add(tpl)
    return {"text": doc.text().rstrip("\n") + "\n", "spans": doc.spans, "negs": doc.negs}


def make_records(rows, n: int, seed: int, style: str, id_prefix: str) -> list[dict]:
    rng = random.Random(seed)
    pools = build_pools(rows)
    out = []
    for i in range(n):
        rec = make_record(pools, rng, style)
        out.append({"id": f"{id_prefix}-{i}", "source": f"snippet_{style}", **rec})
    return out
