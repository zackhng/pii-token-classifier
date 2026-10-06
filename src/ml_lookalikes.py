"""Look-alike form records for the multilingual training data (the multilingual counterpart of
snippets.make_records, which v2 added for English).

Each record is a few "field: value" lines in one language that mix real PII taken from that
language's training documents (names, accounts, phones, e-mails, tax IDs) with values that must
stay O: SWIFT / bank / branch codes, PIN, CVV, OTP, card expiry, amounts. Model A (pilot) tagged
12.4% of such codes as PII on the English stress set (v3: 6.6%).

Generated because no public dataset in these languages labels such codes as not-PII. Only the
field cues and the code / amount values are generated; the PII values are real training spans.
"""
import random
import string

from ml_templates import d, up

# per language: cues for codes (with the kind of value), amounts, and the positive fields
CODE_KINDS = ["swift", "bank", "branch", "pin", "cvv", "otp", "expiry"]
CUES = {
    "zh-Hans": {"code": ["SWIFT代码", "银行代码", "分行号", "交易密码", "CVV码", "验证码", "有效期"],
                "amount": ["交易金额", "余额", "手续费"], "cur": ["人民币{n}元", "¥{n}"],
                "PERSON": ["户名", "客户姓名"], "ACCOUNT": ["卡号", "账号"], "PHONE": ["手机"], "EMAIL": ["邮箱"], "TIN": ["税号"]},
    "ja": {"code": ["SWIFTコード", "金融機関コード", "支店番号", "暗証番号", "セキュリティコード", "認証コード", "有効期限"],
           "amount": ["振込金額", "残高", "手数料"], "cur": ["{n}円", "¥{n}"],
           "PERSON": ["口座名義", "氏名"], "ACCOUNT": ["口座番号", "カード番号"], "PHONE": ["電話"], "EMAIL": ["メール"], "TIN": ["マイナンバー"]},
    "ko": {"code": ["SWIFT 코드", "은행 코드", "지점 번호", "비밀번호", "CVC", "인증번호", "유효기간"],
           "amount": ["이체 금액", "잔액", "수수료"], "cur": ["{n}원", "₩{n}"],
           "PERSON": ["예금주", "성명"], "ACCOUNT": ["계좌번호", "카드번호"], "PHONE": ["휴대폰"], "EMAIL": ["이메일"], "TIN": ["사업자등록번호"]},
    "hi": {"code": ["स्विफ्ट कोड", "आईएफएससी कोड", "शाखा कोड", "पिन", "सीवीवी", "ओटीपी", "समाप्ति तिथि"],
           "amount": ["राशि", "शेष", "शुल्क"], "cur": ["₹{n}", "रु. {n}"],
           "PERSON": ["खाताधारक", "नाम"], "ACCOUNT": ["खाता संख्या", "कार्ड संख्या"], "PHONE": ["मोबाइल"], "EMAIL": ["ईमेल"], "TIN": ["पैन"]},
    "ar": {"code": ["رمز السويفت", "رمز البنك", "رقم الفرع", "الرقم السري", "رمز CVV", "رمز التحقق", "تاريخ الانتهاء"],
           "amount": ["المبلغ", "الرصيد", "الرسوم"], "cur": ["{n} ريال", "{n} درهم", "AED {n}"],
           "PERSON": ["اسم صاحب الحساب", "الاسم"], "ACCOUNT": ["رقم الحساب", "رقم البطاقة"], "PHONE": ["الجوال"], "EMAIL": ["البريد الإلكتروني"], "TIN": ["الرقم الضريبي"]},
    "th": {"code": ["รหัส SWIFT", "รหัสธนาคาร", "รหัสสาขา", "รหัส PIN", "รหัส CVV", "รหัส OTP", "วันหมดอายุ"],
           "amount": ["จำนวนเงิน", "ยอดคงเหลือ", "ค่าธรรมเนียม"], "cur": ["{n} บาท", "฿{n}"],
           "PERSON": ["ชื่อบัญชี", "ชื่อ"], "ACCOUNT": ["เลขที่บัญชี", "หมายเลขบัตร"], "PHONE": ["โทรศัพท์"], "EMAIL": ["อีเมล"], "TIN": ["เลขผู้เสียภาษี"]},
    "vi": {"code": ["Mã SWIFT", "Mã ngân hàng", "Mã chi nhánh", "Mã PIN", "Mã CVV", "Mã OTP", "Ngày hết hạn"],
           "amount": ["Số tiền", "Số dư", "Phí"], "cur": ["{n} VNĐ", "{n} đồng"],
           "PERSON": ["Chủ tài khoản", "Họ tên"], "ACCOUNT": ["Số tài khoản", "Số thẻ"], "PHONE": ["Điện thoại"], "EMAIL": ["Email"], "TIN": ["Mã số thuế"]},
    "ms": {"code": ["Kod SWIFT", "Kod Bank", "Kod Cawangan", "PIN", "CVV", "Kod TAC", "Tarikh Luput"],
           "amount": ["Jumlah", "Baki", "Caj"], "cur": ["RM{n}", "RM {n}"],
           "PERSON": ["Nama Pemegang Akaun", "Nama"], "ACCOUNT": ["No. Akaun", "No. Kad"], "PHONE": ["Telefon"], "EMAIL": ["E-mel"], "TIN": ["No. Cukai"]},
    "id": {"code": ["Kode SWIFT", "Kode Bank", "Kode Cabang", "PIN", "CVV", "Kode OTP", "Masa Berlaku"],
           "amount": ["Jumlah", "Saldo", "Biaya"], "cur": ["Rp {n}", "IDR {n}"],
           "PERSON": ["Atas Nama", "Nama"], "ACCOUNT": ["No. Rekening", "No. Kartu"], "PHONE": ["No. HP"], "EMAIL": ["Email"], "TIN": ["NPWP"]},
    "tl": {"code": ["SWIFT Code", "Bank Code", "Branch Code", "PIN", "CVV", "OTP", "Petsa ng Pag-expire"],
           "amount": ["Halaga", "Balanse", "Bayad"], "cur": ["₱{n}", "PHP {n}"],
           "PERSON": ["Pangalan ng May-ari", "Pangalan"], "ACCOUNT": ["Numero ng Account", "Numero ng Card"], "PHONE": ["Telepono"], "EMAIL": ["Email"], "TIN": ["TIN"]},
}
COUNTRY = {"zh-Hans": "CN", "zh-Hant": "TW", "ja": "JP", "ko": "KR", "hi": "IN", "ar": "AE", "th": "TH",
           "vi": "VN", "ms": "MY", "id": "ID", "tl": "PH"}
SEPS = {"zh-Hans": ["：", ": "], "zh-Hant": ["：", ": "], "ja": ["：", ": "]}


def cues(lang: str) -> dict:
    if lang != "zh-Hant":
        return CUES[lang]
    import opencc
    cc = opencc.OpenCC("s2t")      # character conversion; s2twp turns 代码 "code" into 程式碼 "program code"
    return {k: [cc.convert(x) for x in v] for k, v in CUES["zh-Hans"].items()}


def code_value(kind: str, lang: str, rng) -> str:
    if kind == "swift":
        return up(rng, 4) + COUNTRY[lang] + rng.choice(string.ascii_uppercase + string.digits) + up(rng, 1) + \
            rng.choice(["", "XXX", d(rng, 3)])
    if kind == "bank":
        return d(rng, rng.choice([3, 4]))
    if kind == "branch":
        return d(rng, rng.choice([3, 4, 5]))
    if kind == "pin":
        return d(rng, rng.choice([4, 6]))
    if kind == "cvv":
        return d(rng, 3)
    if kind == "otp":
        return d(rng, 6)
    return f"{rng.randint(1, 12):02d}/{rng.randint(25, 32)}"          # expiry MM/YY


def records(lang: str, pool: dict, n: int, seed: int) -> list[dict]:
    """n look-alike records; `pool` = {label: [real values from this language's training docs]}."""
    c = cues(lang)
    rng = random.Random(f"lookalike-{seed}-{lang}")
    seps = SEPS.get(lang, [": ", " : ", "\t"])
    labels = [l for l in ("PERSON", "ACCOUNT", "PHONE", "EMAIL", "TIN") if pool.get(l)]
    out = []
    for i in range(n):
        fields = [("pos", l) for l in rng.sample(labels, rng.randint(1, min(3, len(labels))))]
        fields += [("code", k) for k in rng.sample(range(len(CODE_KINDS)), rng.randint(2, 4))]
        if rng.random() < 0.6:
            fields.append(("amount", None))
        rng.shuffle(fields)
        sep = rng.choice(seps)
        text, spans = "", []
        for kind, what in fields:
            if kind == "pos":
                cue, value, label = rng.choice(c[what]), rng.choice(pool[what]), what
            elif kind == "code":
                cue, value, label = c["code"][what], code_value(CODE_KINDS[what], lang, rng), "O"
            else:
                n_amt = rng.randint(10, 9_000_000)
                cue, value, label = rng.choice(c["amount"]), rng.choice(c["cur"]).format(n=f"{n_amt:,}"), "O"
            text += cue + sep
            spans.append({"start": len(text), "end": len(text) + len(value), "label": label})
            text += value + "\n"
        out.append({"id": f"lookalike_{lang}-{i}", "source": f"lookalike_{lang}", "lang": lang, "kind": "generated",
                    "partial": "", "text": text, "spans": spans})
    return out
