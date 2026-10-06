"""Native-language wealth-management templates and value pools for the multilingual zero-shot
test set (build_ml_eval.py). Evaluation only - never used for training.

Every language provides
  cues     field labels per entity / look-alike group, used in forms and statement headers
  seps     field separators
  titles   form / statement titles, sign-off closings, job titles (plain text, never PII)
  prose    RM e-mails, transfer instructions, suitability notes with {PLACEHOLDERS}
  gen      value generators: rng -> str (ADDRESS: rng -> list of lines)

Placeholders: PERSON PERSON2 BUSINESS ADDRESS DOB ACCOUNT PHONE EMAIL TIN (gold PII) and
AMOUNT DATE CODE (look-alikes that must stay O). Label rules are the English ones (README).
zh-Hant reuses the zh-Hans text via OpenCC (s2twp) and overrides the Taiwan / Hong Kong pools.
"""
import string

ENTITY_PH = {"PERSON": "PERSON", "PERSON2": "PERSON", "BUSINESS": "BUSINESS", "ADDRESS": "ADDRESS",
             "DOB": "DOB", "ACCOUNT": "ACCOUNT", "PHONE": "PHONE", "EMAIL": "EMAIL", "TIN": "TIN"}
NEG_PH = {"AMOUNT": "amount", "DATE": "date", "CODE": "code"}


# ---------------------------------------------------------------- shared helpers
def d(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def up(rng, n):
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def pick(rng, xs):
    return rng.choice(xs)


def digits_to(s, table):
    return s.translate(str.maketrans("0123456789", table))


ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
DEVANAGARI = "०१२३४५६७८९"


def swift(cc):
    """SWIFT/BIC look-alike: bank(4) + country(2) + location(2) [+ branch(3)]."""
    return lambda rng: up(rng, 4) + cc + up(rng, 2) + rng.choice(["", up(rng, 1) + d(rng, 2)])


def ymd(rng, lo, hi):
    return rng.randint(lo, hi), rng.randint(1, 12), rng.randint(1, 28)


def email_gen(locals_, domains):
    def gen(rng):
        a, b = pick(rng, locals_), pick(rng, [x for x in locals_ if "." not in x] or locals_)
        local = pick(rng, [a, f"{a}{d(rng, 3)}"] if "." in a else
                     [f"{a}.{b}", f"{a}{b}", f"{a}_{b}{d(rng, 2)}", f"{a}{d(rng, 3)}"])
        return f"{local}@{pick(rng, domains)}"
    return gen


def indian_grouping(n: int) -> str:
    s = str(n)
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail = head[-2:] + "," + tail
        head = head[:-2]
    return f"{head},{tail}" if head else tail


def amount_gen(fmts, lo=10_000, hi=5_000_000, sep=",", dec_sep=".", indian=False):
    def gen(rng):
        n = rng.randint(lo, hi) // 100 * 100
        grouped = indian_grouping(n) if indian else f"{n:,}".replace(",", sep)
        return pick(rng, fmts).format(n=grouped, dec=f"{grouped}{dec_sep}00", m=n // 10_000)
    return gen


def date_gen(fmt_fn, lo, hi):
    return lambda rng: fmt_fn(rng, *ymd(rng, lo, hi))


DOB_YEARS, DATE_YEARS = (1945, 2003), (2023, 2026)


# ---------------------------------------------------------------- English (control)
EN_MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def en_date(rng, y, m, dd):
    return pick(rng, [f"{dd:02d}/{m:02d}/{y}", f"{dd} {EN_MON[m-1]} {y}", f"{EN_MON[m-1]} {dd}, {y}",
                      f"{y}-{m:02d}-{dd:02d}", f"{dd} {EN_MON[m-1][:3]} {y}"])


def en_address(rng):
    if rng.random() < 0.5:   # Singapore
        return [f"Blk {rng.randint(1, 999)} {pick(rng, ['Ang Mo Kio Avenue 3', 'Tampines Street 21', 'Bukit Timah Road', 'Orchard Road', 'Marine Parade Road'])}",
                f"#{rng.randint(2, 30):02d}-{rng.randint(1, 120):02d}", f"Singapore {d(rng, 6)}"]
    return [f"{rng.randint(1, 250)} {pick(rng, ['High Street', 'Victoria Road', 'Kings Road', 'Church Lane', 'Park Avenue'])}",
            pick(rng, ["London", "Manchester", "Leeds", "Bristol"]) + f" {up(rng, 2)}{rng.randint(1, 20)} {rng.randint(1, 9)}{up(rng, 2)}",
            "United Kingdom"]


EN = dict(
    cues={"PERSON": ["Name", "Full Name", "Client Name", "Account Holder"],
          "DOB": ["Date of Birth", "DOB", "Birth Date"],
          "ACCOUNT": ["Account No.", "Account Number", "Portfolio No."],
          "PHONE": ["Phone", "Mobile", "Tel"], "EMAIL": ["Email", "E-mail Address"],
          "TIN": ["Tax ID", "TIN", "Tax Identification No."],
          "ADDRESS": ["Address", "Residential Address", "Mailing Address"],
          "BUSINESS": ["Employer", "Company"],
          "amount": ["Initial Deposit", "Annual Income", "Balance"],
          "date": ["Application Date", "Statement Date", "Value Date"],
          "code": ["SWIFT Code", "Branch Code"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["Account Opening Form", "Know Your Customer (KYC) Form", "Client Information Update"],
            "statement": ["Portfolio Statement", "Monthly Account Statement"],
            "closing": ["Best regards,", "Kind regards,", "Sincerely,"],
            "job": ["Relationship Manager", "Senior Wealth Advisor", "Private Banker"]},
    prose=["Dear {PERSON},\nThank you for meeting us on {DATE}. As discussed, {AMOUNT} will be transferred to your account {ACCOUNT} once the subscription is confirmed. Please call me on {PHONE} if you have any questions.",
           "Please transfer {AMOUNT} from account {ACCOUNT} held by {PERSON} to {BUSINESS} (SWIFT {CODE}) on {DATE}.",
           "Client {PERSON}, born {DOB}, has a moderate risk profile. The client is employed by {BUSINESS} and resides at {ADDRESS}.",
           "We have updated the mailing address of {PERSON} to {ADDRESS}. Statements will be sent to {EMAIL} from {DATE}.",
           "{PERSON} (tax ID {TIN}) requested a redemption of {AMOUNT} from portfolio {ACCOUNT}.",
           "Hi {PERSON}, your relationship manager {PERSON2} can be reached at {PHONE} or {EMAIL}."],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['James', 'Sarah', 'Michael', 'Priya', 'Daniel', 'Emily', 'Wei Ling', 'Robert', 'Aisha', 'Thomas', 'Grace', 'David'])} "
                         f"{pick(r, ['Thompson', 'Tan', 'Wright', 'Fernandes', 'Clarke', 'Lim', 'Patel', 'Morgan', 'Hughes', 'Ong', 'Bennett'])}",
        BUSINESS=lambda r: f"{pick(r, ['Harbour', 'Northgate', 'Crestview', 'Silverline', 'Meridian', 'Oakwood', 'Bluewater'])} "
                           f"{pick(r, ['Capital', 'Holdings', 'Trading', 'Advisory', 'Logistics'])} {pick(r, ['Pte. Ltd.', 'Ltd', 'LLC', 'plc', 'Inc.'])}",
        ADDRESS=en_address,
        DOB=date_gen(en_date, *DOB_YEARS), DATE=date_gen(en_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+65 {d(r, 4)} {d(r, 4)}", f"+44 20 {d(r, 4)} {d(r, 4)}", f"07{d(r, 3)} {d(r, 6)}", f"({d(r, 3)}) {d(r, 3)}-{d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 4)}-{d(r, 6)}-{d(r, 1)}", f"GB{d(r, 2)} {up(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 2)}", d(r, 10)]),
        TIN=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 2)}-{d(r, 4)}", d(r, 10), f"S{d(r, 7)}{up(r, 1)}"]),
        EMAIL=email_gen(["james", "sarah", "mtan", "priya", "d.clarke", "emily", "rob", "grace"], ["gmail.com", "outlook.com", "yahoo.co.uk", "hotmail.com"]),
        AMOUNT=amount_gen(["SGD {n}", "USD {dec}", "£{n}", "${dec}"]),
        CODE=lambda r: pick(r, [swift("SG")(r), swift("GB")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Simplified Chinese
ZH_SUR = ["王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周", "徐", "孙", "马", "朱", "胡", "郭", "何", "林", "罗", "高", "欧阳"]
ZH_GIV = ["伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", "军", "洋", "勇", "艳", "杰", "娟", "涛", "明", "超", "建华", "志强", "嘉怡", "子涵", "浩然", "雨萱"]
ZH_CO = (["上海", "北京", "深圳", "广州", "杭州", ""], ["华信", "中泰", "瑞丰", "恒达", "鼎盛", "汇金", "远洋", "博远", "嘉禾", "启明"],
         ["投资", "资产管理", "科技", "贸易", "实业", "控股", "证券"], ["有限公司", "股份有限公司", "集团有限公司"])
ZH_ADDR = [("上海市", "浦东新区", ["世纪大道", "陆家嘴环路", "张杨路"], "200120"),
           ("北京市", "朝阳区", ["建国路", "东三环中路", "朝阳门外大街"], "100020"),
           ("广州市", "天河区", ["天河路", "华夏路", "体育东路"], "510620"),
           ("深圳市", "福田区", ["深南大道", "福华三路", "益田路"], "518048"),
           ("杭州市", "西湖区", ["文三路", "天目山路"], "310012"),
           ("成都市", "武侯区", ["人民南路", "科华北路"], "610041")]


def zh_date(rng, y, m, dd):
    return pick(rng, [f"{y}年{m}月{dd}日", f"{y}年{m:02d}月{dd:02d}日", f"{y}-{m:02d}-{dd:02d}", f"{y}/{m:02d}/{dd:02d}"])


def zh_address(rng):
    city, dist, streets, pc = pick(rng, ZH_ADDR)
    street = f"{pick(rng, streets)}{rng.randint(1, 1999)}号"
    unit = pick(rng, ["", f"{rng.randint(1, 30)}栋{rng.randint(1, 32)}{rng.randint(1, 9):02d}室", f"{pick(rng, ['国金中心', '环球金融中心', '华润大厦'])}{rng.randint(2, 60)}楼"])
    lines = [f"{city}{dist}", f"{street}{unit}"]
    if rng.random() < 0.4:
        lines.append(f"邮编：{pc}")
    return lines


ZH = dict(
    cues={"PERSON": ["姓名", "客户姓名", "账户持有人"], "DOB": ["出生日期", "生日"],
          "ACCOUNT": ["账号", "银行账号", "账户号码", "投资组合编号"], "PHONE": ["电话", "手机号码", "联系电话"],
          "EMAIL": ["电子邮箱", "邮箱"], "TIN": ["纳税人识别号", "税号"], "ADDRESS": ["地址", "住址", "通讯地址"],
          "BUSINESS": ["工作单位", "雇主", "公司名称"], "amount": ["初始存款", "年收入", "账户余额"],
          "date": ["申请日期", "对账单日期", "起息日"], "code": ["SWIFT代码", "分行代码"]},
    seps=["：", "：", ": ", "\t"],
    titles={"kyc": ["开户申请表", "客户身份识别（KYC）表", "客户资料更新表"], "statement": ["投资组合对账单", "月度账户对账单"],
            "closing": ["此致\n敬礼", "顺颂商祺", "祝好"], "job": ["客户经理", "高级财富顾问", "私人银行家"]},
    prose=["尊敬的{PERSON}：\n感谢您于{DATE}与我们会面。如前所述，认购确认后，{AMOUNT}将转入您的账户{ACCOUNT}。如有任何疑问，请致电{PHONE}与我联系。",
           "请于{DATE}从{PERSON}名下账户{ACCOUNT}向{BUSINESS}（SWIFT {CODE}）汇款{AMOUNT}。",
           "客户{PERSON}，出生于{DOB}，风险评级为中等。客户受雇于{BUSINESS}，现居住于{ADDRESS}。",
           "我们已将{PERSON}的通讯地址更新为{ADDRESS}。自{DATE}起，对账单将发送至{EMAIL}。",
           "{PERSON}（纳税人识别号{TIN}）申请从投资组合{ACCOUNT}赎回{AMOUNT}。",
           "{PERSON}您好，您的客户经理{PERSON2}的联系方式为{PHONE}或{EMAIL}。"],
    gen=dict(
        PERSON=lambda r: pick(r, ZH_SUR) + pick(r, ZH_GIV),
        BUSINESS=lambda r: "".join(pick(r, part) for part in ZH_CO),
        ADDRESS=zh_address,
        DOB=date_gen(zh_date, *DOB_YEARS), DATE=date_gen(zh_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+86 1{d(r, 2)} {d(r, 4)} {d(r, 4)}", f"1{d(r, 2)}-{d(r, 4)}-{d(r, 4)}", f"0{d(r, 2)}-{d(r, 8)}", f"1{d(r, 10)}"]),
        ACCOUNT=lambda r: pick(r, [f"6222 {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 3)}", f"62{d(r, 17)}", f"{d(r, 4)}-{d(r, 4)}-{d(r, 8)}"]),
        TIN=lambda r: pick(r, [f"{d(r, 17)}{pick(r, ['X', d(r, 1)])}", f"91{d(r, 6)}{up(r, 2)}{d(r, 7)}{up(r, 1)}"]),
        EMAIL=email_gen(["wang.wei", "zhangli", "chen.min", "liuyang", "huangjie", "zhou.tao", "linfang", "xu"], ["163.com", "qq.com", "126.com", "sina.com", "gmail.com"]),
        AMOUNT=amount_gen(["人民币{n}元", "¥{dec}", "CNY {n}", "{m}万元"]),
        CODE=lambda r: pick(r, [swift("CN")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Traditional Chinese (TW / HK pools)
TW_ADDR = [("臺北市", "信義區", ["信義路五段", "松仁路", "忠孝東路四段"], "110"), ("臺北市", "大安區", ["敦化南路二段", "復興南路一段"], "106"),
           ("新北市", "板橋區", ["文化路一段", "中山路一段"], "220"), ("臺中市", "西屯區", ["臺灣大道三段"], "407"),
           ("高雄市", "前鎮區", ["成功二路", "中山二路"], "806")]
HK_ADDR = [("香港", "中環", ["皇后大道中", "德輔道中"]), ("九龍", "尖沙咀", ["彌敦道", "廣東道"]), ("香港", "銅鑼灣", ["軒尼詩道", "怡和街"])]


def zht_date(rng, y, m, dd):
    return pick(rng, [f"{y}年{m}月{dd}日", f"民國{y - 1911}年{m}月{dd}日", f"{y}/{m:02d}/{dd:02d}", f"{dd:02d}/{m:02d}/{y}"])


def zht_address(rng):
    if rng.random() < 0.6:
        city, dist, streets, pc = pick(rng, TW_ADDR)
        lines = [f"{pc}{city}{dist}", f"{pick(rng, streets)}{rng.randint(1, 300)}號{rng.randint(2, 30)}樓"]
    else:
        reg, dist, streets = pick(rng, HK_ADDR)
        lines = [f"{reg}{dist}{pick(rng, streets)}{rng.randint(1, 300)}號",
                 f"{pick(rng, ['國際金融中心', '時代廣場', '中環中心', '海港城'])}{rng.randint(2, 60)}樓{rng.randint(1, 30):02d}室"]
    return lines


ZHT_GEN = dict(
    ADDRESS=zht_address,
    DOB=date_gen(zht_date, *DOB_YEARS), DATE=date_gen(zht_date, *DATE_YEARS),
    PHONE=lambda r: pick(r, [f"+886 9{d(r, 2)} {d(r, 3)} {d(r, 3)}", f"09{d(r, 2)}-{d(r, 3)}-{d(r, 3)}", f"(02) {d(r, 4)}-{d(r, 4)}", f"+852 {d(r, 4)} {d(r, 4)}"]),
    ACCOUNT=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 2)}-{d(r, 6)}-{d(r, 1)}", d(r, 14), f"{d(r, 3)}-{d(r, 6)}-{d(r, 3)}"]),
    TIN=lambda r: pick(r, [d(r, 8), f"{up(r, 1)}{pick(r, ['1', '2'])}{d(r, 8)}"]),
    EMAIL=email_gen(["chen.yating", "lin.chihao", "wang.mei", "huang", "chang.wei", "lee.kaman", "wong", "cheung"], ["gmail.com", "yahoo.com.tw", "hinet.net", "netvigator.com"]),
    AMOUNT=amount_gen(["新臺幣{n}元", "NT${n}", "HK${dec}", "港幣{m}萬元"]),
    CODE=lambda r: pick(r, [swift("TW")(r), swift("HK")(r), d(r, 3)]),
)

# ---------------------------------------------------------------- Japanese
JA_SUR = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤", "吉田", "山田", "松本", "井上", "木村"]
JA_GIV = ["太郎", "花子", "翔太", "陽菜", "健一", "美咲", "大輔", "由美", "拓也", "さくら", "直樹", "愛", "誠", "恵子", "蓮"]
JA_ADDR = [("東京都", "千代田区", "丸の内", "100-0005"), ("東京都", "港区", "六本木", "106-0032"), ("大阪府", "大阪市北区", "梅田", "530-0001"),
           ("神奈川県", "横浜市西区", "みなとみらい", "220-0012"), ("愛知県", "名古屋市中区", "栄", "460-0008"), ("福岡県", "福岡市中央区", "天神", "810-0001")]


def ja_date(rng, y, m, dd):
    showa = f"昭和{y - 1925}年{m}月{dd}日" if 1926 <= y <= 1988 else f"平成{y - 1988}年{m}月{dd}日" if y >= 1989 else f"{y}年{m}月{dd}日"
    return pick(rng, [f"{y}年{m}月{dd}日", f"{y}年{m:02d}月{dd:02d}日", f"{y}/{m:02d}/{dd:02d}", showa])


def ja_address(rng):
    pref, city, town, pc = pick(rng, JA_ADDR)
    a, b, c = rng.randint(1, 9), rng.randint(1, 30), rng.randint(1, 20)
    block = pick(rng, [f"{a}丁目{b}番{c}号", f"{a}-{b}-{c}"])
    lines = [f"〒{pc}", f"{pref}{city}{town}{block}"]
    if rng.random() < 0.5:
        lines.append(f"{pick(rng, ['パークタワー', 'グランドメゾン', '第一ビル', 'ヒルズレジデンス'])}{rng.randint(2, 40)}{pick(rng, ['階', '0' + str(rng.randint(1, 9)) + '号室'])}")
    return lines


JA = dict(
    cues={"PERSON": ["氏名", "お名前", "口座名義人"], "DOB": ["生年月日"], "ACCOUNT": ["口座番号", "証券口座番号"],
          "PHONE": ["電話番号", "携帯電話", "連絡先"], "EMAIL": ["メールアドレス", "Eメール"], "TIN": ["マイナンバー", "個人番号"],
          "ADDRESS": ["住所", "ご住所", "現住所"], "BUSINESS": ["勤務先", "会社名"], "amount": ["初回入金額", "年収", "残高"],
          "date": ["申込日", "作成日", "受渡日"], "code": ["SWIFTコード", "支店コード"]},
    seps=["：", "：", ": ", "　"],
    titles={"kyc": ["口座開設申込書", "お客様情報確認書（KYC）", "登録内容変更届"], "statement": ["取引残高報告書", "資産残高明細書"],
            "closing": ["よろしくお願いいたします。", "敬具"], "job": ["プライベートバンカー", "ウェルスマネージャー", "営業部 担当"]},
    prose=["{PERSON}様\n{DATE}はお時間をいただきありがとうございました。お申込みが確定次第、{AMOUNT}をお客様の口座{ACCOUNT}へ入金いたします。ご不明な点は{PHONE}までお電話ください。",
           "{DATE}付で、{PERSON}名義の口座{ACCOUNT}から{BUSINESS}（SWIFT {CODE}）宛に{AMOUNT}を送金してください。",
           "顧客{PERSON}（生年月日：{DOB}）のリスク許容度は中程度です。{BUSINESS}に勤務しており、{ADDRESS}に居住しています。",
           "{PERSON}様のご住所を{ADDRESS}に変更いたしました。{DATE}以降、報告書は{EMAIL}宛に送付されます。",
           "{PERSON}様（マイナンバー{TIN}）より、ポートフォリオ{ACCOUNT}から{AMOUNT}の解約のお申し出がありました。",
           "{PERSON}様、担当の{PERSON2}には{PHONE}または{EMAIL}でご連絡いただけます。"],
    gen=dict(
        PERSON=lambda r: pick(r, JA_SUR) + pick(r, ["", "", " ", "　"]) + pick(r, JA_GIV),
        BUSINESS=lambda r: pick(r, ["株式会社{x}", "{x}株式会社", "有限会社{x}"]).format(
            x=pick(r, ["山田商事", "東邦ホールディングス", "みらい証券", "日本テック", "大和物産", "さくら不動産", "光陽インベストメント", "アオバ・キャピタル"])),
        ADDRESS=ja_address,
        DOB=date_gen(ja_date, *DOB_YEARS), DATE=date_gen(ja_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"03-{d(r, 4)}-{d(r, 4)}", f"090-{d(r, 4)}-{d(r, 4)}", f"+81 90 {d(r, 4)} {d(r, 4)}", f"06-{d(r, 4)}-{d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [d(r, 7), f"{d(r, 3)}-{d(r, 7)}", f"{d(r, 4)}-{d(r, 7)}"]),
        TIN=lambda r: pick(r, [f"{d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 12)]),
        EMAIL=email_gen(["taro", "yamada", "hanako", "suzuki", "k.tanaka", "sato", "yumi", "takahashi"], ["docomo.ne.jp", "gmail.com", "yahoo.co.jp", "ezweb.ne.jp"]),
        AMOUNT=amount_gen(["{n}円", "¥{n}", "{m}万円", "JPY {n}"]),
        CODE=lambda r: pick(r, [swift("JP")(r), d(r, 3)]),
    ),
)

# ---------------------------------------------------------------- Korean
KO_ADDR = [("서울특별시", "중구", ["세종대로", "을지로"]), ("서울특별시", "강남구", ["테헤란로", "강남대로"]),
           ("부산광역시", "해운대구", ["센텀중앙로", "해운대로"]), ("경기도", "성남시 분당구", ["판교역로", "정자일로"]),
           ("인천광역시", "연수구", ["컨벤시아대로"]), ("대구광역시", "수성구", ["달구벌대로"])]


def ko_date(rng, y, m, dd):
    return pick(rng, [f"{y}년 {m}월 {dd}일", f"{y}.{m:02d}.{dd:02d}", f"{y}-{m:02d}-{dd:02d}", f"{y}. {m}. {dd}."])


def ko_address(rng):
    city, gu, roads = pick(rng, KO_ADDR)
    road = f"{city} {gu} {pick(rng, roads)} {rng.randint(1, 400)}"
    unit = f"{rng.randint(101, 115)}동 {rng.randint(1, 25)}{rng.randint(1, 4):02d}호"
    pc = f"({d(rng, 5)})"
    return pick(rng, [[road, unit], [road + ",", unit, pc], [pc + " " + road, unit]])


KO = dict(
    cues={"PERSON": ["성명", "이름", "예금주", "고객명"], "DOB": ["생년월일", "출생일"], "ACCOUNT": ["계좌번호", "증권계좌번호"],
          "PHONE": ["전화번호", "휴대폰", "연락처"], "EMAIL": ["이메일", "전자우편"], "TIN": ["납세자번호", "사업자등록번호"],
          "ADDRESS": ["주소", "자택 주소", "거주지"], "BUSINESS": ["직장명", "회사명"], "amount": ["초기 입금액", "연소득", "잔액"],
          "date": ["신청일", "작성일", "결제일"], "code": ["SWIFT 코드", "지점 코드"]},
    seps=[": ", ": ", " : ", "\t"],
    titles={"kyc": ["계좌 개설 신청서", "고객 확인서(KYC)", "고객 정보 변경 신청서"], "statement": ["자산 운용 보고서", "월간 거래 내역서"],
            "closing": ["감사합니다.", "드림"], "job": ["PB 팀장", "자산관리 매니저", "프라이빗 뱅커"]},
    prose=["{PERSON} 고객님께,\n{DATE}에 시간을 내주셔서 감사합니다. 청약이 확정되면 {AMOUNT}이 고객님의 계좌 {ACCOUNT}(으)로 입금될 예정입니다. 문의 사항은 {PHONE}(으)로 연락 주십시오.",
           "{DATE}에 {PERSON} 명의의 계좌 {ACCOUNT}에서 {BUSINESS}(SWIFT {CODE})(으)로 {AMOUNT}을 송금해 주십시오.",
           "고객 {PERSON}(생년월일 {DOB})의 투자 성향은 중립형입니다. 고객은 {BUSINESS}에 재직 중이며 {ADDRESS}에 거주하고 있습니다.",
           "{PERSON} 고객님의 주소가 {ADDRESS}(으)로 변경되었습니다. {DATE}부터 보고서는 {EMAIL}(으)로 발송됩니다.",
           "{PERSON} 고객님(납세자번호 {TIN})께서 포트폴리오 {ACCOUNT}에서 {AMOUNT} 환매를 요청하셨습니다.",
           "{PERSON} 고객님, 담당 PB {PERSON2}에게 {PHONE} 또는 {EMAIL}(으)로 연락하실 수 있습니다."],
    gen=dict(
        PERSON=lambda r: pick(r, ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권"])
                         + pick(r, ["민수", "서연", "지훈", "지우", "현우", "수빈", "도윤", "하은", "준호", "예진", "성민", "은지", "동현", "유진", "재원"]),
        BUSINESS=lambda r: pick(r, ["{x} 주식회사", "(주){x}", "주식회사 {x}"]).format(
            x=pick(r, ["한빛투자", "대양물산", "새벽자산운용", "서울테크", "푸른캐피탈", "새솔건설", "누리증권", "가온홀딩스"])),
        ADDRESS=ko_address,
        DOB=lambda r: (lambda y, m, dd: pick(r, [ko_date(r, y, m, dd), f"{y % 100:02d}{m:02d}{dd:02d}"]))(*ymd(r, *DOB_YEARS)),
        DATE=date_gen(ko_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"010-{d(r, 4)}-{d(r, 4)}", f"02-{d(r, 3)}-{d(r, 4)}", f"+82 10 {d(r, 4)} {d(r, 4)}", f"031-{d(r, 3)}-{d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 3)}-{d(r, 6)}", f"{d(r, 3)}-{d(r, 2)}-{d(r, 6)}", f"{d(r, 4)}-{d(r, 2)}-{d(r, 7)}"]),
        TIN=lambda r: f"{d(r, 3)}-{d(r, 2)}-{d(r, 5)}",
        EMAIL=email_gen(["minsu", "kim", "seoyeon", "park", "jihoon.lee", "choi", "hayun", "jung"], ["naver.com", "daum.net", "gmail.com", "kakao.com"]),
        AMOUNT=amount_gen(["{n}원", "₩{n}", "{m}만 원", "KRW {n}"]),
        CODE=lambda r: pick(r, [swift("KR")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Hindi
HI_MON = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
HI_CITY = [("मुंबई", "महाराष्ट्र", "400050"), ("नई दिल्ली", "दिल्ली", "110001"), ("बेंगलुरु", "कर्नाटक", "560001"),
           ("पुणे", "महाराष्ट्र", "411001"), ("जयपुर", "राजस्थान", "302001"), ("लखनऊ", "उत्तर प्रदेश", "226001"), ("अहमदाबाद", "गुजरात", "380009")]


def hi_date(rng, y, m, dd):
    s = pick(rng, [f"{dd:02d}/{m:02d}/{y}", f"{dd:02d}-{m:02d}-{y}", f"{dd} {HI_MON[m-1]} {y}", f"{y}-{m:02d}-{dd:02d}"])
    return digits_to(s, DEVANAGARI) if rng.random() < 0.2 else s


def hi_address(rng):
    city, state, pin = pick(rng, HI_CITY)
    l1 = f"{pick(rng, ['फ्लैट नं. ', 'मकान नं. ', ''])}{rng.randint(1, 999)}, {pick(rng, ['सनशाइन अपार्टमेंट', 'गोकुल निवास', 'शांति कुंज', 'सागर टावर'])}"
    l2 = pick(rng, ["एम.जी. रोड", "नेहरू मार्ग", "स्टेशन रोड", "लिंक रोड", "गांधी नगर"])
    return pick(rng, [[l1 + ",", l2 + ",", f"{city} - {pin}", state], [f"{l1}, {l2}", f"{city}, {state} {pin}"]])


HI = dict(
    cues={"PERSON": ["नाम", "ग्राहक का नाम", "खाताधारक का नाम"], "DOB": ["जन्म तिथि", "जन्मतिथि"],
          "ACCOUNT": ["खाता संख्या", "खाता नंबर", "डीमैट खाता संख्या"], "PHONE": ["फ़ोन नंबर", "मोबाइल नंबर", "संपर्क नंबर"],
          "EMAIL": ["ईमेल", "ईमेल पता"], "TIN": ["पैन", "पैन नंबर", "कर पहचान संख्या"], "ADDRESS": ["पता", "आवासीय पता", "पत्राचार का पता"],
          "BUSINESS": ["नियोक्ता", "कंपनी का नाम"], "amount": ["प्रारंभिक जमा राशि", "वार्षिक आय", "शेष राशि"],
          "date": ["आवेदन की तिथि", "विवरण की तिथि", "मूल्य तिथि"], "code": ["IFSC कोड", "SWIFT कोड", "शाखा कोड"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["खाता खोलने का आवेदन पत्र", "केवाईसी फ़ॉर्म", "ग्राहक विवरण अद्यतन फ़ॉर्म"], "statement": ["पोर्टफोलियो विवरण", "मासिक खाता विवरण"],
            "closing": ["सादर,", "भवदीय,", "धन्यवाद,"], "job": ["रिलेशनशिप मैनेजर", "वरिष्ठ वेल्थ सलाहकार"]},
    prose=["प्रिय {PERSON},\n{DATE} को हमसे मिलने के लिए धन्यवाद। जैसा कि चर्चा हुई, सब्सक्रिप्शन की पुष्टि होने पर {AMOUNT} आपके खाते {ACCOUNT} में स्थानांतरित कर दिए जाएंगे। कोई प्रश्न हो तो कृपया मुझे {PHONE} पर कॉल करें।",
           "कृपया {DATE} को {PERSON} के खाते {ACCOUNT} से {BUSINESS} (SWIFT {CODE}) को {AMOUNT} स्थानांतरित करें।",
           "ग्राहक {PERSON}, जिनकी जन्म तिथि {DOB} है, का जोखिम प्रोफ़ाइल मध्यम है। वे {BUSINESS} में कार्यरत हैं और {ADDRESS} में रहते हैं।",
           "हमने {PERSON} का पत्राचार पता बदलकर {ADDRESS} कर दिया है। {DATE} से विवरण {EMAIL} पर भेजे जाएंगे।",
           "{PERSON} (पैन {TIN}) ने पोर्टफोलियो {ACCOUNT} से {AMOUNT} के रिडेम्पशन का अनुरोध किया है।",
           "नमस्ते {PERSON} जी, आपके रिलेशनशिप मैनेजर {PERSON2} से {PHONE} या {EMAIL} पर संपर्क किया जा सकता है।"],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['राजेश', 'प्रिया', 'अमित', 'सुनीता', 'विकास', 'अनीता', 'राहुल', 'पूजा', 'संजय', 'नेहा', 'अर्जुन', 'कविता', 'मनोज', 'दीपिका', 'सुरेश'])} "
                         f"{pick(r, ['शर्मा', 'वर्मा', 'गुप्ता', 'सिंह', 'कुमार', 'पटेल', 'मेहता', 'जोशी', 'अग्रवाल', 'रेड्डी', 'चौधरी', 'मिश्रा', 'यादव', 'कपूर', 'नायर'])}",
        BUSINESS=lambda r: f"{pick(r, ['श्री गणेश', 'नवभारत', 'सूर्या', 'अमृत', 'गंगा', 'आदित्य', 'कावेरी'])} "
                           f"{pick(r, ['इन्वेस्टमेंट्स', 'ट्रेडर्स', 'फाइनेंशियल सर्विसेज', 'इंडस्ट्रीज', 'कैपिटल'])} {pick(r, ['प्राइवेट लिमिटेड', 'प्रा. लि.', 'लिमिटेड'])}",
        ADDRESS=hi_address,
        DOB=date_gen(hi_date, *DOB_YEARS), DATE=date_gen(hi_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+91 9{d(r, 4)} {d(r, 5)}", f"9{d(r, 4)}-{d(r, 5)}", f"022-{d(r, 4)} {d(r, 4)}", f"9{d(r, 9)}"]),
        ACCOUNT=lambda r: pick(r, [d(r, 14), f"{d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 12)]),
        TIN=lambda r: f"{up(r, 5)}{d(r, 4)}{up(r, 1)}",
        EMAIL=email_gen(["rajesh", "priya", "amit.sharma", "sunita", "vikas", "pooja", "verma", "gupta"], ["gmail.com", "rediffmail.com", "yahoo.co.in", "outlook.com"]),
        AMOUNT=amount_gen(["₹{n}", "रु. {n}", "INR {n}", "₹{dec}"], indian=True),
        CODE=lambda r: pick(r, [f"{up(r, 4)}0{d(r, 6)}", swift("IN")(r)]),
    ),
)

# ---------------------------------------------------------------- Arabic
AR_MON = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
AR_CITY = [("دبي", "الإمارات العربية المتحدة"), ("أبوظبي", "الإمارات العربية المتحدة"), ("الشارقة", "الإمارات العربية المتحدة"),
           ("الرياض", "المملكة العربية السعودية"), ("جدة", "المملكة العربية السعودية"), ("الدوحة", "قطر"), ("المنامة", "البحرين")]


def ar_date(rng, y, m, dd):
    s = pick(rng, [f"{dd:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{dd:02d}", f"{dd} {AR_MON[m-1]} {y}"])
    return digits_to(s, ARABIC_INDIC) if rng.random() < 0.25 else s


def ar_address(rng):
    city, country = pick(rng, AR_CITY)
    l1 = f"{pick(rng, ['شقة', 'فيلا', 'مكتب'])} {rng.randint(1, 3000)}، {pick(rng, ['برج الماسة', 'برج النخيل', 'برج الصفوة', 'برج اللؤلؤة'])}"
    l2 = f"{pick(rng, ['شارع الشيخ زايد', 'طريق الملك فهد', 'شارع الكورنيش', 'شارع حمدان', 'طريق الملك عبدالعزيز'])}، {pick(rng, ['حي العليا', 'مرسى دبي', 'الخالدية', 'حي الملز', 'البرشاء'])}"
    l3 = f"{city}{pick(rng, ['', ' ' + d(rng, 5), '، ص.ب. ' + d(rng, 5)])}"
    return pick(rng, [[l1 + "،", l2 + "،", l3, country], [f"{l1}، {l2}، {l3}، {country}"]])


AR = dict(
    cues={"PERSON": ["الاسم", "اسم العميل", "اسم صاحب الحساب"], "DOB": ["تاريخ الميلاد"],
          "ACCOUNT": ["رقم الحساب", "رقم الآيبان", "رقم المحفظة"], "PHONE": ["رقم الهاتف", "الجوال", "رقم الاتصال"],
          "EMAIL": ["البريد الإلكتروني"], "TIN": ["الرقم الضريبي", "رقم التعريف الضريبي"], "ADDRESS": ["العنوان", "عنوان السكن", "العنوان البريدي"],
          "BUSINESS": ["جهة العمل", "اسم الشركة"], "amount": ["الإيداع الأولي", "الدخل السنوي", "الرصيد"],
          "date": ["تاريخ الطلب", "تاريخ الكشف", "تاريخ الاستحقاق"], "code": ["رمز السويفت", "رمز الفرع"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["نموذج فتح حساب", "نموذج اعرف عميلك", "نموذج تحديث بيانات العميل"], "statement": ["كشف المحفظة الاستثمارية", "كشف الحساب الشهري"],
            "closing": ["مع خالص التحية،", "وتفضلوا بقبول فائق الاحترام،"], "job": ["مدير علاقات العملاء", "مستشار ثروات أول"]},
    prose=["عزيزي {PERSON}،\nشكراً لاجتماعك معنا بتاريخ {DATE}. كما تم الاتفاق، سيتم تحويل {AMOUNT} إلى حسابك {ACCOUNT} بمجرد تأكيد الاكتتاب. يرجى الاتصال بي على {PHONE} إذا كانت لديك أي أسئلة.",
           "يرجى تحويل {AMOUNT} من الحساب {ACCOUNT} باسم {PERSON} إلى {BUSINESS} (سويفت {CODE}) بتاريخ {DATE}.",
           "العميل {PERSON}، المولود في {DOB}، لديه ملف مخاطر متوسط. يعمل لدى {BUSINESS} ويقيم في {ADDRESS}.",
           "قمنا بتحديث العنوان البريدي للعميل {PERSON} إلى {ADDRESS}. سيتم إرسال الكشوفات إلى {EMAIL} اعتباراً من {DATE}.",
           "طلب {PERSON} (الرقم الضريبي {TIN}) استرداد {AMOUNT} من المحفظة {ACCOUNT}.",
           "مرحباً {PERSON}، يمكنك التواصل مع مدير علاقتك {PERSON2} على {PHONE} أو {EMAIL}."],
    gen=dict(
        PERSON=lambda r: pick(r, ["{g} {f}", "{g} {g2} {f}", "{g} بن {g2} {f}"]).format(
            g=pick(r, ["محمد", "أحمد", "فاطمة", "علي", "عائشة", "خالد", "مريم", "عمر", "نورة", "يوسف", "سارة", "عبدالله", "ليلى", "حسن", "هند"]),
            g2=pick(r, ["سالم", "راشد", "سعيد", "عبدالرحمن", "إبراهيم"]),
            f=pick(r, ["الهاشمي", "المنصوري", "العتيبي", "الشمري", "القحطاني", "الكعبي", "الزعابي", "الحمادي", "النعيمي", "السويدي", "الدوسري", "المطيري"])),
        BUSINESS=lambda r: f"شركة {pick(r, ['الخليج', 'النخبة', 'الريادة', 'الفجر', 'الواحة', 'المستقبل', 'الأفق'])} "
                           f"{pick(r, ['للاستثمار', 'للتجارة العامة', 'القابضة', 'للخدمات المالية', 'العقارية'])}{pick(r, ['', ' ذ.م.م', ' ش.م.ع', ' المحدودة'])}",
        ADDRESS=ar_address,
        DOB=date_gen(ar_date, *DOB_YEARS), DATE=date_gen(ar_date, *DATE_YEARS),
        PHONE=lambda r: (lambda s: digits_to(s, ARABIC_INDIC) if r.random() < 0.2 else s)(
            pick(r, [f"+971 5{d(r, 1)} {d(r, 3)} {d(r, 4)}", f"+966 5{d(r, 1)} {d(r, 3)} {d(r, 4)}", f"05{d(r, 1)} {d(r, 3)} {d(r, 4)}", f"04 {d(r, 3)} {d(r, 4)}"])),
        ACCOUNT=lambda r: pick(r, [f"AE{d(r, 2)} {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 3)}", f"SA{d(r, 2)} {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 13)]),
        TIN=lambda r: pick(r, [f"100{d(r, 12)}", f"3{d(r, 13)}3"]),
        EMAIL=email_gen(["mohammed", "ahmed", "fatima", "k.alhashmi", "omar", "noura", "alotaibi", "sara"], ["gmail.com", "hotmail.com", "emirates.net.ae", "outlook.sa"]),
        AMOUNT=lambda r: (lambda s: digits_to(s, ARABIC_INDIC) if r.random() < 0.2 else s)(amount_gen(["{n} درهم", "AED {n}", "{n} ريال", "SAR {dec}"])(r)),
        CODE=lambda r: pick(r, [swift("AE")(r), swift("SA")(r), d(r, 3)]),
    ),
)

# ---------------------------------------------------------------- Thai
TH_MON = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
TH_MON_S = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
TH_AREA = [("แขวงคลองเตยเหนือ", "เขตวัฒนา", "กรุงเทพมหานคร", "10110"), ("แขวงสีลม", "เขตบางรัก", "กรุงเทพมหานคร", "10500"),
           ("แขวงสามเสนใน", "เขตพญาไท", "กรุงเทพมหานคร", "10400"), ("ตำบลช้างคลาน", "อำเภอเมืองเชียงใหม่", "จังหวัดเชียงใหม่", "50100"),
           ("ตำบลหนองปรือ", "อำเภอบางละมุง", "จังหวัดชลบุรี", "20150"), ("ตำบลตลาดใหญ่", "อำเภอเมืองภูเก็ต", "จังหวัดภูเก็ต", "83000")]


def th_date(rng, y, m, dd):
    be = y + 543
    return pick(rng, [f"{dd} {TH_MON[m-1]} {be}", f"{dd:02d}/{m:02d}/{be}", f"{dd} {TH_MON_S[m-1]} {be}", f"{y}-{m:02d}-{dd:02d}"])


def th_address(rng):
    sub, dist, prov, pc = pick(rng, TH_AREA)
    house = pick(rng, [f"{rng.randint(1, 999)}/{rng.randint(1, 99)}", str(rng.randint(1, 999))])
    l1 = f"{house} {pick(rng, ['', 'หมู่ ' + str(rng.randint(1, 12)) + ' ', 'อาคารสาทรทาวเวอร์ ชั้น ' + str(rng.randint(2, 40)) + ' '])}" \
         f"{pick(rng, ['ซอยสุขุมวิท ' + str(rng.randint(1, 101)) + ' ', 'ซอยอารีย์ ', ''])}{pick(rng, ['ถนนสุขุมวิท', 'ถนนสีลม', 'ถนนพหลโยธิน', 'ถนนรัชดาภิเษก', 'ถนนเพชรบุรี'])}"
    return pick(rng, [[l1, f"{sub} {dist}", f"{prov} {pc}"], [f"{l1} {sub} {dist} {prov} {pc}"]])


TH = dict(
    cues={"PERSON": ["ชื่อ-นามสกุล", "ชื่อลูกค้า", "ชื่อผู้ถือบัญชี"], "DOB": ["วันเกิด", "วัน/เดือน/ปีเกิด"],
          "ACCOUNT": ["เลขที่บัญชี", "หมายเลขบัญชี", "เลขที่บัญชีกองทุน"], "PHONE": ["โทรศัพท์", "เบอร์มือถือ", "หมายเลขติดต่อ"],
          "EMAIL": ["อีเมล"], "TIN": ["เลขประจำตัวผู้เสียภาษี", "เลขผู้เสียภาษี"], "ADDRESS": ["ที่อยู่", "ที่อยู่ตามทะเบียนบ้าน", "ที่อยู่สำหรับจัดส่งเอกสาร"],
          "BUSINESS": ["สถานที่ทำงาน", "ชื่อบริษัท"], "amount": ["เงินฝากเริ่มต้น", "รายได้ต่อปี", "ยอดคงเหลือ"],
          "date": ["วันที่ยื่นคำขอ", "วันที่ออกรายงาน", "วันที่ชำระราคา"], "code": ["รหัส SWIFT", "รหัสสาขา"]},
    seps=[": ", ": ", " ", "\t"],
    titles={"kyc": ["ใบคำขอเปิดบัญชี", "แบบฟอร์มรู้จักลูกค้า (KYC)", "แบบฟอร์มแก้ไขข้อมูลลูกค้า"], "statement": ["รายงานสรุปพอร์ตการลงทุน", "รายการเดินบัญชีประจำเดือน"],
            "closing": ["ขอแสดงความนับถือ", "ด้วยความเคารพ"], "job": ["ผู้จัดการฝ่ายลูกค้าสัมพันธ์", "ที่ปรึกษาการลงทุนอาวุโส"]},
    prose=["เรียน คุณ{PERSON}\nขอขอบคุณที่สละเวลาพบกับเราเมื่อวันที่ {DATE} ตามที่ได้หารือกันไว้ เมื่อการจองซื้อได้รับการยืนยันแล้ว จะมีการโอนเงิน {AMOUNT} เข้าบัญชี {ACCOUNT} ของท่าน หากมีข้อสงสัยกรุณาติดต่อดิฉันที่ {PHONE}",
           "กรุณาโอนเงิน {AMOUNT} จากบัญชี {ACCOUNT} ของ {PERSON} ไปยัง {BUSINESS} (SWIFT {CODE}) ในวันที่ {DATE}",
           "ลูกค้า {PERSON} เกิดวันที่ {DOB} มีระดับความเสี่ยงปานกลาง ปัจจุบันทำงานที่ {BUSINESS} และพักอาศัยอยู่ที่ {ADDRESS}",
           "เราได้เปลี่ยนที่อยู่สำหรับจัดส่งเอกสารของคุณ{PERSON} เป็น {ADDRESS} แล้ว ตั้งแต่วันที่ {DATE} รายงานจะถูกส่งไปที่ {EMAIL}",
           "คุณ{PERSON} (เลขประจำตัวผู้เสียภาษี {TIN}) ขอขายคืนหน่วยลงทุนจำนวน {AMOUNT} จากพอร์ต {ACCOUNT}",
           "สวัสดีค่ะ คุณ{PERSON} ท่านสามารถติดต่อผู้จัดการฝ่ายลูกค้าสัมพันธ์ของท่าน คุณ{PERSON2} ได้ที่ {PHONE} หรือ {EMAIL}"],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['สมชาย', 'สมหญิง', 'วิชัย', 'สุดารัตน์', 'ประเสริฐ', 'กาญจนา', 'ธนพล', 'ณัฐธิดา', 'อนุชา', 'พิมพ์ชนก', 'กิตติ', 'วราภรณ์', 'ศักดิ์ชัย', 'อรุณี', 'ปิยะ'])} "
                         f"{pick(r, ['ใจดี', 'รักไทย', 'ศรีสุข', 'วงศ์สวัสดิ์', 'ทองมา', 'สุวรรณรัตน์', 'บุญมี', 'แก้วกาญจนา', 'จันทร์เพ็ญ', 'พงษ์ไพบูลย์', 'ศรีวงศ์', 'มั่นคง'])}",
        BUSINESS=lambda r: f"บริษัท {pick(r, ['สยามอินเวสต์', 'ไทยรุ่งเรือง', 'กรุงเทพพัฒนา', 'เอเชียแคปปิตอล', 'รุ่งโรจน์เทรดดิ้ง', 'ศรีสยามโฮลดิ้ง', 'บางกอกแอสเซท'])} จำกัด{pick(r, ['', ' (มหาชน)'])}",
        ADDRESS=th_address,
        DOB=date_gen(th_date, *DOB_YEARS), DATE=date_gen(th_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"08{d(r, 1)}-{d(r, 3)}-{d(r, 4)}", f"02-{d(r, 3)}-{d(r, 4)}", f"+66 8{d(r, 1)} {d(r, 3)} {d(r, 4)}", f"08{d(r, 8)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 1)}-{d(r, 5)}-{d(r, 1)}", d(r, 10), f"{d(r, 3)}-{d(r, 6)}-{d(r, 1)}"]),
        TIN=lambda r: pick(r, [f"{d(r, 1)}-{d(r, 4)}-{d(r, 5)}-{d(r, 2)}-{d(r, 1)}", d(r, 13)]),
        EMAIL=email_gen(["somchai", "sudarat", "wichai", "kanjana", "thanapol", "n.srisuk", "piya", "arunee"], ["gmail.com", "hotmail.com", "yahoo.co.th", "outlook.co.th"]),
        AMOUNT=amount_gen(["{n} บาท", "฿{dec}", "THB {n}", "{n}.00 บาท"]),
        CODE=lambda r: pick(r, [swift("TH")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Vietnamese
VI_AREA = [("Phường Bến Nghé", "Quận 1", "TP. Hồ Chí Minh"), ("Phường Võ Thị Sáu", "Quận 3", "TP. Hồ Chí Minh"),
           ("Phường Láng Hạ", "Quận Đống Đa", "Hà Nội"), ("Phường Tràng Tiền", "Quận Hoàn Kiếm", "Hà Nội"),
           ("Phường Hải Châu 1", "Quận Hải Châu", "Đà Nẵng")]


def vi_date(rng, y, m, dd):
    return pick(rng, [f"{dd:02d}/{m:02d}/{y}", f"{dd:02d}-{m:02d}-{y}", f"ngày {dd} tháng {m} năm {y}", f"{y}-{m:02d}-{dd:02d}"])


def vi_address(rng):
    ward, dist, city = pick(rng, VI_AREA)
    l1 = f"{pick(rng, ['Số ', '', 'Căn hộ ' + str(rng.randint(101, 2510)) + ', Tòa ' + up(rng, 1) + ', '])}{rng.randint(1, 300)} " \
         f"{pick(rng, ['Nguyễn Huệ', 'Lê Lợi', 'Trần Hưng Đạo', 'Láng Hạ', 'Hai Bà Trưng', 'Điện Biên Phủ'])}"
    return pick(rng, [[l1 + ",", f"{ward}, {dist},", city], [f"{l1}, {ward}, {dist}, {city}"]])


VI = dict(
    cues={"PERSON": ["Họ và tên", "Họ tên khách hàng", "Chủ tài khoản"], "DOB": ["Ngày sinh"],
          "ACCOUNT": ["Số tài khoản", "Số tài khoản chứng khoán"], "PHONE": ["Điện thoại", "Số di động", "Số điện thoại liên hệ"],
          "EMAIL": ["Email", "Thư điện tử"], "TIN": ["Mã số thuế", "MST"], "ADDRESS": ["Địa chỉ", "Địa chỉ thường trú", "Địa chỉ liên hệ"],
          "BUSINESS": ["Nơi làm việc", "Tên công ty"], "amount": ["Số tiền nộp ban đầu", "Thu nhập hằng năm", "Số dư"],
          "date": ["Ngày đăng ký", "Ngày sao kê", "Ngày giá trị"], "code": ["Mã SWIFT", "Mã chi nhánh"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["Đơn đăng ký mở tài khoản", "Phiếu thông tin khách hàng (KYC)", "Phiếu cập nhật thông tin khách hàng"],
            "statement": ["Báo cáo danh mục đầu tư", "Sao kê tài khoản hằng tháng"], "closing": ["Trân trọng,", "Kính thư,"],
            "job": ["Chuyên viên quan hệ khách hàng", "Giám đốc quản lý tài sản"]},
    prose=["Kính gửi {PERSON},\nCảm ơn Quý khách đã gặp chúng tôi vào ngày {DATE}. Như đã trao đổi, sau khi lệnh đăng ký mua được xác nhận, {AMOUNT} sẽ được chuyển vào tài khoản {ACCOUNT} của Quý khách. Nếu có thắc mắc, vui lòng gọi cho tôi theo số {PHONE}.",
           "Vui lòng chuyển {AMOUNT} từ tài khoản {ACCOUNT} của {PERSON} đến {BUSINESS} (SWIFT {CODE}) vào ngày {DATE}.",
           "Khách hàng {PERSON}, sinh ngày {DOB}, có khẩu vị rủi ro trung bình. Khách hàng đang làm việc tại {BUSINESS} và cư trú tại {ADDRESS}.",
           "Chúng tôi đã cập nhật địa chỉ liên hệ của {PERSON} thành {ADDRESS}. Từ ngày {DATE}, sao kê sẽ được gửi đến {EMAIL}.",
           "{PERSON} (mã số thuế {TIN}) đã yêu cầu bán lại {AMOUNT} từ danh mục {ACCOUNT}.",
           "Xin chào {PERSON}, Quý khách có thể liên hệ chuyên viên quan hệ khách hàng {PERSON2} qua số {PHONE} hoặc {EMAIL}."],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan', 'Vũ', 'Võ', 'Đặng', 'Bùi', 'Đỗ'])} "
                         f"{pick(r, ['Văn', 'Thị', 'Minh', 'Thanh', 'Quốc', 'Ngọc', 'Hữu', 'Đức'])} "
                         f"{pick(r, ['An', 'Bình', 'Hương', 'Lan', 'Dũng', 'Hùng', 'Mai', 'Tuấn', 'Linh', 'Phương', 'Nam', 'Trang', 'Khoa', 'Thảo', 'Long'])}",
        BUSINESS=lambda r: f"{pick(r, ['Công ty TNHH', 'Công ty Cổ phần'])} {pick(r, ['Đầu tư Sao Việt', 'Thương mại Hòa Bình', 'Phát triển Đông Á', 'Tài chính An Phát', 'Bất động sản Thành Công', 'Xuất nhập khẩu Minh Long'])}",
        ADDRESS=vi_address,
        DOB=date_gen(vi_date, *DOB_YEARS), DATE=date_gen(vi_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"09{d(r, 2)} {d(r, 3)} {d(r, 3)}", f"+84 9{d(r, 2)} {d(r, 3)} {d(r, 3)}", f"028 {d(r, 4)} {d(r, 4)}", f"09{d(r, 8)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 4)} {d(r, 4)} {d(r, 5)}", d(r, 14), d(r, 10)]),
        TIN=lambda r: pick(r, [d(r, 10), f"{d(r, 10)}-{d(r, 3)}"]),
        EMAIL=email_gen(["nguyen.an", "tranlan", "hoangdung", "pham.mai", "le.tuan", "vu.linh", "dang", "bui.nam"], ["gmail.com", "yahoo.com.vn", "vnn.vn", "outlook.com"]),
        AMOUNT=amount_gen(["{n} VNĐ", "{n} đồng", "VND {n}", "{n} ₫"], 10_000_000, 9_000_000_000, sep="."),
        CODE=lambda r: pick(r, [swift("VN")(r), d(r, 3)]),
    ),
)

# ---------------------------------------------------------------- Malay
MS_MON = ["Januari", "Februari", "Mac", "April", "Mei", "Jun", "Julai", "Ogos", "September", "Oktober", "November", "Disember"]
MS_AREA = [("50450", "Kuala Lumpur", "Wilayah Persekutuan"), ("47300", "Petaling Jaya", "Selangor"), ("10250", "George Town", "Pulau Pinang"),
           ("80000", "Johor Bahru", "Johor"), ("88000", "Kota Kinabalu", "Sabah")]


def ms_date(rng, y, m, dd):
    return pick(rng, [f"{dd:02d}/{m:02d}/{y}", f"{dd} {MS_MON[m-1]} {y}", f"{y}-{m:02d}-{dd:02d}", f"{dd}.{m:02d}.{y}"])


def ms_person(rng):
    k = rng.random()
    if k < 0.55:
        male = rng.random() < 0.5
        g = pick(rng, ["Ahmad", "Muhammad Faiz", "Hafiz", "Azman", "Zulkifli", "Amirul"] if male else ["Siti Aishah", "Nurul Huda", "Aminah", "Farah", "Noraini"])
        return f"{g} {'bin' if male else 'binti'} {pick(rng, ['Ismail', 'Abdullah', 'Hassan', 'Rahman', 'Yusof', 'Osman'])}"
    if k < 0.8:
        return f"{pick(rng, ['Tan', 'Lim', 'Wong', 'Lee', 'Ng', 'Chong'])} {pick(rng, ['Wei Ming', 'Mei Ling', 'Kah Wai', 'Chee Keong', 'Siew Lan'])}"
    return pick(rng, ["Ravi a/l Muthu", "Priya a/p Subramaniam", "Suresh a/l Krishnan", "Kavitha a/p Rajan"])


def ms_address(rng):
    pc, city, state = pick(rng, MS_AREA)
    l1 = f"{pick(rng, ['No. ' + str(rng.randint(1, 200)), 'Lot ' + str(rng.randint(1, 900)), f'{rng.randint(1, 30)}-{rng.randint(1, 20)}-{rng.randint(1, 12)}'])}, " \
         f"{pick(rng, ['Jalan Ampang', 'Jalan Tun Razak', 'Jalan SS2/24', 'Jalan Bukit Bintang', 'Persiaran Gurney'])}"
    return pick(rng, [[l1 + ",", f"{pc} {city},", state], [f"{l1}, {pc} {city}, {state}"]])


MS = dict(
    cues={"PERSON": ["Nama", "Nama Penuh", "Nama Pemegang Akaun"], "DOB": ["Tarikh Lahir"], "ACCOUNT": ["No. Akaun", "Nombor Akaun"],
          "PHONE": ["No. Telefon", "Telefon Bimbit"], "EMAIL": ["E-mel", "Alamat E-mel"], "TIN": ["No. Cukai Pendapatan", "Nombor Pengenalan Cukai (TIN)"],
          "ADDRESS": ["Alamat", "Alamat Kediaman", "Alamat Surat-menyurat"], "BUSINESS": ["Majikan", "Nama Syarikat"],
          "amount": ["Deposit Permulaan", "Pendapatan Tahunan", "Baki"], "date": ["Tarikh Permohonan", "Tarikh Penyata", "Tarikh Nilai"],
          "code": ["Kod SWIFT", "Kod Cawangan"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["Borang Pembukaan Akaun", "Borang Kenali Pelanggan Anda (KYC)", "Borang Kemas Kini Maklumat Pelanggan"],
            "statement": ["Penyata Portfolio", "Penyata Akaun Bulanan"], "closing": ["Yang benar,", "Sekian, terima kasih."],
            "job": ["Pengurus Perhubungan", "Penasihat Kekayaan Kanan"]},
    prose=["Tuan/Puan {PERSON},\nTerima kasih kerana bertemu dengan kami pada {DATE}. Seperti yang dibincangkan, {AMOUNT} akan dipindahkan ke akaun {ACCOUNT} anda sebaik sahaja langganan disahkan. Sila hubungi saya di {PHONE} jika ada sebarang pertanyaan.",
           "Sila pindahkan {AMOUNT} daripada akaun {ACCOUNT} milik {PERSON} kepada {BUSINESS} (SWIFT {CODE}) pada {DATE}.",
           "Pelanggan {PERSON}, dilahirkan pada {DOB}, mempunyai profil risiko sederhana. Beliau bekerja di {BUSINESS} dan menetap di {ADDRESS}.",
           "Kami telah mengemas kini alamat surat-menyurat {PERSON} kepada {ADDRESS}. Penyata akan dihantar ke {EMAIL} mulai {DATE}.",
           "{PERSON} (No. Cukai {TIN}) telah memohon penebusan sebanyak {AMOUNT} daripada portfolio {ACCOUNT}.",
           "Hai {PERSON}, pengurus perhubungan anda, {PERSON2}, boleh dihubungi di {PHONE} atau {EMAIL}."],
    gen=dict(
        PERSON=ms_person,
        BUSINESS=lambda r: f"{pick(r, ['Mega Jaya', 'Sinar Harapan', 'Bumi Makmur', 'Teratai Holdings', 'Seri Murni Capital', 'Cahaya Timur Trading'])} {pick(r, ['Sdn. Bhd.', 'Sdn Bhd', 'Berhad'])}",
        ADDRESS=ms_address,
        DOB=date_gen(ms_date, *DOB_YEARS), DATE=date_gen(ms_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+60 1{d(r, 1)}-{d(r, 3)} {d(r, 4)}", f"01{d(r, 1)}-{d(r, 3)} {d(r, 4)}", f"03-{d(r, 4)} {d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 10), f"{d(r, 2)}-{d(r, 6)}-{d(r, 4)}"]),
        TIN=lambda r: pick(r, [f"IG{d(r, 11)}", f"SG {d(r, 11)}", f"OG{d(r, 10)}"]),
        EMAIL=email_gen(["ahmad", "siti", "nurul.huda", "tan.weiming", "azman", "farah", "ravi", "lim"], ["gmail.com", "yahoo.com.my", "hotmail.my", "outlook.com"]),
        AMOUNT=amount_gen(["RM{dec}", "RM {n}", "MYR {n}"]),
        CODE=lambda r: pick(r, [swift("MY")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Indonesian
ID_MON = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
ID_AREA = [("Kel. Senayan", "Kec. Kebayoran Baru", "Jakarta Selatan", "12190"), ("Kel. Menteng", "Kec. Menteng", "Jakarta Pusat", "10310"),
           ("Kel. Dago", "Kec. Coblong", "Bandung", "40135"), ("Kel. Gubeng", "Kec. Gubeng", "Surabaya", "60281"), ("Kel. Kuta", "Kec. Kuta", "Badung", "80361")]


def id_date(rng, y, m, dd):
    return pick(rng, [f"{dd:02d}-{m:02d}-{y}", f"{dd} {ID_MON[m-1]} {y}", f"{dd:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{dd:02d}"])


def id_address(rng):
    kel, kec, city, pc = pick(rng, ID_AREA)
    l1 = f"{pick(rng, ['Jl. Jend. Sudirman', 'Jl. M.H. Thamrin', 'Jl. Gatot Subroto', 'Jl. Diponegoro', 'Jl. Asia Afrika'])} " \
         f"{pick(rng, ['No. ' + str(rng.randint(1, 200)), 'Kav. ' + str(rng.randint(1, 90))])}"
    rt = f"RT {rng.randint(1, 15):03d}/RW {rng.randint(1, 12):03d}"
    return pick(rng, [[l1 + ",", f"{rt}, {kel}, {kec},", f"{city} {pc}"], [f"{l1}, {kel}, {kec}, {city} {pc}"]])


ID = dict(
    cues={"PERSON": ["Nama", "Nama Lengkap", "Nama Pemilik Rekening"], "DOB": ["Tanggal Lahir"],
          "ACCOUNT": ["No. Rekening", "Nomor Rekening", "Nomor Rekening Efek"], "PHONE": ["No. Telepon", "No. HP"],
          "EMAIL": ["Email", "Alamat Email"], "TIN": ["NPWP", "Nomor Pokok Wajib Pajak"], "ADDRESS": ["Alamat", "Alamat Domisili", "Alamat Surat Menyurat"],
          "BUSINESS": ["Nama Perusahaan", "Tempat Bekerja"], "amount": ["Setoran Awal", "Penghasilan per Tahun", "Saldo"],
          "date": ["Tanggal Pengajuan", "Tanggal Laporan", "Tanggal Valuta"], "code": ["Kode SWIFT", "Kode Cabang"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["Formulir Pembukaan Rekening", "Formulir Prinsip Mengenal Nasabah (KYC)", "Formulir Pembaruan Data Nasabah"],
            "statement": ["Laporan Portofolio", "Rekening Koran Bulanan"], "closing": ["Hormat kami,", "Salam hangat,"],
            "job": ["Relationship Manager", "Penasihat Keuangan Senior"]},
    prose=["Yth. Bapak/Ibu {PERSON},\nTerima kasih telah bertemu dengan kami pada {DATE}. Sesuai pembahasan, dana sebesar {AMOUNT} akan ditransfer ke rekening {ACCOUNT} setelah pemesanan dikonfirmasi. Silakan hubungi saya di {PHONE} jika ada pertanyaan.",
           "Mohon transfer {AMOUNT} dari rekening {ACCOUNT} atas nama {PERSON} ke {BUSINESS} (SWIFT {CODE}) pada tanggal {DATE}.",
           "Nasabah {PERSON}, lahir pada {DOB}, memiliki profil risiko moderat. Nasabah bekerja di {BUSINESS} dan berdomisili di {ADDRESS}.",
           "Kami telah memperbarui alamat surat menyurat {PERSON} menjadi {ADDRESS}. Mulai {DATE}, laporan akan dikirim ke {EMAIL}.",
           "{PERSON} (NPWP {TIN}) mengajukan pencairan sebesar {AMOUNT} dari portofolio {ACCOUNT}.",
           "Halo {PERSON}, relationship manager Anda, {PERSON2}, dapat dihubungi di {PHONE} atau {EMAIL}."],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['Budi', 'Siti', 'Agus', 'Dewi', 'Andi', 'Rina', 'Hendra', 'Sri', 'Joko', 'Putri', 'Bambang', 'Ayu', 'Rudi', 'Indah', 'Fajar'])} "
                         f"{pick(r, ['Santoso', 'Rahayu', 'Setiawan', 'Lestari', 'Wijaya', 'Kartika', 'Gunawan', 'Wahyuni', 'Susilo', 'Maharani', 'Hartono', 'Pratiwi', 'Hidayat', 'Nugroho'])}",
        BUSINESS=lambda r: f"PT {pick(r, ['Sinar Abadi Jaya', 'Maju Bersama Sejahtera', 'Nusantara Investama', 'Cahaya Gemilang', 'Mitra Karya Utama', 'Bumi Raya Persada'])}{pick(r, ['', ' Tbk'])}",
        ADDRESS=id_address,
        DOB=date_gen(id_date, *DOB_YEARS), DATE=date_gen(id_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+62 8{d(r, 2)}-{d(r, 4)}-{d(r, 4)}", f"08{d(r, 2)} {d(r, 4)} {d(r, 4)}", f"(021) {d(r, 4)} {d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 3)}-{d(r, 4)}", d(r, 10), f"{d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 3)}"]),
        TIN=lambda r: f"{d(r, 2)}.{d(r, 3)}.{d(r, 3)}.{d(r, 1)}-{d(r, 3)}.{d(r, 3)}",
        EMAIL=email_gen(["budi", "siti.rahayu", "agus", "dewi", "andi.wijaya", "rina", "hendra", "putri"], ["gmail.com", "yahoo.co.id", "outlook.co.id", "telkom.net"]),
        AMOUNT=amount_gen(["Rp {n}", "Rp{n},00", "IDR {n}"], 1_000_000, 9_000_000_000, sep="."),
        CODE=lambda r: pick(r, [swift("ID")(r), d(r, 4)]),
    ),
)

# ---------------------------------------------------------------- Tagalog / Filipino
TL_MON = ["Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo", "Hulyo", "Agosto", "Setyembre", "Oktubre", "Nobyembre", "Disyembre"]
TL_AREA = [("Makati City", "1223", "Metro Manila"), ("Pasig City", "1605", "Metro Manila"), ("Quezon City", "1100", "Metro Manila"),
           ("Cebu City", "6000", "Cebu"), ("Davao City", "8000", "Davao del Sur")]


def tl_date(rng, y, m, dd):
    return pick(rng, [f"{TL_MON[m-1]} {dd}, {y}", f"{dd:02d}/{m:02d}/{y}", f"{EN_MON[m-1]} {dd}, {y}", f"{y}-{m:02d}-{dd:02d}"])


def tl_address(rng):
    city, pc, region = pick(rng, TL_AREA)
    l1 = f"{pick(rng, ['Unit ' + str(rng.randint(101, 3500)), f'Blk {rng.randint(1, 40)} Lot {rng.randint(1, 60)}', str(rng.randint(1, 900))])} " \
         f"{pick(rng, ['Ayala Avenue', 'Ortigas Avenue', 'Rizal Street', 'Mabini Street', 'Osmeña Boulevard'])}"
    brgy = pick(rng, ["Brgy. San Lorenzo", "Brgy. Poblacion", "Brgy. Kapitolyo", "Brgy. Lahug"])
    return pick(rng, [[l1 + ",", f"{brgy}, {city}", f"{pc} {region}"], [f"{l1}, {brgy}, {city}, {pc} {region}"]])


TL = dict(
    cues={"PERSON": ["Pangalan", "Buong Pangalan", "Pangalan ng May-ari ng Account"], "DOB": ["Petsa ng Kapanganakan", "Kaarawan"],
          "ACCOUNT": ["Numero ng Account", "Account No."], "PHONE": ["Numero ng Telepono", "Mobile No."], "EMAIL": ["Email", "Email Address"],
          "TIN": ["TIN", "Tax Identification Number"], "ADDRESS": ["Tirahan", "Address", "Kasalukuyang Tirahan"],
          "BUSINESS": ["Pinagtatrabahuhan", "Pangalan ng Kumpanya"], "amount": ["Paunang Deposito", "Taunang Kita", "Balanse"],
          "date": ["Petsa ng Aplikasyon", "Petsa ng Statement", "Petsa ng Halaga"], "code": ["SWIFT Code", "Branch Code"]},
    seps=[": ", ": ", " - ", "\t"],
    titles={"kyc": ["Form sa Pagbubukas ng Account", "KYC Form (Kilalanin ang Iyong Kliyente)", "Form sa Pag-update ng Impormasyon"],
            "statement": ["Pahayag ng Portfolio", "Buwanang Statement ng Account"], "closing": ["Lubos na gumagalang,", "Maraming salamat,"],
            "job": ["Relationship Manager", "Senior Wealth Adviser"]},
    prose=["Mahal na {PERSON},\nSalamat sa pakikipagkita sa amin noong {DATE}. Gaya ng napag-usapan, ililipat ang {AMOUNT} sa iyong account {ACCOUNT} kapag nakumpirma na ang subscription. Mangyaring tawagan ako sa {PHONE} kung may tanong ka.",
           "Pakilipat ang {AMOUNT} mula sa account {ACCOUNT} ni {PERSON} papunta sa {BUSINESS} (SWIFT {CODE}) sa {DATE}.",
           "Ang kliyenteng si {PERSON}, ipinanganak noong {DOB}, ay may katamtamang risk profile. Nagtatrabaho siya sa {BUSINESS} at nakatira sa {ADDRESS}.",
           "Na-update na namin ang tirahan ni {PERSON} sa {ADDRESS}. Simula {DATE}, ipapadala ang mga statement sa {EMAIL}.",
           "Humiling si {PERSON} (TIN {TIN}) ng redemption na {AMOUNT} mula sa portfolio {ACCOUNT}.",
           "Kumusta {PERSON}, maaari mong kontakin ang iyong relationship manager na si {PERSON2} sa {PHONE} o {EMAIL}."],
    gen=dict(
        PERSON=lambda r: f"{pick(r, ['Juan', 'Maria', 'Jose', 'Ana', 'Mark Anthony', 'Kristine', 'Rodel', 'Mary Grace', 'Jerome', 'Liza', 'Ramon', 'Cristina', 'Paolo', 'Angelica'])} "
                         f"{pick(r, ['', 'A. ', 'B. ', 'C. ', 'M. ', 'R. ', 'S. '])}"
                         f"{pick(r, ['Dela Cruz', 'Santos', 'Reyes', 'Garcia', 'Mendoza', 'Bautista', 'Ramos', 'Villanueva', 'Aquino', 'Castillo', 'Gonzales', 'Navarro', 'Torres'])}",
        BUSINESS=lambda r: f"{pick(r, ['Mabuhay Capital', 'Pilipinas Trading', 'Bagong Araw Realty', 'Luzon Ventures', 'Bayanihan Finance', 'Kalayaan Industries'])} {pick(r, ['Inc.', 'Corporation', 'Holdings, Inc.'])}",
        ADDRESS=tl_address,
        DOB=date_gen(tl_date, *DOB_YEARS), DATE=date_gen(tl_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"+63 9{d(r, 2)} {d(r, 3)} {d(r, 4)}", f"09{d(r, 2)}-{d(r, 3)}-{d(r, 4)}", f"(02) 8{d(r, 3)} {d(r, 4)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 4)}-{d(r, 4)}-{d(r, 2)}", d(r, 12), f"{d(r, 3)}-{d(r, 6)}-{d(r, 3)}"]),
        TIN=lambda r: f"{d(r, 3)}-{d(r, 3)}-{d(r, 3)}-{d(r, 3)}",
        EMAIL=email_gen(["juan.delacruz", "maria", "jsantos", "ana.reyes", "kristine", "rodel", "liza", "paolo"], ["gmail.com", "yahoo.com.ph", "outlook.ph", "globe.com.ph"]),
        AMOUNT=amount_gen(["₱{dec}", "PHP {n}", "₱{n}"]),
        CODE=lambda r: pick(r, [swift("PH")(r), d(r, 3)]),
    ),
)


def zh_hant():
    """zh-Hans text converted with OpenCC (Taiwan phrasing); TW / HK value pools."""
    import opencc
    cc = opencc.OpenCC("s2twp")
    conv = lambda x: cc.convert(x)
    lang = dict(
        cues={k: [conv(c) for c in v] for k, v in ZH["cues"].items()},
        seps=ZH["seps"],
        titles={k: [conv(c) for c in v] for k, v in ZH["titles"].items()},
        prose=[conv(p) for p in ZH["prose"]],
        gen=dict(ZH["gen"]),
    )
    lang["gen"]["PERSON"] = lambda r: conv(ZH["gen"]["PERSON"](r))
    lang["gen"]["BUSINESS"] = lambda r: conv(ZH["gen"]["BUSINESS"](r).replace("上海", "臺北").replace("北京", "香港"))
    lang["gen"].update(ZHT_GEN)
    return lang


def languages() -> dict:
    return {"en": EN, "zh-Hans": ZH, "zh-Hant": zh_hant(), "ja": JA, "ko": KO, "hi": HI, "ar": AR,
            "th": TH, "vi": VI, "ms": MS, "id": ID, "tl": TL}
