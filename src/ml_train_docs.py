"""Generated financial training documents for languages without public financial PII data.

Used only where no public source exists (configs/ml_sources.yaml, kind: generated):
  th  no public Thai PII data at all (ThaiNER has names / organisations only)
  ar  no public Arabic tax IDs; few dates of birth (SITR)
  hi  no public Devanagari dates of birth / accounts / tax IDs (AI4Privacy hi entities are
      romanised; Maskara "hi-IN" is Indian English)

TRAINING ONLY. The prose templates, titles and the name / company / street / city pools here
are disjoint from the test-only src/ml_templates.py (tests/test_ml_train.py checks this); form
field cues are ordinary vocabulary and may coincide. The document builders (forms, statements,
signatures, prose) are reused from build_ml_eval.py.
"""
import random

from build_ml_eval import DOC_TYPES, Doc
from ml_templates import (ARABIC_INDIC, DEVANAGARI, DOB_YEARS, DATE_YEARS, amount_gen, ar_date, d,
                          date_gen, digits_to, email_gen, hi_date, pick, swift, th_date, up)

# ---------------------------------------------------------------- Thai
TH_GIVEN = ["อภิชาติ", "จิราพร", "ชัยวัฒน์", "ปวีณา", "ธีรพงษ์", "สุภาวดี", "เอกชัย", "นันทนา", "วรวุฒิ", "รัตนา",
            "สมศักดิ์", "อัญชลี", "ภานุวัฒน์", "ศิริพร", "ณรงค์", "มาลัย", "กฤษดา", "เบญจวรรณ", "ปริญญา", "อารีรัตน์"]
TH_FAMILY = ["สิทธิโชค", "เจริญสุข", "ประเสริฐวงศ์", "ทองดี", "ศรีอ่อน", "บุญเรือง", "แสงทอง", "อินทร์แก้ว",
             "ชัยมงคล", "วัฒนากุล", "พรหมมา", "สุขเจริญ", "นาคสวัสดิ์", "กิตติวงศ์", "รุ่งเรืองศรี"]
TH_COMPANY = ["ทรัพย์มั่นคงลิสซิ่ง", "อุดมทรัพย์แคปปิตอล", "เจ้าพระยาอินเตอร์เทรด", "ไทยพาณิชย์การค้า", "ล้านนาโฮลดิ้ง",
              "อันดามันแอสเซท", "สุวรรณภูมิฟินันซ์", "อีสานรุ่งเรืองกรุ๊ป", "ศรีราชาพร็อพเพอร์ตี้"]
TH_AREA = [("แขวงลุมพินี", "เขตปทุมวัน", "กรุงเทพมหานคร", "10330"), ("แขวงจอมพล", "เขตจตุจักร", "กรุงเทพมหานคร", "10900"),
           ("แขวงบางนาเหนือ", "เขตบางนา", "กรุงเทพมหานคร", "10260"), ("ตำบลในเมือง", "อำเภอเมืองขอนแก่น", "จังหวัดขอนแก่น", "40000"),
           ("ตำบลบ่อผุด", "อำเภอเกาะสมุย", "จังหวัดสุราษฎร์ธานี", "84320"), ("ตำบลหาดใหญ่", "อำเภอหาดใหญ่", "จังหวัดสงขลา", "90110"),
           ("ตำบลในเมือง", "อำเภอเมืองนครราชสีมา", "จังหวัดนครราชสีมา", "30000")]
TH_ROAD = ["ถนนวิทยุ", "ถนนลาดพร้าว", "ถนนบางนา-ตราด", "ถนนมิตรภาพ", "ถนนศรีนครินทร์", "ถนนเจริญกรุง", "ถนนนิมมานเหมินท์"]


def th_address(rng):
    sub, dist, prov, pc = pick(rng, TH_AREA)
    l1 = (f"{rng.randint(1, 899)}/{rng.randint(1, 250)} "
          f"{pick(rng, ['', 'หมู่บ้านพฤกษา ', 'คอนโดศุภาลัย ห้อง ' + str(rng.randint(101, 3020)) + ' ', 'อาคารเอ็มไพร์ ชั้น ' + str(rng.randint(2, 50)) + ' '])}"
          f"{pick(rng, ['ซอย ' + str(rng.randint(1, 80)) + ' ', ''])}{pick(rng, TH_ROAD)}")
    return pick(rng, [[l1, f"{sub} {dist}", f"{prov} {pc}"], [f"{l1} {sub} {dist} {prov} {pc}"], [l1, f"{sub} {dist} {prov}", pc]])


TH = dict(
    cues={"PERSON": ["ชื่อผู้ขอเปิดบัญชี", "ชื่อผู้ลงทุน", "ผู้รับประโยชน์", "ชื่อ-สกุล"],
          "DOB": ["วันเดือนปีเกิด", "เกิดวันที่", "วันเกิด"],
          "ACCOUNT": ["บัญชีเงินฝากเลขที่", "เลขที่บัญชีหลักทรัพย์", "หมายเลขบัตรเครดิต", "เลขที่สัญญา/บัญชี"],
          "PHONE": ["มือถือ", "เบอร์โทรศัพท์", "โทร."], "EMAIL": ["อีเมล", "E-mail"],
          "TIN": ["เลขประจำตัวผู้เสียภาษีอากร", "เลขที่ผู้เสียภาษี", "Tax ID"],
          "ADDRESS": ["ที่อยู่ปัจจุบัน", "ที่อยู่ที่ติดต่อได้", "ที่อยู่ที่ทำงาน"],
          "BUSINESS": ["นายจ้าง", "บริษัทผู้ว่าจ้าง", "ชื่อนิติบุคคล"],
          "amount": ["วงเงินกู้", "มูลค่าการลงทุน", "ยอดเงินโอน", "เงินเดือน"],
          "date": ["วันที่ทำรายการ", "วันครบกำหนด", "ณ วันที่"], "code": ["รหัส SWIFT", "รหัสธนาคาร", "รหัสอ้างอิง"]},
    seps=[": ", " : ", " ", "\t"],
    titles={"kyc": ["คำขอเปิดบัญชีซื้อขายหลักทรัพย์", "แบบประเมินความเหมาะสมในการลงทุน", "แบบฟอร์มข้อมูลผู้ลงทุน"],
            "statement": ["ใบแจ้งยอดบัญชีเงินฝาก", "รายงานมูลค่าทรัพย์สินสุทธิ", "ใบแจ้งหนี้บัตรเครดิต"],
            "closing": ["ด้วยความนับถือ", "ขอบพระคุณค่ะ", "ขอบคุณครับ"],
            "job": ["เจ้าหน้าที่บริหารความสัมพันธ์", "ผู้แนะนำการลงทุน", "ผู้จัดการสาขา"]},
    prose=["ถึง คุณ{PERSON}\nตามที่ท่านแจ้งความประสงค์เมื่อวันที่ {DATE} ธนาคารได้ดำเนินการโอนเงินจำนวน {AMOUNT} เข้าบัญชีเลขที่ {ACCOUNT} เรียบร้อยแล้ว หากมีข้อสงสัยโปรดติดต่อ {PHONE}",
           "ผู้ลงทุน {PERSON} เลขประจำตัวผู้เสียภาษี {TIN} เกิดเมื่อ {DOB} ประสงค์ลงทุนในกองทุนรวมผ่านบัญชี {ACCOUNT}",
           "บริษัท {BUSINESS} ขอแจ้งว่าได้ชำระเงินเดือนให้แก่ {PERSON} จำนวน {AMOUNT} ผ่านธนาคาร (SWIFT {CODE}) เมื่อวันที่ {DATE}",
           "ขอแจ้งเปลี่ยนแปลงที่อยู่ของคุณ{PERSON} จากเดิมเป็น {ADDRESS} โดยมีผลตั้งแต่วันที่ {DATE} และขอให้ส่งเอกสารทางอีเมล {EMAIL}",
           "คุณ{PERSON} (วันเกิด {DOB}) ทำงานกับ {BUSINESS} ระดับความเสี่ยงที่ยอมรับได้อยู่ในระดับสูง ติดต่อได้ที่ {PHONE} หรือ {EMAIL}",
           "รายการนี้เป็นการชำระค่าบัตรเครดิตหมายเลข {ACCOUNT} ของ {PERSON} จำนวน {AMOUNT} ครบกำหนดชำระวันที่ {DATE}",
           "เรียน คุณ{PERSON2}\nผู้ลงทุนชื่อ {PERSON} ซึ่งอาศัยอยู่ที่ {ADDRESS} ได้ยื่นแบบประเมินความเสี่ยงแล้ว เลขผู้เสียภาษี {TIN}",
           "สัญญาสินเชื่อระหว่าง {BUSINESS} กับ {PERSON} วงเงิน {AMOUNT} ผู้กู้เกิดวันที่ {DOB} โทร. {PHONE}"],
    gen=dict(
        PERSON=lambda r: f"{pick(r, TH_GIVEN)} {pick(r, TH_FAMILY)}",
        BUSINESS=lambda r: f"บริษัท {pick(r, TH_COMPANY)} จำกัด{pick(r, ['', '', ' (มหาชน)'])}",
        ADDRESS=th_address,
        DOB=date_gen(th_date, *DOB_YEARS), DATE=date_gen(th_date, *DATE_YEARS),
        PHONE=lambda r: pick(r, [f"09{d(r, 1)}-{d(r, 3)}-{d(r, 4)}", f"06{d(r, 1)} {d(r, 3)} {d(r, 4)}", f"+66 9{d(r, 1)}-{d(r, 3)}-{d(r, 4)}", f"02 {d(r, 3)} {d(r, 4)}", f"0{d(r, 9)}"]),
        ACCOUNT=lambda r: pick(r, [f"{d(r, 3)}-{d(r, 1)}-{d(r, 5)}-{d(r, 1)}", f"{d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 10), f"{d(r, 3)}-{d(r, 7)}"]),
        TIN=lambda r: pick(r, [f"{r.randint(1, 8)} {d(r, 4)} {d(r, 5)} {d(r, 2)} {d(r, 1)}", f"0{d(r, 12)}", f"{d(r, 1)}-{d(r, 4)}-{d(r, 5)}-{d(r, 2)}-{d(r, 1)}"]),
        EMAIL=email_gen(["apichat", "jiraporn", "chaiwat", "paweena", "theerapong", "s.charoensuk", "ekachai", "rattana"],
                        ["gmail.com", "hotmail.co.th", "outlook.com", "icloud.com", "kbank.co.th"]),
        AMOUNT=amount_gen(["{n} บาท", "{dec} บาท", "THB {dec}", "฿{n}"]),
        CODE=lambda r: pick(r, [swift("TH")(r), d(r, 3), "REF" + d(r, 8)]),
    ),
)

# ---------------------------------------------------------------- Arabic
AR_GIVEN = ["سلطان", "منى", "طارق", "رنا", "ماجد", "دانة", "فهد", "لطيفة", "جاسم", "شيخة", "بدر", "ريم", "زياد", "أسماء", "وليد"]
AR_MID = ["ناصر", "حمد", "عبدالعزيز", "محمود", "صالح", "خليفة"]
AR_FAMILY = ["البلوشي", "الظاهري", "المهيري", "الغامدي", "الزهراني", "الحربي", "الرميثي", "الكواري", "الأنصاري", "الفلاسي", "الشحي", "العنزي"]
AR_COMPANY = ["الصقر", "الندى", "البيان", "المرجان", "السنابل", "الوفاق", "الأمانة", "الراية"]
AR_CITY = [("العين", "الإمارات العربية المتحدة"), ("رأس الخيمة", "الإمارات العربية المتحدة"), ("الدمام", "المملكة العربية السعودية"),
           ("مكة المكرمة", "المملكة العربية السعودية"), ("مسقط", "سلطنة عمان"), ("مدينة الكويت", "الكويت"), ("الخبر", "المملكة العربية السعودية")]


def ar_address(rng):
    city, country = pick(rng, AR_CITY)
    l1 = f"{pick(rng, ['منزل رقم', 'وحدة', 'عمارة'])} {rng.randint(1, 900)}، {pick(rng, ['شارع الملك سلمان', 'شارع الأمير سلطان', 'طريق المطار', 'شارع خالد بن الوليد', 'شارع الاتحاد'])}"
    l2 = f"{pick(rng, ['حي الروضة', 'حي النزهة', 'حي السلامة', 'منطقة الخوير', 'حي الشاطئ'])}، {city}{pick(rng, ['', ' ' + d(rng, 5), '، الرمز البريدي ' + d(rng, 5)])}"
    return pick(rng, [[l1 + "،", l2, country], [f"{l1}، {l2}، {country}"], [l1, l2]])


AR = dict(
    cues={"PERSON": ["اسم المستثمر", "اسم المقترض", "المستفيد", "الاسم الثلاثي"], "DOB": ["تاريخ الولادة", "تاريخ الميلاد", "مواليد"],
          "ACCOUNT": ["رقم الحساب البنكي", "رقم بطاقة الائتمان", "رقم الحساب الاستثماري", "آيبان"],
          "PHONE": ["رقم الجوال", "هاتف", "للتواصل"], "EMAIL": ["البريد الإلكتروني", "الإيميل"],
          "TIN": ["رقم التسجيل الضريبي", "الرقم الضريبي للمكلف", "TRN"], "ADDRESS": ["عنوان المراسلة", "مقر الإقامة", "العنوان الوطني"],
          "BUSINESS": ["الجهة الموظفة", "اسم المنشأة"], "amount": ["قيمة القرض", "مبلغ التحويل", "الراتب الشهري", "قيمة الاستثمار"],
          "date": ["تاريخ العملية", "تاريخ الاستحقاق", "تاريخ الإصدار"], "code": ["رمز البنك", "SWIFT", "الرقم المرجعي"]},
    seps=[": ", " : ", " - ", "\t"],
    titles={"kyc": ["استمارة فتح حساب استثماري", "نموذج تقييم ملاءمة الاستثمار", "طلب تمويل شخصي"],
            "statement": ["كشف حساب جاري", "بيان مركز المحفظة", "كشف بطاقة الائتمان"],
            "closing": ["مع التحية،", "شاكرين تعاونكم،", "مع أطيب التمنيات،"],
            "job": ["مسؤول خدمة العملاء", "مستشار الاستثمار", "مدير الفرع"]},
    prose=["السيد/ة {PERSON} المحترم/ة،\nنفيدكم بأنه تم تحويل مبلغ {AMOUNT} إلى حسابكم رقم {ACCOUNT} بتاريخ {DATE}. للاستفسار يرجى الاتصال على {PHONE}.",
           "المستثمر {PERSON}، تاريخ الميلاد {DOB}، الرقم الضريبي {TIN}، يرغب في الاكتتاب عبر الحساب {ACCOUNT}.",
           "تفيد {BUSINESS} بأن السيد {PERSON} يعمل لديها ويتقاضى راتباً شهرياً قدره {AMOUNT}، ويتم تحويله عبر البنك (SWIFT {CODE}).",
           "تم تحديث عنوان العميل {PERSON} ليصبح {ADDRESS}، وسترسل الإشعارات إلى {EMAIL} ابتداءً من {DATE}.",
           "العميلة {PERSON} من مواليد {DOB}، تعمل في {BUSINESS}، ويمكن التواصل معها عبر {PHONE} أو {EMAIL}.",
           "سداد مستحقات البطاقة رقم {ACCOUNT} الخاصة بالعميل {PERSON} بمبلغ {AMOUNT} قبل تاريخ {DATE}.",
           "إلى {PERSON2}،\nتقدم {PERSON} المقيم في {ADDRESS} بطلب تمويل، ورقم تسجيله الضريبي {TIN}.",
           "عقد تمويل بين {BUSINESS} والسيد {PERSON} بقيمة {AMOUNT}، تاريخ ميلاد المتعامل {DOB}، هاتف {PHONE}."],
    gen=dict(
        PERSON=lambda r: pick(r, ["{g} {f}", "{g} {m} {f}"]).format(g=pick(r, AR_GIVEN), m=pick(r, AR_MID), f=pick(r, AR_FAMILY)),
        BUSINESS=lambda r: f"{pick(r, ['مؤسسة', 'شركة', 'مجموعة'])} {pick(r, AR_COMPANY)} {pick(r, ['للتمويل', 'للصرافة', 'للمقاولات', 'للتأمين', 'التجارية'])}{pick(r, ['', ' ذ.م.م', ' المحدودة'])}",
        ADDRESS=ar_address,
        DOB=date_gen(ar_date, *DOB_YEARS), DATE=date_gen(ar_date, *DATE_YEARS),
        PHONE=lambda r: (lambda s: digits_to(s, ARABIC_INDIC) if r.random() < 0.2 else s)(
            pick(r, [f"+968 9{d(r, 3)} {d(r, 4)}", f"+965 {d(r, 4)} {d(r, 4)}", f"055{d(r, 7)}", f"+971-2-{d(r, 3)}-{d(r, 4)}", f"013 {d(r, 3)} {d(r, 4)}"])),
        ACCOUNT=lambda r: pick(r, [f"SA{d(r, 22)}", f"AE{d(r, 2)}{d(r, 19)}", f"{d(r, 4)} {d(r, 4)} {d(r, 4)} {d(r, 4)}", d(r, 12)]),
        TIN=lambda r: (lambda s: digits_to(s, ARABIC_INDIC) if r.random() < 0.15 else s)(pick(r, [f"3{d(r, 14)}", f"100 {d(r, 4)} {d(r, 4)} {d(r, 3)} 3", f"OM{d(r, 10)}"])),
        EMAIL=email_gen(["sultan", "mona", "tariq", "rana.m", "majid", "dana", "fahad", "alblooshi", "reem"],
                        ["gmail.com", "yahoo.com", "outlook.com", "omantel.net.om", "stc.com.sa"]),
        AMOUNT=lambda r: (lambda s: digits_to(s, ARABIC_INDIC) if r.random() < 0.2 else s)(amount_gen(["{n} ر.س", "{n} د.إ", "KWD {dec}", "{n} ريال عماني"])(r)),
        CODE=lambda r: pick(r, [swift("SA")(r), swift("OM")(r), d(r, 6)]),
    ),
)

# ---------------------------------------------------------------- Hindi (Devanagari)
HI_GIVEN = ["आलोक", "रेखा", "नितिन", "स्मिता", "गौरव", "मीनाक्षी", "हरीश", "शालिनी", "प्रकाश", "ज्योति", "विनोद", "अंजलि", "रोहित", "सविता", "तरुण"]
HI_FAMILY = ["त्रिपाठी", "श्रीवास्तव", "पांडे", "दुबे", "सक्सेना", "भट्ट", "ठाकुर", "चतुर्वेदी", "बंसल", "खन्ना", "मल्होत्रा", "देशमुख"]
HI_COMPANY = ["शुभलक्ष्मी", "हिमालय", "विश्वास", "प्रगति", "उत्कर्ष", "नर्मदा", "सरस्वती"]
HI_CITY = [("भोपाल", "मध्य प्रदेश", "462001"), ("पटना", "बिहार", "800001"), ("चंडीगढ़", "पंजाब", "160017"), ("इंदौर", "मध्य प्रदेश", "452001"),
           ("नागपुर", "महाराष्ट्र", "440001"), ("वाराणसी", "उत्तर प्रदेश", "221001"), ("देहरादून", "उत्तराखंड", "248001")]


def hi_address(rng):
    city, state, pin = pick(rng, HI_CITY)
    l1 = f"{pick(rng, ['प्लॉट नं. ', 'भवन सं. ', 'क्वार्टर नं. '])}{rng.randint(1, 600)}, {pick(rng, ['आनंद विहार', 'कृष्णा कॉलोनी', 'सिविल लाइंस', 'राजेंद्र नगर', 'अशोक नगर'])}"
    l2 = pick(rng, ["महात्मा गांधी मार्ग", "कॉलेज रोड", "सदर बाज़ार", "रिंग रोड"])
    return pick(rng, [[l1 + ",", l2 + ",", f"{city} {pin}", state], [f"{l1}, {l2}, {city}, {state} - {pin}"], [l1, f"{city} ({state}) {pin}"]])


HI = dict(
    cues={"PERSON": ["निवेशक का नाम", "आवेदक का नाम", "लाभार्थी", "पूरा नाम"], "DOB": ["जन्म की तारीख", "जन्म दिनांक", "जन्मतिथि"],
          "ACCOUNT": ["बैंक खाता संख्या", "क्रेडिट कार्ड संख्या", "फोलियो संख्या", "बचत खाता नं."],
          "PHONE": ["मोबाइल", "दूरभाष", "फ़ोन"], "EMAIL": ["ई-मेल", "ईमेल आईडी"],
          "TIN": ["पैन कार्ड संख्या", "स्थायी खाता संख्या (पैन)", "PAN"], "ADDRESS": ["स्थायी पता", "वर्तमान पता", "कार्यालय का पता"],
          "BUSINESS": ["कार्यरत संस्था", "फर्म का नाम"], "amount": ["ऋण राशि", "निवेश राशि", "मासिक वेतन", "अंतरित राशि"],
          "date": ["लेनदेन की तिथि", "परिपक्वता तिथि", "जारी करने की तिथि"], "code": ["आईएफएससी", "MICR कोड", "संदर्भ संख्या"]},
    seps=[": ", " : ", " - ", "\t"],
    titles={"kyc": ["म्यूचुअल फंड निवेश आवेदन", "जोखिम प्रोफ़ाइल प्रश्नावली", "व्यक्तिगत ऋण आवेदन पत्र"],
            "statement": ["बचत खाता विवरणी", "क्रेडिट कार्ड विवरण", "निवेश पोर्टफोलियो सारांश"],
            "closing": ["शुभकामनाओं सहित,", "आपका विश्वासी,", "सधन्यवाद,"],
            "job": ["ग्राहक सेवा अधिकारी", "निवेश सलाहकार", "शाखा प्रबंधक"]},
    prose=["आदरणीय {PERSON},\nआपके अनुरोध पर {DATE} को {AMOUNT} की राशि आपके खाता संख्या {ACCOUNT} में जमा कर दी गई है। किसी भी जानकारी के लिए {PHONE} पर संपर्क करें।",
           "निवेशक {PERSON}, जन्म दिनांक {DOB}, पैन {TIN}, ने फोलियो {ACCOUNT} में एसआईपी शुरू करने का अनुरोध किया है।",
           "{BUSINESS} प्रमाणित करती है कि {PERSON} हमारे यहाँ कार्यरत हैं और उनका मासिक वेतन {AMOUNT} है (बैंक कोड {CODE})।",
           "{PERSON} का पता बदलकर {ADDRESS} कर दिया गया है तथा {DATE} से सभी सूचनाएँ {EMAIL} पर भेजी जाएँगी।",
           "ग्राहक {PERSON} (जन्मतिथि {DOB}) {BUSINESS} में कार्यरत हैं; उनसे {PHONE} या {EMAIL} पर संपर्क किया जा सकता है।",
           "क्रेडिट कार्ड संख्या {ACCOUNT} पर {PERSON} की बकाया राशि {AMOUNT} है, जिसकी अंतिम तिथि {DATE} है।",
           "प्रिय {PERSON2},\n{ADDRESS} निवासी {PERSON} ने ऋण के लिए आवेदन किया है; उनका पैन {TIN} है।",
           "{BUSINESS} और {PERSON} के बीच {AMOUNT} का ऋण अनुबंध हुआ; उधारकर्ता की जन्म तिथि {DOB}, मोबाइल {PHONE}।"],
    gen=dict(
        PERSON=lambda r: f"{pick(r, HI_GIVEN)} {pick(r, HI_FAMILY)}",
        BUSINESS=lambda r: f"{pick(r, HI_COMPANY)} {pick(r, ['फिनकॉर्प', 'एंटरप्राइजेज', 'एसेट मैनेजमेंट', 'इंश्योरेंस ब्रोकर्स', 'होल्डिंग्स'])} {pick(r, ['प्राइवेट लिमिटेड', 'लिमिटेड', 'एलएलपी'])}",
        ADDRESS=hi_address,
        DOB=date_gen(hi_date, *DOB_YEARS), DATE=date_gen(hi_date, *DATE_YEARS),
        PHONE=lambda r: (lambda s: digits_to(s, DEVANAGARI) if r.random() < 0.15 else s)(
            pick(r, [f"+91-8{d(r, 9)}", f"7{d(r, 2)} {d(r, 3)} {d(r, 4)}", f"0755-{d(r, 7)}", f"6{d(r, 4)}-{d(r, 5)}"])),
        ACCOUNT=lambda r: (lambda s: digits_to(s, DEVANAGARI) if r.random() < 0.15 else s)(
            pick(r, [d(r, 11), f"{d(r, 4)}-{d(r, 4)}-{d(r, 4)}-{d(r, 4)}", f"{d(r, 3)}/{d(r, 7)}", d(r, 16)])),
        TIN=lambda r: f"{up(r, 3)}P{up(r, 1)}{d(r, 4)}{up(r, 1)}",
        EMAIL=email_gen(["alok", "rekha", "nitin", "smita.s", "gaurav", "harish", "jyoti", "tarun"],
                        ["gmail.com", "yahoo.in", "outlook.com", "sbi.co.in", "hotmail.com"]),
        AMOUNT=amount_gen(["₹{n}", "रुपये {n}", "Rs. {dec}", "{n} रुपये"], indian=True),
        CODE=lambda r: pick(r, [f"{up(r, 4)}0{d(r, 6)}", d(r, 9)]),
    ),
)

LANGS = {"th": TH, "ar": AR, "hi": HI}


def generate(lang: str, n: int, seed: int) -> list[dict]:
    """n training documents in `lang` (KYC forms, statements, prose + signatures)."""
    L = LANGS[lang]
    rng = random.Random(f"train-{seed}-{lang}")
    rows = []
    for i in range(n):
        doc = Doc()
        fn = rng.choices([f for f, _ in DOC_TYPES], [w for _, w in DOC_TYPES])[0]
        fn(doc, L, lang, rng)
        spans = doc.spans + [{"start": g["start"], "end": g["end"], "label": "O"} for g in doc.negs]
        rows.append({"id": f"gen_{lang}-{i}", "source": f"gen_{lang}", "lang": lang, "kind": "generated",
                     "partial": "", "doc_type": fn.__name__, "text": doc.text,
                     "spans": sorted(spans, key=lambda s: s["start"])})
    return rows
