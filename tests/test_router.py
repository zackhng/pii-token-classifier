"""Language router (src/router.py)."""
from router import Router, letter_scripts, segments


def test_script_rules():
    r = Router()
    assert r.route("山田太郎はABCキャピタルに勤務しています。") == "ja"
    assert r.route("김민수는 ABC 캐피탈에서 근무합니다.") == "ko"
    assert r.route("राजेश कुमार का खाता") == "hi"
    assert r.route("العميل محمد أحمد") == "ar"
    assert r.route("ลูกค้า สมชาย ใจดี") == "th"
    assert r.route("张伟任职于ABC资本，联系电话13812345678") == "zh-Hans"
    assert r.route("張偉任職於ABC資本，聯絡電話13812345678") == "zh-Hant"
    assert r.route("1234 5678", default="en") == "en"          # no letters -> default


def test_japanese_kanji_heavy_text_is_not_chinese():
    assert Router().route("東京都千代田区丸の内1-1-1") == "ja"


def test_latin_classifier():
    texts = (["Please transfer the amount to my savings account today."] * 20 +
             ["Vui lòng chuyển khoản vào tài khoản tiết kiệm của tôi hôm nay."] * 20 +
             ["Sila pindahkan jumlah itu ke akaun simpanan saya hari ini."] * 20 +
             ["Mohon transfer jumlahnya ke rekening tabungan saya hari ini."] * 20 +
             ["Pakilipat ang halaga sa aking savings account ngayong araw."] * 20)
    langs = ["en"] * 20 + ["vi"] * 20 + ["ms"] * 20 + ["id"] * 20 + ["tl"] * 20
    r = Router.train(texts, langs)
    assert r.route("Cảm ơn Quý khách đã gửi tiền vào tài khoản.") == "vi"
    assert r.route("Thank you for the transfer to the account.") == "en"


def test_segments_and_mixed_document():
    text = "Dear Mr Tan,\nplease see below.\n\n张伟的账户已更新。\n\n12345\n\nRegards"
    segs = segments(text)
    assert "".join(text[s:e] for s, e in segs) == text
    routed = Router().route_segments(text)
    langs = [l for _, _, l in routed]
    assert langs[0] == "en" and "zh-Hans" in langs
    assert routed[0][0] == 0 and routed[-1][1] == len(text)
    # the digits-only block joins its neighbour instead of getting its own route
    assert all(text[s:e].strip() != "12345" for s, e, _ in routed)


def test_letter_scripts():
    c = letter_scripts("Ab张キ김ส्ำअع")
    assert c["latin"] == 2 and c["han"] == 1 and c["kana"] == 1 and c["hangul"] == 1
