"""Generate golden fixtures for the `gpt_sovits_text::zh_normalization` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures zh_normalization

Writes crates/ports/gpt_sovits_text/tests/fixtures/zh_normalization/:

- normalize.jsonl: `TextNormalizer().normalize(text)` for hand-written edge
  cases, every sentence of parity/corpus/gpt_sovits_zh_en_yue.json and
  bulk inputs generated with a fixed seed.
- normalize_sentence.jsonl: `TextNormalizer().normalize_sentence(text)` (no
  `_split`, so spaces and filtered characters reach the rules).
- functions.json: unit-level outputs (num2str, verbalize_cardinal,
  verbalize_digit, _time_num2str, phone2str, _split with lang != "zh",
  char_convert in both directions, str.isspace, RE_NSW.findall and the
  constants.py translation tables).

Each jsonl line is {"i": input, "o": output} or, when Python raises,
{"i": input, "e": [exception type, str(exception.args[0])]}.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

from text.zh_normalization import char_convert  # noqa: E402
from text.zh_normalization import chronology  # noqa: E402
from text.zh_normalization import constants  # noqa: E402
from text.zh_normalization import num  # noqa: E402
from text.zh_normalization import phonecode  # noqa: E402
from text.zh_normalization.text_normlization import TextNormalizer  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "zh_normalization"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"

TN = TextNormalizer()


def run(fn, arg):
    try:
        return {"i": arg, "o": fn(arg)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return {"i": arg, "e": [type(e).__name__, str(e.args[0]) if e.args else ""]}


HAND = [
    "",
    " ",
    "\u3000",
    "\n\n",
    "你好",
    # README examples
    "电影中梁朝伟扮演的陈永仁的编号27149",
    "这块黄金重达324.75克",
    "我们班的最高总分为583分",
    "12~23",
    "-1.5~2",
    "她出生于86年8月18日，她弟弟出生于1995年3月1日",
    "等会请在12:05请通知我",
    "今天的最低气温达到-10°C",
    "现场有7/12的观众投出了赞成票",
    "明天有62％的概率降雨",
    "随便来几个价格12块5，34.5元，20.1万",
    "这是固话0421-33441122",
    "这是手机+86 18544139121",
    # dates
    "2024年",
    "24年",
    "2024年1月",
    "2024年01月01日",
    "2024年12月31号",
    "2024年13月",
    "2024年2月30日",
    "2024年2月32日",
    "2024年0月",
    "12345年",
    "1年",
    "2024-01-02",
    "2024/01/02",
    "2024.01.02",
    "2024 01 02",
    "2024-01/02",
    "2024-13-01",
    "2024-12-32",
    "2024-00-10",
    "１９９９年１２月３１日",
    # times
    "8:30",
    "08:30",
    "8:00",
    "8:05:09",
    "23:59:59",
    "24:00",
    "25:61",
    "0:00:00",
    "12:30:00",
    "8:30-12:30",
    "8:15~9:30",
    "8:30-9:15",
    "8:30:15-9:45:01",
    "8:00-9:00",
    "18:30~20:30:30",
    "时间是10:3",
    # to range / temperature / measure
    "10cm~20cm",
    "3%~5%",
    "-5°C~10°C",
    ".5kg~1.5kg",
    "1m~2mm",
    "20度~30度",
    "-3°C",
    "3.5℃",
    "20度",
    "37.5摄氏度",
    "-0.5摄氏度",
    "5cm2",
    "5cm²",
    "5cm3",
    "5cm³",
    "5db",
    "5ds",
    "5kg",
    "5km",
    "5m2",
    "5m²",
    "5m³",
    "5m3",
    "5ml",
    "5m",
    "5mm",
    "5s",
    "cms dbs",
    # asmd / power
    "1+2=3",
    "3-2=1",
    "2×3=6",
    "6÷2=3",
    "-1+-2",
    "1.5+.5=2",
    "x²+y²=z²",
    "a+b=c",
    "2ⁿ",
    "10⁻³",
    "eˣ",
    "1+2+3+4+5",
    "A=B",
    "3+",
    "+3",
    # frac / percentage
    "1/2",
    "-1/2",
    "100/3",
    "0/0",
    "50%",
    "-3.5%",
    "0.0%",
    "100.00%",
    # phones
    "13812345678",
    "+8613812345678",
    "8613812345678",
    "+86 13812345678",
    "15012345678",
    "15412345678",
    "17612345678",
    "17512345678",
    "19812345678",
    "138123456789",
    "010-12345678",
    "01012345678",
    "021-1234567",
    "0571-87654321",
    "12345678",
    "0123456",
    "400-123-4567",
    "4001234567",
    "400-1234567",
    # range
    "1-5",
    "1~5",
    "-1-5",
    "1.5-2.5",
    "1-5-9",
    "+1-5",
    "1-5+",
    "10-20个",
    # integers / version / decimal / quantifier / default
    "-10",
    "-0",
    "1.2.3",
    "1.2.3.4",
    "v2.0.1",
    "版本1.0.0.1发布",
    "3.14",
    "-3.14",
    ".5",
    "0.50",
    "3.20",
    "3.0",
    "3.000",
    "10.00100",
    "5个",
    "2个",
    "2多个",
    "2余人",
    "3几天",
    "5+人",
    "100万元",
    "3千万元",
    "5美元",
    "2吨",
    "2十吨",
    "3毫克",
    "3公分",
    "3厘米",
    "2小时",
    "123",
    "00078",
    "001",
    "1000",
    "10",
    "11",
    "100",
    "101",
    "1001",
    "10000",
    "100000",
    "100000000",
    "1000000000000",
    "12345678901234567890",
    "0",
    "00",
    "1",
    "2",
    "20",
    "200",
    # post replace
    "①②③④⑤⑥⑦⑧⑨⑩",
    "αβγΓδΔεζηθΘικλΛμνξΞοπΠρςΣστυφΦχψΨωΩ",
    "a/b",
    "5km/h",
    "1=1",
    "a×b÷c",
    "a+b-c",
    # splitting and filters
    "你好，世界。你好！吗？是的：好；行、",
    "他说：“你好。”然后走了。",
    "他说：‘好’。",
    "a,b;c?d!e",
    "《三体》【科幻】<小说>{好}(看)（真的）#话题&@某人“引用”^_|\\——",
    "你 好 世 界",
    "\u3000你好\u3000",
    "\t你好\t",
    "\x1c你好\x1d",
    "\u00a0你好\u2003",
    "。。。",
    "，",
    "\n你好\n\n世界\n",
    # traditional / full width
    "這是一個測試，我們說繁體字。",
    "一般是指存取一個應用程式啟動時始終顯示在網站或網頁瀏覽器中的一個或多個初始網頁等畫面存在的站點",
    "ＡＢＣａｂｃ１２３",
    "ｈｅｌｌｏ\u3000ｗｏｒｌｄ",
    "１２：３０",
    "５０％",
    "－５",
    # non-ASCII decimal digits (KeyError paths and Unicode version)
    "٣",
    "٣٤٥",
    "價格是٣元",
    "७८९",
    "𝟘𝟙𝟚",
    "\U00011f50\U00011f51",
    "\U0001e4f0",
    "２٣",
    "²³",
    "1٣:30",
    "٣年",
    "12٣4年",
    "8:3٣",
]


HAN = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处理府研质")
WORDS = [
    "今天", "价格", "温度", "电话", "号码", "版本", "我们", "时间", "大约", "需要", "比赛", "比分", "距离", "面积",
    "体积", "速度", "公司", "成立于", "出生", "增长", "下降", "大概", "其中", "总共", "第", "约",
]
TRAD = list(char_convert.traditional_characters)
SIMP = list(char_convert.simplified_charcters)
QUANT = [
    "处", "台", "个", "人", "张", "元", "块", "角", "毛", "分", "米", "千米", "毫米", "厘米", "分米", "微米", "千克",
    "毫克", "微克", "公分", "年", "月", "日", "天", "小时", "秒", "岁", "吨", "万吨", "千万", "百万", "万", "千", "百",
    "亿", "亿元", "万元", "美元", "千块", "十吨", "斤", "两", "杯", "瓶", "箱", "件", "条", "只", "名", "位", "次",
]
UNITS_TO = ["%", "°C", "℃", "度", "摄氏度", "cm2", "cm²", "cm3", "cm³", "cm", "db", "ds", "kg", "km", "m2", "m²", "m³", "m3",
            "ml", "m", "mm", "s"]
PUNCT = ["，", "。", "！", "？", "：", "；", "、", ",", ";", "?", "!", ".", "…", "~", "～", "-", "—", "——", "/", "”", "’", "“",
         "‘", "《", "》", "【", "】", "<", ">", "{", "}", "(", ")", "（", "）", "#", "&", "@", "^", "_", "|", "\\", "=", "+",
         "×", "÷", "%", "％", "：", "\u3000", " ", "\t", "\n", "'", "\"", "*", "$", "¥", "￥"]
GREEK = list("αβγΓδΔεζηθΘικλΛμνξΞοπΠρςΣστυφΦχψΨωΩ①②③④⑤⑥⑦⑧⑨⑩")
SUPER = list("⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ")
ODD_DIGITS = ["٣", "٠", "۵", "७", "০", "๓", "𝟘", "𝟗", "\U00011f50", "\U0001e4f0", "²", "½", "Ⅷ", "〇", "零"]
FULLWIDTH = [chr(ord(c) + 65248) for c in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!%,.:;?-~+=/"]
ENGLISH = ["hello", "GPT", "AI", "iPhone", "x", "a", "b", "OK", "km", "s", "m", "the", "test", "v", "V"]


def digits(rng, lo=1, hi=6, lead_zero=0.1):
    n = rng.randint(lo, hi)
    s = "".join(rng.choice("0123456789") for _ in range(n))
    if rng.random() < lead_zero:
        s = "0" + s
    return s


def number(rng):
    r = rng.random()
    sign = "-" if rng.random() < 0.2 else ""
    if r < 0.35:
        return sign + digits(rng, 1, 4, 0.05)
    if r < 0.5:
        return sign + digits(rng, 5, 20, 0.15)
    if r < 0.75:
        return sign + digits(rng, 1, 4, 0.1) + "." + digits(rng, 1, 4, 0.3)
    if r < 0.85:
        return "." + digits(rng, 1, 3)
    if r < 0.95:
        return ".".join(digits(rng, 1, 2, 0) for _ in range(rng.randint(2, 5)))
    return sign + digits(rng, 1, 3) + "." + digits(rng, 1, 3) + "0" * rng.randint(1, 3)


def date(rng):
    y = digits(rng, 4, 4, 0) if rng.random() < 0.7 else digits(rng, 2, 2, 0)
    m = str(rng.randint(0, 13))
    if rng.random() < 0.3 and len(m) == 1:
        m = "0" + m
    d = str(rng.randint(0, 33))
    if rng.random() < 0.3 and len(d) == 1:
        d = "0" + d
    r = rng.random()
    if r < 0.6:
        s = y + "年"
        if rng.random() < 0.8:
            s += m + "月"
            if rng.random() < 0.8:
                s += d + rng.choice("日号")
        elif rng.random() < 0.3:
            s += d + rng.choice("日号")
        return s
    sep = rng.choice(["-", "/", ".", " ", "-", "/"])
    sep2 = sep if rng.random() < 0.85 else rng.choice(["-", "/", ".", " "])
    m = m.zfill(2) if rng.random() < 0.9 else m
    d = d.zfill(2) if rng.random() < 0.9 else d
    return digits(rng, 4, 4, 0) + sep + m + sep2 + d


def time(rng):
    def one():
        h = str(rng.randint(0, 26))
        if rng.random() < 0.3 and len(h) == 1:
            h = "0" + h
        mm = rng.choice(["00", "30", "05", "15", "45", "59", "60", "07", str(rng.randint(0, 59)).zfill(2)])
        s = h + ":" + mm
        if rng.random() < 0.3:
            s += ":" + rng.choice(["00", "30", "09", "59", "61", str(rng.randint(0, 59)).zfill(2)])
        return s
    if rng.random() < 0.4:
        return one() + rng.choice(["-", "~", "~", "-", "至", "到"]) + one()
    return one()


def phone(rng):
    r = rng.random()
    if r < 0.45:
        prefix = rng.choice(["", "", "+86", "86", "+86 ", "86 ", "+86\u3000"])
        second = rng.choice(["3", "8", "5", "7", "9", "4", "6", "2"])
        third = str(rng.randint(0, 9))
        return prefix + "1" + second + third + digits(rng, 8, 8, 0)[:8] if rng.random() < 0.9 else prefix + "1" + second + third + digits(rng, 7, 9, 0)
    if r < 0.85:
        area = rng.choice(["010", "021", "022", "023", "024", "0571", "0755", "0421", "", "0", "01"])
        sep = rng.choice(["-", "", "-"])
        first = str(rng.randint(0, 9))
        return area + (sep if area else "") + first + digits(rng, 6, 7, 0)[: rng.choice([6, 7])]
    return "400" + rng.choice(["-", ""]) + digits(rng, 3, 3, 0) + rng.choice(["-", ""]) + digits(rng, 4, 4, 0)


def money_quant(rng):
    n = rng.choice([digits(rng, 1, 4, 0.05), digits(rng, 1, 3) + "." + digits(rng, 1, 2)])
    mid = rng.choice(["", "", "", "多", "余", "几", "+"])
    return n + mid + rng.choice(QUANT) + (digits(rng, 1, 1, 0) if rng.random() < 0.2 else "")


def to_range(rng):
    u = rng.choice(UNITS_TO)
    u2 = u if rng.random() < 0.7 else rng.choice(UNITS_TO)
    return number(rng) + u + rng.choice(["~", "~", "-", "～"]) + number(rng) + u2


def temperature(rng):
    return number(rng) + rng.choice(["°C", "℃", "度", "摄氏度", "°F", "°"])


def measure(rng):
    return number(rng) + rng.choice(UNITS_TO[5:] + ["kg/m³", "km/h", "m/s", "mm²", "dbs", "ms"])


def asmd(rng):
    def operand():
        r = rng.random()
        if r < 0.6:
            o = number(rng)
        elif r < 0.8:
            o = rng.choice("abcxyzXYZAB")
        else:
            o = "." + digits(rng, 1, 2)
        if rng.random() < 0.25:
            o += "".join(rng.choice(SUPER) for _ in range(rng.randint(1, 2)))
        return o
    s = operand()
    for _ in range(rng.randint(1, 3)):
        s += rng.choice(["+", "-", "×", "÷", "="]) + operand()
    return s


def frac_pct(rng):
    if rng.random() < 0.5:
        return rng.choice(["", "-"]) + digits(rng, 1, 3, 0.1) + "/" + digits(rng, 1, 3, 0.1)
    return number(rng) + rng.choice(["%", "%", "％"])


def num_range(rng):
    return number(rng) + rng.choice(["-", "~"]) + number(rng)


def text_frag(rng):
    r = rng.random()
    if r < 0.5:
        return "".join(rng.choice(HAN) for _ in range(rng.randint(1, 5)))
    if r < 0.8:
        return rng.choice(WORDS)
    return "".join(rng.choice(TRAD) for _ in range(rng.randint(1, 4)))


def misc(rng):
    r = rng.random()
    if r < 0.25:
        return rng.choice(PUNCT)
    if r < 0.4:
        return rng.choice(GREEK)
    if r < 0.5:
        return "".join(rng.choice(SUPER) for _ in range(rng.randint(1, 3)))
    if r < 0.6:
        return rng.choice(ODD_DIGITS) + (digits(rng, 0, 3, 0) if rng.random() < 0.5 else "")
    if r < 0.8:
        return "".join(rng.choice(FULLWIDTH) for _ in range(rng.randint(1, 4)))
    if r < 0.9:
        return rng.choice(ENGLISH)
    return "".join(rng.choice(SIMP) for _ in range(rng.randint(1, 3)))


GENERATORS = [
    (number, 10), (date, 8), (time, 8), (phone, 6), (money_quant, 8), (to_range, 5), (temperature, 5), (measure, 5),
    (asmd, 6), (frac_pct, 6), (num_range, 6), (text_frag, 30), (misc, 20),
]


def random_text(rng):
    funcs = [g for g, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    parts = []
    for _ in range(rng.randint(1, 7)):
        parts.append(rng.choices(funcs, weights)[0](rng))
        if rng.random() < 0.3:
            parts.append(rng.choice(PUNCT))
    return "".join(parts)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260402)

    corpus = [entry["text"] for entry in json.loads(CORPUS.read_text(encoding="utf-8"))]
    bulk = [random_text(rng) for _ in range(6500)]
    inputs = list(dict.fromkeys(HAND + corpus + bulk))
    write_jsonl(OUT / "normalize.jsonl", [run(TN.normalize, t) for t in inputs])

    sent_bulk = [random_text(rng) for _ in range(2000)]
    sent_inputs = list(dict.fromkeys(HAND + sent_bulk))
    write_jsonl(OUT / "normalize_sentence.jsonl", [run(TN.normalize_sentence, t) for t in sent_inputs])

    nums = sorted(set(
        [str(i) for i in range(0, 1201)]
        + [digits(rng, 1, 24, 0.3) for _ in range(1500)]
        + ["", "0", "00", "000", ".", "0.", ".0", "1.", "1.0", "1.00", "1.010", "0.0001", "1.2.3", "٣", "1٣", "٣.5"]
        + [digits(rng, 1, 6, 0.3) + "." + digits(rng, 1, 5, 0.3) for _ in range(800)]
    ))
    funcs = {
        "num2str": [run(num.num2str, s) for s in nums],
        "verbalize_cardinal": [run(num.verbalize_cardinal, s) for s in nums if "." not in s],
        "verbalize_digit": [run(num.verbalize_digit, s) for s in nums[:400] if "." not in s],
        "verbalize_digit_alt_one": [run(lambda s: num.verbalize_digit(s, alt_one=True), s) for s in nums[:400] if "." not in s],
        "_get_value": [run(num._get_value, s) for s in nums if "." not in s and s][:600],
        "_time_num2str": [run(chronology._time_num2str, f"{i:02d}") for i in range(0, 100)] + [run(chronology._time_num2str, str(i)) for i in range(0, 100)],
        "phone2str_mobile": [run(lambda s: phonecode.phone2str(s, mobile=True), phone(rng)) for _ in range(300)]
        + [run(lambda s: phonecode.phone2str(s, mobile=True), s) for s in ["+86 138 1234 5678", "++86\t138", "", "+", "86\u3000138"]],
        "phone2str_landline": [run(lambda s: phonecode.phone2str(s, mobile=False), phone(rng)) for _ in range(300)]
        + [run(lambda s: phonecode.phone2str(s, mobile=False), s) for s in ["010--1234", "", "-", "400-123-4567"]],
        "_split_en": [run(lambda s: TN._split(s, lang="en"), t) for t in HAND],
        "isspace": [cp for cp in range(sys.maxunicode + 1) if chr(cp).isspace()],
        "re_nsw_findall": [run(constants.RE_NSW.findall, t) for t in HAND + [random_text(rng) for _ in range(300)]],
        "translate": [
            run(lambda t: {
                "F2H_ASCII_LETTERS": t.translate(constants.F2H_ASCII_LETTERS),
                "H2F_ASCII_LETTERS": t.translate(constants.H2F_ASCII_LETTERS),
                "F2H_DIGITS": t.translate(constants.F2H_DIGITS),
                "H2F_DIGITS": t.translate(constants.H2F_DIGITS),
                "F2H_PUNCTUATIONS": t.translate(constants.F2H_PUNCTUATIONS),
                "H2F_PUNCTUATIONS": t.translate(constants.H2F_PUNCTUATIONS),
                "F2H_SPACE": t.translate(constants.F2H_SPACE),
                "H2F_SPACE": t.translate(constants.H2F_SPACE),
            }, t)
            for t in HAND + ["".join(chr(c) for c in range(0x20, 0x7F)), "".join(chr(c) for c in range(0xFF00, 0xFF66))]
        ],
        "t2s_all": run(char_convert.tranditional_to_simplified, char_convert.traditional_characters),
        "s2t_all": run(char_convert.simplified_to_traditional, char_convert.simplified_charcters),
        "t2s_random": [run(char_convert.tranditional_to_simplified, "".join(rng.choice(TRAD + SIMP + HAN) for _ in range(12))) for _ in range(200)],
        "s2t_random": [run(char_convert.simplified_to_traditional, "".join(rng.choice(TRAD + SIMP + HAN) for _ in range(12))) for _ in range(200)],
    }
    (OUT / "functions.json").write_text(json.dumps(funcs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
