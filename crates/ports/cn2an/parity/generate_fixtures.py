"""Generate golden fixtures for the cn2an port.

Usage: pixi run --manifest-path crates/ports/cn2an/pixi.toml fixtures

Writes to tests/fixtures/ in the crate:

- transform_an2cn.jsonl: {"i": text, "o": cn2an.transform(text, "an2cn")}
- transform_an2cn_direct.jsonl: {"i": text, "o": cn2an.transform(text, "an2cn", direct=True)}
- an2cn.jsonl: {"i": text, "low": ..., "up": ..., "rmb": ..., "direct": ...}
  with the output of An2Cn().an2cn(text, mode) per mode, or null when Python
  raises.

Inputs are hand-written edge cases plus random samples with fixed seeds.
"""

import json
import random
import warnings
from pathlib import Path

import cn2an
from cn2an import An2Cn

assert cn2an.__version__ == "0.5.24", cn2an.__version__

warnings.simplefilter("ignore")

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

MEASURE_WORDS = (
    "斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|"
    "辆|台|匹|头|位|亩|小时|分钟|秒|天|半"
).split("|")

FULL_WIDTH_DIGITS = "０１２３４５６７８９"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
DEVANAGARI_DIGITS = "०१२३४५६७८९"
# Kawi digits (Unicode 15): not `\d` in Python 3.11.
KAWI_DIGITS = "".join(chr(0x11F50 + i) for i in range(10))
# Mathematical bold digits: Nd, not converted by full_angle_to_half_angle.
MATH_DIGITS = "".join(chr(0x1D7CE + i) for i in range(10))


def rand_digits(rng: random.Random, n: int, style: str = "uniform") -> str:
    if n <= 0:
        return ""
    if style == "uniform":
        first = rng.choice("123456789")
        return first + "".join(rng.choice("0123456789") for _ in range(n - 1))
    if style == "zeros":
        first = rng.choice("123456789")
        return first + "".join(rng.choice("0000000123456789") for _ in range(n - 1))
    if style == "binary":
        return "1" + "".join(rng.choice("01") for _ in range(n - 1))
    if style == "free":
        return "".join(rng.choice("0123456789") for _ in range(n))
    raise ValueError(style)


def rand_int(rng: random.Random) -> str:
    n = rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20])
    return rand_digits(rng, n, rng.choice(["uniform", "zeros", "binary", "free"]))


def rand_decimal(rng: random.Random) -> str:
    frac_len = rng.choice([1, 1, 2, 2, 3, 4, 8, 15, 16, 17, 20])
    return f"{rand_int(rng)}.{rand_digits(rng, frac_len, rng.choice(['free', 'zeros']))}"


def rand_number(rng: random.Random) -> str:
    x = rand_int(rng) if rng.random() < 0.6 else rand_decimal(rng)
    if rng.random() < 0.25:
        x = "-" + x
    return x


def small_number(rng: random.Random) -> str:
    x = str(rng.choice([rng.randint(0, 10), rng.randint(0, 100), rng.randint(0, 100000)]))
    if rng.random() < 0.3:
        x += "." + rand_digits(rng, rng.randint(1, 3), "free")
    return x


def rand_date(rng: random.Random) -> str:
    y = str(rng.choice([rng.randint(0, 99), rng.randint(100, 999), rng.randint(1000, 2999), rng.randint(10000, 99999)]))
    if rng.random() < 0.2:
        y = y.zfill(rng.randint(2, 5))
    m = str(rng.choice([rng.randint(0, 12), rng.randint(0, 120)]))
    d = str(rng.choice([rng.randint(0, 31), rng.randint(0, 300)]))
    if rng.random() < 0.3:
        m = m.zfill(2)
    if rng.random() < 0.3:
        d = d.zfill(2)
    fmt = rng.choice([
        "{y}年{m}月{d}日", "{y}年{m}月", "{m}月{d}日", "{y}年", "{m}月", "{d}日", "{y}年{d}日",
        "{y}-{m}-{d}", "{y}/{m}/{d}", "{y}.{m}.{d}", "{m}/{d}", "{y}年{m}月{d}号",
        "{y}年度", "公元{y}年{m}月{d}日上午", "{y}年{m}月{d}日{h}点{mi}分",
    ])
    return fmt.format(y=y, m=m, d=d, h=rng.randint(0, 24), mi=str(rng.randint(0, 59)).zfill(2))


def rand_percent(rng: random.Random) -> str:
    x = small_number(rng) if rng.random() < 0.7 else rand_number(rng)
    if rng.random() < 0.2 and not x.startswith("-"):
        x = "-" + x
    return x + rng.choice(["%", "%", "%", "％", "%%", "%的"])


def rand_fraction(rng: random.Random) -> str:
    a = rand_int(rng) if rng.random() < 0.3 else str(rng.randint(0, 100))
    b = rand_int(rng) if rng.random() < 0.3 else str(rng.randint(0, 100))
    return rng.choice(["{a}/{b}", "-{a}/{b}", "{a}/{b}/{c}", "{a}.5/{b}", "{a}/-{b}", "约{a}/{b}的人"]).format(
        a=a, b=b, c=rng.randint(0, 9))


def rand_celsius(rng: random.Random) -> str:
    x = small_number(rng)
    return rng.choice(["{x}℃", "-{x}℃", "零下{x}℃", "{x}°C", "{x}℃左右", "{x}-{y}℃", "{x}~{y}℃"]).format(
        x=x, y=small_number(rng))


def rand_range(rng: random.Random) -> str:
    a = small_number(rng)
    b = small_number(rng)
    unit = rng.choice(MEASURE_WORDS + ["人", "x", "", "-", "."])
    return rng.choice(["{a}-{b}{u}", "约{a}-{b}{u}左右", "{a}-{b}-{c}{u}", "-{a}-{b}{u}", "{a}--{b}{u}"]).format(
        a=a, b=b, c=small_number(rng), u=unit)


def unicode_digits_token(rng: random.Random) -> str:
    table = rng.choice([FULL_WIDTH_DIGITS, ARABIC_INDIC_DIGITS, DEVANAGARI_DIGITS, KAWI_DIGITS, MATH_DIGITS])
    n = rng.randint(1, 5)
    s = "".join(rng.choice(table) for _ in range(n))
    if rng.random() < 0.4:
        s = rng.choice(["", "1", "2"]) + s + rng.choice(["", "3", ".5", "年", "%", "℃", "/7", "米"])
    return s


WORDS = [
    "今天", "我们", "买了", "价格是", "大约", "温度", "公元", "第", "号", "的", "共有", "人",
    "Hello", "GPT", "abc", "v", "x", "kg", "km", "iPhone", "版本", "增长", "下降", "零下",
    "，", "。", "！", "？", "、", "：", "；", "（", "）", "“", "”", " ", "  ", "\t", "\n",
    ",", ".", "-", "/", "%", "℃", "+", "~", "年", "月", "日", "点", "负",
    "－", "．", "／", "％", "　", "臺灣", "萬", "億", "兩", "廿", "半", "一半",
]
WORDS += MEASURE_WORDS


def rand_sentence(rng: random.Random) -> str:
    gens = [rand_int, rand_number, rand_decimal, rand_date, rand_percent, rand_fraction, rand_celsius,
            rand_range, unicode_digits_token, small_number]
    parts = []
    for _ in range(rng.randint(1, 8)):
        if rng.random() < 0.5:
            parts.append(rng.choice(gens)(rng))
        else:
            parts.append(rng.choice(WORDS))
    return "".join(parts)


def rand_noise(rng: random.Random) -> str:
    alphabet = "0123456789-.%/℃年月日米半分小时钟个元 a,０１－．"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))


TRANSFORM_EDGE = [
    "", "0", "00", "007", "-0", "--0", "--0.5", "-", ".", "-.", "1.", ".5", "-.5", "1..2", "1.2.3",
    "10", "11", "19", "20", "100", "101", "110", "1000", "1001", "1010", "1100", "10000", "10001",
    "100000", "100001", "1000000", "10000000", "100000000", "100000001", "100010001", "1000000000000",
    "1000100000000", "10000000000000000", "9999999999999999", "99999999999999999", "12345678901234567890",
    "3.1415926535897932384626", "0.00005", "-12.5", "12.50", "1-2", "1--2", "a-1", "1-a",
    "2023年", "2023年5月1日", "23年", "123年", "12345年", "1年", "5月", "12月", "123月", "31日",
    "1日", "001日", "2023年13月32日", "2023-05-01", "2023/05/01", "2023.05.01", "05/01",
    "1/2", "1/2/3", "0/0", "12345678901234567890/3", "50%", "-50%", "12.5%", "1.2.3%", "%5",
    "100%%", "50％", "25℃", "-25℃", "25.5℃", "零下5℃", "１２３", "２０２３年", "１２．５",
    "１２％", "１/２", "٣", "١٢٣年", "٣٣%", "2٣", "１2", "१२३", chr(0x11F51) + "2", chr(0x1D7CF),
    "1-2米", "1.5-2.5公斤", "3-5个小时", "1-2", "1-2人", "1-2.5", "10-20分钟", "1-2-3米",
    "5-10千克", "1-2克", "1-2千", "1-2小", "1-2半", "-1-2米", "12345678901234567-1米",
    "半斤", "3天", "1.5小时", "第1名", "iPhone 15 Pro", "GPT-4o", "v1.2.3", "COVID-19",
    "2023年5月1日9点30分", "电话：13800138000", "价格是12.50元", "增长了-3.5%",
    "0.5", "0.05", "0.0", "00.00", "-00.010", "1e10", "1,000,000", "1 000", "１０,０００",
    "　１２　", "臺灣2023年", "１２年", "12年２月",
]


def transform_inputs() -> list:
    rng = random.Random(20240501)
    inputs = list(TRANSFORM_EDGE)
    # integers across magnitudes
    for n in range(1, 21):
        inputs.append("1" + "0" * (n - 1))
        inputs.append("9" * n)
        inputs.append("1" * n)
        for style in ["uniform", "zeros", "binary", "free"]:
            for _ in range(8):
                inputs.append(rand_digits(rng, n, style))
    for _ in range(400):
        inputs.append(rand_number(rng))
    for gen, count in [(rand_decimal, 300), (rand_date, 600), (rand_percent, 300), (rand_fraction, 300),
                       (rand_celsius, 300), (rand_range, 400), (unicode_digits_token, 200), (rand_noise, 600)]:
        for _ in range(count):
            inputs.append(gen(rng))
    for _ in range(2500):
        inputs.append(rand_sentence(rng))
    return inputs


def an2cn_inputs() -> list:
    rng = random.Random(20240502)
    inputs = [
        "", "0", "00", "000", "007", "-0", "--0", "--0.5", "--5", "-", "--", ".", "..", "-.", "1.", ".5",
        "-.5", "1..2", "1.2.3", "0.0", "0.00", "0.000", "0.01", "0.10", "0.1", "0.11", "0.011", "0.101",
        "0.001", "1.0", "1.00", "1.000", "1.05", "1.50", "1.55", "1.555", "1.005", "1.050", "10.01",
        "100.10", "1-2", "1-", "-1-", "0-", "1.-2", "1.2-", "a", "1a", "１２", "１２．５", "－１２",
        "　", "1 2", "壹", "萬", "+1", "1_000", "1e5",
        "10", "11", "19", "20", "100", "101", "110", "111", "1000", "1001", "1010", "1011", "1100",
        "10000", "10001", "10010", "10100", "11000", "100000", "100001", "101000", "110000",
        "1000000", "10000000", "10001000", "100000000", "100000001", "100001000", "100010000",
        "1000010000", "100000000000", "1000000000000", "1000100000000", "1000000010000",
        "10000000000000000", "9999999999999999", "99999999999999999", "00000000000000001",
        "0000000000000000000", "1." + "1" * 20, "0." + "0" * 17 + "1",
    ]
    for n in range(1, 19):
        inputs.append("1" + "0" * (n - 1))
        inputs.append("9" * n)
        for style in ["uniform", "zeros", "binary", "free"]:
            for _ in range(12):
                inputs.append(rand_digits(rng, n, style))
    for _ in range(600):
        inputs.append(rand_number(rng))
    for _ in range(300):
        ip = rng.choice(["0", "1", "10", "100", "1000", "10000", rand_int(rng)])
        fp = "".join(rng.choice("0001") for _ in range(rng.randint(0, 4)))
        inputs.append(f"{ip}.{fp}")
    for _ in range(300):
        alphabet = "0123456789-.０１－．a"
        inputs.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10))))
    return inputs


def py_an2cn(ac: An2Cn, x: str, mode: str):
    try:
        return ac.an2cn(x, mode)
    except Exception:
        return None


def dedupe(xs: list) -> list:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{path.relative_to(CRATE).as_posix()}: {len(rows)} cases, {path.stat().st_size} bytes")


def main() -> None:
    t_inputs = dedupe(transform_inputs())
    write_jsonl(OUT / "transform_an2cn.jsonl",
                [{"i": x, "o": cn2an.transform(x, "an2cn")} for x in t_inputs])
    write_jsonl(OUT / "transform_an2cn_direct.jsonl",
                [{"i": x, "o": cn2an.transform(x, "an2cn", direct=True)} for x in t_inputs[::4]])

    ac = An2Cn()
    rows = []
    for x in dedupe(an2cn_inputs()):
        row = {"i": x}
        for mode in ["low", "up", "rmb", "direct"]:
            row[mode] = py_an2cn(ac, x, mode)
        rows.append(row)
    write_jsonl(OUT / "an2cn.jsonl", rows)


if __name__ == "__main__":
    main()
