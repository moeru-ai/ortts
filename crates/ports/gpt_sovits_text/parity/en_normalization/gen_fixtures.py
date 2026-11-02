"""Generate golden fixtures for the `gpt_sovits_text::en_normalization` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures en_normalization

Writes crates/ports/gpt_sovits_text/tests/fixtures/en_normalization/:

- hand.jsonl: hand-written edge cases for every regex and branch of expend.py.
- corpus.jsonl: every sentence of parity/corpus/gpt_sovits_zh_en_yue.json
  and every non-CJK span of it.
- bulk.jsonl: random sentences built from number, currency, time, unit,
  fraction, ordinal, math, abbreviation and acronym tokens (fixed seed).
- py_int.jsonl: `int(s)` (as `str(int(s))`) for the kinds of strings that
  expend.py passes to `int()`, including the 4300 digit limit. normalize()
  cannot reach the limit in reasonable time: `_pounds_re_end` backtracks in
  cubic time on a long run of digits.

Each line is {"i": input, "o": normalize(input)} or, when Python raises,
{"i": input, "e": [exception type, str(exception)]}.
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

from text.en_normalization.expend import normalize  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "en_normalization"
CORPUS = gpt_sovits_env.CORPUS_DIR / "gpt_sovits_zh_en_yue.json"


def run(text):
    try:
        return {"i": text, "o": normalize(text)}
    except Exception as e:  # noqa: BLE001 - every exception is part of the contract
        return {"i": text, "e": [type(e).__name__, str(e)]}


HAND = [
    "",
    " ",
    "hello",
    "Hello, world!",
    # upstream __main__ examples
    "1. test ordinal number 1st",
    "32.3$, $6.24, 1.1£, £7.14.",
    "3/23, 1/2, 3/2, 1/3, 6/1",
    "1st, 22nd",
    "a test 20h, 1.2s, 1L, 0.1km",
    "a test of time 4:00, 13:00, 13:30",
    "a test of temperature 4°F, 23°C, -19°C",
    # _ordinal_number_re
    "1. ",
    "1.",
    "1. a",
    "2. b 3. c 11. d 12. e 13. f 21. g 101. h 111. i",
    "Step 10. Done",
    "a1. b",
    "_1. b",
    "é1. b",
    "\u03011. b",
    "²1. b",
    "0. zero",
    "007. bond",
    "3.  two spaces",
    "3.\tTab",
    "99999999999999999999999999999999999999999. big",
    # RE_ASMD
    "1 + 2",
    "1 + 2 + 3",
    "1 + 2 = 3",
    "10 - 3 = 7",
    "6 × 7 = 42",
    "8 ÷ 2 = 4",
    "x + y = z",
    "a - b",
    "abc + def",
    "-1 + -2",
    "1 - -2",
    "1.5 + 2.25",
    ".5 + .25",
    "x² + y² = z²",
    "2ⁿ - 1",
    "eˣ = y",
    "1+2",
    "1 +2",
    "1  +  2",
    "1\t+\t2",
    "1\u3000+\u30002",
    "1\x1c+\x1c2",
    "1\xa0+\xa02",
    "١ + ٢",
    "3 * 4",
    "3 / 4",
    "A = B = C = D",
    "1 + 2 + 3 + 4 + 5 + 6",
    "e = mc²",
    "2 - 1 - 0",
    # RE_INTEGER
    "-5",
    "-5 degrees",
    "it is -5 outside",
    "it is  -5 outside",
    "it is\t-5",
    "it is\x1f-5",
    "a-5",
    "-١٢",
    "--5",
    "x -0",
    "-3.14",
    "temperature: -19°C",
    # _comma_number_re
    "1,000",
    "12,345,678",
    "1,2,3",
    "1,,2",
    "1,",
    ",1",
    "100,000 people",
    "1,000,000,000,000",
    "1, 2, 3",
    "10,5",
    # _time_re
    "3:30",
    "03:30",
    "0:00",
    "00:00",
    "12:00",
    "12:00 a.m.",
    "12:01",
    "13:00",
    "13:30",
    "23:59",
    "24:00",
    "19:60",
    "3:30pm",
    "3:30 PM",
    "at 7:05.",
    "123:45",
    "1:234",
    "a3:30",
    "3:30a",
    "é3:30",
    "3:30é",
    "\u03013:30",
    "3:30\u0301",
    "²3:30",
    "3:30²",
    "_3:30",
    "1:2:3",
    "10:10:10",
    "2:5",
    # _measurement_re
    "1m",
    "2m",
    "1.0m",
    "0.1km",
    "01m",
    "5km",
    "5km/h",
    "100 km/h",
    "6ft",
    "1ft",
    "2L",
    "1L",
    "3tbsp",
    "1tsp",
    "24h",
    "1h",
    "5min",
    "1min",
    "30s",
    "1s",
    "1.5s",
    "23°C",
    "1°C",
    "4°F",
    "-19°C",
    "5mins",
    "5ms",
    "5m2",
    "a5m",
    "5mé",
    "5m\u0301",
    "5m²",
    "5 m",
    "1.2.3m",
    "10.5km",
    "99999999999999999999999999999999999999999m",
    # pounds
    "£5",
    "£1",
    "£1.01",
    "£1.5",
    "£2.50",
    "£0.01",
    "£0.99",
    "£.5",
    "£0",
    "£0.00",
    "£1.2.3",
    "£1,000",
    "£1,000.50",
    "5£",
    "1.1£",
    "1.01£",
    "£,5",
    "£5,.3",
    "£1.,5",
    "£1.5,5",
    "£007",
    "£1.123",
    # dollars
    "$5",
    "$1",
    "$1.01",
    "$6.24",
    "$0.5",
    "$.99",
    "$0",
    "$0.00",
    "$1.2.3",
    "$1,000,000",
    "$1,000.99",
    "32.3$",
    "1$",
    "(5$",
    "(1.5$)",
    "($5)",
    "$,5",
    "5,$",
    "$5,.3",
    "$1.,5",
    "$100 and $200",
    "$1.001",
    # _decimal_number_re
    "3.14",
    "13.234",
    "0.5",
    "1.0",
    "1.\n5",
    "1.\t5",
    "1.\x1c5",
    "1.\u30005",
    "1..5",
    "1.2.3",
    "1.2.3.4",
    "192.168.0.1",
    "v1.2",
    # _fraction_re
    "1/2",
    "3/2",
    "0/2",
    "1/3",
    "3/4",
    "5/6",
    "10/3",
    "6/1",
    "0/1",
    "1/0",
    "2/0",
    "1/100",
    "2/101",
    "1/1000000",
    "1/2/3",
    "22/7",
    "101/3",
    "1/99999999999999999999999999999999999999999",
    "99999999999999999999999999999999999999999/2",
    # _ordinal_re
    "1st",
    "2nd",
    "3rd",
    "4th",
    "11th",
    "12th",
    "13th",
    "21st",
    "22nd",
    "23rd",
    "101st",
    "103rd",
    "111th",
    "1000th",
    "1nd",
    "2st",
    "0th",
    "1sth",
    "1stst",
    "99999999999999999999999999999999999999999th",
    # _number_re and years
    "0",
    "1",
    "7",
    "13",
    "99",
    "100",
    "101",
    "999",
    "1000",
    "1001",
    "1066",
    "1100",
    "1200",
    "1905",
    "1999",
    "2000",
    "2001",
    "2008",
    "2009",
    "2010",
    "2019",
    "2100",
    "2999",
    "3000",
    "3001",
    "10000",
    "123456",
    "1000000",
    "007",
    "0001999",
    "9" * 36,
    "9" * 37,
    "9" * 39,
    "1" + "0" * 36,
    "1" + "0" * 40,
    "9" * 200,
    "(" + "9" * 300 + "$",
    "£," + "9" * 300,
    # percentages
    "50%",
    "12.5%",
    "100 %",
    "%",
    "%%",
    # accents and filtering
    "café",
    "naïve résumé",
    "Ångström",
    "\u212a",
    "\u212b",
    "ﬁne",
    "ÆØÅ",
    "Straße",
    "Đà Nẵng",
    "日本語 text",
    "full width １２３",
    "emoji 😀 ok",
    "quotes \"double\" and 'single'",
    "semi; colon: dash - en – em —",
    "a/b\\c",
    "tab\there\nnewline",
    "#hash @at &amp *star",
    "(parens) [brackets] {braces}",
    "under_score",
    # i.e. / e.g.
    "i.e.",
    "I.E.",
    "e.g.",
    "E.G.",
    "this, i.e. that",
    "fruit, e.g. apples",
    "ie eg",
    "i.e.e.g.",
    # capitals and acronyms
    "NASA",
    "API",
    "GPU",
    "macOS",
    "iPhone",
    "McDonald",
    "USA and UK",
    "A",
    " A",
    "a A",
    "aA",
    "AbC",
    "OpenAI GPT-4",
    "HTTPS URL",
    "The CPU's cache",
    "I'm OK",
    # abbreviations
    "Dr. Smith",
    "Mr. and Mrs. Jones",
    "St. Louis",
    "Prof. X",
    "Jan. 1st",
    "etc.",
    "vs.",
    "No. 1",
    # ranges and mixed punctuation
    "10-20",
    "1990-2000",
    "pages 3-5",
    "5 - 10",
    "1,000-2,000",
    "$5-$10",
    "3:30-4:30",
    "Wait... what?!",
    "Hello?? Yes!!",
    "--",
    "a - b - c",
    "x--y",
]


def corpus_inputs():
    corpus = json.loads(CORPUS.read_text())
    out = []
    span_re = re.compile(r"[^\u3000-\u303f\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]+")
    for item in corpus:
        text = item["text"]
        out.append(text)
        for span in span_re.findall(text):
            if re.search(r"[A-Za-z0-9]", span):
                out.append(span)
                out.append(span.strip())
    seen = set()
    return [t for t in out if not (t in seen or seen.add(t))]


WORDS = (
    "the a an of to and in is it you that he was for on are with as I his they be at one have "
    "this from or had by hot word but what some we can out other were all there when up use your "
    "how said each she which do their time if will way about many then them write would like so "
    "these her long make thing see him two has look more day could go come did number sound no "
    "most people my over know water than call first who may down side been now find"
).split()
ACRONYMS = ["NASA", "API", "GPU", "CPU", "macOS", "iPhone", "iOS", "USB", "HTTP", "JSON", "OpenAI", "YouTube", "eBay", "PhD", "TV"]
ABBREVIATIONS = ["Dr.", "Mr.", "Mrs.", "Ms.", "St.", "Jr.", "Sr.", "Prof.", "Inc.", "Ltd.", "Co.", "Mt.", "vs.", "etc.", "i.e.", "e.g.", "I.E.", "E.G.", "a.m.", "p.m."]
UNITS = ["m", "km", "km/h", "ft", "L", "tbsp", "tsp", "h", "min", "s", "°C", "°F", "kg", "ms", "mins"]
PUNCT = [",", ".", "!", "?", ";", ":", "...", " -", "'", '"', "(", ")", "—"]
ACCENTED = ["café", "naïve", "résumé", "jalapeño", "Zoë", "façade", "Ångström", "über", "Ⅻ", "²", "½", "١٢", "日本", "中文", "\u0301"]
SPACES = [" ", " ", " ", " ", "  ", "\t", "\n", "\xa0", "\u3000", "\x1c"]


def rand_int(r):
    k = r.random()
    if k < 0.4:
        return str(r.randint(0, 99))
    if k < 0.7:
        return str(r.randint(100, 9999))
    if k < 0.85:
        return str(r.randint(1000, 3100))
    if k < 0.995:
        return str(r.randint(10000, 10**12))
    return str(r.randint(10**30, 10**42))


def with_commas(s):
    return f"{int(s):,}"


def token(r):
    k = r.randrange(24)
    n = rand_int(r)
    if k == 0:
        return n
    if k == 1:
        return with_commas(n)
    if k == 2:
        return f"{n}.{r.randint(0, 999):0{r.randint(1, 3)}d}"
    if k == 3:
        return "-" + r.choice([n, with_commas(n), f"{n}.{r.randint(0, 99)}"])
    if k == 4:
        amount = r.choice([n, with_commas(n), f"{r.randint(0, 999)}.{r.randint(0, 99):02d}", f"{r.randint(0, 9)}.{r.randint(0, 9)}", f".{r.randint(0, 99)}", f"{r.randint(0, 9)}.{r.randint(0, 9)}.{r.randint(0, 9)}", f"{r.randint(0, 9)},"])
        sym = r.choice(["$", "£", "€", "¥", "₹"])
        return sym + amount if r.random() < 0.6 else amount + sym
    if k == 5:
        return str(r.randint(0, 3000)) + r.choice(["st", "nd", "rd", "th"])
    if k == 6:
        return f"{r.randint(1, 120)}. "
    if k == 7:
        t = f"{r.randint(0, 25)}:{r.randint(0, 61):02d}"
        return t + r.choice(["", "", "am", "pm", " a.m.", " p.m.", " AM", ":00"])
    if k == 8:
        return str(r.randint(1000, 2100))
    if k == 9:
        return f"{r.randint(0, 20)}/{r.randint(0, 20)}"
    if k == 10:
        return r.choice([n, f"{r.randint(0, 100)}.{r.randint(0, 9)}"]) + r.choice(["%", " %", " percent"])
    if k == 11:
        num = r.choice([str(r.randint(0, 3)), str(r.randint(0, 500)), f"{r.randint(0, 9)}.{r.randint(0, 9)}"])
        return num + r.choice(["", "", " "]) + r.choice(UNITS)
    if k == 12:
        return r.choice(ABBREVIATIONS)
    if k == 13:
        return r.choice(ACRONYMS)
    if k == 14:
        a = r.choice([str(r.randint(0, 99)), f"{r.randint(0, 9)}.{r.randint(0, 9)}", r.choice("xyzabXY"), r.choice(["x²", "2ⁿ", ".5", "-3"])])
        b = r.choice([str(r.randint(0, 99)), r.choice("xyzab"), "-" + str(r.randint(0, 9)), "y³"])
        op = r.choice(["+", "-", "×", "÷", "=", "*", "/"])
        sp = r.choice([" ", " ", "", "  ", "\t"])
        return f"{a}{sp}{op}{sp}{b}"
    if k == 15:
        return f"{rand_int(r)}-{rand_int(r)}"
    if k == 16:
        return r.choice(ACCENTED)
    if k == 17:
        return f"{r.randint(0, 9)}.{r.choice(SPACES)}{r.randint(0, 9)}"
    if k == 18:
        return "".join(r.choice("0123456789.,$£/:-") for _ in range(r.randint(1, 8)))
    if k == 19:
        return r.choice(WORDS).capitalize()
    if k == 20:
        return r.choice(WORDS).upper()
    return r.choice(WORDS)


def sentence(r):
    parts = []
    for _ in range(r.randint(1, 12)):
        parts.append(token(r))
        if r.random() < 0.2:
            parts[-1] += r.choice(PUNCT)
    out = parts[0]
    for p in parts[1:]:
        out += r.choice(SPACES) + p
    return out


def bulk_inputs():
    r = random.Random(20240501)
    out = [sentence(r) for _ in range(5200)]
    # single tokens, one per kind, to isolate each branch
    out += [token(r) for _ in range(1500)]
    return out


def run_int(text):
    try:
        return {"i": text, "o": str(int(text))}
    except Exception as e:  # noqa: BLE001
        return {"i": text, "e": [type(e).__name__, str(e)]}


INT_INPUTS = [
    "", "0", "00", "007", "1", "10", "123", ",5", "5,", "(5", "(1", ",0", "1,5", ".", "..",
    "9" * 36, "9" * 37, "9" * 39,
    "9" * 4300, "0" * 4300, "9" * 4301, "0" * 4301, "9" * 5000,
    "9" * 5000 + ",", "9" * 4300 + ",", "(" + "9" * 5000, "," + "9" * 5000,
    "9" * 150 + "(", "(" + "9" * 197, "(" + "9" * 198, "(" + "9" * 199, "(" + "9" * 300,
]


def write(name, inputs, fn=run):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w") as f:
        for text in inputs:
            f.write(json.dumps(fn(text), ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(inputs)} cases)")


def main():
    write("hand.jsonl", HAND)
    write("corpus.jsonl", corpus_inputs())
    write("bulk.jsonl", bulk_inputs())
    write("py_int.jsonl", INT_INPUTS, run_int)


if __name__ == "__main__":
    main()
