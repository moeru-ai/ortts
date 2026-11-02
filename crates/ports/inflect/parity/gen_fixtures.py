"""Generate golden fixtures and data tables for the `inflect` port crate.

Run from anywhere with (see parity/README.md):

    pixi run --manifest-path crates/ports/inflect/pixi.toml data
    pixi run --manifest-path crates/ports/inflect/pixi.toml fixtures

or directly with `python parity/gen_fixtures.py [--only data|fixtures]`.
Paths are relative to the crate directory.

Writes:

- data/decimal_digits.tsv: the Unicode decimal digit
  ranges of the pinned Python (`str.isdecimal`, which is what `re` uses for
  `\\d` and what `int()` accepts), with their digit values.
- tests/fixtures/range.json: number_to_words for every int
  in 0..=20000, with the default arguments and with andword="".
- tests/fixtures/cases.jsonl: one case per line,
  number_to_words and ordinal on hand-written edge cases, expend.py shaped
  calls and seeded random inputs.

Each line of cases.jsonl is one block of cases that share a function and its
keyword arguments: {"f": "number_to_words" | "ordinal", "kw": {...}, "c":
[[type, input, result], ...]}. type is "i" (Python int) or "s" (Python str),
input is the input as a string, and result is the returned string or
{"e": <exception class name>} when Python raises.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

import inflect


CRATE = Path(__file__).resolve().parents[1]
FIXTURES = CRATE / "tests" / "fixtures"
DATA = CRATE / "data"

ENGINE = inflect.engine()


def write_decimal_digits() -> None:
    """Write runs of consecutive decimal digits as `start end value_of_start`."""
    runs: list[list[int]] = []
    for cp in range(0x110000):
        ch = chr(cp)
        if not ch.isdecimal():
            continue
        value = unicodedata.decimal(ch)
        if runs and runs[-1][1] == cp - 1 and runs[-1][2] + (cp - runs[-1][0]) == value:
            runs[-1][1] = cp
        else:
            runs.append([cp, cp, value])
    lines = [
        "# Unicode decimal digits (str.isdecimal) of Python "
        f"{sys.version_info.major}.{sys.version_info.minor}, Unicode {unicodedata.unidata_version}",
        "# start\tend\tvalue_of_start (hex code points, decimal value)",
    ]
    lines += [f"{s:X}\t{e:X}\t{v}" for s, e, v in runs]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "decimal_digits.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(fn: str, num, kw: dict) -> dict:
    case = {
        "f": fn,
        "t": "int" if isinstance(num, int) else "str",
        "n": str(num),
    }
    if kw:
        case["kw"] = kw
    try:
        if fn == "number_to_words":
            case["o"] = ENGINE.number_to_words(num, **kw)
        else:
            case["o"] = ENGINE.ordinal(num, **kw)
    except Exception as err:  # noqa: BLE001 - we record the class name
        case["e"] = type(err).__name__
    return case


# Keyword sets passed by GPT-SoVITS expend.py plus variants of each option.
EXPEND_KWS = [
    {},
    {"andword": ""},
    {"andword": "", "zero": "oh", "group": 2},
]
VARIANT_KWS = [
    {"group": 1},
    {"group": 2},
    {"group": 3},
    {"zero": "oh"},
    {"zero": "oh", "group": 1},
    {"zero": "oh", "group": 3},
    {"one": "a"},
    {"one": "a", "group": 1},
    {"one": "a", "group": 3},
    {"andword": "&"},
    {"andword": ","},
    {"comma": ""},
    {"comma": ";"},
    {"comma": "", "group": 2},
    {"comma": ";", "group": 3},
    {"decimal": "dot"},
    {"decimal": ""},
    {"decimal": None},
    {"decimal": "dot", "group": 2},
    {"decimal": None, "group": 1},
]
ALL_KWS = EXPEND_KWS + VARIANT_KWS

EDGE_STRINGS = [
    # plain and padded
    "0", "1", "00", "01", "001", "0001", "007", "10", "100", "101", "110",
    "1000", "1001", "1010", "1100", "1101", "10000", "100000", "1000000",
    "1000001", "1001001", "12345678901234567890", " 12", "12 ", "\t3\n",
    # signs
    "-0", "-1", "-12", "+7", "+0", " -5", "-", "+", "--1", "-+1",
    # decimals
    "3.14", "0.5", ".5", "5.", ".", "..", "1.2.3", "-0.5", "-.5", "+1.50",
    "10.01", "1.000", "0.0", "12.", "1..2", ".0", "3.14159265358979",
    # ordinal suffixes (expend.py passes "21st" style strings)
    "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd",
    "101st", "111th", "1000th", "1000000th", "102nd", "20th", "30th", "40th",
    "90th", "1001st", "1st.", "1.5th", "0th", "st", "nd", "rd", "th", "tt",
    "5t", "xst", "-1st", "1,000th",
    # separators and junk
    "1,234", "1,234,567", "1 234", "1_000", "abc", "a1b2", "one", "1e3",
    "1E3", "0x10", "$100", "100%", "12:30",
    # whitespace-only and other IndexError inputs
    " ", "\t", "\x1c", "　",
    # unicode digits and whitespace
    "١٢٣", "٠١٢٣", "１２",
    "०", "১০০", "1 2", "1 2", "١.٥",
    "\U0001d7ce\U0001d7cf", "1　,2",
    # range limits
    "9" * 36, "1" + "0" * 35, "1" + "0" * 36, "1" + "0" * 38, "1" + "0" * 39,
    "100" + "0" * 36, "9" * 37, "0" * 40 + "1", "0" * 50,
    # empty (typeguard rejects it)
    "",
]

EDGE_INTS = [
    0, 1, 2, 9, 10, 11, 12, 13, 19, 20, 21, 99, 100, 101, 110, 111, 999, 1000,
    1001, 1010, 1100, 1999, 2000, 2001, 2010, 2099, 9999, 10000, 10001, 100000,
    1000000, 1000001, 10**9, 10**12 + 1, 10**15, 10**18 + 7, 10**21, 10**24,
    10**27, 10**30, 10**33, 10**36 - 1, 10**36, 10**37, 10**38, 2**127 - 1,
    -1, -2, -11, -100, -1001, -12345, -(10**20), -(2**127),
]


def number_to_words_cases(rng: random.Random) -> list[dict]:
    cases = []
    for kw in ALL_KWS:
        for num in EDGE_STRINGS:
            cases.append(run("number_to_words", num, kw))
        for num in EDGE_INTS:
            cases.append(run("number_to_words", num, kw))
    for group in (-1, 4):
        cases.append(run("number_to_words", 5, {"group": group}))
        cases.append(run("number_to_words", "5", {"group": group}))

    # expend.py _expand_number: group=2 path for 1000 < num < 3000.
    for num in range(1000, 3001):
        cases.append(run("number_to_words", num, {"andword": "", "zero": "oh", "group": 2}))

    # expend.py _expand_ordinal: strings matching [0-9]+(st|nd|rd|th).
    for num in range(0, 1000):
        for suff in ("st", "nd", "rd", "th"):
            cases.append(run("number_to_words", f"{num}{suff}", {}))

    # Seeded random ints up to 10^36 (and a few beyond, to hit the range limit).
    for _ in range(900):
        digits = rng.randint(1, 38)
        num = rng.randrange(10 ** (digits - 1), 10**digits) if digits > 1 else rng.randrange(10)
        if rng.random() < 0.2:
            num = -num
        kw = rng.choice(ALL_KWS)
        cases.append(run("number_to_words", num, kw))

    # Seeded random ints with many zero groups.
    for _ in range(500):
        digits = rng.randint(1, 37)
        s = "".join(rng.choice("0000000123456789") for _ in range(digits))
        cases.append(run("number_to_words", int(s), rng.choice(ALL_KWS)))

    # Seeded random strings: digits with signs, points, suffixes and junk.
    alphabet = "0123456789" * 4 + "...-+, \t_ab"
    for _ in range(2000):
        length = rng.randint(1, 24)
        s = "".join(rng.choice(alphabet) for _ in range(length))
        if rng.random() < 0.25:
            s += rng.choice(["st", "nd", "rd", "th"])
        cases.append(run("number_to_words", s, rng.choice(ALL_KWS)))
    return cases


ORDINAL_EDGE = [
    "0", "1", "2", "3", "4", "11", "12", "13", "14", "21", "101", "111", "112",
    "-1", "-11", "-21", "-", "-a", "--1", "1.", "1.5", "1.25", "1.x", "1..",
    "-1.", "12.3.4", "1_000", "1__0", "1_", "12 ", "12\x1c", "12　",
    "12\x1f", "\x0b12", "12\x85", "\xa012", "12\x00", "1_\u0662", "\u0661_\u0662",
    "1 1", "1a", "5th", "١٢", "١.", "１３",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "twenty", "thirty", "ninety",
    "twenty-one", "twenty-two", "one hundred", "one hundred and twelve",
    "one thousand", "one million", "zero", "oh", "a", "ty", "", " ", "x",
    "nineteen ninety-nine", "twelve twelve", "first", "1st", "abc", ".5", "Twelve",
]


def ordinal_cases(rng: random.Random) -> list[dict]:
    cases = [run("ordinal", s, {}) for s in ORDINAL_EDGE]
    for num in range(-120, 1200):
        cases.append(run("ordinal", num, {}))
        cases.append(run("ordinal", str(num), {}))
    # expend.py _expand_fraction: ordinal(number_to_words(denominator)).
    for num in range(0, 2001):
        cases.append(run("ordinal", ENGINE.number_to_words(num), {}))
    for _ in range(250):
        num = rng.randrange(10 ** rng.randint(1, 36))
        cases.append(run("ordinal", num, {}))
        cases.append(run("ordinal", ENGINE.number_to_words(num), {}))
        cases.append(run("ordinal", ENGINE.number_to_words(num, andword=""), {}))
    alphabet = "0123456789" * 3 + ".-_ a"
    for _ in range(800):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
        cases.append(run("ordinal", s, {}))
    return cases


def write_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260)
    cases = number_to_words_cases(rng) + ordinal_cases(rng)
    # Group cases by (function, keyword arguments), keeping first-seen order.
    blocks: dict[tuple[str, str], dict] = {}
    for case in cases:
        kw = case.get("kw", {})
        key = (case["f"], json.dumps(kw, sort_keys=True))
        block = blocks.setdefault(key, {"f": case["f"], "kw": kw, "c": []})
        result = case["o"] if "o" in case else {"e": case["e"]}
        block["c"].append([case["t"][0], case["n"], result])
    with (FIXTURES / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for block in blocks.values():
            fh.write(json.dumps(block, ensure_ascii=False, separators=(",", ":")) + "\n")

    rng_out = {
        "start": 0,
        "end": 20000,
        "default": [ENGINE.number_to_words(n) for n in range(0, 20001)],
        "andword_empty": [ENGINE.number_to_words(n, andword="") for n in range(0, 20001)],
    }
    (FIXTURES / "range.json").write_text(
        json.dumps(rng_out, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"cases: {len(cases)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        choices=("data", "fixtures"),
        help="write only data/ or only tests/fixtures/ (default: both)",
    )
    args = parser.parse_args()
    if args.only in (None, "data"):
        write_decimal_digits()
    if args.only in (None, "fixtures"):
        write_fixtures()


if __name__ == "__main__":
    main()
