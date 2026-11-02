"""Generate golden fixtures for the `tojyutping` port crate.

Run with the crate's pinned pixi environment:

    pixi run --manifest-path crates/ports/tojyutping/pixi.toml fixtures

Writes to tests/fixtures/ of the crate:

- constants.json: `punct_dict` and the punctuation / digit sets of utils.py.
- chars.json: `get_jyutping_list` on every character with a node in the trie.
- words.json: `get_jyutping_list` on a seeded sample of multi-character trie
  words, plus every word whose stored syllable count differs from its length.
- sentences.json: `get_jyutping_list`, `get_jyutping` and `get_jyutping_text`
  on hand-written edge cases, the GPT-SoVITS corpus (parity/corpus/) and seeded random
  sentences that mix Han (simplified and traditional), punctuation, digits,
  Latin, spaces and control characters.
"""

import json
import random
from importlib.metadata import version
from pathlib import Path

import ToJyutping
from ToJyutping import Trie as T
from ToJyutping import utils as U
from ToJyutping.Jyutping import JyutpingList

assert version("ToJyutping") == "3.2.0", version("ToJyutping")

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "tests/fixtures"
CORPUS = CRATE / "parity/corpus/gpt_sovits_zh_en_yue.json"

WORD_SAMPLE = 12000
RANDOM_SENTENCES = 1500


def dump(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{name}: {path.stat().st_size} bytes")


def walk_words():
    words = []
    stack = [(T.root, "")]
    while stack:
        node, prefix = stack.pop()
        if node.v:
            words.append((prefix, node.v))
        for c, child in node.items():
            stack.append((child, prefix + c))
    words.sort(key=lambda w: w[0])
    return words


def jl(s):
    return [[k, v] for k, v in ToJyutping.get_jyutping_list(s)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20240302)

    dump(
        "constants.json",
        {
            "punct_dict": U.punct_dict,
            "left_bracket": sorted(U.left_bracket),
            "right_bracket": sorted(U.right_bracket),
            "left_punct": sorted(U.left_punct),
            "right_punct": sorted(U.right_punct),
            "other_punct": sorted(U.other_punct),
            "left_or_other_punct": sorted(U.left_or_other_punct),
            "right_or_other_punct": sorted(U.right_or_other_punct),
            "minus_signs": sorted(U.minus_signs),
            "decimal_seps": sorted(U.decimal_seps),
            "digits": sorted(U.digits),
            "isspace": [c for c in map(chr, range(0x110000)) if c.isspace()],
        },
    )

    words = walk_words()
    singles = [w for w, _ in words if len(w) == 1]
    root_chars = sorted(T.root)
    assert len(root_chars) == len(singles)
    dump("chars.json", [[c, jl(c)] for c in root_chars])

    multi = [w for w, _ in words if len(w) > 1]
    odd = {
        w
        for w, v in words
        if len(w) > 1 and any((len(x) if isinstance(x, JyutpingList) else 1) != len(w) for x in v)
    }
    sample = sorted(set(rng.sample(multi, WORD_SAMPLE)) | odd)
    dump("words.json", [[w, jl(w)] for w in sample])

    puncts = list(U.punct_dict) + list("@#$%^&*_+=|\\<>`") + ["\u2212", "😀", "🇭🇰", "©"]
    digits = sorted(U.digits)
    spaces = [" ", " ", " ", "\u3000", "\xa0", "\u2003", "\u202f", "\t", "\n", "\r\n", "\x0b", "\x1c", "\x85"]
    controls = ["\x00", "\x07", "\x1f", "\x7f", "\x80", "\x9f"]
    latin = ["Hello", "world", "ABC", "x", "iPhone", "café", "naïve", "OK", "GPT", "Ωmega", "ｆｕｌｌ", "e.g.", "U.S.A."]
    numbers = ["3.14", "-5", "1,000", "１２．５", "-0.5", "2024", "10-20", "1·5", "1'000", "﹣3", "－７", "3⸱4", "𝟏𝟐.𝟑", "--1", "a-1", "1-a"]
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    han_extra = "".join(item["text"] for item in corpus)
    # Han characters that are not in the trie (sampled from CJK blocks).
    not_in_trie = [c for c in map(chr, list(range(0x4E00, 0xA000)) + list(range(0x20000, 0x2A6E0)) + list(range(0x30000, 0x3134B))) if c not in T.root]

    def piece():
        r = rng.random()
        if r < 0.30:
            return rng.choice(multi)
        if r < 0.40:
            w = rng.choice(multi)
            return w[: rng.randint(1, len(w) - 1)]
        if r < 0.50:
            return "".join(rng.choice(singles) for _ in range(rng.randint(1, 4)))
        if r < 0.55:
            return rng.choice(han_extra)
        if r < 0.58:
            return rng.choice(not_in_trie)
        if r < 0.72:
            return "".join(rng.choice(puncts) for _ in range(rng.randint(1, 3)))
        if r < 0.78:
            return rng.choice(numbers) if rng.random() < 0.6 else "".join(rng.choice(digits) for _ in range(rng.randint(1, 4)))
        if r < 0.84:
            return rng.choice(latin)
        if r < 0.97:
            return rng.choice(spaces)
        return rng.choice(controls)

    hand = [
        "",
        " ",
        "\n",
        "咁啱 ABC123，你好！",
        "我哋今日去飲茶。",
        "佢係我嘅朋友，佢好鍾意食嘢。",
        "你食咗飯未呀？",
        "银行行长",
        "銀行行長",
        "重慶重新",
        "「你好」，『世界』。",
        "（括號）[方括號]{花括號}",
        "“引號”‘單引號’",
        "\"雙\" '單' \"未閉合",
        "他說：\"你好\"嗎？",
        "一--二",
        "一-二",
        "一 - 二",
        "(一)-(二)",
        "一~二",
        "一~~二",
        "一~~~二",
        "(一)~(二)",
        "一·二",
        "-5度",
        "溫度係-5度",
        "--5",
        "a-5",
        "3.14",
        "價錢係3.14蚊",
        "1,000,000人",
        "１２．５",
        "﹣３",
        "-a",
        "1.a",
        "a.1",
        "第1-2章",
        "行行行行行",
        "長長長長長",
        "㖊噚吋呎哩唡",
        "卅卌兙兛",
        "一\t二\n三\r\n四",
        "一\x00二\x9f三\x85四",
        "一\u3000二\xa0三",
        "你好……",
        "你好！！！？？",
        "⋯⋮",
        "〈一〉《二》〔三〕【四】〖五〗",
        "﹁直排﹂﹃引號﹄",
        "｟一｠｢二｣､･",
        "!\"'(),-./:;?[]{}~",
        "，。！？；：",
        "😀🇭🇰©",
        "𠀋𠮩𡃁",
        "Hello world",
        "ToJyutping 3.2.0",
        "e.g. 一",
        "一 (二) 三",
        "一(二)三",
        "一((二))三",
        "一)二(三",
        "“一”二“三”",
        "‘’“”",
        "'一'二'",
        "一'二'三'",
        "-",
        "~",
        "·",
        "…",
        "-1",
        "1-",
        "1.",
        ".1",
        "1.1.1",
        "a~b",
        "一~",
        "~一",
        "一-",
        "-一",
        "(一-二)",
        "[一]-[二]",
        "一-(二)",
        "一·(二)",
        "Ａ－１",
    ]

    sentences = list(hand)
    sentences += [item["text"] for item in corpus]
    for _ in range(RANDOM_SENTENCES):
        n = rng.randint(1, 14)
        sentences.append("".join(piece() for _ in range(n)))
    # dedupe, keep order
    seen = set()
    sentences = [s for s in sentences if not (s in seen or seen.add(s))]

    dump(
        "sentences.json",
        [
            {
                "text": s,
                "list": jl(s),
                "jyutping": ToJyutping.get_jyutping(s),
                "jyutping_text": ToJyutping.get_jyutping_text(s),
            }
            for s in sentences
        ],
    )
    print(f"chars={len(root_chars)} words={len(sample)} (odd={len(odd)}) sentences={len(sentences)}")


if __name__ == "__main__":
    main()
