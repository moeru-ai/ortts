"""Generate golden fixtures for the `fast_langdetect` port crate.

Runs the pinned fast-langdetect 1.1.0 (backed by fasttext-predict 0.9.2.4) with
the full `lid.176.bin` model, configured like GPT-SoVITS
(`LangDetector(LangDetectConfig(cache_dir=...))`), and writes
`tests/fixtures/detect.jsonl` of this crate.

Usage (from the repository root):

    pixi run --manifest-path crates/ports/fast_langdetect/pixi.toml fixtures

The model is read from `parity/.cache/lid.176.bin` (the `lid-model` task
downloads it there). `FAST_LANGDETECT_MODEL` overrides the path; the file must
be named `lid.176.bin`, because fast-langdetect looks it up in `cache_dir`.

Each fixture line is one `LangDetector.detect()` call:

    {"text": str, "k": int, "threshold": float, "normalize_input": bool,
     "max_input_length": int | null, "result": [[lang, score], ...]}
"""

import json
import os
import random
import struct
from pathlib import Path

from fast_langdetect import LangDetectConfig, LangDetector

CRATE = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get("FAST_LANGDETECT_MODEL", CRATE / "parity" / ".cache" / "lid.176.bin")
).expanduser()
MODEL_DIR = MODEL_PATH.parent
OUT = CRATE / "tests" / "fixtures" / "detect.jsonl"
CORPUS = CRATE / "parity" / "corpus" / "gpt_sovits_zh_en_yue.json"


def read_vocab(path):
    """Read the word entries of a fasttext .bin dictionary (no matrices)."""
    with open(path, "rb") as f:
        f.read(8 + 12 * 4 + 8)  # magic, version, args
        size, _nwords, _nlabels = struct.unpack("<iii", f.read(12))
        f.read(16)  # ntokens, pruneidx_size
        words = []
        for _ in range(size):
            buf = bytearray()
            while (c := f.read(1)) != b"\x00":
                buf += c
            _count, typ = struct.unpack("<qb", f.read(9))
            if typ == 0:
                words.append(buf.decode("utf-8", errors="replace"))
        return words


HAND = [
    "", " ", "  ", "\n", "\n\n", "\t", "\r", "\v", "\f", "\x00", "a\x00b", "a\tb\rc\vd\fe",
    "a", "A", "I", "é", "ß", "我", "あ", "ア", "한", "ㅋ", "z", "1", "0", "42", "3.14", "2024年",
    "123456789", "1 2 3", "!", "?", "...", "，", "。", "！？", "!!!???", "-", "_", "__label__en",
    "__label__zh hello", "__label__xx", "__label__", "</s>", "<s>", "a </s> b", "😀", "😀😀😀",
    "👍 ok", "🇨🇳", "❤️", "👨‍👩‍👧",
    "hello", "Hello", "HELLO", "HELLO WORLD", "HELLO world", "Hello World", "HeLLo WoRLD",
    "ABCDEF", "ABCDE", "ABCDEf", "ABCDEFg", "ABCDEFGHIj", "AB12CD", "ÀÉÎÕÜ", "ΑΒΓΔΕ", "ПРИВЕТ МИР",
    "Привет мир", "ǅemal", "ǅ", "ᾈ", "İSTANBUL", "ΟΔΟΣ ΣΟΦΟΣ", "ΣΊΣΥΦΟΣ", "ǄEMAL ABCDE",
    "THIS IS A LONG UPPERCASE SENTENCE", "I AM HERE 123", "123 ABC", "abc", "ABC",
    "ABCDEFGHIJ ǅ", "ABCDEFGHIJ ᾈ", "ΑΒΓΔΕΖΗ abc", "ABCDEFGHIJKLMN abc",
    "你好", "你好世界", "衬衫", "今天天气真不错", "我爱北京天安门", "一", "的",
    "こんにちは", "ありがとう", "カタカナ", "ひらがなとカタカナ", "東京タワー", "日本語のテキスト",
    "안녕하세요", "감사합니다", "한국어 텍스트", "サンキュー", "ㅎㅎ",
    "hello 你好", "你好 hello", "我喜欢 Python 编程", "iPhone 15 Pro", "GPT-SoVITS 是一个项目",
    "東京 Tokyo 도쿄", "Bonjour", "Hola", "Guten Tag", "Ciao", "Olá", "Merhaba", "Xin chào",
    "สวัสดี", "नमस्ते", "مرحبا", "שלום", "Γειά σου", "Здравствуйте", "გამარჯობა", "Բարեւ",
    "ሰላም", "வணக்கம்", "ನಮಸ್ಕಾರ", "హలో", "ഹലോ", "নমস্কার", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ສະບາຍດີ", "សួស្តី",
    "မင်္ဂလာပါ", "བཀྲ་ཤིས་བདེ་ལེགས", "Сайн байна уу", "Salam", "Habari", "Sawubona",
    "line one\nline two", "\nleading newline", "trailing newline\n", "a\n\nb", "中文\n日本語\nEnglish",
    "a" * 79, "a" * 80, "a" * 81, "a" * 200, "我" * 81, "A" * 100, ("Hello world. " * 10),
    ("今天天气很好。" * 15), ("THE QUICK BROWN FOX " * 6), "x" * 80 + "我", "a\n" * 50,
    "the the the the", "of", "and", "is", "le", "la", "der", "die", "el", "de",
    "   spaced   out   ", "tab\tseparated\twords", "mixed\r\nline endings",
    "https://example.com/path?q=1", "user@example.com", "#hashtag @mention", "C++ / Rust / Go",
    "©®™", "°C", "½ ¾", "∑∫√", "→←↑↓", "★☆", "♪♫", "「こんにちは」", "（你好）", "【测试】",
    "ｈｅｌｌｏ", "ＡＢＣＤＥＦ", "１２３", "　", " nbsp", "zero​width", "﻿bom",
    "á", "é̂", "ﬁ ligature", "Ⅻ", "ⅻ", "ǈ", "ǋ", "ǲ", "ⅫⅫⅫⅫⅫⅫ",
]

SCRIPT_RANGES = [
    (0x0041, 0x007A), (0x00C0, 0x024F), (0x0370, 0x03FF), (0x0400, 0x04FF), (0x0530, 0x058F),
    (0x0590, 0x05FF), (0x0600, 0x06FF), (0x0900, 0x097F), (0x0980, 0x09FF), (0x0B80, 0x0BFF),
    (0x0E00, 0x0E7F), (0x10A0, 0x10FF), (0x1100, 0x11FF), (0x3040, 0x309F), (0x30A0, 0x30FF),
    (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0x1F300, 0x1F64F), (0x2000, 0x206F), (0x3000, 0x303F),
    (0xFF00, 0xFFEF), (0x0030, 0x0039), (0x0021, 0x002F), (0x20000, 0x2A6DF), (0x1E00, 0x1EFF),
]


def bulk(rng, vocab, corpus):
    out = []
    # Single vocabulary words (in-vocabulary path) and case variants.
    out += rng.sample(vocab, 900)
    out += [w.upper() for w in rng.sample(vocab, 150)]
    out += [w.capitalize() for w in rng.sample(vocab, 100)]
    # Phrases built from vocabulary words.
    for _ in range(900):
        words = [rng.choice(vocab) for _ in range(rng.randint(2, 12))]
        s = rng.choice([" ", " ", " ", "", "\n", "\t", ", "]).join(words)
        r = rng.random()
        if r < 0.1:
            s = s.upper()
        elif r < 0.15:
            s = s.title()
        out.append(s)
    # Random code points from many scripts (out-of-vocabulary path).
    for _ in range(500):
        parts = []
        for _ in range(rng.randint(1, 4)):
            lo, hi = rng.choice(SCRIPT_RANGES)
            parts.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 12))))
        out.append(rng.choice([" ", "", "\n", " 123 ", "，"]).join(parts))
    # Corpus sentences, whole and sliced (LangSegmenter feeds short slices).
    for s in corpus:
        out.append(s)
        if len(s) > 2:
            a = rng.randrange(len(s))
            out.append(s[a:rng.randint(a + 1, len(s))])
    # Corpus text mixed with vocabulary words.
    for _ in range(300):
        out.append(rng.choice(corpus) + rng.choice(["", " ", "\n"]) + rng.choice(vocab))
    return out


def main():
    if MODEL_PATH.name != "lid.176.bin":
        raise SystemExit(f"FAST_LANGDETECT_MODEL must point to a file named lid.176.bin: {MODEL_PATH}")
    model_path = MODEL_PATH
    vocab = sorted(set(read_vocab(model_path)))
    corpus = [item["text"] for item in json.loads(CORPUS.read_text(encoding="utf-8"))]
    rng = random.Random(176)

    texts = []
    seen = set()
    for t in HAND + bulk(rng, vocab, corpus):
        if t not in seen:
            seen.add(t)
            texts.append(t)

    # GPT-SoVITS configuration (normalize_input=True, max_input_length=80).
    gpt = LangDetector(LangDetectConfig(cache_dir=str(MODEL_DIR)))
    raw = LangDetector(
        LangDetectConfig(cache_dir=str(MODEL_DIR), normalize_input=False, max_input_length=None)
    )

    lines = []

    def emit(det, norm, maxlen, text, k, threshold):
        res = det.detect(text, model="full", k=k, threshold=threshold)
        lines.append({
            "text": text, "k": k, "threshold": threshold, "normalize_input": norm,
            "max_input_length": maxlen, "result": [[r["lang"], r["score"]] for r in res],
        })

    for i, t in enumerate(texts):
        # split_lang: fast_lang_detect() and possible_detection_list().
        emit(gpt, True, 80, t, 1, 0.0)
        emit(gpt, True, 80, t, 5, 0.01)
        if i % 10 == 0:
            emit(raw, False, None, t, 3, 0.0)
        if i % 50 == 0:
            emit(gpt, True, 80, t, -1, 0.0)  # unlimited k: exposes the threshold pruning
            emit(gpt, True, 80, t, 176, 0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(texts)} texts, {len(lines)} cases -> {OUT}")


if __name__ == "__main__":
    main()
