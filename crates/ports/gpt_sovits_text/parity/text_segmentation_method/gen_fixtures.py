"""Golden fixtures for TTS_infer_pack/text_segmentation_method.py.

Run:

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures text_segmentation_method

Writes, under crates/ports/gpt_sovits_text/tests/fixtures/text_segmentation_method/:

- constants.json: `punctuation`, `splits`, method names, `get_method` errors.
- cases.jsonl: one input text per line with the output of `split`, every
  `cutN` method, and `split_big_text` (default `max_len` and a small one).
  A Python exception is stored as {"error": "<exception name>"}.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

from TTS_infer_pack import text_segmentation_method as tsm  # noqa: E402

OUT_DIR = gpt_sovits_env.FIXTURES / "text_segmentation_method"
SEED = 1234
BULK = 1600

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可也你说年过下能而后多自好学天心现最只公开看样做想起长没还很知些己今"
JA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコガギグ日本語東京私猫"
EN = ["hello", "world", "the", "quick", "brown", "fox", "GPT", "SoVITS", "is", "a", "TTS", "model", "e.g", "Mr", "U.S", "i.e", "OK"]
NUMBERS = [
    "3.14", "1,000", "10,000.5", "0.5", "2024", "1.2.3", "v2.0", "5.", ".5", "12.",
    "３.１４", "٣.١٤", "²․³", "².³", "①.②", "1.a", "a.1", "Ⅳ.Ⅴ", "½.½", "9.9.9",
]
PUNCT = [
    "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", ";", "、", "-", " ",
    "……", "——", "...", "。。", "!!", "?!", "…。", "——。", "；", "「", "」", "《", "》", '"', "'",
    "\n", "\n\n", "\n\n\n", "\t", "  ", "(", ")", "~~", "--",
]

EDGE = [
    "", "\n", "\n\n", " ", "  ", "。", "。。", "……", "——", "…", "—", ".", "..", "...", ",", "!", "?",
    "-", "~", ":", "、", ";", "a", "你", "你好", "3.14", "1,000", "3.14.", ".3.14", "a.b", "a. b",
    "1. 2", "1 .2", "1.2.3", "。abc。", ".abc.", "\nabc\n", "\n\nabc", "abc\n\n", "\n。\n", "。\n。",
    " . ", "！？", "，。", "!?…,.- ", "你好。", "你好", "你好，世界。今天天气很好！我们去公园吧？好的~",
    "一。二。三。四。", "一。二。三。四。五。", "一。二。三。四。五。六。七。八。", "一。二。三。四。五。六。七。八。九。",
    "a.b.c.d.e.f.g.h.i", "Hello world. This is a test... Is it OK? Yes!",
    "价格是3.14元，共1,000件。", "价格是３.１４元。", "٣.١٤ and ².³ and ①.② and Ⅳ.Ⅴ.",
    "……你好……", "——你好——", "你好…………世界", "你好————世界", "你好———世界", "你好……。",
    "。" * 10, "." * 10, "\n".join(["a", "", "b", "", "", "c"]),
    "日本語のテキストです。次の文、そして最後！",
    "x" * 50 + "。", "x" * 49 + "。", "x" * 51 + "。", "x" * 25 + "。" + "y" * 25 + "。",
    "x" * 50 + "。" + "y" * 49 + "。", "x" * 50 + "。" + "y" * 50 + "。", "x" * 50 + "。" + "y" * 48 + "。",
    "z" * 509, "z" * 510, "z" * 511, "z" * 600, "z" * 300 + "。" + "w" * 300,
    "z" * 509 + "。", "z" * 510 + "。", "。" + "z" * 510, "z" * 1100 + "，" + "q" * 20,
]


def piece(rng):
    r = rng.random()
    if r < 0.30:
        return "".join(rng.choice(ZH) for _ in range(rng.randint(1, 12)))
    if r < 0.45:
        return " ".join(rng.choice(EN) for _ in range(rng.randint(1, 6)))
    if r < 0.55:
        return "".join(rng.choice(JA) for _ in range(rng.randint(1, 10)))
    if r < 0.65:
        return rng.choice(NUMBERS)
    return rng.choice(PUNCT)


def paragraph(rng):
    kind = rng.random()
    if kind < 0.035:
        target = rng.randint(511, 700)  # very long
    elif kind < 0.15:
        target = rng.randint(100, 200)
    else:
        target = rng.randint(1, 70)
    out = ""
    while len(out) < target:
        out += piece(rng)
    if rng.random() < 0.2:
        out = "\n" * rng.randint(1, 3) + out
    if rng.random() < 0.2:
        out = out + "\n" * rng.randint(1, 3)
    return out


def call(func, *args):
    try:
        return func(*args)
    except Exception as e:  # noqa: BLE001  (record the Python exception name)
        return {"error": type(e).__name__}


def main():
    rng = random.Random(SEED)
    texts = EDGE + [paragraph(rng) for _ in range(BULK)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_texts = 0
    with open(OUT_DIR / "cases.jsonl", "w", encoding="utf-8") as f:
        for text in texts:
            long_texts += len(text) > 510
            max_len = rng.choice([1, 5, 10, 20, 50, 100, 200, 300])
            case = {
                "text": text,
                "split": call(tsm.split, text),
                **{name: call(tsm.get_method(name), text) for name in tsm.get_method_names()},
                "split_big_text": call(tsm.split_big_text, text),
                "small_max_len": max_len,
                "split_big_text_small": call(tsm.split_big_text, text, max_len),
            }
            f.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")

    def get_method_error(name):
        try:
            tsm.get_method(name)
        except ValueError as e:
            return str(e)

    constants = {
        "punctuation": sorted(tsm.punctuation),
        "splits": sorted(tsm.splits),
        "method_names": tsm.get_method_names(),
        "get_method_errors": {name: get_method_error(name) for name in ["cut6", "", "CUT0", "cut 1"]},
    }
    with open(OUT_DIR / "constants.json", "w", encoding="utf-8") as f:
        json.dump(constants, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {len(texts)} cases ({len(EDGE)} edge, {BULK} bulk, {long_texts} over 510 chars)")


if __name__ == "__main__":
    main()
