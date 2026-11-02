"""Generate golden fixtures for the `gpt_sovits_text::lang_segmenter` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures lang_segmenter

Imports upstream `text/LangSegmenter/langsegmenter.py` (which replaces
fast-langdetect's default detector), then points the detector at
`parity/.cache/fast_langdetect/lid.176.bin` (the same file as
upstream `pretrained_models/fast_langdetect`). Writes
`crates/ports/gpt_sovits_text/tests/fixtures/lang_segmenter/`:

- `get_texts.jsonl`: `LangSegmenter.getTexts(text, default_lang)` for
  `default_lang` in `""`, `"zh"`, `"ja"`, `"ko"`, for every input of
  `parity/lang_segmenter/lang_inputs.py`. Each result is
  `[[lang, text], ...]` or `{"error": "<exception class>"}`.
- `functions.jsonl`: `full_en`, `full_cjk`, `split_jako("ja", ...)` and
  `split_jako("ko", ...)` (with `lang` `"x"`) for the hand-written and word
  inputs, every 4th bulk input, and single-character probes around the
  character class boundaries.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import gpt_sovits_env  # noqa: E402

OUT = gpt_sovits_env.FIXTURES / "lang_segmenter"

gpt_sovits_env.setup()

import fast_langdetect  # noqa: E402
from fast_langdetect.infer import LangDetectConfig, LangDetector  # noqa: E402
from text.LangSegmenter import langsegmenter as ls  # noqa: E402

from lang_inputs import inputs  # noqa: E402

fast_langdetect.infer._default_detector = LangDetector(
    LangDetectConfig(cache_dir=gpt_sovits_env.LID_MODEL.parent)
)

DEFAULT_LANGS = ["", "zh", "ja", "ko"]

PROBE_RANGES = [
    (0x00, 0x7F), (0x80, 0xA0), (0x1100, 0x1101), (0x11FF, 0x1200), (0x1680, 0x1680),
    (0x1FFF, 0x2001), (0x2026, 0x2026), (0x206F, 0x2070), (0x2FFF, 0x3002), (0x301B, 0x301D),
    (0x303F, 0x3042), (0x3095, 0x309B), (0x309F, 0x30A2), (0x30F9, 0x30FD), (0x3129, 0x3131),
    (0x318E, 0x3190), (0x33FF, 0x3400), (0x4DB4, 0x4DB6), (0x4DFF, 0x4E00), (0x9FFE, 0xA000),
    (0xABFF, 0xAC00), (0xD7AE, 0xD7B0), (0xFEFF, 0xFF00), (0xFF01, 0xFF01), (0xFF10, 0xFF11),
    (0xFF21, 0xFF22), (0xFF61, 0xFF62), (0xFFEF, 0xFFF0), (0x1F600, 0x1F600),
    (0x1FFFF, 0x20000), (0x2A6DD, 0x2A6DE), (0x2A6FF, 0x2A700), (0x2B73F, 0x2B741),
    (0x2B81F, 0x2B821), (0x2CEAF, 0x2CEB0), (0x2EBEF, 0x2EBF0), (0x2EE5D, 0x2EE5E),
    (0x2FFFF, 0x30000), (0x3134A, 0x3134B), (0x3134F, 0x31350), (0x323AF, 0x323B0),
]


def encode_texts(r):
    return [[d["lang"], d["text"]] for d in r]


def get_texts(text, default_lang):
    try:
        return encode_texts(ls.LangSegmenter.getTexts(text, default_lang))
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = inputs()
    with open(OUT / "get_texts.jsonl", "w") as f:
        for cat, text in items:
            rec = {"cat": cat, "text": text, "get_texts": {d: get_texts(text, d) for d in DEFAULT_LANGS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    func_texts = []
    bulk_i = 0
    for cat, text in items:
        if cat == "bulk":
            bulk_i += 1
            if bulk_i % 4:
                continue
        if cat == "corpus":
            continue
        func_texts.append(text)
    for a, b in PROBE_RANGES:
        func_texts.extend(chr(c) for c in range(a, b + 1))
    func_texts += ["a\nあ", "\nabc", "abc\n", "1\na", "あ1、い", "あ 。ア", "한1。가", "한 。", "あ。", "。あ"]
    with open(OUT / "functions.jsonl", "w") as f:
        for text in func_texts:
            item = {"lang": "x", "text": text}
            rec = {
                "text": text,
                "full_en": ls.full_en(text),
                "full_cjk": ls.full_cjk(text),
                "split_jako_ja": encode_texts(ls.split_jako("ja", item)),
                "split_jako_ko": encode_texts(ls.split_jako("ko", item)),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(items)} inputs, {len(func_texts)} function inputs", file=sys.stderr)


if __name__ == "__main__":
    main()
