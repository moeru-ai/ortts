"""Golden fixtures for the model-free parts of `TTS_infer_pack/TextPreprocessor.py`.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures text_preprocessor

Writes `crates/ports/gpt_sovits_text/tests/fixtures/text_preprocessor/`:

- `functions.jsonl`: for each input text, `get_first`,
  `replace_consecutive_punctuation`, `filter_text(text.split("\\n"))` and
  `pre_seg_text(text, lang, cutN)` for `lang` in `zh`, `en` (the only
  distinction `pre_seg_text` makes) and N = 0..5. Inputs: the edge
  paragraphs of `e2e_common.py`, hand-written cases below, every zh/en/yue corpus item,
  every eighth ja corpus item, and 50 random paragraphs (seed 7).
- `merge_short_text_in_array.jsonl`: random lists of short strings with
  thresholds 0..8 (seed 11).

`get_phones_and_bert` and `preprocess` are covered by the e2e fixtures.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
import e2e_common as c  # noqa: E402

OUT = c.FIXTURES / "text_preprocessor"
METHODS = [f"cut{i}" for i in range(6)] + ["cut9"]

HAND = [
    "abc",
    "abcd",
    "ab cd",
    " abc",
    "  abcd  ",
    "　abc",
    "a，bcdef",
    "abcd，ef",
    "，abc",
    "。",
    "_",
    "__。",
    "_a",
    "²³",
    "½",
    "Ⅳ",
    "①②",
    "٣",
    "é",
    "\t\n",
    "\n \n",
    " \n ",
    "a\n\nb\n\n\nc",
    "a\n \nb",
    "!!!???...,,,---",
    "!-!-!-",
    "……——……",
    "a..b",
    "a,,b!!c??d--e……f",
    "😀😀😀",
    "😀。好",
    "x" * 520,
    "x" * 520 + "。",
    "，".join(["一二三四五六"] * 120),
    "一二\n三四\n五六\n七八\n九十",
    "a\nb\nc\nd\ne\nf\ng",
    "一\n二三四五六七\n八",
]


def safe(f, *args):
    try:
        return c.quiet(f, *args)
    except Exception as e:  # noqa: BLE001
        return c.err(e)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tp = c.text_preprocessor()
    zh_en_yue = [x["text"] for x in c.load_corpus("gpt_sovits_zh_en_yue.json")]
    ja = [x["text"] for x in c.load_corpus("gpt_sovits_ja.json")][::8]
    paras = [p["text"] for p in c.paragraphs(50, 7)]
    texts = list(dict.fromkeys(c.EDGE_PARAGRAPHS + HAND + zh_en_yue + ja + paras))

    with open(OUT / "functions.jsonl", "w") as f:
        for text in texts:
            rec = {
                "text": text,
                "get_first": c.TPM.get_first(text),
                "replace_consecutive_punctuation": tp.replace_consecutive_punctuation(text),
                "filter_text": safe(tp.filter_text, text.split("\n")),
                "pre_seg_text": {
                    lang: {m: safe(tp.pre_seg_text, text, lang, m) for m in METHODS} for lang in ["zh", "en"]
                },
            }
            f.write(c.dumps(rec) + "\n")

    rng = random.Random(11)
    alphabet = ["", "a", "ab", "abc", "你好", "。", " ", "abcdef", "一二三四五", "x"]
    with open(OUT / "merge_short_text_in_array.jsonl", "w") as f:
        for _ in range(600):
            texts = [rng.choice(alphabet) for _ in range(rng.randint(0, 8))]
            threshold = rng.randint(0, 8)
            rec = {"texts": texts, "threshold": threshold, "out": c.TPM.merge_short_text_in_array(texts, threshold)}
            f.write(c.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
