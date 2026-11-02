"""Golden fixtures for `gpt_sovits_text::cleaner` (`GPT_SoVITS/text/cleaner.py`).

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures cleaner

Uses the setup of `parity/e2e/e2e_common.py` (G2PW model,
`NotPortedError` for `text.korean` and `text.chinese`). The `version`
environment variable is removed first, so `version=None` means `"v2"`.
Writes `crates/ports/gpt_sovits_text/tests/fixtures/cleaner/`:

- `clean_text.jsonl`: `clean_text(text, language, version)` for the
  zh/en/yue corpus, every third item of the ja corpus, the edge paragraphs
  and special-symbol texts, in `zh`, `ja`, `en`, `yue` with `version=None`;
  every fifth text also in `ko`, `x`, `all_zh`, `""` and with `version`
  `"v1"` (`zh`, `ja`, `en`, `yue`, `ko`) and `"v3"` (`zh`, `en`). The
  zh/en/yue corpus is not run in `ja`: the Japanese frontend port
  (jpreprocess) differs from Open JTalk on many Chinese-only sentences (see
  `crates/ports/gpt_sovits_text/parity/README.md`), and GPT-SoVITS only gives it
  segments that `LangSegmenter` labels `ja` or text in the `ja` modes.
- `clean_special.jsonl`: `clean_special(text, language, special_s,
  target_symbol)` for `zh` and `yue` (both specials), and an unknown
  language (`KeyError`).
- `text_to_sequence.jsonl`: `text_to_sequence(text, language)` (always
  `TypeError` upstream).

An output is `{"phones": "<space separated>", "word2ph": [..] | null,
"norm": str}` or `{"error": "<exception class>"}`.
"""

import os
import sys
from pathlib import Path

os.environ.pop("version", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
import e2e_common as c  # noqa: E402

OUT = c.FIXTURES / "cleaner"

SPECIAL_TEXTS = [
    "价格￥100元。",
    "￥",
    "^",
    "^_^",
    "你好^啊，^^今天。",
    "￥￥￥五十",
    "a^b",
    "今天^天气^很好。",
    "Hello^world",
    "こんにちは^さようなら",
    "佢^唔係^好鍾意。",
    "UNK测试",
    "「こんにちは」『世界』",
    "a",
    "ab",
    "I",
    "OK.",
    "，",
    "",
    " ",
]


def out(f, *args):
    try:
        phones, word2ph, norm = c.quiet(f, *args)
    except Exception as e:  # noqa: BLE001
        return c.err(e)
    return {"phones": " ".join(phones), "word2ph": word2ph, "norm": norm}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zh_en_yue = [x["text"] for x in c.load_corpus("gpt_sovits_zh_en_yue.json")]
    ja = [x["text"] for x in c.load_corpus("gpt_sovits_ja.json")][::3]
    texts = list(dict.fromkeys(zh_en_yue + ja + c.EDGE_PARAGRAPHS + SPECIAL_TEXTS))

    with open(OUT / "clean_text.jsonl", "w") as f:
        for i, text in enumerate(texts):
            # `ja` only for text that GPT-SoVITS would give the Japanese frontend
            # (see the module docs).
            ja_ok = text not in zh_en_yue
            cases = [(lang, None) for lang in ["zh", "ja", "en", "yue"] if lang != "ja" or ja_ok]
            if i % 5 == 0:
                cases += [(lang, None) for lang in ["ko", "x", "all_zh", ""]]
                cases += [(lang, "v1") for lang in ["zh", "ja", "en", "yue", "ko"] if lang != "ja" or ja_ok]
                cases += [(lang, "v3") for lang in ["zh", "en"]]
            for lang, version in cases:
                rec = {"text": text, "language": lang, "version": version}
                rec["out"] = out(c.cleaner.clean_text, text, lang, version)
                f.write(c.dumps(rec) + "\n")

    with open(OUT / "clean_special.jsonl", "w") as f:
        for text in SPECIAL_TEXTS + zh_en_yue[::10]:
            for lang in ["zh", "yue", "fr"]:
                for special_s, _, target in c.cleaner.special:
                    rec = {"text": text, "language": lang, "special_s": special_s, "target_symbol": target}
                    rec["out"] = out(c.cleaner.clean_special, text, lang, special_s, target)
                    f.write(c.dumps(rec) + "\n")

    with open(OUT / "text_to_sequence.jsonl", "w") as f:
        for text, lang in [("你好", "zh"), ("hello", "en"), ("", "ja")]:
            try:
                res = c.cleaner.text_to_sequence(text, lang)
                rec = {"text": text, "language": lang, "out": res}
            except Exception as e:  # noqa: BLE001
                rec = {"text": text, "language": lang, "out": c.err(e)}
            f.write(c.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
