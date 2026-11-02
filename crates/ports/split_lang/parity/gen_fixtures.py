"""Generate golden fixtures for the `split_lang` port crate.

    pixi run --manifest-path crates/ports/split_lang/pixi.toml fixtures

Runs the pinned split-lang 2.1.1 (budoux 0.10.2, fast-langdetect 1.1.0) with
the full `lid.176.bin` (default `parity/.cache/fast_langdetect/lid.176.bin`,
fetched by the `lid-model` task; override with `FAST_LANGDETECT_MODEL`), set
as fast-langdetect's default detector the way GPT-SoVITS'
`text/LangSegmenter/langsegmenter.py` does, and writes the crate's
`tests/fixtures/`:

- `split_by_lang.jsonl`: one line per input (see `lang_inputs.py`) with
  `LangSplitter.split_by_lang(text)` for
  - `gsv`: `LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)`
    (the GPT-SoVITS configuration),
  - `gsv_no_digit`: the same with `merge_across_digit = False` (what
    `LangSegmenter.getTexts` runs),
  - `default`: `LangSplitter()` (split-lang's own `DEFAULT_LANG_MAP`), for
    every 5th input.
  Each result is `[[lang, text, index, length], ...]` or
  `{"error": "<exception class>"}`.
- `budoux.jsonl`: `budoux` Japanese and Simplified Chinese `parse` of every
  input that contains a CJK character, and of every ZH_JA section that
  `pre_split` produces.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CRATE = HERE.parent
OUT = CRATE / "tests/fixtures"
LID_MODEL = Path(os.environ.get("FAST_LANGDETECT_MODEL", HERE / ".cache/fast_langdetect/lid.176.bin"))

import budoux  # noqa: E402
import fast_langdetect  # noqa: E402
from fast_langdetect.infer import LangDetectConfig, LangDetector  # noqa: E402
from split_lang import LangSectionType, LangSplitter  # noqa: E402

from lang_inputs import inputs  # noqa: E402

# `LangSegmenter.DEFAULT_LANG_MAP` of GPT-SoVITS 04a44eb
# (GPT_SoVITS/text/LangSegmenter/langsegmenter.py), copied so that this
# script does not need the upstream checkout.
GSV_LANG_MAP = {
    "zh": "zh",
    "yue": "zh",
    "wuu": "zh",
    "zh-cn": "zh",
    "zh-tw": "x",
    "ko": "ko",
    "ja": "ja",
    "en": "en",
}

# Same model file as GPT-SoVITS' pretrained_models/fast_langdetect.
# fast-langdetect loads `<cache_dir>/lid.176.bin`.
if LID_MODEL.name != "lid.176.bin" or not LID_MODEL.is_file():
    sys.exit(f"{LID_MODEL}: not found (run the lid-model task, or point FAST_LANGDETECT_MODEL at lid.176.bin)")
fast_langdetect.infer._default_detector = LangDetector(LangDetectConfig(cache_dir=LID_MODEL.parent))

JA = budoux.load_default_japanese_parser()
ZH = budoux.load_default_simplified_chinese_parser()


def run(splitter, text):
    try:
        return [[s.lang, s.text, s.index, s.length] for s in splitter.split_by_lang(text)]
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gsv = LangSplitter(lang_map=GSV_LANG_MAP)
    gsv_no_digit = LangSplitter(lang_map=GSV_LANG_MAP)
    gsv_no_digit.merge_across_digit = False
    default = LangSplitter()

    items = inputs()
    budoux_texts = []
    seen = set()
    with open(OUT / "split_by_lang.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, (cat, text) in enumerate(items):
            rec = {"cat": cat, "text": text, "gsv": run(gsv, text), "gsv_no_digit": run(gsv_no_digit, text)}
            if i % 5 == 0:
                rec["default"] = run(default, text)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cands = [text] + [
                s.text for s in gsv.pre_split(text) if s.lang_section_type is LangSectionType.ZH_JA
            ]
            for t in cands:
                if t not in seen and any("\u3040" <= c <= "\u9fff" for c in t):
                    seen.add(t)
                    budoux_texts.append(t)
    with open(OUT / "budoux.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for t in [""] + budoux_texts:
            f.write(json.dumps({"text": t, "ja": JA.parse(t), "zh": ZH.parse(t)}, ensure_ascii=False) + "\n")
    print(f"{len(items)} inputs, {len(budoux_texts) + 1} budoux texts", file=sys.stderr)


if __name__ == "__main__":
    main()
