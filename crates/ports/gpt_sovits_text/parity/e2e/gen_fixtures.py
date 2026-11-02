"""End-to-end golden fixtures for `TextPreprocessor` (GPT-SoVITS v2 frontend).

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures e2e

Runs upstream `TTS_infer_pack/TextPreprocessor.py` with the G2PW model, the
fasttext language ID model and the fake BERT model of `e2e_common.py`, and
writes `crates/ports/gpt_sovits_text/tests/fixtures/e2e/`:

- `get_phones_and_bert.jsonl`: `get_phones_and_bert(text, mode, "v2")` for
  every item of `parity/corpus/gpt_sovits_zh_en_yue.json` (modes: the
  item's `lang`, `auto`, `all_zh`, `en`, `auto_yue`; `yue` items also
  `all_yue`) and `gpt_sovits_ja.json` (`ja`, `all_ja`, `auto`), plus the edge
  paragraphs of `e2e_common.py` in `zh`, `ja`, `en`, `yue`, `auto`,
  `auto_yue`, `all_zh`, `all_ja`, `all_yue`, `all_ko` and `ko`.
- `preprocess.jsonl`: `preprocess(text, lang, cutN, "v2")` for N = 0..5, for
  every corpus item (in its `lang`), the edge paragraphs (in `zh`, `en`,
  `ja`, `auto`), and 120 random mixed paragraphs (seed 20261102, two modes
  each).

A `get_phones_and_bert` output is `{"phones": [ids], "norm": str, "bert":
[shape, checksum]}` (see `e2e_common.bert_summary`) or `{"error":
"<exception class>"}`. A `preprocess` output is a list of them (one per
segment) or an error; when it
equals the output of an earlier method of the same record it is stored as
that method's name (`"cut0"`).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_common as c  # noqa: E402

OUT = c.FIXTURES / "e2e"
METHODS = [f"cut{i}" for i in range(6)]
EDGE_MODES = ["zh", "ja", "en", "yue", "auto", "auto_yue", "all_zh", "all_ja", "all_yue", "all_ko", "ko"]
EDGE_PRE_MODES = ["zh", "en", "ja", "auto"]
SEED = 20261102
N_PARAGRAPHS = 120


def gpb(tp, text, mode):
    try:
        phones, bert, norm = c.quiet(tp.get_phones_and_bert, text, mode, "v2")
    except Exception as e:  # noqa: BLE001
        return c.err(e)
    return {"phones": phones, "norm": norm, "bert": c.bert_summary(bert)}


def pre(tp, text, mode, method):
    try:
        segs = c.quiet(tp.preprocess, text, mode, method, "v2")
    except Exception as e:  # noqa: BLE001
        return c.err(e)
    out = []
    for s in segs:
        out.append({"phones": s["phones"], "norm": s["norm_text"], "bert": c.bert_summary(s["bert_features"])})
    return out


def dedupe(outs):
    """Replace an output equal to an earlier method's output by that method's name."""
    res = {}
    for m, o in outs.items():
        res[m] = next((k for k in res if not isinstance(res[k], str) and res[k] == o), o)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tp = c.text_preprocessor()
    zh_en_yue = c.load_corpus("gpt_sovits_zh_en_yue.json")
    ja = c.load_corpus("gpt_sovits_ja.json")

    t0 = time.time()
    with open(OUT / "get_phones_and_bert.jsonl", "w") as f:
        def write(corpus, cat, text, modes):
            for mode in modes:
                rec = {"corpus": corpus, "cat": cat, "text": text, "mode": mode, "out": gpb(tp, text, mode)}
                f.write(c.dumps(rec) + "\n")

        for item in zh_en_yue:
            modes = [item["lang"], "auto", "all_zh", "en", "auto_yue"]
            if item["lang"] == "yue":
                modes.append("all_yue")
            write("zh_en_yue", item["cat"], item["text"], modes)
        print(f"zh_en_yue get_phones_and_bert {time.time() - t0:.0f}s", file=sys.stderr)
        for item in ja:
            write("ja", item["cat"], item["text"], ["ja", "all_ja", "auto"])
        print(f"ja get_phones_and_bert {time.time() - t0:.0f}s", file=sys.stderr)
        for text in c.EDGE_PARAGRAPHS:
            write("edge", "edge", text, EDGE_MODES)

    with open(OUT / "preprocess.jsonl", "w") as f:
        def write(corpus, cat, text, modes):
            for mode in modes:
                out = dedupe({m: pre(tp, text, mode, m) for m in METHODS})
                rec = {"corpus": corpus, "cat": cat, "text": text, "mode": mode, "out": out}
                f.write(c.dumps(rec) + "\n")

        for item in zh_en_yue:
            write("zh_en_yue", item["cat"], item["text"], [item["lang"]])
        for item in ja:
            write("ja", item["cat"], item["text"], [item["lang"]])
        print(f"corpus preprocess {time.time() - t0:.0f}s", file=sys.stderr)
        for text in c.EDGE_PARAGRAPHS:
            write("edge", "edge", text, EDGE_PRE_MODES)
        for p in c.paragraphs(N_PARAGRAPHS, SEED):
            write("paragraphs", p["cat"], p["text"], p["modes"])
        print(f"paragraph preprocess {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
