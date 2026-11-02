"""Dump the upstream GPT-SoVITS frontend output (phone ids) for a corpus file.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml python \
        crates/ports/gpt_sovits_text/parity/python_ref.py \
        crates/ports/gpt_sovits_text/parity/corpus/gpt_sovits_zh_en_yue.json > python_out.json

Needs the `assets` pixi task to have run.

BERT features are not needed for phoneme parity, so get_bert_inf returns zeros.
"""

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
corpus_path = Path(sys.argv[1]).resolve()

import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

with contextlib.redirect_stdout(sys.stderr):
    import torch
    from text.symbols2 import symbols
    from TTS_infer_pack.TextPreprocessor import TextPreprocessor

    class NoBertPreprocessor(TextPreprocessor):
        def get_bert_inf(self, phones, word2ph, norm_text, language):
            return torch.zeros((1024, len(phones)), dtype=torch.float32)

    tp = NoBertPreprocessor(None, None, None)


def main():
    corpus = json.loads(corpus_path.read_text())
    out = []
    for item in corpus:
        text, lang = item["text"], item["lang"]
        rec = {"text": text, "lang": lang, "cat": item["cat"]}
        with contextlib.redirect_stdout(sys.stderr):
            try:
                phones, _, norm = tp.get_phones_and_bert(text, lang, "v2")
                rec["whole"] = {"ids": phones, "phones": [symbols[i] for i in phones], "norm": norm}
            except Exception as e:  # noqa: BLE001
                rec["whole"] = {"error": repr(e)}
            try:
                segs = tp.preprocess(text, lang, "cut0", "v2")
                rec["full"] = [
                    {"ids": s["phones"], "phones": [symbols[i] for i in s["phones"]], "norm": s["norm_text"]}
                    for s in segs
                ]
            except Exception as e:  # noqa: BLE001
                rec["full"] = {"error": repr(e)}
        out.append(rec)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
