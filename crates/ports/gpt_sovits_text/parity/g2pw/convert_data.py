"""Collect the data files of the g2pw port into crates/ports/gpt_sovits_text/data/g2pw/.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data g2pw

Writes:
- polyphonic.rep, polyphonic-fix.rep: copied byte for byte from GPT-SoVITS @ 04a44eb
  (GPT_SoVITS/text/g2pw). Upstream caches them into polyphonic.pickle; the port parses
  the .rep files itself, so the pickle is not needed.
- tokenizer.json: the chinese-roberta-wwm-ext-large tokenizer that GPT-SoVITS loads with
  AutoTokenizer.from_pretrained(bert_path), copied byte for byte from the cache dir
  (the `bert-tokenizer` pixi task downloads it).
  The script checks that the backend tokenizer transformers builds from that directory
  serializes to the same JSON, so the file is what Python really runs.
- opencc/*.txt and opencc/s2tw.json: the dictionaries of the OpenCC("s2tw") config of the
  pinned opencc wheel, dumped from the shipped .ocd2 (marisa trie) files to the OpenCC text
  format with the wheel's own opencc_dict tool, and the config file itself.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

import opencc  # noqa: E402
from transformers.models.auto.tokenization_auto import AutoTokenizer  # noqa: E402

OUT = gpt_sovits_env.DATA / "g2pw"
UPSTREAM_G2PW = gpt_sovits_env.UPSTREAM / "GPT_SoVITS/text/g2pw"
BERT_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"  # chinese2.py default bert_path

OPENCC_DIR = Path(opencc.__file__).resolve().parent / "clib"
S2TW_DICTS = [
    "CJK_Compatibility_Ideographs",
    "STPhrases",
    "STPhrases_GeneratedFromRegionalPhrases",
    "STCharacters",
    "TWVariantsPhrases",
    "TWVariants",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("polyphonic.rep", "polyphonic-fix.rep"):
        shutil.copyfile(UPSTREAM_G2PW / name, OUT / name)

    tokenizer_json = Path(BERT_PATH) / "tokenizer.json"  # cwd is <cache>/run
    tok = AutoTokenizer.from_pretrained(BERT_PATH)
    effective = json.loads(tok.backend_tokenizer.to_str())
    original = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    assert effective == original, "transformers rebuilt the tokenizer; commit its effective JSON instead"
    shutil.copyfile(tokenizer_json, OUT / "tokenizer.json")

    (OUT / "opencc").mkdir(exist_ok=True)
    config = json.loads((OPENCC_DIR / "share/opencc/s2tw.json").read_text(encoding="utf-8"))
    shutil.copyfile(OPENCC_DIR / "share/opencc/s2tw.json", OUT / "opencc/s2tw.json")
    used = sorted(set(_dict_files(config)))
    assert used == sorted(f"{d}.ocd2" for d in S2TW_DICTS), used
    for name in S2TW_DICTS:
        subprocess.run(
            [
                str(OPENCC_DIR / "bin/opencc_dict"),
                "-i",
                str(OPENCC_DIR / f"share/opencc/{name}.ocd2"),
                "-o",
                str(OUT / f"opencc/{name}.txt"),
                "-f",
                "ocd2",
                "-t",
                "text",
            ],
            check=True,
        )
    print(f"opencc {opencc.__version__}")


def _dict_files(node):
    if isinstance(node, dict):
        if node.get("type") == "ocd2":
            yield node["file"]
        for v in node.values():
            yield from _dict_files(v)
    elif isinstance(node, list):
        for v in node:
            yield from _dict_files(v)


if __name__ == "__main__":
    main()
