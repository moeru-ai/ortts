"""Crate-local paths for the nltk parity scripts.

Every default is inside this crate. The cache lives in `parity/.cache/`
(gitignored) and is filled by the download tasks in `pixi.toml`. The
environment variables below override single locations.

    NLTK_PARITY_CACHE      cache root, default parity/.cache
    GPT_SOVITS_ROOT        upstream GPT-SoVITS checkout, default <cache>/GPT-SoVITS
    G2PW_MODEL_DIR         G2PW ONNX model, default <cache>/G2PWModel
    BERT_TOKENIZER_DIR     chinese-roberta-wwm-ext-large tokenizer,
                           default <cache>/chinese-roberta-wwm-ext-large
    FAST_LANGDETECT_MODEL  fastText lid.176.bin, default <cache>/fast_langdetect/lid.176.bin
    NLTK_DATA              nltk_data directory, default <cache>/nltk_data

Importing this module sets `NLTK_DATA` (if unset) before nltk is imported, so
nltk and g2p_en look in the crate cache first.
"""

import os
from pathlib import Path

PARITY = Path(__file__).resolve().parent
CRATE = PARITY.parent
FIXTURES = CRATE / "tests" / "fixtures"
DATA = CRATE / "data"
CORPUS = PARITY / "corpus" / "gpt_sovits_zh_en_yue.json"


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


CACHE = _env_path("NLTK_PARITY_CACHE", PARITY / ".cache")
UPSTREAM = _env_path("GPT_SOVITS_ROOT", CACHE / "GPT-SoVITS")
G2PW_MODEL_DIR = _env_path("G2PW_MODEL_DIR", CACHE / "G2PWModel")
BERT_TOKENIZER_DIR = _env_path("BERT_TOKENIZER_DIR", CACHE / "chinese-roberta-wwm-ext-large")
FAST_LANGDETECT_MODEL = _env_path("FAST_LANGDETECT_MODEL", CACHE / "fast_langdetect" / "lid.176.bin")
NLTK_DATA = _env_path("NLTK_DATA", CACHE / "nltk_data")

# GPT-SoVITS commit the golden fixtures are generated from.
UPSTREAM_COMMIT = "04a44eb977586645f33594b72af13a035a2056c6"

os.environ["NLTK_DATA"] = str(NLTK_DATA)
