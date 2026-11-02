"""Paths of the crate-local parity tooling, and the upstream import helper.

Every generator imports this module (directly or through `e2e/e2e_common.py`)
for the crate paths. `setup()` puts the upstream GPT-SoVITS checkout on
`sys.path`; call it before importing GPT-SoVITS modules:

    import gpt_sovits_env

    gpt_sovits_env.setup()
    from text import chinese2

Environment (all optional; the defaults are inside the crate):
    GPT_SOVITS_TEXT_PARITY_CACHE  cache directory, default `parity/.cache`
    GPT_SOVITS_ROOT               upstream checkout, default `<cache>/GPT-SoVITS`
                                  (fixtures are generated from commit 04a44eb)

The pixi tasks `upstream`, `g2pw-model`, `bert-tokenizer` and `lid-model`
fill the cache (see `parity/fetch.py`).

Side effects of `setup()`: changes the working directory to `<cache>/run`, so
the relative paths hardcoded in upstream (GPT_SoVITS/text/G2PWModel,
GPT_SoVITS/pretrained_models/...) resolve inside the cache. Upstream
`langsegmenter.py` loads `lid.176.bin` from
`<upstream>/GPT_SoVITS/pretrained_models/fast_langdetect` instead, which the
`lid-model` task links to `<cache>/fast_langdetect`.
"""

import os
import sys
import types
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
CRATE_DIR = PARITY_DIR.parent
FIXTURES = CRATE_DIR / "tests" / "fixtures"
DATA = CRATE_DIR / "data"
SRC = CRATE_DIR / "src"
CORPUS_DIR = PARITY_DIR / "corpus"

CACHE_DIR = Path(os.environ.get("GPT_SOVITS_TEXT_PARITY_CACHE", PARITY_DIR / ".cache")).expanduser().resolve()
UPSTREAM = Path(os.environ.get("GPT_SOVITS_ROOT", CACHE_DIR / "GPT-SoVITS")).expanduser().resolve()
UPSTREAM_COMMIT = "04a44eb977586645f33594b72af13a035a2056c6"
RUN_DIR = CACHE_DIR / "run"
G2PW_MODEL_DIR = CACHE_DIR / "G2PWModel"
LID_MODEL = CACHE_DIR / "fast_langdetect" / "lid.176.bin"
BERT_DIR = RUN_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
# Where upstream langsegmenter.py points fast-langdetect at import time.
UPSTREAM_LID_DIR = UPSTREAM / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect"

_done = False


def setup():
    """Put upstream on `sys.path` and change to `<cache>/run`. Idempotent."""
    global _done
    if _done:
        return
    if not (UPSTREAM / "GPT_SoVITS" / "text").is_dir():
        sys.exit(f"upstream GPT-SoVITS checkout not found at {UPSTREAM}; run the `upstream` pixi task")
    if not RUN_DIR.is_dir():
        sys.exit(f"{RUN_DIR} not found; run the `assets` pixi task")
    if not (UPSTREAM_LID_DIR / "lid.176.bin").is_file():
        print(
            f"warning: {UPSTREAM_LID_DIR / 'lid.176.bin'} is missing (run the `lid-model` pixi task); "
            "LangSegmenter would fall back to fast-langdetect's small model",
            file=sys.stderr,
        )
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / "GPT_SoVITS"))
    os.chdir(RUN_DIR)

    # TTS_infer_pack/__init__.py imports the whole torch inference pipeline; register
    # the package without executing it so TextPreprocessor only loads its own imports.
    pkg = types.ModuleType("TTS_infer_pack")
    pkg.__path__ = [str(UPSTREAM / "GPT_SoVITS" / "TTS_infer_pack")]
    sys.modules.setdefault("TTS_infer_pack", pkg)
    _done = True
