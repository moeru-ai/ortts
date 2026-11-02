"""Import helper for running the upstream GPT-SoVITS text frontend as a reference.

Usage (from a script in this directory):

    import gpt_sovits_env  # noqa: F401  (must be imported before nltk, g2p_en and GPT-SoVITS)
    from text import english

Environment:
    GPT_SOVITS_ROOT  upstream checkout, default parity/.cache/GPT-SoVITS
                     (`pixi run upstream` fetches commit 04a44eb there)
    NLTK_DATA        nltk data directory, default parity/.cache/nltk_data
                     (`pixi run nltk-data` downloads the packages there)
"""

import os
import sys
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
CRATE_DIR = PARITY_DIR.parent
CACHE_DIR = PARITY_DIR / ".cache"
UPSTREAM = Path(os.environ.get("GPT_SOVITS_ROOT", CACHE_DIR / "GPT-SoVITS"))
UPSTREAM_COMMIT = "04a44eb977586645f33594b72af13a035a2056c6"
NLTK_DATA = Path(os.environ.setdefault("NLTK_DATA", str(CACHE_DIR / "nltk_data")))

if not (UPSTREAM / "GPT_SoVITS" / "text" / "english.py").is_file():
    sys.exit(f"GPT-SoVITS checkout not found at {UPSTREAM}; run `pixi run upstream` or set GPT_SOVITS_ROOT")

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "GPT_SoVITS"))
