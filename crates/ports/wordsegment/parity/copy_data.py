"""Copy data/unigrams.txt, data/bigrams.txt and data/words.txt of this crate
unmodified from the installed wordsegment 1.3.1 package.

Run: pixi run --manifest-path crates/ports/wordsegment/pixi.toml data
"""

import shutil
from pathlib import Path

import wordsegment

assert wordsegment.__version__ == "1.3.1", wordsegment.__version__

CRATE = Path(__file__).resolve().parents[1]
SRC = Path(wordsegment.__file__).resolve().parent
OUT = CRATE / "data"
OUT.mkdir(parents=True, exist_ok=True)

for name in ("unigrams.txt", "bigrams.txt", "words.txt"):
    shutil.copyfile(SRC / name, OUT / name)
    print(f"wrote {OUT / name}")
