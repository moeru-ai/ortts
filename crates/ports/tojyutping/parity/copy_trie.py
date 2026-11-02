"""Copy `trie.txt` from the installed ToJyutping 3.2.0 wheel to data/trie.txt.

Run with the crate's pinned pixi environment:

    pixi run --manifest-path crates/ports/tojyutping/pixi.toml data

The file is copied byte for byte. The Rust crate embeds it with `include_str!`.
"""

import shutil
from importlib.metadata import version
from pathlib import Path

import ToJyutping

assert version("ToJyutping") == "3.2.0", version("ToJyutping")

CRATE = Path(__file__).resolve().parents[1]
SRC = Path(ToJyutping.__file__).resolve().parent / "trie.txt"
DST = CRATE / "data/trie.txt"


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, DST)
    print(f"trie.txt: {DST.stat().st_size} bytes")


if __name__ == "__main__":
    main()
