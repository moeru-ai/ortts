"""Copy the pypinyin 0.55.0 dictionaries into data/ of the crate.

data/pinyin_dict.json and data/phrases_dict.json are the files that the
pypinyin package ships, unchanged (byte for byte).

Usage: pixi run --manifest-path crates/ports/pypinyin/pixi.toml data
"""

import shutil
from pathlib import Path

import pypinyin

assert pypinyin.__version__ == "0.55.0", pypinyin.__version__

CRATE = Path(__file__).resolve().parents[1]
SRC = Path(pypinyin.__file__).resolve().parent
OUT = CRATE / "data"
NAMES = ["pinyin_dict.json", "phrases_dict.json"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copyfile(SRC / name, OUT / name)
        print(f"copied {name} ({(OUT / name).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
