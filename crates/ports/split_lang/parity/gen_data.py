"""Generate the data files of the `split_lang` port crate.

    pixi run --manifest-path crates/ports/split_lang/pixi.toml data

Writes into the crate's `data/`:

- `python_isdigit.txt`: the code point ranges for which Python 3.11
  `str.isdigit()` is true (Unicode 14.0, `Numeric_Type` `Digit` or
  `Decimal`), one `XXXX..YYYY` hex range per line. `LangSplitter.pre_split`
  classifies characters with `char.isdigit()`, and Rust has no equivalent.
- `budoux/ja.json`, `budoux/zh-hans.json`: verbatim copies of the budoux
  0.10.2 default models that split-lang loads
  (`budoux.load_default_japanese_parser`,
  `budoux.load_default_simplified_chinese_parser`).
"""

import shutil
import sys
import unicodedata
from pathlib import Path

import budoux

CRATE = Path(__file__).resolve().parents[1]
DATA = CRATE / "data"

assert sys.version_info[:2] == (3, 11), sys.version
assert unicodedata.unidata_version == "14.0.0", unicodedata.unidata_version
assert budoux.__version__ == "0.10.2", budoux.__version__


def ranges(pred):
    out = []
    start = None
    for cp in range(0x110000 + 1):
        ok = cp < 0x110000 and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{a:04X}..{b:04X}" for a, b in ranges(str.isdigit)]
    (DATA / "python_isdigit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    models = Path(budoux.__file__).parent / "models"
    (DATA / "budoux").mkdir(exist_ok=True)
    for name in ["ja.json", "zh-hans.json"]:
        shutil.copyfile(models / name, DATA / "budoux" / name)


if __name__ == "__main__":
    main()
