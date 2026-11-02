"""Export the data that the cn2an port embeds.

Usage: pixi run --manifest-path crates/ports/cn2an/pixi.toml data

Writes to data/ in the crate:

- t2s.tsv: proces 0.1.7 `T2S_DICT` (a Python dict literal in proces/conf.py),
  one `traditional<TAB>simplified` pair per line, in dict order. cn2an's
  An2Cn calls proces.preprocess(..., ["traditional_to_simplified", ...]).
- python_decimal_ranges.txt: the code points that Python's `re` matches with
  `\\d` in a str pattern (Unicode category Nd, i.e. str.isdecimal()), as
  inclusive hex ranges `start-end`, one per line. Generated from the pinned
  interpreter so the Rust port matches its Unicode version exactly.
"""

import re
import sys
import unicodedata
from pathlib import Path

import proces
from proces.conf import T2S_DICT

assert proces.__version__ == "0.1.7", proces.__version__

CRATE = Path(__file__).resolve().parents[1]
DATA = CRATE / "data"
DATA.mkdir(parents=True, exist_ok=True)

with open(DATA / "t2s.tsv", "w", encoding="utf-8", newline="\n") as f:
    for k, v in T2S_DICT.items():
        assert len(k) == 1 and len(v) == 1 and "\t" not in k + v and "\n" not in k + v
        f.write(f"{k}\t{v}\n")

DIGIT = re.compile(r"\d")
ranges = []
for cp in range(sys.maxunicode + 1):
    if DIGIT.fullmatch(chr(cp)):
        if ranges and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])

with open(DATA / "python_decimal_ranges.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"# Python {sys.version.split()[0]} re \\d, Unicode {unicodedata.unidata_version}\n")
    for a, b in ranges:
        f.write(f"{a:X}-{b:X}\n")

print(f"t2s: {len(T2S_DICT)} entries, decimal ranges: {len(ranges)}")
