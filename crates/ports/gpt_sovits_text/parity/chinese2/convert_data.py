"""Collect the data files of the `gpt_sovits_text::chinese2` port.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data chinese2

Writes `crates/ports/gpt_sovits_text/data/chinese2/`:

- `opencpop-strict.txt`: `GPT_SoVITS/text/opencpop-strict.txt` of the upstream
  checkout (`parity/.cache/GPT-SoVITS` or `$GPT_SOVITS_ROOT`, commit 04a44eb), byte for byte.
- `python_isalpha.txt`: the code point ranges where Python 3.11 `str.isalpha()`
  is true (Unicode general category Lu, Ll, Lt, Lm or Lo; Unicode 14.0.0),
  one `XXXX-YYYY` hex range per line. `_g2p` calls `pinyin[0].isalpha()`.
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

UPSTREAM = gpt_sovits_env.UPSTREAM
OUT = gpt_sovits_env.DATA / "chinese2"


def main():
    assert sys.version_info[:2] == (3, 11), sys.version
    assert unicodedata.unidata_version == "14.0.0", unicodedata.unidata_version
    OUT.mkdir(parents=True, exist_ok=True)
    src = UPSTREAM / "GPT_SoVITS/text/opencpop-strict.txt"
    (OUT / "opencpop-strict.txt").write_bytes(src.read_bytes())

    ranges = []
    start = None
    for cp in range(0x110000 + 1):
        ok = cp < 0x110000 and chr(cp).isalpha()
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            ranges.append((start, cp - 1))
            start = None
    with open(OUT / "python_isalpha.txt", "w", encoding="ascii", newline="\n") as f:
        for a, b in ranges:
            f.write(f"{a:04X}-{b:04X}\n")
    print(f"{len(ranges)} isalpha ranges", file=sys.stderr)


if __name__ == "__main__":
    main()
