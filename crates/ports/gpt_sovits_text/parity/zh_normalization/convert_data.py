"""Extract the data tables of GPT_SoVITS/text/zh_normalization into
crates/ports/gpt_sovits_text/data/zh_normalization/.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data zh_normalization

Writes:
- simplified_charcters.txt / traditional_characters.txt: the two literal strings
  of char_convert.py, verbatim, without a trailing newline.
- python_decimal_digits.txt: the code point ranges that Python's `re` matches
  with `\\d` on str patterns (Unicode category Nd in the interpreter's Unicode
  database, 14.0.0 for Python 3.11), one `XXXX-XXXX` range per line. The Rust
  port compiles `\\d` to this class so it does not depend on the Unicode
  version of the regex crate.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

gpt_sovits_env.setup()

from text.zh_normalization import char_convert  # noqa: E402

OUT = gpt_sovits_env.DATA / "zh_normalization"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "simplified_charcters.txt").write_text(char_convert.simplified_charcters, encoding="utf-8")
    (OUT / "traditional_characters.txt").write_text(char_convert.traditional_characters, encoding="utf-8")

    digit = re.compile(r"\d")
    ranges = []
    for cp in range(sys.maxunicode + 1):
        if digit.match(chr(cp)):
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
    lines = [f"{a:04X}-{b:04X}\n" for a, b in ranges]
    (OUT / "python_decimal_digits.txt").write_text(
        f"# Python {sys.version.split()[0]} re \\d, Unicode {unicodedata.unidata_version}\n" + "".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
