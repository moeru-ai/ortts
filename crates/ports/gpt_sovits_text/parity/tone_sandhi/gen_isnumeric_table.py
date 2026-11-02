"""Print the Python `str.isnumeric()` table embedded in tone_sandhi.rs.

`_neural_sandhi` and `_yi_sandhi` call `str.isnumeric()` on single
characters. The result depends on the Unicode version of the interpreter
(Python 3.11: Unicode 14.0.0), so the table is generated from the pinned
interpreter instead of using Rust's `char::is_numeric` (which differs, for
example for CJK numerals such as 一 and 两).

Run: pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data tone_sandhi
(`parity/regen.py` replaces the generated block in
crates/ports/gpt_sovits_text/src/tone_sandhi.rs with the output).
"""

import sys
import unicodedata


def ranges(pred):
    out = []
    start = None
    for cp in range(sys.maxunicode + 2):
        ok = cp <= sys.maxunicode and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    return out


rs = ranges(str.isnumeric)
print(f"// BEGIN GENERATED (gen_isnumeric_table.py, Unicode {unicodedata.unidata_version})")
print("/// Code points where Python `str.isnumeric()` is true.")
print("const PY_NUMERIC_RANGES: &[(char, char)] = &[")
for a, b in rs:
    print(f"  ('\\u{{{a:04x}}}', '\\u{{{b:04x}}}'),")
print("];")
print("// END GENERATED")
