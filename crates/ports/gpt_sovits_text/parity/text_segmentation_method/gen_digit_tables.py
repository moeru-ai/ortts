"""Print the Python digit tables embedded in text_segmentation_method.rs.

`cut4` uses the regex `(?<!\\d)\\.(?!\\d)`; for `str` patterns, Python `re`
matches `\\d` with `str.isdecimal()` (Unicode category Nd). `cut5` uses
`str.isdigit()` (Numeric_Type Decimal or Digit). Both depend on the Unicode
version of the interpreter (Python 3.11: Unicode 14.0.0), so the tables are
generated from the pinned interpreter instead of using Rust's char methods.

Run: pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data text_segmentation_method
(`parity/regen.py` replaces the generated block in
crates/ports/gpt_sovits_text/src/tts_infer_pack/text_segmentation_method.rs with the output).
"""

import re
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


def emit(name, doc, rs):
    print(f"/// {doc}")
    print(f"const {name}: &[(char, char)] = &[")
    for a, b in rs:
        print(f"  ('\\u{{{a:04x}}}', '\\u{{{b:04x}}}'),")
    print("];")


re_digit = re.compile(r"\d")
decimal = ranges(str.isdecimal)
assert decimal == ranges(lambda ch: re_digit.fullmatch(ch) is not None)
print(f"// BEGIN GENERATED (gen_digit_tables.py, Unicode {unicodedata.unidata_version})")
emit("PY_DECIMAL_RANGES", "Code points where Python `str.isdecimal()` (and `re` `\\d`) is true.", decimal)
emit("PY_DIGIT_RANGES", "Code points where Python `str.isdigit()` is true.", ranges(str.isdigit))
print("// END GENERATED")
