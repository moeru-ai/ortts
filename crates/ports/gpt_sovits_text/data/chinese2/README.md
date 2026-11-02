# chinese2 data

Written by `parity/chinese2/convert_data.py`. Do not edit by
hand. Both files are embedded with `include_str!`.

- `opencpop-strict.txt`: `GPT_SoVITS/text/opencpop-strict.txt` from
  GPT-SoVITS commit `04a44eb` (MIT, Copyright (c) 2024 RVC-Boss), byte for
  byte. One `pinyin<TAB>initial final` line per syllable (429 lines). It is
  `pinyin_to_symbol_map` in `chinese2.py`.
- `python_isalpha.txt`: the code point ranges where Python 3.11
  `str.isalpha()` is true (general category Lu, Ll, Lt, Lm or Lo, Unicode
  14.0.0), one `XXXX-YYYY` hex range per line. `_g2p` calls
  `pinyin[0].isalpha()`, and Rust `char::is_alphabetic` is a different set
  (it includes Nl and Other_Alphabetic marks).
