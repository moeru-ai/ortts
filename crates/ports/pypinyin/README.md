# pypinyin

Rust port of [pypinyin](https://github.com/mozillazg/python-pinyin) 0.55.0,
limited to what the GPT-SoVITS text frontend uses. The output is the same as
the Python package.

```rust
use pypinyin::{LazyPinyinOptions, PinyinOptions, Style, lazy_pinyin, pinyin};

let opts = PinyinOptions { style: Style::TONE3, neutral_tone_with_five: true, ..Default::default() };
assert_eq!(pinyin("中国", &opts), [["zhong1"], ["guo2"]]);

let opts = LazyPinyinOptions { style: Style::INITIALS, ..Default::default() };
assert_eq!(lazy_pinyin("中国", &opts), ["zh", "g"]);
```

## Ported

- `pinyin`, `lazy_pinyin` and the `Pinyin` class (`core`)
- `Converter`, `DefaultConverter`, `UltimateConverter`, as traits so that
  Python subclasses (the GPT-SoVITS G2PW converter) can be ported
- The styles `NORMAL`, `TONE`, `TONE2`, `TONE3`, `INITIALS`, `FIRST_LETTER`,
  `FINALS`, `FINALS_TONE`, `FINALS_TONE2`, `FINALS_TONE3` (`style`)
- `contrib::tone_convert`, `contrib::uv`, `contrib::neutral_tone`
- `seg::simpleseg`, `seg::mmseg`, `standard`, `phonetic_symbol`
- The bundled dictionaries, unchanged and embedded with `include_str!`

The module table and the subclassing guide are in
[`parity/README.md`](parity/README.md).

## Not ported

- `tone_sandhi` (GPT-SoVITS has its own tone sandhi)
- `errors='exception'`
- The styles `BOPOMOFO*`, `CYRILLIC*`, `WADEGILES`, `GWOYEU`, `BRAILLE_*`,
  and custom styles (`pypinyin.style.register`)
- Custom dictionaries (`load_*_dict`), `slug`, legacy helpers,
  `pypinyin.runner`, `PYPINYIN_NO_PHRASES`

## Parity

Fixtures come from the pinned Python package:
`pixi run --manifest-path crates/ports/pypinyin/pixi.toml fixtures`. No model
is needed. All 407,587 cases in 6 fixture sets match exactly; there are no
known differences. See [`parity/README.md`](parity/README.md) for details.

## License

MIT, like pypinyin. See `LICENSE` (Copyright (c) 2016 mozillazg, 闲耘).
