# inflect

Rust port of [inflect](https://github.com/jaraco/inflect) 7.6.0, limited to
`number_to_words` and `ordinal`, which GPT-SoVITS uses to read numbers as
English words.

```rust
use inflect::{NumberToWordsOptions, engine};

let mut p = engine::new();
assert_eq!(p.number_to_words(101, &NumberToWordsOptions::default())?, "one hundred and one");
let opts = NumberToWordsOptions { andword: "", zero: "oh", group: 2, ..Default::default() };
assert_eq!(p.number_to_words(1905, &opts)?, "nineteen, oh five");
assert_eq!(p.ordinal("twenty-one")?, "twenty-first");
assert_eq!(p.ordinal("3")?, "3rd");
# Ok::<(), inflect::Error>(())
```

## Ported

- `engine::number_to_words` with `group`, `comma`, `andword`, `zero`, `one`
  and `decimal` (`NumberToWordsOptions`).
- `engine::ordinal`.
- `NumberOrWord` for the Python `int` or `str` argument, and `Error` for the
  Python exceptions on this path.

## Not ported

- Everything else in inflect (plurals, articles, currency, classical modes, ...).
- `wantlist=True`, `threshold`, `float` and `Decimal` inputs.
- Python `int` values outside `i128` (pass them as a string).
- An `andword` with a backslash is inserted literally.

## Parity

Fixtures come from the pinned Python package:
`pixi run --manifest-path crates/ports/inflect/pixi.toml fixtures` (no model
or download needed). All 59704 cases match exactly, outputs and errors. There
are no known differences. See `parity/README.md` for the
fixture sets, the call sites in GPT-SoVITS and the details of the scope.

## License

MIT, copyright Paul Dyson, Jason R. Coombs and inflect contributors. See
`LICENSE`.
