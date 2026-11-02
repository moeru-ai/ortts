# cn2an

Rust port of [cn2an](https://github.com/Ailln/cn2an) 0.5.24, limited to what
the GPT-SoVITS text frontend uses: `cn2an.transform(text, "an2cn")`.

```rust
let text = cn2an::transform("小王捡了100块钱", "an2cn", false)?;
assert_eq!(text, "小王捡了一百块钱");

let up = cn2an::an2cn("123", "up")?; // "壹佰贰拾叁"
```

## Ported

- `cn2an::transform` / `Transform`: the `"an2cn"` method, with and without `direct`
- `cn2an::an2cn` / `An2Cn`: all four modes (`low`, `up`, `rmb`, `direct`)
- `cn2an::conf`: the an2cn number and unit tables
- `cn2an::proces::preprocess`: the two pipelines from proces 0.1.7 that `An2Cn` runs

## Not ported

- The `"cn2an"` direction and all of `Cn2An` (returns `Error::NotPorted`)
- `int` / `float` / `None` inputs to `an2cn` (the Rust API takes `&str`)
- `warnings.warn` calls (the behaviour is ported, the warnings are not emitted)
- The other `proces` pipelines, `cn2an.utils`, and the test and benchmark modules

## Parity

Fixtures come from the pinned Python package
(`pixi run --manifest-path crates/ports/cn2an/pixi.toml fixtures`; no model or
download needed) and are checked by `tests/parity.rs`. All 15232 cases
(6390 `transform`, 1598 `transform` with `direct`, 1811 inputs x 4 `an2cn`
modes) match exactly, with no known differences. See
[`parity/README.md`](parity/README.md) for details.

## License

MIT, copyright Ailln (cn2an and proces). See [`LICENSE`](LICENSE).
