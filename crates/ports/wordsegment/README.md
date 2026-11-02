# wordsegment

Rust port of [wordsegment](http://www.grantjenks.com/docs/wordsegment/) 1.3.1.
Full port of `wordsegment/__init__.py` except the CLI. GPT-SoVITS uses it to
split out-of-vocabulary English words in `en_G2p.qryword`.

```rust
wordsegment::load();
assert_eq!(wordsegment::segment("thisisatest"), ["this", "is", "a", "test"]);

let mut segmenter = wordsegment::Segmenter::new();
segmenter.load();
let words = segmenter.segment("camelCaseWords");
```

## Ported

- `Segmenter` with `load`, `parse`, `score`, `isegment`, `segment`, `divide`, `clean`
- Module-level `load`, `clean`, `isegment`, `segment`, `UNIGRAMS`, `BIGRAMS`, `WORDS` (global segmenter)
- Data files `data/{unigrams,bigrams,words}.txt`, embedded with `include_str!`

## Not ported

- `main` (the CLI) and `__main__.py`
- File names: `parse` takes file contents and `load` reads the embedded data
- Lazy `isegment`: the whole result is computed first (same words)

## Parity

Fixtures are generated from the pinned Python package with
`pixi run --manifest-path crates/ports/wordsegment/pixi.toml fixtures`.
All 10,582 cases match exactly (scores compare the f64 bit pattern), with no
known differences. See [parity/README.md](parity/README.md) for details.

## License

Apache-2.0, Copyright 2018 Grant Jenks (based on code by Peter Norvig). See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
