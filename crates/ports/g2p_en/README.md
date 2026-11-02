# g2p_en

Rust port of [g2p_en](https://github.com/Kyubyong/g2p) 2.1.0, limited to what
the GPT-SoVITS text frontend uses: the GRU model that `en_G2p` calls for
out-of-vocabulary words.

```rust
use g2p_en::G2p;

let g2p = G2p::new();
let phonemes = g2p.predict("ortts"); // e.g. ["AO1", "R", "T", "S"]
```

## Ported

- `G2p::new()` (`G2p.__init__` without `self.cmu`): vocab tables, GRU weights
  from the embedded `data/checkpoint20.npz`, and `homograph2features`
- `G2p::predict()`, with `encode`, `gru`, `grucell` and `sigmoid`
- `g2p::construct_homograph_dictionary()`

## Not ported

- `G2p.__call__` (nltk tokenizer, POS tagger, text preprocessing): GPT-SoVITS
  overrides it
- `self.cmu` (nltk CMUdict): GPT-SoVITS replaces it
- `expand.py` (`normalize_numbers`) and the `nltk.download` calls

## Parity

Fixtures are generated from the pinned Python package with
`pixi run fixtures` in this directory. All 6901 cases match exactly (6530
`predict` words and 371 `homograph2features` entries), and
`KNOWN_MISMATCHES` is empty. Hidden states can differ from numpy/OpenBLAS in
the last float bits, but no `argmax` differs. See
[parity/README.md](parity/README.md) for the fixture sets, the float notes and
the data checksums.

## License

Apache-2.0, by Kyubyong Park and Jongseok Kim. See [NOTICE](NOTICE) and
[LICENSE](LICENSE).
