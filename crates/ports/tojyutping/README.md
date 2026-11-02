# tojyutping

Rust port of [ToJyutping](https://github.com/CanCLID/ToJyutping) 3.2.0,
limited to what the GPT-SoVITS text frontend uses
(`ToJyutping.get_jyutping_list` in `GPT_SoVITS/text/cantonese.py`).

```rust
let syllables = tojyutping::get_jyutping_list("廣東話");
// [("廣", Some("gwong2")), ("東", Some("dung1")), ("話", Some("waa2"))]

let text = tojyutping::get_jyutping("廣東話");
// "廣(gwong2)東(dung1)話(waa2)"
```

## Ported

- `get_jyutping_list`, `get_jyutping`, `get_jyutping_text` (also on
  `ToJyutping` / `JyutpingConverter`)
- `trie`: the embedded `data/trie.txt` (unmodified from the 3.2.0 wheel) and
  its greedy longest-match lookup
- `jyutping` and `utils` (punctuation tables, `format_romanization_text`)

## Not ported

- Candidates: `get_jyutping_candidates`, `Trie.get_all`
- IPA: `get_ipa*`, `jyutping2ipa`
- Customization: `customize`, `CustomizableTrie`, `Jyutping(str)` parsing
- `g2p` and phoneme lists

## Parity

Fixtures are generated from ToJyutping 3.2.0 with
`pixi run --manifest-path crates/ports/tojyutping/pixi.toml fixtures`
(no model or download needed). All 47379 cases across every trie character,
12000 trie words, 1724 sentences and the constants match exactly. There are
no known differences. See [`parity/README.md`](parity/README.md).

## License

BSD-2-Clause, copyright the Cantonese Computational Linguistics Infrastructure
Development Workgroup. See `LICENSE`.
