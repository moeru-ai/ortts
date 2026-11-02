# jieba_fast

Rust port of [jieba_fast](https://github.com/deepcs233/jieba_fast) 0.53,
limited to what the GPT-SoVITS text frontend uses. Where jieba_fast's C
extension and its pure Python code differ, the port follows the code that
jieba_fast actually runs.

```rust
let words = jieba_fast::lcut("我来到北京清华大学", true);
let search = jieba_fast::lcut_for_search("中华人民共和国", true);
let tagged = jieba_fast::posseg::lcut("我爱北京天安门", true); // Vec<Pair { word, flag }>
```

## Ported

- `Tokenizer` and the module functions `cut`, `lcut`, `cut_for_search`,
  `lcut_for_search` (accurate mode, with and without HMM, default dictionary)
- `_jieba_fast_functions`: the C extension's `_get_DAG_and_calc` and `_viterbi`
- `finalseg`: the HMM segmenter for words that are not in the dictionary
- `posseg`: `POSTokenizer`, `Pair`, `cut`, `lcut`

## Not ported

- `cut_all=True`
- User dictionaries (`load_userdict`, `add_word`, `del_word`, `suggest_freq`,
  `Force_Split_Words`), `set_dictionary` and the dictionary cache file
- `tokenize`, parallel mode (`enable_parallel`) and `jieba_fast.analyse`
- Python's `logging` output

## Parity

Fixtures come from the pinned Python package through the crate's pixi
workspace:
`pixi run --manifest-path crates/ports/jieba_fast/pixi.toml fixtures`.
All 13,858 cases in 7 fixture sets match exactly for `lcut`,
`lcut_for_search` and `posseg::lcut`, so the exception list is empty. See
[parity/README.md](parity/README.md) for the fixture sets, the data files and
the port notes.

## License

MIT, copyright Sun Junyi (jieba) and deepcs233 (jieba_fast). See
[LICENSE](LICENSE).
