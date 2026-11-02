# lang_segmenter parity

Fixtures for `gpt_sovits_text::lang_segmenter`, the port of
`GPT_SoVITS/text/LangSegmenter/langsegmenter.py` (GPT-SoVITS @ 04a44eb).

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures lang_segmenter
FAST_LANGDETECT_MODEL=$PWD/crates/ports/gpt_sovits_text/parity/.cache/fast_langdetect/lid.176.bin \
    cargo test -p gpt_sovits_text --test lang_segmenter
```

## Scope

Ported: `full_en`, `full_cjk`, `split_jako`, `merge_lang`, `LangSegmenter`
with `DEFAULT_LANG_MAP` and `getTexts(text, default_lang)`, and the
module-level replacement of fast-langdetect's default detector
(`set_fast_langdetect_cache_dir`). `getTexts` returns `Vec<LangText>`
(`{'lang', 'text'}` dicts) or the Python exception as
`split_lang::SplitLangError` (`IndexError` for empty or whitespace-only
text). Internally the dicts are shared `Rc<RefCell<_>>` values, so the
aliasing of the Python lists (the `ko_list` that is not reset per item, the
in-place `lang` updates in the digit and `x` passes) behaves the same.

Left out: `jieba.setLogLevel` (logging only) and the `__main__` demo.

Model: Python replaces `fast_langdetect.infer._default_detector` at import
time. Rust has no import-time side effects; call
`lang_segmenter::set_fast_langdetect_cache_dir(dir)` or
`fast_langdetect::set_default_detector(..)` before `getTexts`.

## Fixtures

`crates/ports/gpt_sovits_text/tests/fixtures/lang_segmenter/` (1.2 MB):

- `get_texts.jsonl`: `getTexts(text, d)` for `d` in `""`, `"zh"`, `"ja"`,
  `"ko"` over the 3571 inputs of `parity/lang_segmenter/lang_inputs.py`
  (hand-written edge cases, single fragments, 2300 seeded random mixes, every
  text of `parity/corpus/*.json`).
- `functions.jsonl`: `full_en`, `full_cjk`, `split_jako("ja"|"ko", ...)` for
  1508 texts, including single-character probes at every character class
  boundary of the regexes and CJK ranges.

Result (macOS arm64): `getTexts` 3571/3571 for each of the four
`default_lang` values, functions 1508/1508. The documented exception list in
`tests/lang_segmenter.rs` is empty.
