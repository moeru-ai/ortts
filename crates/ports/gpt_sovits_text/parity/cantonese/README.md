# cantonese parity fixtures

`gen_fixtures.py` runs upstream `GPT_SoVITS/text/cantonese.py` (04a44eb) with
ToJyutping 3.2.0 from the crate's pixi environment. It writes
`crates/ports/gpt_sovits_text/tests/fixtures/cantonese/`:

```sh
pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml fixtures cantonese
```

## Scope of the port

Ported: `INITIALS`, `rep_map`, `replace_punctuation`, `text_normalize`,
`punctuation_set`, `jyuping_to_initials_finals_tones`, `get_jyutping`, `g2p`.
Left out: `normalizer` (a `cn2an` lambda that nothing calls),
`get_bert_feature` (BERT) and the `__main__` demo.

## Inputs

- `sentences.jsonl` (3357 texts): about 100 hand-written edge cases (every
  `rep_map` key, `...` runs, Latin, digits in several scripts, dates, times,
  units, phone numbers, characters outside `一-龥`, control
  characters, colloquial particles such as 嘅 咗 嘢 佢 唔 哋 喺 啲 冇 嚟), every
  item of `parity/corpus/gpt_sovits_zh_en_yue.json`, and 3200 seeded
  random mixes of Cantonese and Mandarin words (traditional and simplified),
  traditional characters, random Han characters, punctuation, numbers, Latin
  and whitespace. Each record has `rp`, `norm`, `jy`, `g2p` (on `norm`, as
  `cleaner.clean_text` calls it) and `raw_g2p` (on the raw text, which
  reaches the `TypeError` branch for characters that have no reading).
- `syllables.jsonl` (2190 cases): `jyuping_to_initials_finals_tones` for each
  distinct reading of a BMP character in the trie, for hand-written syllables
  (empty string, which gives `IndexError`, `_`, punctuation, `nga`, toneless
  syllables, non-ASCII decimal tones, syllables with no matching initial,
  multi-syllable readings), and for 300 seeded random lists.

The `ValueError` branch of `get_jyutping` cannot be reached, because every
trie reading matches the jyutping pattern.
