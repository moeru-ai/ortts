# g2p_en

Subset port of the Python [g2p_en](https://github.com/Kyubyong/g2p) 2.1.0
package (Apache-2.0, by Kyubyong Park and Jongseok Kim). It holds the parts
that GPT-SoVITS `GPT_SoVITS/text/english.py` uses.

## What GPT-SoVITS uses

`english.py` subclasses `g2p_en.G2p` as `en_G2p`. It overrides `__call__`,
replaces `self.cmu` with its own dictionary and adds `qryword` and
`namedict`. From the parent it uses only:

- `G2p.__init__`: the vocab tables, the GRU weights from `checkpoint20.npz`
  and `homograph2features` from `homographs.en`. `en_G2p.__init__` then
  overrides the `read` and `complex` entries.
- `G2p.predict(word)`: the out-of-vocabulary fallback at the end of
  `en_G2p.qryword`.

No other GPT-SoVITS module imports g2p_en.

## Ported

All of these are in `g2p_en/g2p.py`, which is `g2p_en::g2p` in Rust. `G2p` is also
re-exported at the crate root, as `g2p_en/__init__.py` does.

| Python | Rust |
|---|---|
| `construct_homograph_dictionary()` | `g2p::construct_homograph_dictionary()` |
| `G2p.__init__` (without `self.cmu`) | `G2p::new()` |
| `G2p.graphemes`, `phonemes`, `g2idx`, `idx2g`, `p2idx`, `idx2p` | public fields |
| `G2p.homograph2features` | public field (mutable, for the `en_G2p` overrides) |
| `G2p.load_variables` (`enc_*`, `dec_*`, `fc_*` arrays) | public `Array2` / `Vec<f32>` fields, read from the embedded npz |
| `G2p.sigmoid` | `G2p::sigmoid` |
| `G2p.grucell` | `G2p::grucell` |
| `G2p.gru` | `G2p::gru` |
| `G2p.encode` | `G2p::encode` |
| `G2p.predict` | `G2p::predict` |

The numpy arrays are all `float32` (`<f4`). The Rust code computes in `f32`
and uses the numpy operation order in `grucell`. `gru`, `grucell` and
`encode` drop the batch axis, because Python always uses a batch of 1.
`predict` iterates over `list(word)`, which is Unicode code points, as Rust
`chars()` does. Every character outside `a`..`z` maps to `<unk>`.

## Left out

- `G2p.__call__`: GPT-SoVITS overrides it. The left-out code includes the
  nltk `TweetTokenizer`, `pos_tag` and the text preprocessing.
- `self.cmu = cmudict.dict()` (nltk CMUdict): GPT-SoVITS replaces it.
- `expand.py` (`normalize_numbers`): only the parent `__call__` uses it.
- The module-level `nltk.download` calls.
- `self.variables`, the open `NpzFile`. Its arrays are the weight fields.

## Data

- `data/checkpoint20.npz`: copied unchanged from g2p_en 2.1.0
  (SHA-1 `d36cdbec25bc361e7806e1f38d2933b3e3d07e7b`). It is an uncompressed
  zip of `.npy` files. `src/npz.rs` reads it directly, so no conversion
  script is necessary.
- `data/homographs.en`: copied unchanged from g2p_en 2.1.0
  (SHA-1 `b4082bad02a3dfcc84b3a100608ba0d26e0fa248`).

## Parity

Run `pixi run fixtures` in this directory (or
`pixi run --manifest-path crates/ports/g2p_en/pixi.toml fixtures` from the
repository root) to generate the fixtures. The task fetches GPT-SoVITS
commit `04a44eb` and the nltk data into the gitignored `parity/.cache`, to
collect the words that `en_G2p` sends to `predict`. `pixi run data` copies
`data/` from the pinned g2p_en package. See `parity/README.md`. The
fixtures are `predict.jsonl` (6530 words),
`homograph2features.json` (371 entries) and `vocab.json`.

| Fixture set | Cases | Exact |
|---|---|---|
| edge (empty string, `<unk>` characters, 20-step cap, long words) | 65 | 65 |
| oov | 76 | 76 |
| names | 51 | 51 |
| products | 85 | 85 |
| gpt_sovits (words that `en_G2p.qryword` passes to `predict`) | 23 | 23 |
| cmudict (seeded sample) | 2597 | 2597 |
| namedict (seeded sample) | 600 | 600 |
| random_az (seeded, length 1..20) | 2486 | 2486 |
| random_mixed (seeded, with digits, punctuation, uppercase, non-ASCII) | 547 | 547 |
| homograph2features | 371 | 371 |

The documented exception list (`KNOWN_MISMATCHES` in `tests/parity.rs`) is
empty.

`predict` does about 7 million multiply-adds for a 10-letter word.
It takes about 10 s with `cargo test -p g2p_en --release` and several minutes
in the unoptimized dev profile. To make the dev profile faster, add
`[profile.dev.package.g2p_en] opt-level = 3` to the workspace `Cargo.toml`.
This does not change the f32 results, because Rust does not contract
`a * b + c` into fused multiply-add.

### Float summation order

numpy sends `np.matmul(x, w.T)` to OpenBLAS (0.3.34 in the crate's pixi
environment). The OpenBLAS summation order depends on the CPU kernel, and
this port does not copy it. The Rust code sums each dot product in index
order without fused multiply-add. Because of this, hidden states can differ
from numpy in the last bits. A probe on Apple Silicon found that no simple
order (sequential, or 4/8/16/32 partial sums, with or without FMA) gives
the same bits as OpenBLAS for all rows. `np.exp` and `np.tanh` can also
differ from Rust `f32::exp` and `f32::tanh` by an ulp.

`predict` returns only `argmax` results. These are the same in all fixture
cases. The smallest gap between the top two logits in the fixtures is
`9.5e-7`, at a decoder step of `avfbqaiiilksrcroqao`, and that case also
matches. A word with an even smaller gap could produce a different
phoneme. If that happens, add the word to `KNOWN_MISMATCHES` and to this
README.
