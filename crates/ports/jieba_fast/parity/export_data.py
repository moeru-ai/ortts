"""Export the jieba_fast 0.53 data files for the Rust port.

Run: pixi run --manifest-path crates/ports/jieba_fast/pixi.toml data

Writes to data/ in the crate:
- dict.txt                    copied unmodified from the package
- finalseg/prob_start.txt     "<state>\t<log prob>" per line
- finalseg/prob_trans.txt     "<from>\t<to>:<log prob>\t..." per line
- finalseg/prob_emit.txt      "@<state>" header, then "<char>\t<log prob>" lines
- posseg/char_state_tab.txt   "<char>\t<state>\t<state>..." per line
- posseg/prob_start.txt       same formats as finalseg; a posseg state
- posseg/prob_trans.txt       ("B", "n") is written as "B,n"
- posseg/prob_emit.txt

The Python modules are dict literals (prob_*.py, char_state_tab.py), which Rust
cannot read. Floats are written with repr(), which round-trips exactly through
Rust's correctly rounded f64 parser. Dict and tuple order is kept.
"""

import shutil
from pathlib import Path

import jieba_fast
from jieba_fast import finalseg, posseg

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "data"
PKG = Path(jieba_fast.__file__).resolve().parent


def state_name(state):
  if isinstance(state, tuple):
    assert len(state) == 2 and all("," not in s and "\t" not in s for s in state), state
    return f"{state[0]},{state[1]}"
  return state


def check_char(ch):
  assert len(ch) == 1 and not ch.isspace() and ord(ch) >= 0x20, repr(ch)
  return ch


def fl(x):
  assert isinstance(x, float), x
  s = repr(x)
  assert float(s) == x
  return s


def write(path, lines):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    for line in lines:
      f.write(line + "\n")
  print(f"{path.relative_to(CRATE)}: {len(lines)} lines")


def export_hmm(sub, start_p, trans_p, emit_p):
  write(OUT / sub / "prob_start.txt", [f"{state_name(s)}\t{fl(p)}" for s, p in start_p.items()])
  write(
    OUT / sub / "prob_trans.txt",
    ["\t".join([state_name(s)] + [f"{state_name(t)}:{fl(p)}" for t, p in nxt.items()]) for s, nxt in trans_p.items()],
  )
  lines = []
  for s, table in emit_p.items():
    lines.append("@" + state_name(s))
    lines.extend(f"{check_char(c)}\t{fl(p)}" for c, p in table.items())
  write(OUT / sub / "prob_emit.txt", lines)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  shutil.copyfile(PKG / "dict.txt", OUT / "dict.txt")
  print("data/dict.txt: copied")
  export_hmm("finalseg", finalseg.start_P, finalseg.trans_P, finalseg.emit_P)
  export_hmm("posseg", posseg.start_P, posseg.trans_P, posseg.emit_P)
  write(
    OUT / "posseg/char_state_tab.txt",
    ["\t".join([check_char(c)] + [state_name(s) for s in states]) for c, states in posseg.char_state_tab_P.items()],
  )


if __name__ == "__main__":
  main()
