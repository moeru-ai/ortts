"""Run the fixture or data generators of the crate.

    python parity/regen.py fixtures [module ...]
    python parity/regen.py data [name ...]

Run through the `fixtures` and `data` pixi tasks, which fetch what the
generators need first. With no names, every generator of the group runs, in
the order below. Each generator runs in its own interpreter, as when it is run
by hand.
"""

import subprocess
import sys
import time
from pathlib import Path

PARITY = Path(__file__).resolve().parent
SRC = PARITY.parent / "src"

FIXTURES = [
    "symbols",
    "text_segmentation_method",
    "zh_normalization",
    "tone_sandhi",
    "chinese2",
    "g2pw",
    "english",
    "en_normalization",
    "japanese",
    "cantonese",
    "lang_segmenter",
    "cleaner",
    "text_preprocessor",
    "e2e",
]

# name -> (script, Rust file whose `// BEGIN GENERATED` ... `// END GENERATED`
# block is replaced with the script's stdout, or None if the script writes its files).
DATA = {
    "chinese2": ("chinese2/convert_data.py", None),
    "english": ("english/gen_data.py", None),
    "g2pw": ("g2pw/convert_data.py", None),
    "zh_normalization": ("zh_normalization/convert_data.py", None),
    "en_normalization": ("en_normalization/gen_tables.py", None),
    "tone_sandhi": ("tone_sandhi/gen_isnumeric_table.py", SRC / "tone_sandhi.rs"),
    "text_segmentation_method": (
        "text_segmentation_method/gen_digit_tables.py",
        SRC / "tts_infer_pack" / "text_segmentation_method.rs",
    ),
}


def splice(path, block):
    """Replace the lines from `// BEGIN GENERATED` to `// END GENERATED` in `path`."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    begin = [i for i, line in enumerate(lines) if line.startswith("// BEGIN GENERATED")]
    end = [i for i, line in enumerate(lines) if line.startswith("// END GENERATED")]
    if len(begin) != 1 or len(end) != 1 or begin[0] > end[0]:
        sys.exit(f"{path}: expected one // BEGIN GENERATED ... // END GENERATED block")
    lines[begin[0] : end[0] + 1] = [block]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(lines))


def run(script, splice_into=None):
    t = time.time()
    print(f"[regen] {script}", file=sys.stderr, flush=True)
    cmd = [sys.executable, str(PARITY / script)]
    if splice_into is None:
        subprocess.run(cmd, check=True)
    else:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout
        splice(splice_into, out.replace("\r\n", "\n"))
        print(f"wrote {splice_into}", file=sys.stderr)
    print(f"[regen] {script}: {time.time() - t:.0f}s", file=sys.stderr, flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("fixtures", "data"):
        sys.exit(__doc__)
    group, names = sys.argv[1], sys.argv[2:]
    known = FIXTURES if group == "fixtures" else list(DATA)
    for name in names:
        if name not in known:
            sys.exit(f"unknown {group} generator {name!r}; expected one of {', '.join(known)}")
    for name in names or known:
        if group == "fixtures":
            run(f"{name}/gen_fixtures.py")
        else:
            run(*DATA[name])


if __name__ == "__main__":
    main()
