"""Generate the embedded data of `gpt_sovits_text::english`.

    pixi run --manifest-path crates/ports/gpt_sovits_text/pixi.toml data english

Writes crates/ports/gpt_sovits_text/data/english/:

- cmudict.rep, cmudict-fast.rep, engdict-hot.rep: copied unchanged from
  upstream GPT_SoVITS/text (04a44eb). The Rust port builds the dictionary from
  them the way `read_dict_new` and `hot_reload_hot` do. This script checks that
  `read_dict_new()` gives the same dict as upstream `engdict_cache.pickle`,
  which `get_dict` loads when it exists.
- namedict.tsv: `namedict_cache.pickle` (a pickled `dict[str, list[list[str]]]`)
  as text, in dict insertion order. One line per entry: the key, then one
  tab-separated field per pronunciation, each pronunciation being its phones
  joined by single spaces. The script reads the file back and checks that it
  gives the same dict.
- py_str.json: code point ranges of the CPython 3.11 single-character string
  predicates that `english.py` calls (Unicode `unicodedata.unidata_version`):
  `isalpha` (`str.isalpha`), `upper_or_title` (`Py_UNICODE_ISUPPER or
  Py_UNICODE_ISTITLE`, which is `c.istitle()` for one character) and `lower`
  (`Py_UNICODE_ISLOWER`, which is `c.islower()` for one character). The Rust
  `str.istitle` port reads the last two.
"""

import hashlib
import json
import pickle
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpt_sovits_env  # noqa: E402

UPSTREAM = gpt_sovits_env.UPSTREAM
TEXT = UPSTREAM / "GPT_SoVITS" / "text"
OUT = gpt_sovits_env.DATA / "english"

REP_FILES = ["cmudict.rep", "cmudict-fast.rep", "engdict-hot.rep"]


def read_dict_new(cmu_path, fast_path):
    """Verbatim copy of upstream `read_dict_new` with the paths as arguments."""
    g2p_dict = {}
    with open(cmu_path) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(fast_path) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def ranges(pred):
    out = []
    start = None
    for cp in range(0x110000):
        ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append([start, cp - 1])
            start = None
    if start is not None:
        out.append([start, 0x10FFFF])
    return out


def write_namedict(path):
    with open(TEXT / "namedict_cache.pickle", "rb") as f:
        name_dict = pickle.load(f)
    assert isinstance(name_dict, dict)
    lines = []
    for key, prons in name_dict.items():
        assert isinstance(key, str) and not any(c in key for c in "\t\n\r"), key
        fields = [key]
        for pron in prons:
            assert all(isinstance(p, str) and p and not any(c in p for c in " \t\n\r") for p in pron), (key, pron)
            fields.append(" ".join(pron))
        lines.append("\t".join(fields) + "\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")

    # Read it back as the Rust port does.
    back = {}
    for line in path.read_text(encoding="utf-8").split("\n")[:-1]:
        key, *fields = line.split("\t")
        back[key] = [f.split(" ") if f else [] for f in fields]
    assert back == name_dict and list(back) == list(name_dict)
    return len(name_dict)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in REP_FILES:
        shutil.copyfile(TEXT / name, OUT / name)
        print(name, hashlib.sha256((OUT / name).read_bytes()).hexdigest())

    with open(TEXT / "engdict_cache.pickle", "rb") as f:
        cache = pickle.load(f)
    built = read_dict_new(OUT / "cmudict.rep", OUT / "cmudict-fast.rep")
    assert built == cache, "read_dict_new() differs from engdict_cache.pickle"
    print(f"engdict_cache.pickle == read_dict_new(): {len(built)} entries")

    n = write_namedict(OUT / "namedict.tsv")
    print(f"namedict.tsv: {n} entries")

    tables = {
        "unidata_version": unicodedata.unidata_version,
        "python": sys.version.split()[0],
        "isalpha": ranges(str.isalpha),
        "upper_or_title": ranges(str.istitle),
        "lower": ranges(str.islower),
    }
    (OUT / "py_str.json").write_text(json.dumps(tables, separators=(",", ":")) + "\n")
    print(f"py_str.json: Unicode {unicodedata.unidata_version}")


if __name__ == "__main__":
    main()
