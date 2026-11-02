"""Download the full fasttext language identification model `lid.176.bin`.

    pixi run --manifest-path crates/ports/split_lang/pixi.toml lid-model

Writes `parity/.cache/fast_langdetect/lid.176.bin` (131 MB), or the path in
`FAST_LANGDETECT_MODEL`. Skips the download when the file is already there
with the MD5 that fast-langdetect 1.1.0 checks.
"""

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = Path(os.environ.get("FAST_LANGDETECT_MODEL", HERE / ".cache/fast_langdetect/lid.176.bin"))
URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MD5 = "01810bc59c6a3d2b79c79e6336612f65"  # fast_langdetect.infer.FASTTEXT_LARGE_MODEL_MD5


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if TARGET.is_file() and md5(TARGET) == MD5:
        print(f"{TARGET}: present", file=sys.stderr)
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_name(TARGET.name + ".part")
    print(f"downloading {URL} -> {TARGET}", file=sys.stderr)
    with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = md5(tmp)
    if got != MD5:
        tmp.unlink()
        sys.exit(f"{URL}: MD5 {got}, expected {MD5}")
    os.replace(tmp, TARGET)


if __name__ == "__main__":
    main()
