"""Download the fastText `lid.176.bin` model into the crate cache.

The default target is `parity/.cache/lid.176.bin`; `FAST_LANGDETECT_MODEL`
overrides it. The download is skipped when the file exists with the expected
MD5 (the checksum that fast-langdetect 1.1.0 verifies).
"""

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MD5 = "01810bc59c6a3d2b79c79e6336612f65"

CRATE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = CRATE / "parity" / ".cache" / "lid.176.bin"


def model_path():
    return Path(os.environ.get("FAST_LANGDETECT_MODEL", DEFAULT_MODEL)).expanduser()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    path = model_path()
    if path.is_file():
        if md5(path) == MD5:
            print(f"{path} is present")
            return
        sys.exit(f"{path} exists but its MD5 is not {MD5}; remove it and retry")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"downloading {URL} -> {path}")
    with urllib.request.urlopen(URL) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, 1 << 20)
    if md5(tmp) != MD5:
        tmp.unlink()
        sys.exit(f"MD5 mismatch for {URL}")
    os.replace(tmp, path)
    print("done")


if __name__ == "__main__":
    main()
