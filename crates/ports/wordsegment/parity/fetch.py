"""Download the files the parity scripts need into parity/.cache/.

The download is skipped when the file is already present with the expected
SHA-256. Set WORDSEGMENT_PARITY_CACHE to use another cache directory.

Run: pixi run --manifest-path crates/ports/wordsegment/pixi.toml apache-license
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("WORDSEGMENT_PARITY_CACHE") or CRATE / "parity" / ".cache")

APACHE_LICENSE = CACHE / "LICENSE-2.0.txt"
APACHE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
APACHE_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

TARGETS = {"apache-license": (APACHE_LICENSE_URL, APACHE_LICENSE, APACHE_LICENSE_SHA256)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, digest: str) -> None:
    if dest.is_file() and sha256(dest) == digest:
        print(f"{dest} is present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        raise SystemExit(f"{url}: sha256 {got} != {digest}")
    tmp.replace(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    for name in sys.argv[1:] or list(TARGETS):
        fetch(*TARGETS[name])
