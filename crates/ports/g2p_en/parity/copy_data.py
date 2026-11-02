"""Copy data/checkpoint20.npz and data/homographs.en from the pinned g2p_en package.

    python parity/copy_data.py

The files are copied byte for byte from the installed g2p_en 2.1.0 and their
SHA-1 is checked against the values in parity/README.md.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import g2p_en

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "checkpoint20.npz": "d36cdbec25bc361e7806e1f38d2933b3e3d07e7b",
    "homographs.en": "b4082bad02a3dfcc84b3a100608ba0d26e0fa248",
}


def main():
    src_dir = Path(g2p_en.__file__).resolve().parent
    DATA.mkdir(exist_ok=True)
    for name, sha1 in FILES.items():
        got = hashlib.sha1((src_dir / name).read_bytes()).hexdigest()
        if got != sha1:
            sys.exit(f"{src_dir / name}: SHA-1 {got}, expected {sha1}")
        shutil.copyfile(src_dir / name, DATA / name)
        print(f"copied {name} ({sha1})")


if __name__ == "__main__":
    main()
