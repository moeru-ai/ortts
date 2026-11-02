"""Download the inputs of the parity scripts into parity/.cache.

    python parity/fetch.py upstream    # GPT-SoVITS source at UPSTREAM_COMMIT
    python parity/fetch.py nltk-data   # nltk packages that g2p_en and pos_tag load

Each command does nothing when its output is already present. The
GPT_SOVITS_ROOT and NLTK_DATA environment variables override the target
directories, as in gpt_sovits_env.py.
"""

import io
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
UPSTREAM_REPO = "RVC-Boss/GPT-SoVITS"
UPSTREAM_COMMIT = "04a44eb977586645f33594b72af13a035a2056c6"
NLTK_PACKAGES = {
    "averaged_perceptron_tagger": "taggers",
    "averaged_perceptron_tagger_eng": "taggers",
    "cmudict": "corpora",
}


def upstream():
    dest = Path(os.environ.get("GPT_SOVITS_ROOT", CACHE_DIR / "GPT-SoVITS"))
    marker = dest / ".ortts-commit"
    if marker.is_file() and marker.read_text().strip() == UPSTREAM_COMMIT:
        print(f"{dest} is at {UPSTREAM_COMMIT[:7]}, skipping")
        return
    if "GPT_SOVITS_ROOT" in os.environ and dest.exists():
        print(f"GPT_SOVITS_ROOT={dest} is set and exists, not touching it")
        return
    url = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{UPSTREAM_COMMIT}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            # Strip the "GPT-SoVITS-<commit>/" top-level directory.
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            m.name = parts[1]
            members.append(m)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(tmp, members=members, filter="data")
        else:
            tar.extractall(tmp, members=members)
    (tmp / ".ortts-commit").write_text(UPSTREAM_COMMIT + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"extracted to {dest}")


def nltk_data():
    dest = Path(os.environ.get("NLTK_DATA", CACHE_DIR / "nltk_data"))
    os.environ["NLTK_DATA"] = str(dest)
    import nltk

    for pkg, kind in NLTK_PACKAGES.items():
        if (dest / kind / f"{pkg}.zip").is_file():
            continue
        if not nltk.download(pkg, download_dir=str(dest), quiet=True, raise_on_error=True):
            sys.exit(f"failed to download nltk package {pkg}")
    print(f"nltk packages {list(NLTK_PACKAGES)} are in {dest}")


COMMANDS = {"upstream": upstream, "nltk-data": nltk_data}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: fetch.py {{{','.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
