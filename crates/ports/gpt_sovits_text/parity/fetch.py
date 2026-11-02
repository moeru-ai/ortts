"""Fetch the inputs of the parity generators into the crate cache.

    python parity/fetch.py upstream|g2pw-model|bert-tokenizer|lid-model ...

Run through the pixi tasks of the same names. Every step is idempotent: it
skips what is already in place. Everything is written under the cache
directory (`parity/.cache`, or `$GPT_SOVITS_TEXT_PARITY_CACHE`):

    GPT-SoVITS/                 upstream checkout at the pinned commit (needs `git`)
      GPT_SoVITS/pretrained_models/fast_langdetect  -> ../../../fast_langdetect
    G2PWModel/                  extracted G2PWModel_1.1.zip
    fast_langdetect/lid.176.bin fasttext language ID model
    run/                        working directory of the generators:
      GPT_SoVITS/text/G2PWModel                          -> ../../../G2PWModel
      GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/  tokenizer files

`run/` exists so that the relative paths hardcoded in upstream
(`GPT_SoVITS/text/G2PWModel`, `GPT_SoVITS/pretrained_models/...`) resolve.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpt_sovits_env as env  # noqa: E402

UPSTREAM_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"

# Same URL as upstream `text/g2pw/onnx_api.py` `download_and_decompress`.
G2PW_URL = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
G2PW_FILES = {
    "bopomofo_to_pinyin_wo_tune_dict.json": "0fe90e26be7653023772b3ad2d08952961bbf295fb39ba685c7172d3d9769f52",
    "char_bopomofo_dict.json": "4832fb6a553a9c5ee0c18b5cb698ce0acc312f324ee77ee61c340266d228184d",
    "config.py": "b6154e494355d14a8d2c8a38d07720f77b34479e26aab21a5f63495ae904da76",
    "g2pW.onnx": "2eb3c71fd95117b2e1abef8d2d0cd78aae894bbe7f0fac105ddc9c32ce63cbd0",
    "MONOPHONIC_CHARS.txt": "fe7ed53098648441a94f414ed5fd3395789d1dceb98dc1e584489c2d43036b69",
    "POLYPHONIC_CHARS.txt": "cc7e5a3eb377e3289af5e3a6be7505ac11a0d21422f80cfbf161059798f8671b",
    "version": "cf063c2ce8ac1403d4dca358f0568dfd32684e7a4412e24bc6e7cd1a7cd0cefb",
}

BERT_REPO = "hfl/chinese-roberta-wwm-ext-large"
BERT_REVISION = "a25cc9e05974bd9687e528edd516f2cfdb3f5db9"
# Only the tokenizer files: the generators replace the BERT model with a fake.
BERT_FILES = {
    "config.json": "53d086daf0ccdddbeb78f8798f34c685a3c48089fa21ec61300527f083fa2563",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874",
    "tokenizer.json": "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"


def log(msg):
    print(f"[fetch] {msg}", file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected, remove=True):
    """Exit if `path` does not have the sha256 `expected`, removing it first by default."""
    got = sha256(path)
    if got != expected:
        if remove:
            path.unlink()
        sys.exit(f"sha256 mismatch for {path}: expected {expected}, got {got}")


def download(url, dest):
    """Download `url` to `dest` through a temporary file in the same directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            step = 1
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                done += len(chunk)
                if total > 16 << 20 and done * 10 >= total * step:
                    log(f"  {done >> 20} / {total >> 20} MiB")
                    step += 1
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def link_dir(link, target):
    """Make `link` point at the directory `target` (symlink, or a junction on Windows)."""
    if link.is_dir():
        return
    if link.is_symlink():  # dangling
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def upstream():
    root = env.UPSTREAM
    commit = env.UPSTREAM_COMMIT
    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if head == commit:
            log(f"upstream: {root} is at {commit[:7]}")
            link_upstream_lid()
            return
        if root != env.CACHE_DIR / "GPT-SoVITS":
            # A checkout given through GPT_SOVITS_ROOT belongs to the user; do not move it.
            sys.exit(f"{root} is at {head[:7]}, but the fixtures are generated from {commit[:7]}")
        log(f"upstream: {root} is at {head[:7]}, checking out {commit[:7]}")
    else:
        if root.exists() and any(root.iterdir()):
            sys.exit(f"{root} exists and is not a git checkout")
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", UPSTREAM_URL], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "-q", "--depth", "1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "advice.detachedHead=false", "checkout", "-q", commit], check=True)
    log(f"upstream: {root} at {commit[:7]}")
    link_upstream_lid()


def g2pw_model():
    model = env.G2PW_MODEL_DIR
    if not all((model / name).is_file() for name in G2PW_FILES):
        if model.exists():
            sys.exit(f"{model} is incomplete; remove it and run again")
        # Same steps as upstream download_and_decompress: the zip holds `G2PWModel_1.1/`.
        cache = model.parent
        zip_path = cache / "G2PWModel_1.1.zip"
        if not zip_path.is_file():
            download(G2PW_URL, zip_path)
        log(f"extracting {zip_path}")
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
            os.replace(Path(tmp) / "G2PWModel_1.1", model)
        zip_path.unlink()
        for name, digest in G2PW_FILES.items():
            check(model / name, digest, remove=False)
    log(f"g2pw-model: {model}")
    link_dir(env.RUN_DIR / "GPT_SoVITS" / "text" / "G2PWModel", model)


def bert_tokenizer():
    out = env.BERT_DIR
    for name, digest in BERT_FILES.items():
        path = out / name
        if not path.is_file():
            download(f"https://huggingface.co/{BERT_REPO}/resolve/{BERT_REVISION}/{name}", path)
            check(path, digest)
    log(f"bert-tokenizer: {out}")


def link_upstream_lid():
    """Link upstream `GPT_SoVITS/pretrained_models/fast_langdetect` to the cached model.

    Upstream `langsegmenter.py` points fast-langdetect at that directory when it
    is imported. Generators that do not replace the detector (english,
    python_ref.py) need the model there, or fast-langdetect silently falls back
    to its small bundled model.
    """
    link = env.UPSTREAM_LID_DIR
    if not env.LID_MODEL.is_file() or not env.UPSTREAM.is_dir() or (link / "lid.176.bin").is_file():
        return
    if env.UPSTREAM != env.CACHE_DIR / "GPT-SoVITS":
        log(f"warning: {link / 'lid.176.bin'} is missing; copy or link {env.LID_MODEL} there")
        return
    if link.is_dir() and not link.is_symlink():
        link.rmdir()  # an empty directory left by a previous run
    link_dir(link, env.LID_MODEL.parent)
    log(f"linked {link} -> {env.LID_MODEL.parent}")


def lid_model():
    path = env.LID_MODEL
    if not path.is_file():
        download(LID_URL, path)
        check(path, LID_SHA256)
    log(f"lid-model: {path}")
    link_upstream_lid()


STEPS = {"upstream": upstream, "g2pw-model": g2pw_model, "bert-tokenizer": bert_tokenizer, "lid-model": lid_model}


def main():
    names = sys.argv[1:] or list(STEPS)
    for name in names:
        if name not in STEPS:
            sys.exit(f"unknown step {name!r}; expected one of {', '.join(STEPS)}")
    env.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        STEPS[name]()


if __name__ == "__main__":
    if shutil.which("git") is None and "upstream" in (sys.argv[1:] or STEPS):
        sys.exit("the upstream step needs git on PATH")
    main()
