"""Download the reference assets for the nltk parity scripts into parity/.cache.

    python parity/fetch.py upstream|g2pw|bert-tokenizer|lid-model|nltk-data

Every target is idempotent: it does nothing when its files are present.
Downloads go to a temporary name first, so an interrupted run leaves no
partial files behind. Locations come from `paths.py` and can be overridden
with the environment variables documented there.
"""

import contextlib
import hashlib
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"

# Same URL as upstream GPT_SoVITS/text/g2pw/onnx_api.py:download_and_decompress.
G2PW_URL = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
G2PW_SHA256 = "b116f6930a7ee55eef6576a8d8e14bf40c1106583439e8ae924b901512379c64"  # 589 MB
G2PW_FILES = (
    "bopomofo_to_pinyin_wo_tune_dict.json",
    "char_bopomofo_dict.json",
    "config.py",
    "g2pW.onnx",
    "MONOPHONIC_CHARS.txt",
    "POLYPHONIC_CHARS.txt",
    "version",
)

# The tokenizer files of hfl/chinese-roberta-wwm-ext-large (the G2PW
# `model_source`), at a fixed revision.
BERT_REPO = "hfl/chinese-roberta-wwm-ext-large"
BERT_REVISION = "a25cc9e05974bd9687e528edd516f2cfdb3f5db9"
BERT_FILES = {
    "config.json": "53d086daf0ccdddbeb78f8798f34c685a3c48089fa21ec61300527f083fa2563",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6",
    "tokenizer_config.json": "61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}

# nltk: the tagger the port embeds. g2p_en (used by the GPT-SoVITS frontend)
# downloads the other two at import time if they are missing.
NLTK_PACKAGES = {
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng.zip",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger.zip",
    "cmudict": "corpora/cmudict.zip",
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def download(url, dest, sha256=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        try:
            with urllib.request.urlopen(url) as r:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                next_report = 0
                while chunk := r.read(1 << 20):
                    tmp.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if total > 8 << 20 and done >= next_report:
                        log(f"  {done >> 20} / {total >> 20} MiB")
                        next_report += total // 10 or 1
            tmp.close()
            if sha256 and h.hexdigest() != sha256:
                raise RuntimeError(f"{url}: sha256 {h.hexdigest()} != {sha256}")
            Path(tmp.name).replace(dest)
        except BaseException:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def upstream():
    """Upstream GPT-SoVITS source tree at UPSTREAM_COMMIT (GitHub archive, no git needed)."""
    dest = paths.UPSTREAM
    stamp = dest / ".ortts-commit"
    if stamp.exists() and stamp.read_text().strip() == paths.UPSTREAM_COMMIT:
        log(f"{dest} is at {paths.UPSTREAM_COMMIT[:7]}")
        return
    if dest.exists() and not stamp.exists() and (dest / ".git").exists():
        log(f"{dest} is a git checkout; using it as is (expected commit {paths.UPSTREAM_COMMIT[:7]})")
        return
    if dest.exists():
        sys.exit(f"{dest} exists but is not at {paths.UPSTREAM_COMMIT[:7]}; remove it and rerun")
    url = f"https://github.com/RVC-Boss/GPT-SoVITS/archive/{paths.UPSTREAM_COMMIT}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "src.zip"
        download(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        (root,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
        (root / ".ortts-commit").write_text(paths.UPSTREAM_COMMIT + "\n")
        root.replace(dest)
    log(f"extracted {dest}")


def g2pw():
    dest = paths.G2PW_MODEL_DIR
    if all((dest / f).exists() for f in G2PW_FILES):
        log(f"{dest} is present")
        return
    if dest.exists():
        sys.exit(f"{dest} exists but is incomplete; remove it and rerun")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "G2PWModel_1.1.zip"
        download(G2PW_URL, archive, G2PW_SHA256)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        src = Path(tmp) / "G2PWModel_1.1"
        missing = [f for f in G2PW_FILES if not (src / f).exists()]
        if missing:
            raise RuntimeError(f"{G2PW_URL}: missing {missing}")
        src.replace(dest)
    log(f"extracted {dest}")


def bert_tokenizer():
    dest = paths.BERT_TOKENIZER_DIR
    for name, sha in BERT_FILES.items():
        f = dest / name
        if f.exists():
            if sha256_file(f) != sha:
                sys.exit(f"{f}: unexpected sha256; remove it and rerun")
            continue
        download(f"https://huggingface.co/{BERT_REPO}/resolve/{BERT_REVISION}/{name}", f, sha)
    log(f"{dest} is present")


def lid_model():
    dest = paths.FAST_LANGDETECT_MODEL
    if dest.exists():
        log(f"{dest} is present")
        return
    download(LID_URL, dest, LID_SHA256)


def nltk_data():
    import nltk

    dest = paths.NLTK_DATA
    dest.mkdir(parents=True, exist_ok=True)
    for pkg, rel in NLTK_PACKAGES.items():
        if (dest / rel).exists():
            log(f"{dest / rel} is present")
            continue
        # nltk prints progress to stdout; keep stdout clean.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = nltk.download(pkg, download_dir=str(dest), quiet=True, raise_on_error=True)
        if not ok:
            sys.exit(f"nltk.download({pkg!r}) failed:\n{out.getvalue()}")
        log(f"downloaded {dest / rel}")


TARGETS = {
    "upstream": upstream,
    "g2pw": g2pw,
    "bert-tokenizer": bert_tokenizer,
    "lid-model": lid_model,
    "nltk-data": nltk_data,
}


def main(argv):
    if not argv or any(a not in TARGETS for a in argv):
        sys.exit(f"usage: fetch.py {{{'|'.join(TARGETS)}}}...")
    for a in argv:
        TARGETS[a]()


if __name__ == "__main__":
    main(sys.argv[1:])
