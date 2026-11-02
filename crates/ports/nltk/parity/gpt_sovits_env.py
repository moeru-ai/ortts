"""Import helper for running the upstream GPT-SoVITS text frontend as a reference.

Usage:

    import gpt_sovits_env  # noqa: F401  (must be imported before GPT-SoVITS modules)
    from text import english

Run the `upstream`, `g2pw`, `bert-tokenizer` and `lid-model` tasks first
(`fixtures` depends on them). Paths come from `paths.py`.

Side effects: changes the working directory to `<cache>/run`. Upstream
hardcodes the relative paths `GPT_SoVITS/text/G2PWModel` and
`GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`, so this module
links (or copies) the cached model and tokenizer to those paths under
`<cache>/run`. fast_langdetect reads the model from
`<upstream>/GPT_SoVITS/pretrained_models/fast_langdetect/`, so the cached
`lid.176.bin` is linked (or copied) there.
"""

import os
import shutil
import sys
import types

import paths

UPSTREAM = paths.UPSTREAM
UPSTREAM_COMMIT = paths.UPSTREAM_COMMIT


def _require(path, task):
    if not path.exists():
        sys.exit(f"missing {path}; run `pixi run --manifest-path {paths.CRATE / 'pixi.toml'} {task}`")


def _link_dir(target, link):
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():  # dangling link
        link.unlink()
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if sys.platform == "win32":
            import _winapi

            try:
                _winapi.CreateJunction(str(target), str(link))
                return
            except OSError:
                pass
        shutil.copytree(target, link)


def _link_file(target, link):
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(target, link)
    except OSError:
        shutil.copyfile(target, link)


_require(UPSTREAM / "GPT_SoVITS" / "text" / "english.py", "upstream")
_require(paths.G2PW_MODEL_DIR / "g2pW.onnx", "g2pw")
_require(paths.BERT_TOKENIZER_DIR / "vocab.txt", "bert-tokenizer")
_require(paths.FAST_LANGDETECT_MODEL, "lid-model")

RUN_DIR = paths.CACHE / "run"
_link_dir(paths.G2PW_MODEL_DIR, RUN_DIR / "GPT_SoVITS" / "text" / "G2PWModel")
_link_dir(
    paths.BERT_TOKENIZER_DIR,
    RUN_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large",
)
_link_file(
    paths.FAST_LANGDETECT_MODEL,
    UPSTREAM / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect" / "lid.176.bin",
)

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "GPT_SoVITS"))
sys.path.insert(0, str(paths.PARITY))
os.chdir(RUN_DIR)

# TTS_infer_pack/__init__.py imports the whole torch inference pipeline; register
# the package without executing it so TextPreprocessor only loads its own imports.
_pkg = types.ModuleType("TTS_infer_pack")
_pkg.__path__ = [str(UPSTREAM / "GPT_SoVITS" / "TTS_infer_pack")]
sys.modules.setdefault("TTS_infer_pack", _pkg)
