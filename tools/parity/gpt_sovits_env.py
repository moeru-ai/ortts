"""Import helper for running the upstream GPT-SoVITS text frontend as a reference.

Usage (from any tools/parity script):

    import gpt_sovits_env  # noqa: F401  (must be imported before GPT-SoVITS modules)
    from text import chinese2

Environment:
    GPT_SOVITS_ROOT        upstream checkout, default ~/Git/github.com/RVC-Boss/GPT-SoVITS
                           (golden fixtures are generated from commit 04a44eb)
    ORTTS_PARITY_ASSETS    assets directory, default tools/parity/assets (set it when
                           running from a git worktree, which has no gitignored assets)

Side effects: changes the working directory to tools/parity/assets/run so the
relative paths hardcoded in upstream (GPT_SoVITS/text/G2PWModel,
GPT_SoVITS/pretrained_models/...) resolve to the gitignored assets directory.
"""

import os
import sys
import types
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.environ.get("ORTTS_PARITY_ASSETS", PARITY_DIR / "assets")).expanduser()
UPSTREAM = Path(os.environ.get("GPT_SOVITS_ROOT", "~/Git/github.com/RVC-Boss/GPT-SoVITS")).expanduser()
UPSTREAM_COMMIT = "04a44eb"

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "GPT_SoVITS"))
sys.path.insert(0, str(PARITY_DIR))
os.chdir(ASSETS_DIR / "run")

# TTS_infer_pack/__init__.py imports the whole torch inference pipeline; register
# the package without executing it so TextPreprocessor only loads its own imports.
_pkg = types.ModuleType("TTS_infer_pack")
_pkg.__path__ = [str(UPSTREAM / "GPT_SoVITS" / "TTS_infer_pack")]
sys.modules.setdefault("TTS_infer_pack", _pkg)
