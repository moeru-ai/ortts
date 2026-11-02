"""Fetch the pinned GPT-SoVITS checkout that gen_fixtures.py reads.

Run: pixi run --manifest-path crates/ports/jieba_fast/pixi.toml upstream

Makes a shallow, sparse checkout of RVC-Boss/GPT-SoVITS at COMMIT with only
the paths the "prose" fixture set reads (docs/cn and tools/i18n/locale). The
target is parity/.cache/GPT-SoVITS, or $GPT_SOVITS_DIR when it is set. When the
target is already a checkout of COMMIT, nothing is done. Needs git on PATH.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
URL = "https://github.com/RVC-Boss/GPT-SoVITS"
COMMIT = "04a44eb977586645f33594b72af13a035a2056c6"
PATHS = ["docs/cn", "tools/i18n/locale"]


def upstream_dir():
  env = os.environ.get("GPT_SOVITS_DIR")
  return Path(env).expanduser().resolve() if env else CRATE / "parity/.cache/GPT-SoVITS"


def git(dest, *args, capture=False):
  return subprocess.run(["git", "-C", str(dest), *args], check=True, text=True, capture_output=capture)


def is_ready(dest):
  if not (dest / ".git").exists():
    return False
  try:
    head = git(dest, "rev-parse", "HEAD", capture=True).stdout.strip()
  except subprocess.CalledProcessError:
    return False
  return head == COMMIT and all((dest / p).is_dir() for p in PATHS)


def main():
  dest = upstream_dir()
  if is_ready(dest):
    print(f"{dest}: already at {COMMIT[:7]}")
    return
  if dest.exists():
    if os.environ.get("GPT_SOVITS_DIR"):
      sys.exit(f"{dest} exists but is not a checkout of {COMMIT[:7]}; fix or remove it")
    shutil.rmtree(dest)
  dest.mkdir(parents=True)
  git(dest, "init", "-q")
  git(dest, "remote", "add", "origin", URL)
  git(dest, "sparse-checkout", "set", *PATHS)
  git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
  git(dest, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
  if not is_ready(dest):
    sys.exit(f"{dest}: checkout of {COMMIT[:7]} failed")
  print(f"{dest}: checked out {COMMIT[:7]}")


if __name__ == "__main__":
  main()
