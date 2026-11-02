"""Write LICENSE of this crate: upstream copyright plus the Apache-2.0 text.

wordsegment 1.3.1 does not ship its LICENSE file in the wheel, so the Apache-2.0
text is downloaded from apache.org (see fetch.py, task "apache-license") into
parity/.cache/LICENSE-2.0.txt. The trailing newline is dropped, so the file
matches the license text that earlier versions of this script took from
another Apache-2.0 package in the environment.

Run: pixi run --manifest-path crates/ports/wordsegment/pixi.toml license
"""

from pathlib import Path

from fetch import APACHE_LICENSE

CRATE = Path(__file__).resolve().parents[1]
OUT = CRATE / "LICENSE"

HEADER = """\
wordsegment 1.3.1
Copyright 2018 Grant Jenks

Based on code from the chapter "Natural Language Corpus Data" from the book
"Beautiful Data" (Segaran and Hammerbacher, 2009).
Original Copyright (c) 2008-2009 by Peter Norvig

"""


def apache_text() -> str:
    if not APACHE_LICENSE.is_file():
        raise SystemExit(f"{APACHE_LICENSE} is missing; run the apache-license task first")
    text = APACHE_LICENSE.read_text(encoding="utf-8")
    if not (text.lstrip().startswith("Apache License") and "Version 2.0, January 2004" in text):
        raise SystemExit(f"{APACHE_LICENSE} is not the Apache-2.0 license text")
    return text.rstrip("\n")


with open(OUT, "w", encoding="utf-8", newline="") as f:
    f.write(HEADER + apache_text())
print(f"wrote {OUT}")
