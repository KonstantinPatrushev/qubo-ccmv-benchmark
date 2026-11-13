"""Decompress the raw rows of every registered run (runs/*/results.jsonl.gz -> results.jsonl), keeping the archives;
the analyses read results.jsonl."""

import gzip
import shutil
from pathlib import Path

for gz in sorted(Path(__file__).resolve().parents[1].glob("runs/*/results.jsonl.gz")):
    out = gz.with_suffix("")
    if not out.exists():
        with gzip.open(gz, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(out)
