#!/usr/bin/env python
"""Build the FF49 dataset (RQ3 primary market) from the Kenneth French data library.

Raw zips are stored verbatim in ``data/raw/ff49`` (with ``<name>.fetched_at.json``) and reused
unless ``--refresh`` is given. Processed files go to ``data/processed/ff49``:

* ``returns_vw.parquet`` / ``returns_ew.parquet``: decimal daily returns of the 49 industries;
* ``prices_vw.parquet``: level index for the walk-forward engine (``cif.data.ff49.price_index``);
* ``rf_daily.parquet``: decimal daily risk-free rate (column ``rf``): from the monthly factors file
  if it is in the raw directory (``--monthly-rf`` downloads it), else the daily file's rate rounded
  to 0.01 %/day (audit F-1); ``summary.rf_source`` says which;
* ``factors_daily.parquet``: Mkt-RF, SMB, HML, RF in decimals;
* ``provenance.json``: URLs, download times, CRSP vintage, SHA-256 of every raw and processed file,
  git state, summary. ``python -m cif.data.provenance verify`` re-hashes them.

    python scripts/data/build_ff49.py [--refresh] [--monthly-rf]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cif.data import ff49, moex  # noqa: E402
from cif.experiments.registry import _git_state  # noqa: E402

logger = logging.getLogger("build_ff49")

CONFIG = {
    "raw_dir": "data/raw/ff49",
    "processed_dir": "data/processed/ff49",
    "base_url": ff49.BASE_URL,
    "files": [ff49.INDUSTRY_ZIP, ff49.FACTORS_ZIP],
    "optional_files": [ff49.FACTORS_MONTHLY_ZIP],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="download again even if cached")
    parser.add_argument("--monthly-rf", action="store_true",
                        help="also download the monthly factors file for the precise daily rf")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_dir = REPO / CONFIG["raw_dir"]
    out_dir = REPO / CONFIG["processed_dir"]
    fetched = {}
    names = CONFIG["files"] + [n for n in CONFIG["optional_files"]
                               if args.monthly_rf or (raw_dir / n).exists()]
    for name in names:
        meta_path = raw_dir / f"{name}.fetched_at.json"
        if args.refresh or not (raw_dir / name).exists():
            logger.info("downloading %s", name)
            fetched[name] = ff49.download_zip(name, raw_dir)
        else:
            fetched[name] = json.loads(meta_path.read_text(encoding="utf-8"))
            logger.info("using cached %s (fetched %s)", name, fetched[name]["fetched_at"])

    summary = ff49.build_dataset(raw_dir, out_dir)
    logger.info("FF49: %s, %s … %s, %d days, vintage %s", summary["n_industries"],
                summary["first_date"], summary["last_date"], summary["n_days"], summary["vintage"])

    git = _git_state()
    prov = {
        "dataset": "ff49",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {"sha": git["sha"], "branch": git["branch"], "dirty": git["dirty"]},
        "config": CONFIG,
        "code_sha256": {p: moex.sha256_file(REPO / p)
                        for p in ("src/cif/data/ff49.py", "scripts/data/build_ff49.py")},
        "sources": {
            "library": "Kenneth R. French Data Library (Tuck School of Business, Dartmouth)",
            "fetched": fetched,
            "vintage": summary["vintage"],
            "missing_codes": list(ff49.MISSING_CODES),
            "units": "files in percent; processed files in decimals",
        },
        "summary": summary,
        "files": {
            "raw": moex.file_manifest(raw_dir),
            "processed": moex.file_manifest(out_dir),
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    logger.info("wrote %s", out_dir / "provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
