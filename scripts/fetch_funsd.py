"""One-time FUNSD dataset download/extract helper (Phase 3).

The FUNSD zip extracts a `dataset/` tree with training_data/ and testing_data/, each holding
annotations/ (the JSON V1 needs) and images/. It lands under data/raw/funsd/ (gitignored), so
config.FUNSD_ROOT resolves to data/raw/funsd/dataset.

    python scripts/fetch_funsd.py                 # download + extract
    python scripts/fetch_funsd.py --url <mirror>  # if the default URL is unreachable

If the download fails (the host is occasionally down), grab dataset.zip manually and unzip it
into data/raw/funsd/ so that data/raw/funsd/dataset/training_data/annotations/ exists. Tests
never touch this; it only feeds scripts/evaluate_funsd.py.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

DEFAULT_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="FUNSD dataset.zip URL")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    dest = config.FUNSD_ROOT.parent          # data/raw/funsd (the zip carries dataset/)
    if config.FUNSD_TRAIN.is_dir() and config.FUNSD_TEST.is_dir() and not args.force:
        print(f"FUNSD already present at {config.FUNSD_ROOT} (use --force to re-download)")
        return

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "funsd.zip"
        print(f"downloading {args.url} ...")
        try:
            urllib.request.urlretrieve(args.url, zip_path)
        except Exception as exc:  # network/host failure -> point at the manual path
            raise SystemExit(
                f"download failed: {exc}\n"
                f"  fetch dataset.zip manually and unzip into {dest} so that\n"
                f"  {config.FUNSD_TRAIN} exists, then re-run scripts/evaluate_funsd.py")
        print(f"extracting -> {dest}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)

    if not config.FUNSD_TRAIN.is_dir():
        raise SystemExit(
            f"extracted, but {config.FUNSD_TRAIN} is missing - the archive layout may differ; "
            f"check {dest} and move annotations so config.FUNSD_TRAIN/TEST resolve.")
    n_train = len(list(config.FUNSD_TRAIN.glob("*.json")))
    n_test = len(list(config.FUNSD_TEST.glob("*.json")))
    print(f"ready: {n_train} train + {n_test} test annotation files under {config.FUNSD_ROOT}")


if __name__ == "__main__":
    main()
