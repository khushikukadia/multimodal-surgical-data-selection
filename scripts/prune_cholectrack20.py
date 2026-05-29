"""Free disk space after a partial CholecTrack20 download.

The Synapse sync downloads ~23 GB including raw ``.mp4`` videos. This project
only needs ``Frames/*.png`` (or ``images/``) plus ``VIDxx.json``.

Run after a failed or complete download to remove bulky files:

    python scripts/prune_cholectrack20.py --root_dir data/cholectrack20

Use ``--dry_run`` first to see what would be deleted.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import log  # noqa: E402


def prune(root_dir: str, dry_run: bool = False) -> None:
    patterns = [
        os.path.join(root_dir, "**", "*.mp4"),
        os.path.join(root_dir, "**", "*.synapse_download_*"),
    ]
    total_bytes = 0
    n_files = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            total_bytes += size
            n_files += 1
            if dry_run:
                log(f"would delete ({size / 1e6:.1f} MB): {path}")
            else:
                os.remove(path)
                log(f"deleted ({size / 1e6:.1f} MB): {path}")
    action = "Would free" if dry_run else "Freed"
    log(f"{action} ~{total_bytes / 1e9:.2f} GB from {n_files} file(s).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root_dir", default="data/cholectrack20")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    if not os.path.isdir(args.root_dir):
        raise FileNotFoundError(args.root_dir)
    prune(args.root_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
