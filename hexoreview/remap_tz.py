"""One-off: add the Eastern 'start_datetime' / UTC 'start_datetime_utc' columns
to an existing blinding map, without re-scanning the EDFs.

    uv run python -m hexoreview.remap_tz private/blinding_map.csv

Safe to run more than once. Writes a .bak copy first.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from .blinding import local_start_str


def upgrade(map_path: Path):
    map_path = Path(map_path)
    df = pd.read_csv(map_path, dtype=str).fillna("")

    if "start_datetime" not in df.columns:
        raise SystemExit("No 'start_datetime' column found; is this a blinding map?")

    # Figure out what the existing start_datetime holds. A '+00:00' or 'T'
    # marks the raw UTC iso; a plain 'YYYY-MM-DD HH:MM:SS' is already Eastern.
    def is_utc_iso(v: str) -> bool:
        return "T" in v or v.endswith("+00:00")

    utc_col = df["start_datetime_utc"] if "start_datetime_utc" in df.columns else None
    if utc_col is None:
        # derive: if start_datetime looks like UTC iso, keep it as the UTC source
        utc_col = df["start_datetime"].where(df["start_datetime"].map(is_utc_iso), "")

    df["start_datetime_utc"] = utc_col
    df["start_datetime"] = [
        local_start_str(u) if u else sd
        for u, sd in zip(df["start_datetime_utc"], df["start_datetime"])
    ]

    shutil.copy2(map_path, map_path.with_suffix(".csv.bak"))
    df.to_csv(map_path, index=False)
    print(f"Upgraded {map_path}")
    print(f"  backup: {map_path.with_suffix('.csv.bak')}")
    print(df[["blind_label", "start_datetime", "start_datetime_utc"]].head().to_string(index=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "private/blinding_map.csv"
    upgrade(Path(path))
