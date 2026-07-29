"""Turn a folder of real EDF recordings into a blinded review set.

Produces two things:

  blinded/night_001.edf ...   symlinks (or copies) the clinicians work from
  private/blinding_map.csv    label -> patient_id, record_name, real start time

Keep the private folder away from the machines clinicians use. Labels are
shuffled with a recorded seed so the ordering carries no information about
patient or date, and nothing tells a reviewer how many patients are in the set.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from .data import read_header, to_local


def local_start_str(utc_iso: str) -> str:
    """UTC ISO start time -> 'YYYY-MM-DD HH:MM:SS' in the reporting timezone."""
    if not utc_iso:
        return ""
    from datetime import datetime

    try:
        moment = datetime.fromisoformat(utc_iso)
    except ValueError:
        return utc_iso
    return to_local(moment).strftime("%Y-%m-%d %H:%M:%S")

MAP_COLUMNS = [
    "blind_label",
    "patient_id",
    "record_name",
    "source_path",
    "start_datetime",        # reporting timezone (Eastern), matches the export
    "start_datetime_utc",    # raw EDF meas_date, for cross-checking
    "duration_s",
    "sfreq",
]


def load_map(map_path: Path) -> dict[str, dict]:
    if not Path(map_path).exists():
        return {}
    with open(map_path, newline="") as fh:
        return {row["blind_label"]: row for row in csv.DictReader(fh)}


def write_map(map_path: Path, rows: list[dict]):
    Path(map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MAP_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["blind_label"]):
            writer.writerow({k: row.get(k, "") for k in MAP_COLUMNS})


def _is_under(path: Path, roots: list[Path]) -> bool:
    """True if path is one of roots or lives inside one of them."""
    rp = path.resolve()
    for root in roots:
        if rp == root or root in rp.parents:
            return True
    return False


def scan(
    source_dir: Path,
    blinded_dir: Path,
    map_path: Path,
    seed: int = 20260720,
    copy: bool = False,
    exclude_dirs=None,
) -> list[dict]:
    """Assign blind labels to any EDF in source_dir that does not have one yet.

    Anything under a directory listed in exclude_dirs is skipped -- used to keep
    the training-sample recordings out of the scored review set even if they sit
    inside source_dir.
    """
    source_dir = Path(source_dir)
    blinded_dir = Path(blinded_dir)
    blinded_dir.mkdir(parents=True, exist_ok=True)

    excluded = [Path(d).resolve() for d in (exclude_dirs or [])]

    existing = load_map(map_path)
    known_sources = {row["source_path"] for row in existing.values()}
    next_index = len(existing) + 1

    found = sorted(
        p for p in source_dir.rglob("*")
        if p.suffix.lower() == ".edf" and not _is_under(p, excluded)
    )
    new_files = [p for p in found if str(p.resolve()) not in known_sources]
    random.Random(seed + next_index).shuffle(new_files)

    added = []
    for path in new_files:
        label = f"night_{next_index:03d}"
        next_index += 1
        try:
            header = read_header(path)
        except Exception as exc:  # unreadable file: record it, skip it
            print(f"  ! could not read {path.name}: {exc}")
            continue

        target = blinded_dir / f"{label}.edf"
        if not target.exists():
            if copy:
                shutil.copy2(path, target)
            else:
                try:
                    target.symlink_to(path.resolve())
                except OSError:
                    shutil.copy2(path, target)

        row = {
            "blind_label": label,
            "patient_id": header["patient_id"],
            "record_name": path.name,
            "source_path": str(path.resolve()),
            "start_datetime": local_start_str(header["start_datetime"]),
            "start_datetime_utc": header["start_datetime"],
            "duration_s": f"{header['duration_s']:.3f}",
            "sfreq": f"{header['sfreq']:.3f}",
        }
        existing[label] = row
        added.append(row)
        print(f"  + {label}  <-  {path.name}  ({header['duration_s'] / 3600:.1f} h)")

    write_map(map_path, list(existing.values()))
    return added