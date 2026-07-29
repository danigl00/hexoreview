"""Training recordings for reviewers to learn from.

Point `--samples-dir` at a folder of full recordings (keep it OUTSIDE the scan
source so these never enter the scored worklist). Given a seizure CSV, each
recording that contains at least one seizure is placed (symlinked, or copied)
under a neutral `sample_NNN.edf` label into the samples output folder, and every
seizure it contains is recorded in a manifest as a position in seconds from the
recording start.

The dashboard lists these under "Training samples". Loading one opens the whole
recording and shows the labelled seizures as read-only marks the reviewer can
select and jump to. They are never scored and never enter the worklist.

Identity is handled exactly like the blinded night set: reviewers see only the
neutral label, elapsed time, and the waveform. The real filename and patient id
go into a private index written beside the samples source, never into the
samples output folder that reviewers use.

Only the header is read here (no signal decoding, no EDF export), so this needs
mne but NOT edfio.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Timezone the CSV timestamps are written in. DST-aware.
CSV_TZ = ZoneInfo("America/New_York")

# CSV columns. Adjust here if your headers differ.
ONSET_COL = "electric_onset"
OFFSET_COL = "sz_offset"
DATE_COL = "sz_date"       # only used if the onset/offset are time-of-day only
TYPE_COL = "sz_type"       # shown in the marks table; not identifying

MANIFEST_NAME = "samples_manifest.csv"       # reviewer-facing: label + marks
MANIFEST_COLUMNS = ["label", "clip_file", "mark", "sz_type", "onset_s", "offset_s"]

PRIVATE_MAP_NAME = "samples_private_map.csv"  # coordinator-only: label -> source
PRIVATE_COLUMNS = [
    "label", "record_name", "patient_id",
    "start_datetime_utc", "duration_s", "n_marks", "source_path",
]


def _as_eastern(t: datetime) -> datetime:
    return t.replace(tzinfo=CSV_TZ) if t.tzinfo is None else t


# --------------------------------------------------------------------------- #
# manifest (no mne needed to read it back)
# --------------------------------------------------------------------------- #
def write_manifest(path: Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})


def load_manifest(path: Path) -> dict[str, dict]:
    """label -> {"clip_file": str, "marks": [{onset_s, offset_s, sz_type}, ...]}."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            info = out.setdefault(label, {"clip_file": row.get("clip_file", ""), "marks": []})
            try:
                on = float(row.get("onset_s") or 0.0)
                off = float(row.get("offset_s") or 0.0)
            except ValueError:
                continue
            info["marks"].append(
                {"onset_s": on, "offset_s": off, "sz_type": row.get("sz_type", "") or ""}
            )
    for info in out.values():
        info["marks"].sort(key=lambda m: m["onset_s"])
    return out


def _write_private(path: Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRIVATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PRIVATE_COLUMNS})


# --------------------------------------------------------------------------- #
# building the sample set (needs mne, header only)
# --------------------------------------------------------------------------- #
def _open_lazy(edf_path: Path):
    """Open an EDF header without loading data, with the latin-1 fallback."""
    import mne

    try:
        return mne.io.read_raw_edf(str(edf_path), preload=False, verbose="error")
    except Exception:
        return mne.io.read_raw_edf(
            str(edf_path), preload=False, encoding="latin1", verbose="error"
        )


def _parse_event(row: dict, col: str):
    """Parse an Eastern event time. Full ISO timestamps are used as-is; a bare
    time-of-day is combined with the sz_date column. Returns tz-aware, or None."""
    value = (row.get(col) or "").strip()
    if not value:
        return None
    try:
        return _as_eastern(datetime.fromisoformat(value))
    except ValueError:
        pass
    date = (row.get(DATE_COL) or "").strip()
    if not date:
        return None
    try:
        return _as_eastern(datetime.fromisoformat(f"{date}T{value}"))
    except ValueError:
        return None


def _link(src: Path, target: Path, copy: bool):
    """Place src at target as a neutral label: symlink, or copy on failure."""
    src, target = Path(src), Path(target)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy:
        shutil.copy2(src, target)
        return
    try:
        target.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, target)


def build_samples(
    samples_src: Path,
    samples_out: Path,
    csv_path: Path,
    *,
    copy: bool = False,
) -> list[dict]:
    """Build the training sample set from the recordings in samples_src.

    Every recording containing at least one seizure (by onset falling inside its
    interval) is given a neutral label and linked into samples_out; its seizures
    are written to the manifest as second-offsets from the recording start.
    Rebuilt from scratch each run. Returns the manifest rows.
    """
    samples_src = Path(samples_src)
    samples_out = Path(samples_out)

    if not samples_src.is_dir():
        print(f"  samples: {samples_src} is not a directory; skipping.")
        return []
    if not Path(csv_path).exists():
        print(f"  samples: no CSV at {csv_path}; skipping.")
        return []

    samples_out.mkdir(parents=True, exist_ok=True)
    for stale in samples_out.glob("sample_*.edf"):  # clean previous runs
        try:
            stale.unlink()
        except OSError:
            pass

    with open(csv_path, newline="") as fh:
        seizures = list(csv.DictReader(fh))

    edfs = sorted(samples_src.glob("*.edf"))
    print(f"  {len(edfs)} recording(s) in {samples_src}; {len(seizures)} seizure row(s) in CSV")

    manifest: list[dict] = []
    private: list[dict] = []
    index = 0
    matched = 0
    for edf in edfs:
        try:
            full = _open_lazy(edf)
        except Exception as exc:
            print(f"  ! could not read {edf.name}: {exc}")
            continue

        start = full.info["meas_date"]
        if start is None:
            print(f"  ! {edf.name} has no start date in its header; skipping.")
            continue
        dur = float(full.n_times) / float(full.info["sfreq"])
        end = start + timedelta(seconds=dur)

        hits = []
        for row in seizures:
            onset = _parse_event(row, ONSET_COL)
            if onset is not None and start <= onset <= end:
                hits.append((onset, row))
        if not hits:
            print(f"  · {edf.name}: no seizures fall inside this recording")
            continue
        hits.sort(key=lambda pair: pair[0])

        index += 1
        label = f"sample_{index:03d}"
        target = samples_out / f"{label}.edf"
        try:
            _link(edf, target, copy)
        except Exception as exc:
            print(f"  ! {label}: could not place {edf.name}: {exc}")
            index -= 1
            continue

        subj = full.info.get("subject_info") or {}
        patient_id = (
            f"{subj.get('first_name', '') or ''}{subj.get('last_name', '') or ''}"
        ).strip().lower() or "unknown"

        for mark_no, (onset, row) in enumerate(hits, start=1):
            offset = _parse_event(row, OFFSET_COL) or onset
            manifest.append(
                {
                    "label": label,
                    "clip_file": target.name,
                    "mark": mark_no,
                    "sz_type": (row.get(TYPE_COL) or "").strip(),
                    "onset_s": round((onset - start).total_seconds(), 3),
                    "offset_s": round((offset - start).total_seconds(), 3),
                }
            )
            matched += 1

        private.append(
            {
                "label": label,
                "record_name": edf.name,
                "patient_id": patient_id,
                "start_datetime_utc": start.isoformat(),
                "duration_s": f"{dur:.3f}",
                "n_marks": len(hits),
                "source_path": str(edf.resolve()),
            }
        )
        print(f"  + {label}  <-  {edf.name}  ({len(hits)} seizure(s), {dur / 3600:.1f} h)")

    write_manifest(samples_out / MANIFEST_NAME, manifest)
    # private index stays beside the samples source (coordinator side), never in
    # the reviewer-facing samples output folder
    try:
        _write_private(samples_src / PRIVATE_MAP_NAME, private)
        print(f"  private index: {samples_src / PRIVATE_MAP_NAME}  <- keep from reviewers")
    except Exception as exc:
        print(f"  (could not write private index: {exc})")

    print(f"\n{index} sample recording(s), {matched} labelled seizure(s) -> {samples_out}")
    return manifest