"""Cut short training clips out of full overnight EDF recordings.

The idea is to give reviewers a handful of known seizures to learn on: a small
EDF that opens instantly and shows one event with a little context either side.

Times in the seizure CSV are wall-clock Eastern. EDF start times (meas_date)
arrive UTC-aware from MNE. So the only real work is turning an Eastern
start/end into an offset in seconds from the recording start, which is the same
timezone reasoning as the annotate step:

    offset_s = (eastern_time_with_tz - meas_date_utc).total_seconds()

Requires edfio for the export step (MNE >= 1.7 uses it for fmt="edf"):

    uv add edfio
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import mne

# The timezone your CSV timestamps are written in. DST-aware, so EST/EDT is
# handled from the tz database rather than a fixed offset.
CSV_TZ = ZoneInfo("America/New_York")

# Context to include around the event, in seconds, when building a clip from an
# onset/offset pair. 60 s each side gives reviewers some baseline to compare to.
PAD_BEFORE_S = 60.0
PAD_AFTER_S = 60.0


def _read_raw(edf_path: Path):
    """Load an EDF fully (needed to crop and export), with a latin-1 fallback."""
    try:
        return mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    except Exception:
        return mne.io.read_raw_edf(
            edf_path, preload=True, encoding="latin1", verbose="error"
        )


def _as_eastern(t: datetime) -> datetime:
    """Attach Eastern tzinfo to a naive CSV timestamp; leave aware ones alone."""
    return t.replace(tzinfo=CSV_TZ) if t.tzinfo is None else t


def crop_edf(
    edf_path: Path,
    out_path: Path,
    start_eastern: datetime,
    end_eastern: datetime,
    *,
    anonymize: bool = False,
) -> dict:
    """Write the segment [start_eastern, end_eastern] of one EDF to out_path.

    Returns a small summary dict. Raises ValueError if the requested window does
    not overlap the recording at all (wrong file, or a timezone mistake).

    All original channels and their names are kept, so the clip flows through
    the existing scan / blinding / Recording pipeline unchanged.
    """
    edf_path = Path(edf_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_raw(edf_path)
    meas = raw.info["meas_date"]  # tz-aware UTC from MNE
    if meas is None:
        raise ValueError(f"{edf_path.name} has no start date in its header")

    start = _as_eastern(start_eastern)
    end = _as_eastern(end_eastern)
    if end <= start:
        raise ValueError(f"end ({end}) is not after start ({start})")

    # offsets in seconds from the recording start
    tmin = (start - meas).total_seconds()
    tmax = (end - meas).total_seconds()

    duration = raw.times[-1]  # last valid sample time in seconds
    if tmax <= 0 or tmin >= duration:
        raise ValueError(
            f"requested window {start:%Y-%m-%d %H:%M:%S} – {end:%H:%M:%S} "
            f"does not overlap {edf_path.name} "
            f"(recording starts {meas.astimezone(CSV_TZ):%Y-%m-%d %H:%M:%S} "
            f"Eastern, runs {duration / 3600:.1f} h). Check the file match."
        )

    # clamp to what the recording actually contains
    tmin_c = max(tmin, 0.0)
    tmax_c = min(tmax, duration)
    raw.crop(tmin=tmin_c, tmax=tmax_c)

    # crop leaves meas_date at the ORIGINAL start and tracks the offset in
    # first_samp; reset it so the clip's header reports the real wall-clock
    # start of the segment (this is what the dashboard's clock mode reads).
    raw.set_meas_date(meas + timedelta(seconds=tmin_c))
    if anonymize:
        # strips subject_info but keeps the (now segment-correct) meas_date
        raw.anonymize(daysback=0)

    mne.export.export_raw(str(out_path), raw, fmt="edf", overwrite=True)

    return {
        "out_path": str(out_path),
        "clip_seconds": round(tmax_c - tmin_c, 3),
        "clamped": tmin < 0 or tmax > duration,
        "segment_start_eastern": (meas + timedelta(seconds=tmin_c))
        .astimezone(CSV_TZ)
        .strftime("%Y-%m-%d %H:%M:%S"),
    }


def clip_around_event(
    edf_path: Path,
    out_path: Path,
    onset_eastern: datetime,
    offset_eastern: datetime | None = None,
    *,
    pad_before_s: float = PAD_BEFORE_S,
    pad_after_s: float = PAD_AFTER_S,
    anonymize: bool = False,
) -> dict:
    """Clip a seizure with padding either side.

    If offset is None the event is treated as an instant and the clip spans
    pad_before_s before to pad_after_s after the onset.
    """
    onset = _as_eastern(onset_eastern)
    end_event = _as_eastern(offset_eastern) if offset_eastern else onset
    return crop_edf(
        edf_path,
        out_path,
        onset - timedelta(seconds=pad_before_s),
        end_event + timedelta(seconds=pad_after_s),
        anonymize=anonymize,
    )


# --------------------------------------------------------------------------- #
# CSV driver
# --------------------------------------------------------------------------- #
# Adjust these two names to match your CSV. The annotate step already added an
# 'edf_path' column pointing at the matched recording; ONSET_COL is whichever
# onset you want the clip centred on (electric_onset / clinical_onset / ...).
EDF_PATH_COL = "edf_path"
ONSET_COL = "electric_onset"
OFFSET_COL = "sz_offset"      # set to None to make fixed-length clips
ID_COLS = ("p_num", "sz_id")  # used only to name the output files


def _parse_ts(value) -> datetime | None:
    """Parse a CSV timestamp. Assumes a full 'YYYY-MM-DD HH:MM:SS'-style string.

    If your onset columns are time-of-day only, combine them with sz_date here
    before this point instead.
    """
    if value is None or str(value).strip() == "":
        return None
    return datetime.fromisoformat(str(value).strip())


def clips_from_csv(csv_path: Path, out_dir: Path, *, anonymize: bool = True) -> list[dict]:
    """Build one training clip per seizure row that has a matched EDF.

    Returns a list of per-clip summaries; failures are recorded, not raised, so
    one bad row does not stop the batch.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            edf = row.get(EDF_PATH_COL, "").strip()
            onset = _parse_ts(row.get(ONSET_COL))
            if not edf or onset is None:
                continue  # unmatched row or no onset time
            offset = _parse_ts(row.get(OFFSET_COL)) if OFFSET_COL else None

            name = "_".join(str(row.get(c, "")).strip() for c in ID_COLS if row.get(c))
            out_path = out_dir / f"clip_{name or onset:%Y%m%d_%H%M%S}.edf"

            try:
                summary = clip_around_event(
                    Path(edf), out_path, onset, offset, anonymize=anonymize
                )
                summary["status"] = "ok"
            except Exception as exc:
                summary = {"out_path": str(out_path), "status": "failed", "error": str(exc)}
            summary.update({c: row.get(c, "") for c in ID_COLS})
            results.append(summary)
            print(f"  {summary['status']:<6} {out_path.name}"
                  + (f"  ({summary.get('error')})" if summary["status"] == "failed" else ""))

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} clip(s) written to {out_dir}")
    return results


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "seizures.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "training_clips"
    clips_from_csv(Path(csv_path), Path(out_dir))
