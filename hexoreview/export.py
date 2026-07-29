"""Unblind the collected marks into two analysis-ready CSV files.

annotations.csv  one row per seizure mark
coverage.csv     one row per (clinician, recording), including nights reviewed
                 with no seizures found -- the denominator you need for
                 sensitivity and false-alarm rates
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .blinding import load_map
from .data import to_local
from .store import Store


def _clock(start_iso: str, offset_s: float) -> str:
    """Wall-clock timestamp as 'YYYY-MM-DD HH:MM:SS' in the reporting timezone.

    The offset is added in UTC and only then converted, so a recording that
    crosses a daylight-saving change still lands on the right wall time.
    """
    if not start_iso:
        return ""
    try:
        start = datetime.fromisoformat(start_iso)
    except ValueError:
        return ""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    instant = start + timedelta(seconds=float(offset_s))
    return to_local(instant).strftime("%Y-%m-%d %H:%M:%S")


def _stamp(iso: str) -> str:
    """Reformat a stored ISO timestamp in the reporting timezone."""
    if not iso:
        return ""
    try:
        return to_local(datetime.fromisoformat(iso)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso


def _rate(active_seconds, duration_s) -> float:
    """Minutes of review per hour of recording: pace, comparable across nights."""
    try:
        hours = float(duration_s) / 3600
        if hours <= 0:
            return 0.0
        return round((float(active_seconds or 0) / 60) / hours, 2)
    except (TypeError, ValueError):
        return 0.0


def export(db_path: Path, map_path: Path, out_dir: Path) -> dict[str, Path]:
    store = Store(db_path)
    mapping = load_map(map_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_rows = []
    for a in store.all_annotations():
        m = mapping.get(a["blind_label"], {})
        # map now stores Eastern in 'start_datetime' and UTC in
        # 'start_datetime_utc'; older maps kept UTC in 'start_datetime'
        start_iso = m.get("start_datetime_utc") or m.get("start_datetime", "")
        ann_rows.append(
            {
                "clinician_id": a["clinician_id"],
                "patient_id": m.get("patient_id", "UNMAPPED"),
                "record_name": m.get("record_name", "UNMAPPED"),
                "blind_label": a["blind_label"],
                "onset": round(float(a["onset_s"]), 3),
                "offset": round(float(a["offset_s"]), 3),
                "duration_s": round(float(a["offset_s"]) - float(a["onset_s"]), 3),
                "onset_clock": _clock(start_iso, a["onset_s"]),
                "offset_clock": _clock(start_iso, a["offset_s"]),
                "note": a["note"] or "",
                "annotation_id": a["id"],
                "created_at": _stamp(a["created_at"]),
            }
        )

    annotations = pd.DataFrame(
        ann_rows,
        columns=[
            "clinician_id", "patient_id", "record_name", "blind_label",
            "onset", "offset", "duration_s", "onset_clock", "offset_clock",
            "note", "annotation_id", "created_at",
        ],
    ).sort_values(["clinician_id", "patient_id", "onset"], ignore_index=True)

    # coverage: every clinician x every recording, so "reviewed, nothing found"
    # is explicit rather than an absence of rows
    counts = (
        annotations.groupby(["clinician_id", "blind_label"]).size()
        if len(annotations)
        else pd.Series(dtype=int)
    )
    reviews = {(r["clinician_id"], r["blind_label"]): r for r in store.all_reviews()}
    clinician_ids = [c["clinician_id"] for c in store.reviewers()]
    sittings = store.session_counts()

    cov_rows = []
    for label in sorted(mapping):
        m = mapping[label]
        for cid in clinician_ids:
            r = reviews.get((cid, label))
            status = r["status"] if r else "not_started"
            n = int(counts.get((cid, label), 0)) if len(counts) else 0
            cov_rows.append(
                {
                    "clinician_id": cid,
                    "patient_id": m.get("patient_id", ""),
                    "record_name": m.get("record_name", ""),
                    "blind_label": label,
                    "review_status": status,
                    "n_annotations": n,
                    "outcome": (
                        "reviewed_with_seizures" if status == "reviewed" and n
                        else "reviewed_no_seizures" if status == "reviewed"
                        else status
                    ),
                    "recording_start": _stamp(
                        m.get("start_datetime_utc") or m.get("start_datetime", "")
                    ),
                    "recording_duration_s": m.get("duration_s", ""),
                    "review_seconds": round(float((r or {}).get("active_seconds") or 0), 1),
                    "review_minutes": round(float((r or {}).get("active_seconds") or 0) / 60, 2),
                    "n_sittings": sittings.get((cid, label), 0),
                    "review_min_per_recorded_hour": _rate(
                        (r or {}).get("active_seconds"), m.get("duration_s")
                    ),
                    "completed_at": _stamp((r or {}).get("completed_at", "") or ""),
                }
            )
    coverage = pd.DataFrame(cov_rows)

    paths = {
        "annotations": out_dir / "annotations.csv",
        "coverage": out_dir / "coverage.csv",
    }
    annotations.to_csv(paths["annotations"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    return paths