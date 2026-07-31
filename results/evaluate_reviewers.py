"""
Evaluate seizure-review performance against a ground-truth seizure list.
=======================================================================

Reads the two exports produced by hexoreview (annotations.csv, coverage.csv)
plus a ground-truth seizure CSV, and scores each reviewer on their ability to
*distinguish* seizures in the Hexoskin recordings.

--------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------
Fill in the four paths in the CONFIG block below, then run:

    uv run python evaluate_reviewers.py       (or: python evaluate_reviewers.py)

Everything is written into OUTPUT_DIR, and the same information is printed to
the console and saved to OUTPUT_DIR/evaluation.log.

--------------------------------------------------------------------------
SCORING MODEL  (agreed design)
--------------------------------------------------------------------------
Matching key
  A ground-truth seizure is tied to a recording by a stable record id extracted
  from its `edf_path`, matched against the annotation/coverage `record_name`.
  Overlap is only ever tested within the same recording.

What counts as a seizure interval
  [electric_onset, sz_offset].  `electric_onset` is the only onset used.

Detection (any overlap, positive duration)
  A ground-truth seizure is a TRUE POSITIVE for a reviewer if at least one of
  that reviewer's marks overlaps it. Otherwise it is a FALSE NEGATIVE.
  Marks are scored on the mark side:
    - a mark overlapping 0 seizures            -> 1 false positive
    - a mark overlapping exactly 1 seizure     -> 0 false positives
    - a mark overlapping k >= 2 seizures       -> (k - 1) false positives

Which nights are included
    reviewed     -> detection + timing metrics
    in_progress  -> detection only (a warning is printed); excluded from timing
    not_started  -> excluded from EVERYTHING (a warning is printed)

Per-reviewer metrics
    sensitivity, precision, false_alarms, FAR_24h, review pace, mean review
    time per record, and TP timing errors (see below).

Detection timing error (on TRUE POSITIVES only)
    For each correctly detected seizure, the reviewer's *effective* mark
    boundaries are the union edges of their overlapping marks (earliest onset,
    latest offset -- so "one seizure marked in two pieces" is handled). Then:
        onset_error  = effective mark onset  - seizure electric_onset
        offset_error = effective mark offset - seizure sz_offset
    We report the mean and median ABSOLUTE error (seconds) per reviewer and
    overall (mean across reviewers, and pooled across all TP events).

Inter-rater agreement
  Detection agreement on real seizures: pairwise % agreement + Cohen's kappa,
  and Fleiss' kappa across all reviewers on commonly-reviewed seizures.

False-alarm events (do reviewers make the SAME mistakes?)
  Every false-positive mark is an "event". Overlapping FP marks -- across
  reviewers, within a recording -- are merged into a single false-alarm event
  whose span is the union (earliest onset .. latest offset). For each event we
  record which reviewers flagged it (1), which engaged with the recording but
  did not (0), and which never reviewed it (NA). Events flagged by >= 2
  reviewers are "shared". A pairwise agreement table follows.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================================
# CONFIG  -- edit these four paths, then run.
# =========================================================================
ANNOTATIONS_CSV = Path("export/annotations.csv")
COVERAGE_CSV    = Path("export/coverage.csv")
GROUND_TRUTH_CSV = Path("data/seizures_with_edf.csv")
OUTPUT_DIR      = Path("results/")

# Recording matching: extract this token from each filename as the join key.
# Set to None to match on the full basename instead.
RECORD_ID_PATTERN = r"record[-_]?\d+"

# Timezone handling for the SEIZURE csv (annotations are already local).
REPORT_TZ                   = "America/Toronto"
SEIZURE_TIMES_ARE_UTC       = False
MANUAL_SEIZURE_OFFSET_HOURS = 0.0

# Overlap / FP edge-case rules
OVERLAP_TOUCHING = False   # False = require positive shared duration
FP_SPAN_FLAT     = False   # False = mark over k>=2 seizures charges (k-1) FP
# =========================================================================


# ----------------------------------------------------------------- logging
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(out_dir / "evaluation.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ------------------------------------------------------------- record keys
_RID = re.compile(RECORD_ID_PATTERN, re.IGNORECASE) if RECORD_ID_PATTERN else None


def rec_key(path_or_name) -> str:
    """Stable join key for a recording, robust to differing name prefixes."""
    s = str(path_or_name).strip().replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    if _RID:
        m = _RID.search(s)
        if m:
            digits = re.search(r"\d+", m.group(0)).group(0)
            return f"record-{digits}"
    return s.rsplit(".", 1)[0] if "." in s else s


# ------------------------------------------------------------- small helpers
def truthy(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def fnum(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def overlaps(a0, a1, b0, b1) -> bool:
    lo, hi = max(a0, b0), min(a1, b1)
    return lo <= hi if OVERLAP_TOUCHING else lo < hi


def _touch(a_start, b_end) -> bool:
    """Does an interval starting at a_start reach an interval ending at b_end?"""
    return a_start <= b_end if OVERLAP_TOUCHING else a_start < b_end


def _mean_median(vals):
    if not vals:
        return np.nan, np.nan
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(np.median(a))


def _local_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(REPORT_TZ)
    except Exception:  # noqa: BLE001
        return None


LOCAL_TZ = _local_tz()


def to_local_naive(ts):
    """Normalise a seizure timestamp to naive local wall-clock."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return None
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.astimezone(LOCAL_TZ).replace(tzinfo=None) if LOCAL_TZ else dt.replace(tzinfo=None)
    elif SEIZURE_TIMES_ARE_UTC and LOCAL_TZ is not None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ).replace(tzinfo=None)
    if MANUAL_SEIZURE_OFFSET_HOURS:
        dt = dt + timedelta(hours=MANUAL_SEIZURE_OFFSET_HOURS)
    return pd.Timestamp(dt)


def parse_seizure_dt(date_str, time_or_dt_str):
    t = str(time_or_dt_str).strip()
    d = str(date_str).strip()
    if not t or t.lower() in {"nan", "nat", "none"}:
        return pd.NaT
    has_date = bool(re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", t)) or "T" in t
    if has_date:
        return pd.to_datetime(t, errors="coerce")
    return pd.to_datetime(f"{d} {t}", errors="coerce")


# --------------------------------------------------------------- agreement
def cohen_kappa(labels1, labels2):
    n = len(labels1)
    if n == 0:
        return np.nan, np.nan
    po = sum(int(a == b) for a, b in zip(labels1, labels2)) / n
    p1, p2 = sum(labels1) / n, sum(labels2) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe >= 1.0:
        return po, np.nan
    return po, (po - pe) / (1 - pe)


def fleiss_kappa(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.shape[0] == 0:
        return np.nan
    n_items = counts.shape[0]
    n_raters = counts.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] < 2:
        return np.nan
    n = n_raters[0]
    p_j = counts.sum(axis=0) / (n_items * n)
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_e = np.sum(p_j ** 2)
    if P_e >= 1.0:
        return np.nan
    return (P_i.mean() - P_e) / (1 - P_e)


# ---------------------------------------------------------- false-alarm events
def build_fp_events(mark_records, engaged, reviewers):
    """Cluster overlapping false-positive marks into shared false-alarm events.

    Returns (events_list, events_df). Each event spans the union of its marks;
    per-reviewer flag is 1 (flagged), 0 (engaged but did not flag) or 'NA'.
    """
    engaged_by_rec = {}
    for reviewer, recs in engaged.items():
        for rk in recs:
            engaged_by_rec.setdefault(rk, set()).add(reviewer)

    fp_by_rec = {}
    for mr in mark_records:
        if mr["is_false_alarm"]:
            fp_by_rec.setdefault(mr["record"], []).append(mr)

    events, eid = [], 0
    for rk in sorted(fp_by_rec):
        fps = sorted(fp_by_rec[rk], key=lambda m: m["onset_clock"])
        clusters = []
        for m in fps:
            if clusters and _touch(m["onset_clock"], clusters[-1]["end"]):
                c = clusters[-1]
                c["end"] = max(c["end"], m["offset_clock"])
                c["marks"].append(m)
            else:
                clusters.append({"start": m["onset_clock"],
                                 "end": m["offset_clock"], "marks": [m]})
        for c in clusters:
            eid += 1
            flaggers = sorted({m["reviewer"] for m in c["marks"]})
            engaged_here = engaged_by_rec.get(rk, set())
            row = {
                "event_id": f"FA{eid:04d}", "record": rk,
                "event_onset": c["start"], "event_offset": c["end"],
                "event_duration_s": (c["end"] - c["start"]).total_seconds(),
                "n_marks": len(c["marks"]),
                "n_reviewers_engaged": len(engaged_here),
                "n_reviewers_flagged": len(flaggers),
                "reviewers_flagged": ";".join(flaggers),
                "shared": len(flaggers) >= 2,
            }
            for reviewer in reviewers:
                row[reviewer] = 1 if reviewer in flaggers else (
                    0 if reviewer in engaged_here else "NA")
            events.append(row)
    return events, pd.DataFrame(events)


def fp_agreement_pairwise(events, reviewers):
    """Per-pair false-alarm agreement: of events both could see, how many both
    flagged (Jaccard on event flags over common-record events)."""
    rows = []
    for r1, r2 in combinations(reviewers, 2):
        both = either = 0
        for ev in events:
            e1, e2 = ev.get(r1), ev.get(r2)
            if e1 == "NA" or e2 == "NA":       # only events both engaged with
                continue
            f1, f2 = (e1 == 1), (e2 == 1)
            if f1 or f2:
                either += 1
            if f1 and f2:
                both += 1
        rows.append({
            "reviewer_a": r1, "reviewer_b": r2,
            "events_either_flagged": either, "events_both_flagged": both,
            "agreement_ratio": round(both / either, 4) if either else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ loading
def load_inputs():
    ann = pd.read_csv(ANNOTATIONS_CSV)
    cov = pd.read_csv(COVERAGE_CSV)
    gt = pd.read_csv(GROUND_TRUTH_CSV)

    def need(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{name} missing column(s): {', '.join(missing)}\n"
                f"Found: {', '.join(df.columns)}"
            )

    need(ann, ["clinician_id", "record_name", "onset", "offset",
               "onset_clock", "offset_clock"], "annotations.csv")
    need(cov, ["clinician_id", "record_name", "review_status",
               "recording_duration_s", "review_minutes"], "coverage.csv")
    need(gt, ["p_num", "sz_id", "electric_onset", "sz_offset",
              "edf_file_found", "edf_path"], "ground-truth CSV")
    return ann, cov, gt


# --------------------------------------------------------------------- main
def main():
    out_dir = Path(OUTPUT_DIR)
    log = setup_logging(out_dir)

    log.info("=" * 72)
    log.info("SEIZURE-REVIEW EVALUATION")
    log.info("=" * 72)
    log.info(f"annotations : {ANNOTATIONS_CSV}")
    log.info(f"coverage    : {COVERAGE_CSV}")
    log.info(f"ground truth: {GROUND_TRUTH_CSV}")
    log.info(f"output dir  : {out_dir.resolve()}")
    log.info("")

    ann, cov, gt = load_inputs()

    # -- ground truth: drop unfound, parse intervals, key by recording --------
    n_gt_all = len(gt)
    gt = gt.copy()
    gt["_found"] = gt["edf_file_found"].map(truthy)
    n_dropped = int((~gt["_found"]).sum())
    gt = gt[gt["_found"]].reset_index(drop=True)
    sz_date_col = "sz_date" if "sz_date" in gt.columns else None

    seizures, sz_by_rec = [], {}
    bad_times = 0
    gt_key_sample = {}
    for idx, row in gt.iterrows():
        d = row[sz_date_col] if sz_date_col else ""
        on = to_local_naive(parse_seizure_dt(d, row["electric_onset"]))
        off = to_local_naive(parse_seizure_dt(d, row["sz_offset"]))
        if on is None or off is None:
            bad_times += 1
            continue
        if off < on:
            off = off + pd.Timedelta(days=1)
        rk = rec_key(row["edf_path"])
        gt_key_sample.setdefault(rk, str(row["edf_path"]))
        sz = {"uid": f"{row['p_num']}|{row['sz_id']}|{idx}",
              "p_num": row["p_num"], "sz_id": row["sz_id"],
              "sz_type": row.get("sz_type", ""), "record": rk,
              "onset": on, "offset": off}
        seizures.append(sz)
        sz_by_rec.setdefault(rk, []).append(sz)

    log.info(f"Ground-truth seizures            : {n_gt_all}")
    log.info(f"  dropped (edf_file_found=False) : {n_dropped}")
    if bad_times:
        log.info(f"  dropped (unparseable times)    : {bad_times}")
    log.info(f"  evaluated                      : {len(seizures)}")
    log.info(f"  recordings with seizures       : {len(sz_by_rec)}")
    log.info("")

    # -- coverage: recording duration + engagement per reviewer ---------------
    rec_dur = {}
    ann_key_sample, cov_key_sample = {}, {}
    for _, r in cov.iterrows():
        rk = rec_key(r["record_name"])
        cov_key_sample.setdefault(rk, str(r["record_name"]))
        d = fnum(r["recording_duration_s"])
        if rk not in rec_dur and np.isfinite(d):
            rec_dur[rk] = d

    # -- annotations -> marks per (reviewer, record) --------------------------
    marks = {}
    n_bad_marks = 0
    for _, r in ann.iterrows():
        reviewer = str(r["clinician_id"])
        rk = rec_key(r["record_name"])
        ann_key_sample.setdefault(rk, str(r["record_name"]))
        on = pd.to_datetime(str(r.get("onset_clock", "")), errors="coerce")
        off = pd.to_datetime(str(r.get("offset_clock", "")), errors="coerce")
        on_s, off_s = fnum(r["onset"]), fnum(r["offset"])
        if pd.isna(on) or pd.isna(off):
            n_bad_marks += 1
            continue
        if off < on:
            on, off = off, on
        if np.isfinite(on_s) and np.isfinite(off_s) and off_s < on_s:
            on_s, off_s = off_s, on_s
        marks.setdefault((reviewer, rk), []).append(
            {"on": on, "off": off, "on_s": on_s, "off_s": off_s})
    if n_bad_marks:
        log.info(f"WARNING: {n_bad_marks} mark(s) had unreadable onset_clock/offset_clock.\n")

    # -- engagement per reviewer ---------------------------------------------
    reviewers = sorted(cov["clinician_id"].astype(str).unique())
    engaged, timing_rows = {}, {}
    for _, r in cov.iterrows():
        reviewer = str(r["clinician_id"])
        rk = rec_key(r["record_name"])
        status = str(r["review_status"]).strip()
        if status == "in_progress":
            log.info(f"WARNING: '{r['record_name']}' by '{reviewer}' is IN PROGRESS "
                     f"-> detection only, timing excluded.")
        elif status == "not_started":
            n_here = len(sz_by_rec.get(rk, []))
            extra = f" (contains {n_here} seizure(s))" if n_here else ""
            log.info(f"WARNING: '{r['record_name']}' by '{reviewer}' is NOT ANNOTATED "
                     f"-> excluded entirely{extra}.")
            continue
        engaged.setdefault(reviewer, {})[rk] = status
        if status == "reviewed":
            timing_rows.setdefault(reviewer, []).append(r)
    log.info("")

    # =====================================================================
    # DETECTION per reviewer  (wall-clock overlap) + TP timing errors
    # =====================================================================
    per_seizure, mark_records, reviewer_metrics = {}, [], {}
    timing_records = []          # one row per TP (reviewer x seizure)
    for reviewer in reviewers:
        recs = engaged.get(reviewer, {})
        tp = fn = fp = 0
        det_rec_hours = 0.0
        abs_on_err, abs_off_err = [], []
        for rk in recs:
            d = rec_dur.get(rk)
            if d is not None:
                det_rec_hours += d / 3600.0
            szs = sz_by_rec.get(rk, [])
            mk = marks.get((reviewer, rk), [])
            for sz in szs:
                ov = [m for m in mk if overlaps(sz["onset"], sz["offset"], m["on"], m["off"])]
                caught = bool(ov)
                per_seizure.setdefault(sz["uid"], {})[reviewer] = "TP" if caught else "FN"
                if caught:
                    eff_on = min(m["on"] for m in ov)     # union edges
                    eff_off = max(m["off"] for m in ov)
                    on_err = (eff_on - sz["onset"]).total_seconds()
                    off_err = (eff_off - sz["offset"]).total_seconds()
                    abs_on_err.append(abs(on_err))
                    abs_off_err.append(abs(off_err))
                    timing_records.append({
                        "reviewer": reviewer, "record": rk,
                        "p_num": sz["p_num"], "sz_id": sz["sz_id"],
                        "seizure_onset": sz["onset"], "seizure_offset": sz["offset"],
                        "mark_onset": eff_on, "mark_offset": eff_off,
                        "onset_error_s": round(on_err, 3),
                        "offset_error_s": round(off_err, 3),
                        "abs_onset_error_s": round(abs(on_err), 3),
                        "abs_offset_error_s": round(abs(off_err), 3),
                        "n_marks_used": len(ov),
                    })
                tp += int(caught)
                fn += int(not caught)
            for m in mk:
                k = sum(1 for sz in szs if overlaps(sz["onset"], sz["offset"], m["on"], m["off"]))
                if k == 0:
                    mfp = 1
                elif FP_SPAN_FLAT:
                    mfp = 1 if k >= 2 else 0
                else:
                    mfp = k - 1
                fp += mfp
                mark_records.append({"reviewer": reviewer, "record": rk,
                                     "onset_clock": m["on"], "offset_clock": m["off"],
                                     "onset_s": m["on_s"], "offset_s": m["off_s"],
                                     "n_seizures_overlapped": k,
                                     "false_positives": mfp, "is_false_alarm": k == 0})

        sens = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        far24 = (fp / det_rec_hours * 24.0) if det_rec_hours > 0 else np.nan

        on_mean, on_med = _mean_median(abs_on_err)
        off_mean, off_med = _mean_median(abs_off_err)

        trows = timing_rows.get(reviewer, [])
        rev_min = sum(fnum(r["review_minutes"], 0.0) for r in trows)
        rev_hours = sum(fnum(r["recording_duration_s"], 0.0) for r in trows) / 3600.0
        n_reviewed = len(trows)
        pace_h = (rev_min / rev_hours) if rev_hours > 0 else np.nan
        pace_24h = pace_h * 24 if np.isfinite(pace_h) else np.nan
        mean_min = (rev_min / n_reviewed) if n_reviewed else np.nan

        reviewer_metrics[reviewer] = {
            "reviewer": reviewer, "n_recordings_reviewed": n_reviewed,
            "n_recordings_in_progress": sum(1 for s in recs.values() if s == "in_progress"),
            "n_seizures_evaluated": tp + fn, "TP": tp, "FN": fn, "FP": fp,
            "sensitivity": sens, "precision": prec, "false_alarms": fp,
            "FAR_per_24h": far24, "detection_recorded_hours": round(det_rec_hours, 3),
            "onset_err_mean_abs_s": round(on_mean, 2) if np.isfinite(on_mean) else np.nan,
            "onset_err_median_abs_s": round(on_med, 2) if np.isfinite(on_med) else np.nan,
            "offset_err_mean_abs_s": round(off_mean, 2) if np.isfinite(off_mean) else np.nan,
            "offset_err_median_abs_s": round(off_med, 2) if np.isfinite(off_med) else np.nan,
            "review_min_per_recorded_hour": round(pace_h, 3) if np.isfinite(pace_h) else np.nan,
            "review_minutes_per_24h": round(pace_24h, 2) if np.isfinite(pace_24h) else np.nan,
            "review_hours_per_24h": round(pace_24h / 60, 3) if np.isfinite(pace_24h) else np.nan,
            "mean_review_min_per_record": round(mean_min, 2) if np.isfinite(mean_min) else np.nan,
            "total_review_minutes": round(rev_min, 2), "total_reviewed_hours": round(rev_hours, 3)}

    metrics_df = pd.DataFrame([reviewer_metrics[r] for r in reviewers])
    timing_df = pd.DataFrame(timing_records)

    # -- per-seizure table ----------------------------------------------------
    seiz_rows = []
    for sz in seizures:
        res = per_seizure.get(sz["uid"], {})
        if not res:
            continue
        row = {"p_num": sz["p_num"], "sz_id": sz["sz_id"], "sz_type": sz["sz_type"],
               "record": sz["record"], "electric_onset": sz["onset"], "sz_offset": sz["offset"]}
        caught_n = 0
        for reviewer in reviewers:
            val = res.get(reviewer, "NA")
            row[reviewer] = val
            caught_n += int(val == "TP")
        row["n_reviewers_evaluated"] = sum(1 for r in reviewers if res.get(r, "NA") != "NA")
        row["n_caught"] = caught_n
        row["any_reviewer_caught"] = caught_n > 0
        seiz_rows.append(row)
    seizure_df = pd.DataFrame(seiz_rows)

    # -- overall --------------------------------------------------------------
    def macro(col):
        return metrics_df[col].mean(skipna=True) if len(metrics_df) else np.nan

    tot_tp = int(metrics_df["TP"].sum()); tot_fn = int(metrics_df["FN"].sum())
    tot_fp = int(metrics_df["FP"].sum())
    tot_hours = metrics_df["detection_recorded_hours"].sum()
    pooled_sens = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else np.nan
    pooled_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else np.nan
    pooled_far = (tot_fp / tot_hours * 24) if tot_hours > 0 else np.nan

    overall_df = pd.DataFrame([
        {"aggregate": "mean_across_reviewers", "sensitivity": macro("sensitivity"),
         "precision": macro("precision"), "false_alarms": macro("false_alarms"),
         "FAR_per_24h": macro("FAR_per_24h"),
         "review_min_per_recorded_hour": macro("review_min_per_recorded_hour"),
         "review_minutes_per_24h": macro("review_minutes_per_24h"),
         "mean_review_min_per_record": macro("mean_review_min_per_record")},
        {"aggregate": "pooled_across_reviewers", "sensitivity": pooled_sens,
         "precision": pooled_prec, "false_alarms": tot_fp, "FAR_per_24h": pooled_far,
         "review_min_per_recorded_hour": np.nan, "review_minutes_per_24h": np.nan,
         "mean_review_min_per_record": np.nan}])

    # -- timing-error overall (macro across reviewers + pooled over TP events)-
    pooled_on_mean, pooled_on_med = _mean_median(
        list(timing_df["abs_onset_error_s"]) if len(timing_df) else [])
    pooled_off_mean, pooled_off_med = _mean_median(
        list(timing_df["abs_offset_error_s"]) if len(timing_df) else [])
    timing_overall_df = pd.DataFrame([
        {"aggregate": "mean_across_reviewers",
         "onset_err_mean_abs_s": macro("onset_err_mean_abs_s"),
         "onset_err_median_abs_s": macro("onset_err_median_abs_s"),
         "offset_err_mean_abs_s": macro("offset_err_mean_abs_s"),
         "offset_err_median_abs_s": macro("offset_err_median_abs_s"),
         "n_tp_events": int(metrics_df["TP"].sum())},
        {"aggregate": "pooled_over_tp_events",
         "onset_err_mean_abs_s": round(pooled_on_mean, 2) if np.isfinite(pooled_on_mean) else np.nan,
         "onset_err_median_abs_s": round(pooled_on_med, 2) if np.isfinite(pooled_on_med) else np.nan,
         "offset_err_mean_abs_s": round(pooled_off_mean, 2) if np.isfinite(pooled_off_mean) else np.nan,
         "offset_err_median_abs_s": round(pooled_off_med, 2) if np.isfinite(pooled_off_med) else np.nan,
         "n_tp_events": len(timing_df)},
    ])

    # -- agreement: detection -------------------------------------------------
    det_pairs = []
    for r1, r2 in combinations(reviewers, 2):
        common = set(engaged.get(r1, {})) & set(engaged.get(r2, {}))
        l1, l2 = [], []
        for sz in seizures:
            if sz["record"] not in common:
                continue
            v1 = per_seizure.get(sz["uid"], {}).get(r1)
            v2 = per_seizure.get(sz["uid"], {}).get(r2)
            if v1 is None or v2 is None:
                continue
            l1.append(int(v1 == "TP")); l2.append(int(v2 == "TP"))
        po, kappa = cohen_kappa(l1, l2)
        det_pairs.append({"reviewer_a": r1, "reviewer_b": r2, "n_common_seizures": len(l1),
                          "percent_agreement": round(po, 4) if np.isfinite(po) else np.nan,
                          "cohen_kappa": round(kappa, 4) if np.isfinite(kappa) else np.nan})
    det_pair_df = pd.DataFrame(det_pairs)

    all_common = set.intersection(*[set(engaged.get(r, {})) for r in reviewers]) if reviewers else set()
    fleiss_counts = []
    for sz in seizures:
        if sz["record"] not in all_common:
            continue
        vals = [per_seizure.get(sz["uid"], {}).get(r) for r in reviewers]
        if any(v is None for v in vals):
            continue
        caught = sum(int(v == "TP") for v in vals)
        fleiss_counts.append([len(reviewers) - caught, caught])
    fleiss = fleiss_kappa(fleiss_counts) if fleiss_counts else np.nan

    # -- false-alarm events + agreement --------------------------------------
    fa_events, fa_events_df = build_fp_events(mark_records, engaged, reviewers)
    fa_pair_df = fp_agreement_pairwise(fa_events, reviewers)
    n_fa_total = len(fa_events)
    n_fa_shared = sum(1 for e in fa_events if e["shared"])

    # -- write files ----------------------------------------------------------
    files = {"per_reviewer_metrics.csv": metrics_df, "overall_summary.csv": overall_df,
             "per_seizure_detection.csv": seizure_df,
             "mark_classification.csv": pd.DataFrame(mark_records),
             "detection_timing_errors.csv": timing_df,
             "timing_error_overall.csv": timing_overall_df,
             "agreement_detection_pairwise.csv": det_pair_df,
             "false_alarm_events.csv": fa_events_df,
             "false_alarm_agreement_pairwise.csv": fa_pair_df}
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False)

    # -- print summary --------------------------------------------------------
    log.info("=" * 72)
    log.info("PER-REVIEWER RESULTS")
    log.info("=" * 72)
    for reviewer in reviewers:
        m = reviewer_metrics[reviewer]
        log.info(f"\nReviewer: {reviewer}")
        log.info(f"  nights reviewed / in-progress : {m['n_recordings_reviewed']} / {m['n_recordings_in_progress']}")
        log.info(f"  seizures evaluated            : {m['n_seizures_evaluated']}")
        log.info(f"  TP / FN / FP                  : {m['TP']} / {m['FN']} / {m['FP']}")
        log.info(f"  sensitivity                   : {fmt(m['sensitivity'])}")
        log.info(f"  precision                     : {fmt(m['precision'])}")
        log.info(f"  false alarms (count)          : {m['false_alarms']}")
        log.info(f"  FAR per 24 h                  : {fmt(m['FAR_per_24h'], 2)}")
        log.info(f"  onset error  |mean| / |median|: {fmt(m['onset_err_mean_abs_s'], 1)} / {fmt(m['onset_err_median_abs_s'], 1)} s")
        log.info(f"  offset error |mean| / |median|: {fmt(m['offset_err_mean_abs_s'], 1)} / {fmt(m['offset_err_median_abs_s'], 1)} s")
        log.info(f"  review time per 24 h (min)    : {fmt(m['review_minutes_per_24h'], 1)} ({fmt(m['review_hours_per_24h'], 2)} h)")
        log.info(f"  mean review time per record   : {fmt(m['mean_review_min_per_record'], 1)} min")

    log.info("")
    log.info("=" * 72)
    log.info("OVERALL")
    log.info("=" * 72)
    log.info("Mean across reviewers:")
    log.info(f"  sensitivity                   : {fmt(macro('sensitivity'))}")
    log.info(f"  precision                     : {fmt(macro('precision'))}")
    log.info(f"  false alarms (count)          : {fmt(macro('false_alarms'), 1)}")
    log.info(f"  FAR per 24 h                  : {fmt(macro('FAR_per_24h'), 2)}")
    log.info(f"  onset error  |mean| / |median|: {fmt(macro('onset_err_mean_abs_s'), 1)} / {fmt(macro('onset_err_median_abs_s'), 1)} s")
    log.info(f"  offset error |mean| / |median|: {fmt(macro('offset_err_mean_abs_s'), 1)} / {fmt(macro('offset_err_median_abs_s'), 1)} s")
    log.info(f"  review time per 24 h (min)    : {fmt(macro('review_minutes_per_24h'), 1)}")
    log.info(f"  mean review time per record   : {fmt(macro('mean_review_min_per_record'), 1)} min")
    log.info("Pooled across reviewers:")
    log.info(f"  sensitivity                   : {fmt(pooled_sens)}  ({tot_tp}/{tot_tp + tot_fn})")
    log.info(f"  precision                     : {fmt(pooled_prec)}  ({tot_tp}/{tot_tp + tot_fp})")
    log.info(f"  FAR per 24 h                  : {fmt(pooled_far, 2)}")
    log.info("Pooled over TP events (timing):")
    log.info(f"  onset error  |mean| / |median|: {fmt(pooled_on_mean, 1)} / {fmt(pooled_on_med, 1)} s")
    log.info(f"  offset error |mean| / |median|: {fmt(pooled_off_mean, 1)} / {fmt(pooled_off_med, 1)} s")

    log.info("")
    log.info("=" * 72)
    log.info("INTER-RATER AGREEMENT")
    log.info("=" * 72)
    if len(reviewers) < 2:
        log.info("Only one reviewer present -- agreement not computable.")
    else:
        log.info("Detection agreement on real seizures (pairwise):")
        for _, r in det_pair_df.iterrows():
            log.info(f"  {r['reviewer_a']} vs {r['reviewer_b']}: n={r['n_common_seizures']:>4}  "
                     f"agreement={fmt(r['percent_agreement'])}  kappa={fmt(r['cohen_kappa'])}")
        log.info(f"  Fleiss' kappa (all reviewers, {len(fleiss_counts)} common seizures): {fmt(fleiss)}")

    log.info("")
    log.info("=" * 72)
    log.info("FALSE-ALARM EVENTS")
    log.info("=" * 72)
    log.info(f"Total false-alarm events        : {n_fa_total}")
    log.info(f"  shared by >=2 reviewers       : {n_fa_shared}"
             + (f"  ({n_fa_shared / n_fa_total:.1%})" if n_fa_total else ""))
    log.info("Per reviewer (false-alarm events flagged, of which shared):")
    for reviewer in reviewers:
        flagged = [e for e in fa_events if e.get(reviewer) == 1]
        n_sh = sum(1 for e in flagged if e["shared"])
        log.info(f"  {reviewer:<12} {len(flagged):>4} flagged, {n_sh:>4} shared")
    if len(reviewers) >= 2:
        log.info("Pairwise false-alarm agreement (both / either flagged):")
        for _, r in fa_pair_df.iterrows():
            log.info(f"  {r['reviewer_a']} vs {r['reviewer_b']}: "
                     f"{int(r['events_both_flagged'])} / {int(r['events_either_flagged'])}  "
                     f"ratio={fmt(r['agreement_ratio'])}")

    log.info("")
    log.info("=" * 72)
    log.info("FILES WRITTEN")
    log.info("=" * 72)
    for name in files:
        log.info(f"  {out_dir / name}")
    log.info(f"  {out_dir / 'evaluation.log'}")
    log.info("\nDone.")


if __name__ == "__main__":
    main()