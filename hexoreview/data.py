"""Reading Hexoskin EDF recordings and serving decimated windows to the viewer.

An overnight recording is ~8 h x 256 Hz x 6 channels (~44 M samples). Re-reading
the EDF on every scroll is too slow, so the first time a recording is opened it is
converted once into a float32 .npy file that is then memory-mapped. Every scroll
step reads only the samples for the visible window straight off disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import CHANNELS, DEFAULT_HALF_SCALE, EXPORT_TIMEZONE

_TZ_CACHE = {}


def target_tz():
    """The timezone recordings are reported in, or None if it cannot be loaded."""
    if "tz" not in _TZ_CACHE:
        try:
            from zoneinfo import ZoneInfo

            _TZ_CACHE["tz"] = ZoneInfo(EXPORT_TIMEZONE)
        except Exception as exc:
            print(
                f"Timezone {EXPORT_TIMEZONE!r} could not be loaded ({exc}); "
                "times will be reported in UTC. Install the 'tzdata' package."
            )
            _TZ_CACHE["tz"] = None
    return _TZ_CACHE["tz"]


def to_local(moment):
    """Move an instant into the reporting timezone.

    A naive datetime is taken to be UTC, which is how EDF start times arrive.
    """
    if moment is None:
        return None
    from datetime import timezone as _timezone

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_timezone.utc)
    tz = target_tz()
    return moment.astimezone(tz) if tz is not None else moment


def _read_raw(edf_path: Path):
    """Open an EDF header without loading data, with the latin-1 fallback."""
    import mne

    try:
        return mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    except Exception:
        return mne.io.read_raw_edf(
            edf_path, preload=False, encoding="latin1", verbose="error"
        )


def resolve_picks(ch_names: list[str]) -> list[str]:
    """Map the six wanted channels onto the EDF's actual names.

    Hexoskin exports names like '4113:ECG_I'; the numeric prefix is not stable
    across firmware versions, so match on the part after the colon.
    """
    lookup = {}
    for name in ch_names:
        key = name.split(":")[-1].strip().lower()
        lookup.setdefault(key, name)

    picks, missing = [], []
    for wanted in CHANNELS:
        actual = lookup.get(wanted.lower())
        if actual is None:
            missing.append(wanted)
        else:
            picks.append(actual)
    if missing:
        raise ValueError(
            f"Channels not found in EDF: {', '.join(missing)}. "
            f"Available: {', '.join(ch_names)}"
        )
    return picks


SI_FACTORS = {
    "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3,
    "c": 1e-2, "d": 1e-1, "k": 1e3, "M": 1e6,
}


def parse_half_scale(value) -> float | None:
    """Read a pinned scale like 2.2, '2.2 g', '220 mg' or '50 uV'.

    Returns the half-height in the units the data are plotted in, or None if
    the entry is missing or unreadable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    match = re.match(
        r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)\s*$", str(value)
    )
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    # a prefix only counts if something follows it: 'mg' is milli-grams, but a
    # bare 'm' is a unit in its own right
    if len(unit) > 1 and unit[0] in SI_FACTORS:
        number *= SI_FACTORS[unit[0]]
    return number if number > 0 else None


def read_edf_units(edf_path: Path) -> dict[str, str]:
    """Physical dimension of each signal, read from the raw EDF header.

    The header is fixed-width ASCII, so this is exact and avoids guessing at
    whatever scaling a reader applied.
    """
    try:
        with open(edf_path, "rb") as fh:
            header = fh.read(256)
            n_signals = int(header[252:256].decode("ascii", "ignore").strip() or 0)
            if n_signals <= 0:
                return {}
            labels = fh.read(16 * n_signals)
            fh.read(80 * n_signals)  # transducer type
            units = fh.read(8 * n_signals)
    except Exception:
        return {}

    out = {}
    for i in range(n_signals):
        label = labels[i * 16 : (i + 1) * 16].decode("latin-1").strip()
        unit = units[i * 8 : (i + 1) * 8].decode("latin-1").strip()
        if label:
            out[label] = unit
    return out


def read_header(edf_path: Path) -> dict:
    """Header-only metadata, used by the scan step to build the blinding map."""
    raw = _read_raw(edf_path)
    info = raw.info
    subj = info.get("subject_info") or {}
    patient_id = f"{subj.get('first_name', '') or ''}{subj.get('last_name', '') or ''}"
    meas_date = info.get("meas_date")
    return {
        "patient_id": patient_id.strip().lower() or "unknown",
        "sfreq": float(info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_s": float(raw.n_times) / float(info["sfreq"]),
        "start_datetime": meas_date.isoformat() if meas_date else "",
        "channels": list(raw.ch_names),
    }


class Recording:
    """Memory-mapped access to one blinded overnight recording."""

    def __init__(self, edf_path: Path, cache_dir: Path):
        self.edf_path = Path(edf_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._npy = self.cache_dir / f"{self.edf_path.stem}.npy"
        self._meta_path = self.cache_dir / f"{self.edf_path.stem}.json"
        self.meta = self._ensure_cache()
        self.sfreq = float(self.meta["sfreq"])
        self.n_times = int(self.meta["n_times"])
        self.duration_s = self.n_times / self.sfreq
        self.data = np.load(self._npy, mmap_mode="r")
        self._row_order = self._resolve_order()
        self.units = self._resolve_units()
        self.baseline = self._baseline()
        self.start_datetime = self._start_datetime()

    def _resolve_order(self):
        """Row indices that put the cached file into the current channel order.

        Caches written before the display order changed are reused as they are;
        only the row mapping differs, so nothing has to be converted again.
        """
        cached = list(self.meta.get("channels") or CHANNELS)
        if cached == list(CHANNELS):
            return None
        try:
            return np.array([cached.index(name) for name in CHANNELS])
        except ValueError:
            # cache predates a channel being renamed: rebuild it
            self._npy.unlink(missing_ok=True)
            self._meta_path.unlink(missing_ok=True)
            self.meta = self._ensure_cache()
            self.data = np.load(self._npy, mmap_mode="r")
            return None

    def _resolve_units(self) -> list[str]:
        units = self.meta.get("units")
        if not units:
            picks = self.meta.get("edf_channels") or []
            found = read_edf_units(self.edf_path)
            units = [found.get(p, "") for p in picks]
            self.meta["units"] = units
            self._meta_path.write_text(json.dumps(self.meta, indent=2))
        if self._row_order is not None:
            units = [units[i] for i in self._row_order]
        return list(units)

    def _rows(self, block: np.ndarray) -> np.ndarray:
        """Put a block of channel rows into display order."""
        if self._row_order is None:
            return block
        return block[self._row_order]

    def _start_datetime(self):
        """Recording start as a datetime, or None if the EDF has no date.

        Caches written before this field existed fall back to the EDF header.
        """
        from datetime import datetime

        iso = self.meta.get("start_datetime")
        if iso is None:
            try:
                raw = _read_raw(self.edf_path)
                meas = raw.info.get("meas_date")
                iso = meas.isoformat() if meas else ""
            except Exception:
                iso = ""
            self.meta["start_datetime"] = iso
            self._meta_path.write_text(json.dumps(self.meta, indent=2))
        if not iso:
            return None
        try:
            return datetime.fromisoformat(iso)
        except ValueError:
            return None

    # -- cache ---------------------------------------------------------------
    def is_cached(self) -> bool:
        return self._npy.exists() and self._meta_path.exists()

    def _ensure_cache(self) -> dict:
        if self.is_cached():
            return json.loads(self._meta_path.read_text())

        raw = _read_raw(self.edf_path)
        picks = resolve_picks(list(raw.ch_names))
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)

        tmp = self._npy.with_suffix(".npy.part")
        arr = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(len(picks), n_times)
        )
        chunk = int(sfreq * 600)  # 10 min at a time
        for start in range(0, n_times, chunk):
            stop = min(start + chunk, n_times)
            arr[:, start:stop] = raw.get_data(picks=picks, start=start, stop=stop)
        arr.flush()
        del arr
        tmp.replace(self._npy)

        meas_date = raw.info.get("meas_date")
        edf_units = read_edf_units(self.edf_path)
        meta = {
            "sfreq": sfreq,
            "n_times": n_times,
            "channels": list(CHANNELS),
            "edf_channels": picks,
            "units": [edf_units.get(p, "") for p in picks],
            "start_datetime": meas_date.isoformat() if meas_date else "",
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    # -- scaling -------------------------------------------------------------
    def _baseline(self) -> list[dict]:
        """Per-channel centre and span used as the default vertical scale.

        Measured as the *typical amplitude over a few seconds*, not the range
        across the whole night. Taking the global range gets ruined by slow DC
        drift and by isolated artifacts: one big movement, or a baseline that
        wanders over eight hours, and every trace is squashed into a flat line.
        So the span is sampled in short chunks and the median chunk wins.
        """
        n_probe = 240
        probe_len = max(int(self.sfreq * 5), 1)
        starts = np.linspace(
            0, max(self.n_times - probe_len, 1), n_probe, dtype=np.int64
        )

        centers = np.full((len(CHANNELS), len(starts)), np.nan, dtype=np.float64)
        spans = np.full((len(CHANNELS), len(starts)), np.nan, dtype=np.float64)

        for j, s in enumerate(starts):
            chunk = self._rows(
                np.asarray(self.data[:, s : s + probe_len], dtype=np.float64)
            )
            if chunk.size == 0:
                continue
            with np.errstate(all="ignore"):
                lo = np.nanpercentile(chunk, 1, axis=1)
                hi = np.nanpercentile(chunk, 99, axis=1)
                centers[:, j] = np.nanmedian(chunk, axis=1)
                spans[:, j] = hi - lo

        out = []
        for ch in range(len(CHANNELS)):
            center = float(np.nanmedian(centers[ch]))
            span = float(np.nanmedian(spans[ch]))
            if not np.isfinite(center):
                center = 0.0
            # a flat or dead channel still needs a usable window
            if not np.isfinite(span) or span <= 0:
                span = max(abs(center) * 0.1, 1e-9)
            # 2.0 leaves a typical page filling ~85% of its lane: readable,
            # with headroom so ordinary peaks are not clipped
            span *= 2.0

            # a pinned scale wins: it exists precisely for channels where rare
            # huge excursions would otherwise dictate the whole night's view
            pinned = parse_half_scale(DEFAULT_HALF_SCALE.get(CHANNELS[ch]))
            if pinned is not None:
                span = pinned * 2

            out.append({"center": center, "span": span})
        return out

    def default_baseline(self) -> list[dict]:
        """Recompute the recording-wide default scale."""
        return self._baseline()

    def window_medians(self, t0: float, t1: float, max_samples: int = 20000):
        """Median of each channel over one page, used to keep traces centred."""
        i0 = max(int(t0 * self.sfreq), 0)
        i1 = min(int(t1 * self.sfreq), self.n_times)
        if i1 <= i0:
            return np.array([b["center"] for b in self.baseline], dtype=np.float64)

        step = max(1, (i1 - i0) // max_samples)
        block = self._rows(
            np.asarray(self.data[:, i0:i1:step], dtype=np.float64)
        )
        with np.errstate(all="ignore"):
            meds = np.nanmedian(block, axis=1)
        fallback = np.array([b["center"] for b in self.baseline], dtype=np.float64)
        return np.where(np.isfinite(meds), meds, fallback)

    def window_baseline(self, t0: float, t1: float) -> list[dict]:
        """Same measurement, but over one visible page. Used by Auto scale."""
        i0 = max(int(t0 * self.sfreq), 0)
        i1 = min(int(t1 * self.sfreq), self.n_times)
        if i1 <= i0:
            return self.baseline

        block = self._rows(np.asarray(self.data[:, i0:i1], dtype=np.float64))
        out = []
        for ch in range(block.shape[0]):
            v = block[ch]
            v = v[np.isfinite(v)]
            if v.size == 0:
                out.append(dict(self.baseline[ch]))
                continue
            # true extremes, not percentiles: "fit this page" has to mean the
            # whole page fits, including the peak the reviewer is looking at
            lo, hi = float(v.min()), float(v.max())
            center = (lo + hi) / 2
            span = (hi - lo) * 1.15
            if span <= 0:
                span = max(abs(center) * 0.1, 1e-9)
            out.append({"center": center, "span": span})
        return out

    # -- windows -------------------------------------------------------------
    def get_window(self, t0: float, t1: float, max_points: int = 1800):
        """Return (x, y) per channel for [t0, t1), min/max decimated if needed."""
        i0 = max(int(t0 * self.sfreq), 0)
        i1 = min(int(t1 * self.sfreq), self.n_times)
        if i1 <= i0:
            empty = np.array([], dtype=np.float32)
            return [(empty, empty) for _ in CHANNELS]

        block = self._rows(np.asarray(self.data[:, i0:i1], dtype=np.float32))
        n = block.shape[1]

        if n <= max_points:
            x = (np.arange(i0, i1, dtype=np.float64)) / self.sfreq
            return [(x, block[c]) for c in range(block.shape[0])]

        # min/max envelope: preserves spikes that plain decimation would drop
        n_bins = max_points // 2
        per_bin = n // n_bins
        usable = n_bins * per_bin
        binned = block[:, :usable].reshape(block.shape[0], n_bins, per_bin)
        mins = binned.min(axis=2)
        maxs = binned.max(axis=2)

        bin_starts = i0 + np.arange(n_bins, dtype=np.float64) * per_bin
        x = np.repeat(bin_starts / self.sfreq, 2)

        out = []
        for c in range(block.shape[0]):
            y = np.empty(n_bins * 2, dtype=np.float32)
            y[0::2] = mins[c]
            y[1::2] = maxs[c]
            out.append((x, y))
        return out