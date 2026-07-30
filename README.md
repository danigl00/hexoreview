# hexoreview

A blinded review dashboard for Hexoskin recordings. Reviwers page
through each recording, mark seizures by dragging on the trace, and the tool
exports analysis-ready CSVs.

Reviewers only ever see neutral labels (`night_001`, …) and elapsed time; they are blinded to
the patient identity or the recording date. Identity lives in a private map.

---

## For reviewers

If your machine has already been set up for you (a **Seizure Review** icon on the
desktop), reviewing takes six steps:

1. **Double-click the Seizure Review icon.** The dashboard opens in your browser
   after a few seconds. (Double-clicking again is safe — it just reopens the
   tab.)
2. **Sign in.** Pick your reviewer ID, type your passcode if you have one, and
   click **Start reviewing**.
3. **Open a recording.** Choose one from the worklist and click **Open
   recording**.
4. **Mark seizures.** Drag left-to-right anywhere on a trace to mark a seizure.
   Fine-tune the exact boundaries with the onset/offset boxes, or use **Mark
   whole page as seizure**.
5. **Finish the night.** Click **Finish this night** when done! This records
   the review even if you found nothing.
6. **Save and exit** when you're finished for the session. You can then close the
   window.

Your worklist shows only your own recordings and marks. Take your time, the
timer only counts while you're actually working, and pauses when the window sits
idle.

New to reading these? Open a **Training sample** from the sidebar first: those
are example recordings with the seizures already marked, so you can see what to
look for. Nothing you do on a training sample is scored.

There's a **Using the viewer** section further down with the full set of controls
and keyboard shortcuts.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10. From the project
folder:

```powershell
uv sync
```

On Windows, make sure `tzdata` is installed (uv handles this from the project's
dependencies) so the Eastern timezone resolves correctly.

To give a reviewer a desktop shortcut, run once on their machine from the project
root:

```powershell
powershell -ExecutionPolicy Bypass -File launch\create_shortcut.ps1
```

This creates a **Seizure Review** shortcut that launches the dashboard directly.
Reviewers never need a terminal.

---

## For the study coordinator

### 1. Register reviewers

```powershell
uv run hexoreview reviewer add --id rev_01 --name "J. Smith" --passcode 1234
uv run hexoreview reviewer add --id coord  --name "Coordinator" --coordinator
uv run hexoreview reviewer list
```

Passcodes are optional. A **coordinator** account can see study-wide progress and
run the export, but cannot open recordings — register it only on the machine that
holds the private map.

### 2. Blind the recordings

```powershell
uv run hexoreview scan path/to/Hexoskin_recordings/
    --samples-dir path/to/sample_recordings/
    --samples-csv  path/to/samples_edf.csv
```

Every EDF under the source folder is given a neutral `night_NNN` label, linked
into `blinded/`, and its real identity written to `private/blinding_map.csv`.
Rerun any time you add recordings — existing labels are left untouched.

Use `--copy` if symlinks are blocked on your machines, and `--seed` to control
the label shuffle.

Building the training samples is **optional**. Point `--samples-dir` at a folder
of full recordings and pass a seizure CSV with `--samples-csv`. The CSV lists the
seizures to mark, one per row, with at least an onset and offset timestamp in
Eastern time:

- `electric_onset` — seizure onset (required)
- `sz_offset` — seizure end (optional; defaults to the onset)
- `sz_date` — only needed if the onset/offset are a time of day without a date
- `sz_type` — a label shown in the marks table (optional)

Timestamps may be full (`2026-01-05 03:12:00`) or a bare time paired with
`sz_date`. Each recording in `--samples-dir` whose interval contains a seizure
onset is added to the training set with those seizures pre-marked. Keep
`--samples-dir` **outside** the scan source so these example recordings never
enter the scored worklist. See **Training samples** below.

### 3. Pre-convert recordings (optional but recommended)

```powershell
uv run hexoreview precache
```

The first time a recording opens it is converted to a fast memory-mapped cache,
which takes a while for hours-long recordings. Precaching does this ahead of time so
reviewers never wait.

### 4. Deploy to reviewer machines

Copy the project folder to each reviewer's machine, **excluding the private material**: 
leave out `./private/blinding_map.csv` and `path/to/sample_recordings/samples_private_map.csv`. Reviewers need `./blinded/`,
`./review/`, and (if built) `./review/samples/`; they must not receive anything that maps a label back to a patient.

Then run `create_shortcut.ps1` on each machine (see Installation).

### 5. Collect results

The analysis CSVs are rebuilt automatically whenever a reviewer finishes a night
or exits. To regenerate them on demand, or from the coordinator machine:

```powershell
uv run hexoreview export
uv run hexoreview status
```

`export` writes `annotations.csv` and `coverage.csv`; `status` prints per-reviewer
progress (nights finished, marks, minutes spent).

---


## Using the viewer

- **Marking** — drag left-to-right on any trace to mark a seizure. Select a row
  and use the onset/offset boxes to adjust it exactly, or **Mark whole page as seizure**. 
  Select a mark and **Delete mark** to remove it.
- **Navigation** — `←` `→` page forward/back, `Shift+←` `Shift+→` nudge; the
  position slider and the whole-night strip jump anywhere. **Go to mark** jumps
  to the selected mark.
- **Page length** — choose how much time is on screen, from 10 s to 1 h.
- **Scale** — scroll inside a channel to zoom it vertically; `▲` `▼` move it up
  and down; `+` `−` zoom in steps. The figure under each channel name is the
  half-height of that lane. **Reset scale** returns to defaults.
- **Auto-centre** keeps each trace centred as the baseline drifts. **Fit to  window** shares the height evenly across the six lanes; 
  turn it off to set a fixed lane height (or drag a lane's bottom edge).
- **Finish this night** records the review, including "no seizures found."
  **Reopen for editing** reverts a finished night. **Save and exit** saves,
  refreshes the export, and shuts the server down.

---

## How it stays blinded

- Reviewers see only `night_NNN` labels and elapsed time; the real filename,
  patient identity, and recording date never appear in the interface.
- Patient identity is written only to `./private/blinding_map.csv`, which stays on
  the coordinator machine. Without it, exports still run — the patient and record
  columns simply read `UNMAPPED`.
- Labels are shuffled with a recorded seed, and nothing tells a reviewer how many
  patients are in the set.
- Training samples follow the same rules: neutral labels, elapsed time, identity
  only in the coordinator-side private index.

---

## Configuration

Defaults, all overridable with flags or environment variables:

| Path | Holds | Env var |
|------|-------|---------|
| `blinded/` | blinded `night_NNN.edf` links | `HEXOREVIEW_BLINDED_DIR` |
| `private/blinding_map.csv` | label → patient/record/start (**private**) | `HEXOREVIEW_MAP` |
| `review/reviews.sqlite` | reviewers, status, marks, timings | `HEXOREVIEW_DB` |
| `review/cache/` | per-recording caches | `HEXOREVIEW_CACHE` |
| `review/samples/` | training samples + manifest | `HEXOREVIEW_SAMPLES` |
| `export/` | `annotations.csv`, `coverage.csv` | `HEXOREVIEW_EXPORT` |

`HEXOREVIEW_CLOCK=1` (or `run --clock`) shows time of day instead of elapsed time;
off by default, since the wall-clock time hints at which recording is on screen.

**Channels.** The viewer shows six channels (set in `hexoreview/__init__.py`):
`ECG_I`, `resp_thorac`, `resp_abdomi`, `accel_X`, `accel_Y`, `accel_Z`. Names are
matched on the part after the colon (Hexoskin exports names like `4113:ECG_I`).
`DEFAULT_HALF_SCALE` pins the opening vertical scale for a channel — the
accelerometer axes open at ±2.2 g so rare large excursions don't flatten
everything else.

**Timezones.** CSV timestamps are Eastern; EDF start times are UTC. The tool
attaches the Eastern zone (DST-aware) before comparing, so the EST/EDT switch is
automatic. The display/export zone is `EXPORT_TIMEZONE` in
`hexoreview/__init__.py`; the samples CSV zone is `CSV_TZ` in
`hexoreview/samples.py`. Both default to Eastern.

---

## Export files

`annotations.csv` — one row per seizure mark:
`clinician_id`, `patient_id`, `record_name`, `blind_label`, `onset`, `offset`,
`duration_s`, `onset_clock`, `offset_clock`, `note`, `annotation_id`,
`created_at`.

`coverage.csv` — one row per (reviewer, recording), so reviewed-with-nothing-found
is explicit: `clinician_id`, `patient_id`, `record_name`, `blind_label`,
`review_status`, `n_annotations`, `outcome`, `recording_start`,
`recording_duration_s`, `review_seconds`, `review_minutes`, `n_sittings`,
`review_min_per_recorded_hour`, `completed_at`.

---

## Command reference

| Command | Purpose | Key flags |
|---------|---------|-----------|
| `scan SOURCE_DIR` | blind new EDFs; optionally build samples | `--samples-dir`, `--samples-csv`, `--copy`, `--seed` |
| `reviewer add\|list` | register or list reviewers | `--id`, `--name`, `--passcode`, `--coordinator` |
| `precache` | pre-convert recordings for instant opening | |
| `run` | start the dashboard | `--port`, `--no-browser`, `--allow-remote`, `--clock`, `--out-dir` |
| `export` | write `annotations.csv` + `coverage.csv` | `--out` |
| `status` | progress per reviewer | |

Global flags (before the subcommand): `--db`, `--blinded-dir`, `--cache`,
`--map`, `--export-dir`, `--samples-out`.

**`run` flags:** `--port` (default 5006) sets the local port; `--no-browser`
starts without opening a tab; `--allow-remote` serves to other machines on the
network (local-only otherwise); `--clock` shows time of day; `--out-dir` sets
where the analysis CSVs are written.

---

## Troubleshooting

**"That passcode does not match."** Check the reviewer ID is the right one and the
passcode matches what was registered. Reviewers with no passcode leave the box
empty.

**The dashboard didn't open / the icon does nothing.** A failed launch is logged
to `launch\last_run.log`; send that file to the coordinator. If a server is
already running from an earlier session, launching again just reopens the tab.

**"Port already in use."** Another dashboard is already running on that port —
launching again reopens it, or use `run --port <other>` for a second instance.

**Times look off by an hour.** Confirm `tzdata` is installed and that
`EXPORT_TIMEZONE` / `CSV_TZ` are the intended Eastern zones.

**A training sample is missing.** The build prints how many recordings and CSV
rows it read and a per-recording match count. A sample only appears if its
recording is in `--samples-dir` and a seizure onset falls inside that recording.