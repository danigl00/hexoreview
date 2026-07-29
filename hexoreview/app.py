"""Seizure review dashboard.

Launch with:  hexoreview run --blinded-dir blinded --db review/reviews.sqlite
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import SelectionGeometry, Tap
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    CustomJSTickFormatter,
    PrintfTickFormatter,
    Range1d,
    WheelZoomTool,
)
from bokeh.plotting import figure

from . import CHANNELS
from .config import config
from .data import Recording, to_local
from .store import Store

pn.extension("tabulator", notifications=True, sizing_mode="stretch_width")

TRACE = "#16181d"
MARK = "#e8871a"
WINDOW = "#3d5a80"
GRID = "#dfe2e8"

WINDOW_CHOICES = {
    "10 s": 10, "30 s": 30, "1 min": 60, "2 min": 120, "5 min": 300,
    "10 min": 600, "30 min": 1800, "1 h": 3600,
}
PLOT_HEIGHT = 150        # only used when Fit to window is switched off
MIN_LANE = 56
MAX_LANE = 560
AXIS_STRIP = 38          # the time axis lives in its own strip below the lanes

# Review time is only counted while the reviewer is actually doing something.
# The browser reports activity (mouse, keys, scrolling) at most every 20 s; if
# nothing arrives for IDLE_LIMIT the clock stops, so a window left open all
# afternoon does not count as review time.
TICK_SECONDS = 10
IDLE_LIMIT = 90
FLUSH_EVERY = 30


def hms(seconds: float) -> str:
    """Elapsed seconds as HH:MM:SS."""
    seconds = max(float(seconds), 0.0)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


SI_PREFIXES = {"n", "u", "µ", "μ", "m", "c", "d", "k", "M"}
BASE_UNITS = {"V", "A", "g", "s", "m", "Ohm", "ohm", "Hz", "T", "W", "Pa"}


def base_unit(unit: str) -> str:
    """Drop the SI prefix from an EDF physical dimension.

    MNE rescales EDF signals to SI base units when it reads them, so a channel
    the header calls 'uV' arrives as volts. Keeping the header's prefix would
    label a nanovolt reading 'nuV'.
    """
    unit = (unit or "").strip()
    if len(unit) > 1 and unit[0] in SI_PREFIXES and unit[1:] in BASE_UNITS:
        return unit[1:]
    return unit


def eng(value: float, unit: str = "") -> str:
    """Compact number with an SI prefix, e.g. 1.2 mV."""
    value = float(value)
    if not np.isfinite(value) or value == 0:
        return f"0 {unit}".strip()
    prefixes = [(1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1, ""), (1e3, "k")]
    scale, prefix = prefixes[0]
    for factor, name in prefixes:
        if abs(value) >= factor:
            scale, prefix = factor, name
    shown = value / scale
    digits = 0 if abs(shown) >= 100 else (1 if abs(shown) >= 10 else 2)
    return f"{shown:.{digits}f} {prefix}{unit}".strip()


def clock_of(start, seconds: float) -> str:
    """Time of day at `seconds` into a recording that began at `start`.

    Time of day only, never the date: a reviewer needs to know it is 03:12 in
    the night, but the date would identify the recording. Reported in the same
    timezone as the export.
    """
    from datetime import timedelta

    if start is None:
        return hms(seconds)
    return to_local(start + timedelta(seconds=float(seconds))).strftime("%H:%M:%S")


class ReviewApp:
    def __init__(
        self, store: Store, blinded_dir: Path, cache_dir: Path,
        clock_mode: bool = False, export_dir: Path | None = None,
        map_path: Path | None = None, samples_dir: Path | None = None,
    ):
        self.store = store
        self.blinded_dir = Path(blinded_dir)
        self.cache_dir = Path(cache_dir)
        self.clock_mode = clock_mode
        self.export_dir = Path(export_dir) if export_dir else Path("export")
        self.map_path = Path(map_path) if map_path else Path("private/blinding_map.csv")
        self.samples_dir = Path(samples_dir) if samples_dir else Path("review/samples")

        # training samples are display-only clips; loading one sets this so the
        # marking, timing and worklist machinery all stand down
        self.training_mode = False
        from .samples import MANIFEST_NAME, load_manifest

        self.sample_manifest = load_manifest(self.samples_dir / MANIFEST_NAME)

        self.clinician_name: str | None = None
        self.clinician_id: str | None = None
        self.role = "reviewer"
        self.rec: Recording | None = None
        self.label: str | None = None
        self.t0 = 0.0
        self.window = 60.0
        self.gains = [1.0] * len(CHANNELS)

        self._session_id: int | None = None
        self._last_activity = 0.0
        self._pending_seconds = 0.0
        self._periodic = None

        self._register_recordings()
        self._build_login()
        self._build_viewer()

        self.layout = pn.Column(self.login_card, sizing_mode="stretch_both")

    # ------------------------------------------------------------------ setup
    def _register_recordings(self):
        self.store.register_dir(self.blinded_dir)

    # ------------------------------------------------------------------ login
    def _build_login(self):
        ids = [c["clinician_id"] for c in self.store.clinicians()]
        self.login_select = pn.widgets.Select(
            name="Reviewer", options=ids or ["(no reviewers registered)"], width=260
        )
        self.login_pass = pn.widgets.PasswordInput(
            name="Passcode", placeholder="leave empty if none", width=260
        )
        self.login_btn = pn.widgets.Button(
            name="Start reviewing", button_type="primary", width=260
        )
        self.login_msg = pn.pane.Markdown("", width=260)
        self.login_btn.on_click(self._do_login)

        self.login_card = pn.Column(
            pn.pane.Markdown("## Overnight seizure review"),
            pn.pane.Markdown(
                "Sign in with your reviewer ID. You will only see your own marks."
            ),
            self.login_select, self.login_pass, self.login_btn, self.login_msg,
            width=320,
        )

    def _do_login(self, _):
        cid = self.login_select.value
        if not self.store.check_login(cid, self.login_pass.value):
            self.login_msg.object = "That passcode does not match. Try again."
            return
        self.clinician_id = cid
        row = next(
            (c for c in self.store.clinicians() if c["clinician_id"] == cid), None
        )
        self.clinician_name = (row["display_name"] if row else cid) or cid
        self.role = self.store.role_of(cid)

        if self.role == "coordinator":
            # coordinators see study-wide progress and the export, never the
            # recordings themselves
            self._build_coordinator_view()
            self.layout[:] = [self.coordinator]
            return

        self.who.object = (
            "<div style='padding:8px 10px;background:#eef1f6;border-radius:6px;"
            "border:1px solid #dfe2e8'>"
            "<div style='font:11px system-ui;color:#6b7280'>Signed in as</div>"
            "<div style='font:600 14px system-ui;color:#16181d'>"
            f"{self.clinician_name}</div>"
            f"<div style='font:11px ui-monospace,monospace;color:#6b7280'>{cid}</div>"
            "</div>"
        )
        self._touch()
        self._start_timer()
        self.refresh_worklist()
        self.layout[:] = [self.viewer]

    # ------------------------------------------------------------ coordinator
    def _build_coordinator_view(self):
        self.export_btn = pn.widgets.Button(
            name="Generate export", button_type="primary", width=180
        )
        self.export_btn.on_click(self._run_export)
        self.export_msg = pn.pane.Markdown("", width=620)
        self.download_row = pn.Row()
        self.progress_table = pn.widgets.Tabulator(
            self._progress_frame(), show_index=False, height=260,
            layout="fit_columns", disabled=True,
        )
        refresh = pn.widgets.Button(name="Refresh", width=100)
        refresh.on_click(lambda e: setattr(
            self.progress_table, "value", self._progress_frame()
        ))

        map_ok = Path(config.map_path).exists()
        note = (
            f"Blinding map: `{config.map_path}`"
            if map_ok
            else f"**No blinding map at `{config.map_path}`.** The export can still "
            "be generated, but patient and recording columns will read UNMAPPED."
        )

        self.coordinator = pn.Column(
            pn.pane.Markdown(f"## Study progress — {self.clinician_name}"),
            pn.pane.Markdown(
                "Recordings are not viewable from this account. "
                "Exports join the blinded labels back to patients using the "
                "private map, so keep this account off reviewer machines."
            ),
            pn.Row(refresh),
            self.progress_table,
            pn.layout.Divider(),
            pn.pane.Markdown("### Export"),
            pn.pane.Markdown(note),
            self.export_btn,
            self.export_msg,
            self.download_row,
            width=680,
        )

    def _progress_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.store.reviewers():
            cid = c["clinician_id"]
            work = self.store.worklist(cid)
            done = [r for r in work if r["status"] == "reviewed"]
            with_marks = [r for r in done if r["n_marks"]]
            minutes = sum(
                self.store.active_seconds(cid, r["blind_label"]) for r in work
            ) / 60
            rows.append(
                {
                    "reviewer": cid,
                    "name": c["display_name"],
                    "nights": len(work),
                    "finished": len(done),
                    "with marks": len(with_marks),
                    "marks": sum(r["n_marks"] for r in work),
                    "minutes": round(minutes, 1),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "reviewer", "name", "nights", "finished", "with marks",
                "marks", "minutes",
            ],
        )

    def _run_export(self, _):
        from .export import export

        self.export_btn.loading = True
        try:
            paths = export(config.db_path, config.map_path, config.export_dir)
            ann = pd.read_csv(paths["annotations"])
            cov = pd.read_csv(paths["coverage"])
            self.export_msg.object = (
                f"Exported **{len(ann)}** marks across "
                f"**{cov['blind_label'].nunique()}** recordings to "
                f"`{config.export_dir}`."
            )
            self.download_row[:] = [
                pn.widgets.FileDownload(
                    file=str(paths["annotations"]), filename="annotations.csv",
                    button_type="success", width=200,
                ),
                pn.widgets.FileDownload(
                    file=str(paths["coverage"]), filename="coverage.csv",
                    button_type="success", width=200,
                ),
            ]
            self.progress_table.value = self._progress_frame()
        except Exception as exc:
            self.export_msg.object = f"Export failed: `{exc}`"
        finally:
            self.export_btn.loading = False

    # ----------------------------------------------------------------- viewer
    def _build_viewer(self):
        self.worklist = pn.widgets.Select(name="Recording", options=[], width=260)
        self.show_reviewed = pn.widgets.Checkbox(name="Include finished nights")
        self.load_btn = pn.widgets.Button(
            name="Open recording", button_type="primary", width=260
        )
        self.status_md = pn.pane.Markdown("No recording open.", width=260)
        self.show_reviewed.param.watch(lambda e: self.refresh_worklist(), "value")
        self.load_btn.on_click(self._load_selected)

        # training samples menu (one entry per recording that contains seizures)
        sample_opts = {
            f"{label}  ·  {len(info['marks'])} seizure(s)": label
            for label, info in self.sample_manifest.items()
        }
        self.sample_select = pn.widgets.Select(
            name="Training sample",
            options=sample_opts or {"(none available)": ""},
            width=260,
        )
        self.load_sample_btn = pn.widgets.Button(
            name="Load sample", width=260, disabled=not sample_opts,
        )
        self.load_sample_btn.on_click(self._load_sample)
        self.sample_note = pn.pane.Markdown("", width=260)

        self.win_select = pn.widgets.Select(
            name="Page length", options=list(WINDOW_CHOICES), value="1 min", width=110
        )
        self.win_select.param.watch(self._on_window_change, "value")

        self.pos_slider = pn.widgets.FloatSlider(
            name="Position", start=0, end=1, value=0, step=1, format="0.0",
            show_value=False,
        )
        self.pos_slider.param.watch(self._on_slider, "value_throttled")

        self.btn_back = pn.widgets.Button(
            name="◀ page", width=80, css_classes=["kb-prev"]
        )
        self.btn_fwd = pn.widgets.Button(
            name="page ▶", width=80, css_classes=["kb-next"]
        )
        self.btn_back.on_click(lambda e: self._step(-1.0))
        self.btn_fwd.on_click(lambda e: self._step(1.0))
        self.btn_back_s = pn.widgets.Button(name="◀", width=44, css_classes=["kb-prevs"])
        self.btn_fwd_s = pn.widgets.Button(name="▶", width=44, css_classes=["kb-nexts"])
        self.btn_back_s.on_click(lambda e: self._step(-0.2))
        self.btn_fwd_s.on_click(lambda e: self._step(0.2))

        self.clock = pn.pane.HTML("", width=230)

        self.fit_window = pn.widgets.Checkbox(name="Fit to window", value=True)
        self.fit_window.param.watch(self._on_fit_toggle, "value")
        self.lane_height = pn.widgets.IntSlider(
            name="Lane height", start=MIN_LANE, end=MAX_LANE, step=10,
            value=PLOT_HEIGHT, width=140, disabled=True,
        )
        self.lane_height.param.watch(self._on_lane_height, "value_throttled")

        self.autocenter = pn.widgets.Checkbox(name="Auto-centre", value=True)
        self.autocenter.param.watch(self._on_autocenter_toggle, "value")

        self.reset_scale_btn = pn.widgets.Button(name="Reset scale", width=100)
        self.reset_scale_btn.on_click(self._reset_scale)

        self.mark_btn = pn.widgets.Button(
            name="Mark whole page as seizure", button_type="warning", width=210
        )
        self.mark_btn.on_click(lambda e: self._add_mark(self.t0, self.t0 + self.window))

        self.finish_btn = pn.widgets.Button(
            name="Finish this night", button_type="success", width=260, disabled=True
        )
        self.finish_btn.on_click(self._finish)
        self.reopen_btn = pn.widgets.Button(
            name="Reopen for editing", width=260, visible=False
        )
        self.reopen_btn.on_click(self._reopen)

        self.exit_btn = pn.widgets.Button(
            name="Save and exit", button_type="primary", width=260
        )
        self.exit_btn.on_click(self._save_and_exit)

        self.table = pn.widgets.Tabulator(
            self._empty_table(),
            hidden_columns=["id", "onset_s", "offset_s"],
            editors={
                "start": None, "end": None, "dur": None,
                "note": {"type": "input"},
            },
            titles={
                "start": "start", "end": "end", "dur": "dur (s)", "note": "note",
            },
            widths={"start": 76, "end": 76, "dur": 58},
            selectable=1, show_index=False, height=220, layout="fit_columns",
        )
        self.table.on_edit(self._on_table_edit)
        self.table.param.watch(self._on_row_select, "selection")

        # exact boundaries are adjusted here rather than in the table, so the
        # table itself stays narrow enough to read
        self.onset_input = pn.widgets.FloatInput(
            name="onset (s)", width=86, step=0.5, disabled=True
        )
        self.offset_input = pn.widgets.FloatInput(
            name="offset (s)", width=86, step=0.5, disabled=True
        )
        self.apply_btn = pn.widgets.Button(name="Apply", width=76, disabled=True)
        self.apply_btn.on_click(self._apply_edit)

        self.goto_btn = pn.widgets.Button(name="Go to mark", width=124)
        self.del_btn = pn.widgets.Button(name="Delete mark", width=124)
        self.goto_btn.on_click(self._goto_selected)
        self.del_btn.on_click(self._delete_selected)

        self._build_figures()

        self.ping_btn = pn.widgets.Button(
            name="ping", css_classes=["hexo-ping"], width=1, height=1, margin=0
        )
        self.ping_btn.on_click(lambda e: self._touch())

        self.who = pn.pane.HTML("", width=280)

        sidebar = pn.Column(
            self.who,
            pn.pane.Markdown("### Worklist"),
            self.worklist, self.show_reviewed, self.load_btn, self.status_md,
            pn.layout.Divider(),
            pn.pane.Markdown("### Training samples"),
            pn.pane.Markdown(
                "Example seizures to learn from. Not scored.",
                styles={"font-size": "12px", "color": "#6b7280"},
            ),
            self.sample_select, self.load_sample_btn, self.sample_note,
            pn.layout.Divider(),
            pn.pane.Markdown("### Marks on this night"),
            self.table,
            pn.Row(self.goto_btn, self.del_btn),
            pn.Row(self.onset_input, self.offset_input, self.apply_btn, margin=(4, 0)),
            pn.layout.Divider(),
            self.finish_btn, self.reopen_btn,
            pn.layout.Divider(),
            self.exit_btn,
            pn.pane.Markdown(
                "**Marking** — drag left-to-right anywhere on a trace to mark a "
                "seizure. Select a row and use the onset/offset boxes to "
                "adjust it exactly.\n\n"
                "**Scale** — scroll inside a channel to zoom it vertically, `▲` `▼` to "
                "move it up and down, `+` `−` to zoom in steps. The figure under "
                "each channel name is the half-height of that lane.\n\n"
                "**Lane size** — drag the bottom edge of any channel to make it "
                "taller, or use the lane height slider for all six.\n\n"
                "**Keys** — `←` `→` page, `Shift+←` `Shift+→` nudge.",
                styles={"font-size": "12px", "color": "#4a4f57"},
            ),
            width=300,
        )

        controls = pn.Row(
            self.btn_back, self.btn_back_s, self.btn_fwd_s, self.btn_fwd,
            pn.Spacer(width=16), self.win_select,
            pn.Spacer(width=16), self.clock,
            pn.Spacer(width=16), self.reset_scale_btn,
            pn.Spacer(width=16),
            pn.Column(self.autocenter, self.fit_window, margin=0, width=130),
            self.lane_height,
            pn.Spacer(width=16), self.mark_btn,
            sizing_mode="stretch_width",
        )

        self.plot_area = pn.Column(
            controls,
            self.pos_slider,
            self.overview_pane,
            *self.channel_rows,
            pn.Row(
                pn.Spacer(width=92),
                self.axis_pane,
                sizing_mode="stretch_width",
                height=AXIS_STRIP,
                margin=0,
            ),
            sizing_mode="stretch_both",
        )
        self.viewer = pn.Row(
            sidebar,
            pn.Column(
                self.plot_area, self.ping_btn, self._keyboard_hook(),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

    def _empty_table(self):
        return pd.DataFrame(
            columns=["id", "start", "end", "onset_s", "offset_s", "dur", "note"]
        )

    # ---------------------------------------------------------------- figures
    def _build_figures(self):
        # Every figure gets its own Bokeh models. A single ColumnDataSource or
        # Range1d shared across figures that live in separate panes belongs to
        # only one of them, so updates reach some plots and not others and stale
        # marks survive a recording change. Six cheap copies, kept in step by
        # hand, behave predictably.
        self.mark_sources = []
        self.x_ranges = []
        self.scale_labels = []
        self.lanes = []
        self.pan_buttons = []
        self.sources, self.figs, self.y_ranges = [], [], []
        self.mark_bands = []
        self.channel_rows = []

        for i, name in enumerate(CHANNELS):
            src = ColumnDataSource(dict(x=[], y=[]))
            mark_src = ColumnDataSource(dict(left=[], right=[]))
            x_range = Range1d(0, 60)
            y_range = Range1d(-1, 1)
            fig = figure(
                sizing_mode="stretch_both",
                x_range=x_range, y_range=y_range,
                tools="", toolbar_location=None, output_backend="webgl",
                min_border_left=8, min_border_right=8,
                min_border_top=2, min_border_bottom=2,
            )
            box = BoxSelectTool(dimensions="width", persistent=False)
            # scrolling inside a lane zooms it vertically, which is how you get
            # at a channel whose excursions run off the top and bottom
            wheel = WheelZoomTool(dimensions="height")
            fig.add_tools(box, wheel)
            fig.toolbar.active_drag = box
            fig.toolbar.active_scroll = wheel
            fig.on_event(SelectionGeometry, self._on_drag_select)

            # top/bottom are set per channel in _apply_y_ranges once a recording
            # is loaded. They must stay near the channel's own amplitude: a huge
            # constant loses all precision in the WebGL float32 transform and the
            # band silently fails to draw on low-amplitude channels.
            band = fig.quad(
                left="left", right="right", top=1.0, bottom=-1.0,
                source=mark_src, fill_color=MARK, fill_alpha=0.22,
                line_color=MARK, line_width=1, level="underlay",
            )
            self.mark_bands.append(band)
            fig.line("x", "y", source=src, line_width=1, line_color=TRACE)

            fig.yaxis.visible = True
            fig.yaxis.ticker.desired_num_ticks = 3
            fig.yaxis.formatter = PrintfTickFormatter(format="%.3g")
            fig.yaxis.axis_line_color = None
            fig.yaxis.major_tick_line_color = GRID
            fig.yaxis.minor_tick_line_color = None
            fig.ygrid.visible = False
            fig.xgrid.grid_line_color = GRID
            fig.xaxis.visible = False
            fig.background_fill_color = "#ffffff"
            fig.border_fill_color = None
            fig.outline_line_color = GRID
            fig.min_border_left = 58

            up = pn.widgets.Button(name="+", width=30, height=22, margin=(0, 2))
            down = pn.widgets.Button(name="−", width=30, height=22, margin=(0, 2))
            up.on_click(lambda e, i=i: self._zoom(i, 2.0))
            down.on_click(lambda e, i=i: self._zoom(i, 0.5))
            shift_up = pn.widgets.Button(name="▲", width=30, height=22, margin=(0, 2))
            shift_dn = pn.widgets.Button(name="▼", width=30, height=22, margin=(0, 2))
            shift_up.on_click(lambda e, i=i: self._pan(i, 0.25))
            shift_dn.on_click(lambda e, i=i: self._pan(i, -0.25))
            shift_up.disabled = True     # auto-centre is on by default
            shift_dn.disabled = True
            self.pan_buttons.append((shift_up, shift_dn))

            scale_label = pn.pane.HTML("", width=88, height=16, margin=(0, 0, 0, 4))
            self.scale_labels.append(scale_label)

            side = pn.Column(
                pn.pane.HTML(
                    f"<div style='font:600 11px system-ui;color:#16181d'>{name}</div>",
                    height=16, margin=(2, 0, 0, 4),
                ),
                scale_label,
                pn.Row(down, up, margin=0),
                pn.Row(shift_dn, shift_up, margin=0),
                width=92, margin=0,
            )

            plot_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_width")
            # CSS resize handle: drag the bottom edge of a lane to make a
            # channel taller
            lane = pn.Column(
                plot_pane,
                sizing_mode="stretch_both",
                margin=0,
                styles={
                    "overflow": "hidden",
                    "min-height": f"{MIN_LANE}px",
                    "border-bottom": f"1px solid {GRID}",
                },
            )
            self.lanes.append(lane)

            self.sources.append(src)
            self.mark_sources.append(mark_src)
            self.x_ranges.append(x_range)
            self.figs.append(fig)
            self.y_ranges.append(y_range)
            self.channel_rows.append(
                pn.Row(side, lane, sizing_mode="stretch_both", margin=0)
            )

        # whole-night position strip
        self.ov_marks = ColumnDataSource(dict(left=[], right=[]))
        self.ov_window = ColumnDataSource(dict(left=[0], right=[60]))
        ov = figure(
            height=46, sizing_mode="stretch_width", tools="tap",
            toolbar_location=None, y_range=Range1d(0, 1), x_range=Range1d(0, 1),
            min_border_left=8, min_border_right=8, min_border_top=2,
            min_border_bottom=2,
        )
        ov.quad(left="left", right="right", top=1, bottom=0, source=self.ov_window,
                fill_color=WINDOW, fill_alpha=0.18, line_color=WINDOW)
        ov.quad(left="left", right="right", top=1, bottom=0, source=self.ov_marks,
                fill_color=MARK, fill_alpha=0.95, line_color=MARK)
        ov.yaxis.visible = False
        ov.ygrid.visible = False
        ov.xgrid.visible = False
        ov.background_fill_color = "#f2f3f5"
        ov.border_fill_color = None
        ov.outline_line_color = GRID
        ov.on_event(Tap, self._on_overview_tap)
        self.overview = ov
        self.overview_pane = pn.pane.Bokeh(ov, sizing_mode="stretch_width")

        # The time axis gets its own strip so all six channel lanes stay exactly
        # the same height; otherwise the bottom one loses ~36 px to the axis and
        # its trace is visibly smaller than the rest.
        axis_range = Range1d(0, 60)
        axis_fig = figure(
            height=AXIS_STRIP, sizing_mode="stretch_width",
            x_range=axis_range, y_range=Range1d(0, 1),
            tools="", toolbar_location=None,
            min_border_left=58, min_border_right=8,
            min_border_top=0, min_border_bottom=0,
        )
        axis_fig.yaxis.visible = False
        axis_fig.ygrid.visible = False
        axis_fig.xgrid.visible = False
        axis_fig.background_fill_color = None
        axis_fig.border_fill_color = None
        axis_fig.outline_line_color = None
        axis_fig.xaxis.axis_label = "elapsed time (s)"
        self.axis_range = axis_range
        self.axis_fig = axis_fig
        self.axis_pane = pn.pane.Bokeh(axis_fig, sizing_mode="stretch_width")
        self.x_ranges.append(axis_range)

    def _keyboard_hook(self):
        return pn.pane.HTML(
            """
<style>.hexo-ping { opacity: 0; pointer-events: none; height: 1px; overflow: hidden; }</style>
<script>
(function () {
  if (window.__hexoKeys) return;
  window.__hexoKeys = true;
  const click = (cls) => {
    const el = document.querySelector('.' + cls + ' button, button.' + cls);
    if (el) el.click();
  };
  document.addEventListener('keydown', (e) => {
    const t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || e.metaKey || e.ctrlKey) return;
    if (e.key === 'ArrowLeft')  { click(e.shiftKey ? 'kb-prevs' : 'kb-prev'); }
    else if (e.key === 'ArrowRight') { click(e.shiftKey ? 'kb-nexts' : 'kb-next'); }
    else return;
    e.preventDefault();
  });

  // Tell the server the reviewer is still working, at most once every 20 s.
  // Reading a page without clicking anything still counts as review time.
  let last = 0;
  const ping = () => {
    const now = Date.now();
    if (now - last < 20000 || document.hidden) return;
    last = now;
    click('hexo-ping');
  };
  ['mousemove', 'keydown', 'wheel', 'click', 'scroll'].forEach((ev) =>
    document.addEventListener(ev, ping, { passive: true })
  );
})();
</script>""",
            height=0, margin=0,
        )

    # ------------------------------------------------------------- navigation
    def refresh_worklist(self):
        rows = self.store.worklist(self.clinician_id)
        if not self.show_reviewed.value:
            rows = [r for r in rows if r["status"] != "reviewed"]
        options = {}
        for r in rows:
            tag = {"reviewed": "done", "in_progress": "in progress"}.get(
                r["status"], "new"
            )
            options[f"{r['blind_label']}  ·  {tag}"] = r["blind_label"]
        keep = self.worklist.value
        self.worklist.options = options
        if keep in options.values():
            self.worklist.value = keep
        remaining = sum(1 for r in rows if r["status"] != "reviewed")
        self.status_md.object = (
            f"**{remaining}** night(s) left to review."
            if not self.show_reviewed.value
            else f"**{len(rows)}** night(s) listed."
        )

    def _clear_plots(self):
        """Blank every trace, band and table entry.

        Called before a new recording is loaded so nothing from the previous one
        can survive if the new load is slow or fails partway.
        """
        empty = dict(x=[], y=[])
        for src in self.sources:
            src.data = dict(empty)
        for mark_src in self.mark_sources:
            mark_src.data = dict(left=[], right=[])
        self.ov_marks.data = dict(left=[], right=[])
        self.ov_window.data = dict(left=[0], right=[0])
        self.table.selection = []
        self.table.value = self._empty_table()

    def _load_selected(self, _):
        label = self.worklist.value
        if not label:
            return
        self.load_btn.loading = True
        self.load_btn.name = "Preparing recording…"
        try:
            self._flush_time()
            self._session_id = None
            self.rec = None
            self.label = None
            self._clear_plots()
            self.training_mode = False
            self.sample_note.object = ""

            path = self.blinded_dir / f"{label}.edf"
            self.rec = Recording(path, self.cache_dir)
            self.label = label
            self.store.open_recording(self.clinician_id, label)
            self._session_id = self.store.start_session(self.clinician_id, label)
            self._touch()
            self.gains = [1.0] * len(CHANNELS)
            self.t0 = 0.0
            self.pos_slider.end = max(self.rec.duration_s - self.window, 1.0)
            self.pos_slider.value = 0.0
            self.overview.x_range.start = 0
            self.overview.x_range.end = self.rec.duration_s
            self._apply_y_ranges()
            self._apply_time_axis()
            self._refresh_table()
            self._update_status_controls()
            self._redraw()
            pn.state.notifications.success(
                f"{label} open — {hms(self.rec.duration_s)} of recording", duration=4000
            )
        except Exception as exc:
            pn.state.notifications.error(f"Could not open {label}: {exc}", duration=8000)
        finally:
            self.load_btn.loading = False
            self.load_btn.name = "Open recording"

    def _load_sample(self, _):
        label = self.sample_select.value
        if not label:
            return
        info = self.sample_manifest.get(label)
        if info is None:
            pn.state.notifications.error("That sample is not available.", duration=5000)
            return
        self.load_sample_btn.loading = True
        self.load_sample_btn.name = "Preparing sample…"
        try:
            self._flush_time()
            self._session_id = None
            self.rec = None
            self.label = None
            self._clear_plots()
            self.training_mode = True

            path = self.samples_dir / info["clip_file"]
            self.rec = Recording(path, self.cache_dir)
            self.label = label
            self.gains = [1.0] * len(CHANNELS)
            self.t0 = 0.0
            self.pos_slider.end = max(self.rec.duration_s - self.window, 1.0)
            self.pos_slider.value = 0.0
            self.overview.x_range.start = 0
            self.overview.x_range.end = self.rec.duration_s
            self._apply_y_ranges()
            self._apply_time_axis()
            self._fill_sample_marks(info)
            self._update_status_controls()
            self._redraw()
            n = len(info["marks"])
            self.sample_note.object = (
                f"**{label}** — full recording, {hms(self.rec.duration_s)}, "
                f"{n} labelled seizure(s). Pick a row and 'Go to mark' to jump."
            )
            pn.state.notifications.success(f"{label} loaded", duration=3500)
        except Exception as exc:
            pn.state.notifications.error(
                f"Could not load {label}: {exc}", duration=8000
            )
        finally:
            self.load_sample_btn.loading = False
            self.load_sample_btn.name = "Load sample"

    def _fill_sample_marks(self, info: dict):
        """Show every labelled seizure (read-only), fill the table, jump to the first."""
        marks = info.get("marks", [])
        lefts = [float(m["onset_s"]) for m in marks]
        rights = [float(m["offset_s"]) for m in marks]
        for mark_src in self.mark_sources:
            mark_src.data = dict(left=lefts, right=rights)
        tick = self.rec.duration_s / 400 if self.rec else 1.0
        self.ov_marks.data = dict(
            left=lefts, right=[max(r, l + tick) for l, r in zip(lefts, rights)]
        )
        self.table.value = (
            pd.DataFrame(
                [
                    {
                        "id": -(i + 1),  # negative: cannot collide with real marks
                        "start": hms(m["onset_s"]),
                        "end": hms(m["offset_s"]),
                        "onset_s": round(float(m["onset_s"]), 1),
                        "offset_s": round(float(m["offset_s"]), 1),
                        "dur": round(float(m["offset_s"]) - float(m["onset_s"]), 1),
                        "note": m.get("sz_type", "") or "seizure",
                    }
                    for i, m in enumerate(marks)
                ]
            )
            if marks
            else self._empty_table()
        )
        self.table.selection = []
        if marks:
            self._set_t0(float(marks[0]["onset_s"]) - self.window / 4)

    def _update_status_controls(self):
        if self.training_mode:
            # a training clip: nothing here is scored or editable
            self.finish_btn.disabled = True
            self.finish_btn.name = "Finish this night"
            self.reopen_btn.visible = False
            for w in (self.mark_btn, self.del_btn,
                      self.onset_input, self.offset_input, self.apply_btn):
                w.disabled = True
            return
        done = self.store.status_of(self.clinician_id, self.label) == "reviewed"
        self.finish_btn.disabled = done or self.rec is None
        self.finish_btn.name = "Finished" if done else "Finish this night"
        self.reopen_btn.visible = done
        for w in (self.mark_btn, self.del_btn):
            w.disabled = done
        if done:
            for w in (self.onset_input, self.offset_input, self.apply_btn):
                w.disabled = True

    def _step(self, pages: float):
        if not self.rec:
            return
        self._set_t0(self.t0 + pages * self.window)

    def _set_t0(self, t0: float):
        limit = max(self.rec.duration_s - self.window, 0.0)
        self.t0 = float(np.clip(t0, 0.0, limit))
        self.pos_slider.param.update(end=max(limit, 1.0), value=self.t0)
        self._redraw()

    def _on_slider(self, event):
        if self.rec and abs(event.new - self.t0) > 1e-6:
            self.t0 = float(event.new)
            self._redraw()

    def _on_window_change(self, event):
        self.window = float(WINDOW_CHOICES[event.new])
        if self.rec:
            self.pos_slider.step = max(self.window / 20, 0.1)
            self._set_t0(self.t0)

    def _on_overview_tap(self, event):
        if self.rec:
            self._set_t0(float(event.x) - self.window / 2)

    def _on_fit_toggle(self, event):
        """Share the window height between the six lanes, or use a fixed size."""
        self._touch()
        fit = bool(event.new)
        self.lane_height.disabled = fit
        for lane, row in zip(self.lanes, self.channel_rows):
            if fit:
                lane.height = None
                lane.sizing_mode = "stretch_both"
                row.sizing_mode = "stretch_both"
                row.height = None
            else:
                lane.sizing_mode = "stretch_width"
                lane.height = int(self.lane_height.value)
                row.sizing_mode = "stretch_width"
                row.height = int(self.lane_height.value)
        self.plot_area.sizing_mode = "stretch_both" if fit else "stretch_width"

    def _on_lane_height(self, event):
        """Fixed lane height, used when Fit to window is off."""
        self._touch()
        if self.fit_window.value:
            return
        for lane, row in zip(self.lanes, self.channel_rows):
            lane.height = int(event.new)
            row.height = int(event.new)

    def _on_autocenter_toggle(self, event):
        """Panning by hand only makes sense when auto-centring is off."""
        self._touch()
        on = bool(event.new)
        for up, down in self.pan_buttons:
            up.disabled = on
            down.disabled = on
        if self.rec:
            self._redraw()

    def _reset_scale(self, _=None):
        """Back to the recording-wide default scale."""
        if not self.rec:
            return
        self._touch()
        self.rec.baseline = self.rec.default_baseline()
        self.gains = [1.0] * len(CHANNELS)
        self._apply_y_ranges()

    def _view(self, idx: int) -> tuple[float, float]:
        """Current centre and half-height of one lane, as displayed."""
        y = self.y_ranges[idx]
        start, end = y.start, y.end
        if start is None or end is None or not np.isfinite(end - start):
            base = self.rec.baseline[idx] if self.rec else {"center": 0, "span": 1}
            return base["center"], base["span"] / 2
        return (start + end) / 2, (end - start) / 2

    def _zoom(self, idx: int, factor: float):
        """Vertical zoom, relative to what is on screen now."""
        if not self.rec:
            return
        self._touch()
        center, half = self._view(idx)
        half = float(np.clip(half / factor, 1e-12, 1e12))
        self.y_ranges[idx].start = center - half
        self.y_ranges[idx].end = center + half
        self._update_scale_label(idx)

    def _pan(self, idx: int, fraction: float):
        """Shift one lane up or down without changing its zoom."""
        if not self.rec:
            return
        self._touch()
        center, half = self._view(idx)
        center += fraction * half * 2
        self.y_ranges[idx].start = center - half
        self.y_ranges[idx].end = center + half
        self._update_scale_label(idx)

    def _update_scale_label(self, idx: int):
        """Show the full height of a lane in the channel's own units."""
        if not self.rec:
            self.scale_labels[idx].object = ""
            return
        _, half = self._view(idx)
        unit = base_unit(self.rec.units[idx] if idx < len(self.rec.units) else "")
        self.scale_labels[idx].object = (
            "<div style='font:10px ui-monospace,monospace;color:#6b7280'>"
            f"± {eng(half, unit)}</div>"
        )

    def _apply_y_ranges(self):
        if not self.rec:
            return
        for i, base in enumerate(self.rec.baseline):
            half = (base["span"] / 2) / self.gains[i]
            self.y_ranges[i].start = base["center"] - half
            self.y_ranges[i].end = base["center"] + half
            # wide enough to fill the plot at any gain, small enough to stay
            # precise in float32
            reach = base["span"] * 100
            self.mark_bands[i].glyph.top = base["center"] + reach
            self.mark_bands[i].glyph.bottom = base["center"] - reach
            self._update_scale_label(i)

    # ----------------------------------------------------------------- render
    def _redraw(self):
        if not self.rec:
            return
        self._touch()
        t1 = self.t0 + self.window
        for src, (x, y) in zip(self.sources, self.rec.get_window(self.t0, t1)):
            src.data = dict(x=x, y=y)

        # Keep every trace sitting in the middle of its lane. The span is left
        # alone, so sensitivity stays comparable between pages and between
        # channels; only the offset follows the signal. Without this, slow
        # baseline wander pushes traces out of view and the reviewer has to
        # keep panning.
        if self.autocenter.value:
            for i, med in enumerate(self.rec.window_medians(self.t0, t1)):
                _, half = self._view(i)
                self.y_ranges[i].start = float(med) - half
                self.y_ranges[i].end = float(med) + half
                self.mark_bands[i].glyph.top = float(med) + half * 200
                self.mark_bands[i].glyph.bottom = float(med) - half * 200
        for x_range in self.x_ranges:
            x_range.start, x_range.end = self.t0, t1
        self.ov_window.data = dict(left=[self.t0], right=[t1])
        self.clock.object = (
            "<div style='font:600 13px ui-monospace,monospace;color:#16181d'>"
            f"{self._fmt(self.t0)} – {self._fmt(t1)}</div>"
            "<div style='font:11px system-ui;color:#6b7280'>"
            f"{hms(self.t0)} into {hms(self.rec.duration_s)}</div>"
        )

    # ------------------------------------------------------------------ timer
    def _touch(self):
        """Record that the reviewer just did something."""
        self._last_activity = time.time()

    def _tick(self):
        """Called on a fixed interval; only counts time if recently active."""
        if not (self.rec and self._session_id):
            return
        if time.time() - self._last_activity > IDLE_LIMIT:
            return
        self._pending_seconds += TICK_SECONDS
        if self._pending_seconds >= FLUSH_EVERY:
            self._flush_time()

    def _flush_time(self):
        """Write buffered review time to the database."""
        if self._session_id and self._pending_seconds > 0:
            self.store.log_activity(
                self._session_id, self.clinician_id, self.label,
                self._pending_seconds,
            )
            self._pending_seconds = 0.0

    def _start_timer(self):
        """Start the periodic tick once a browser session exists."""
        if self._periodic is not None:
            return
        try:
            self._periodic = pn.state.add_periodic_callback(
                self._tick, period=TICK_SECONDS * 1000
            )
        except Exception:
            self._periodic = None  # no live session (tests, import time)

    def _fmt(self, seconds: float) -> str:
        """Time label for the reviewer: time of day, or elapsed if unavailable."""
        if self.clock_mode and not self.training_mode and self.rec is not None:
            return clock_of(self.rec.start_datetime, seconds)
        return hms(seconds)

    def _apply_time_axis(self):
        """Label the shared x axis as time of day, keeping the data in seconds."""
        axis = self.axis_fig.xaxis
        start = self.rec.start_datetime if self.rec else None
        if self.clock_mode and not self.training_mode and start is not None:
            axis.formatter = CustomJSTickFormatter(
                args=dict(t0=start.timestamp()),
                code="""
                    const d = new Date((t0 + tick) * 1000);
                    const p = (n) => String(n).padStart(2, '0');
                    return p(d.getHours()) + ':' + p(d.getMinutes())
                           + ':' + p(d.getSeconds());
                """,
            )
            axis.axis_label = "time of day"
        else:
            axis.axis_label = "elapsed time (s)"

    def _refresh_table(self):
        anns = self.store.annotations(self.clinician_id, self.label)
        df = pd.DataFrame(
            [
                {
                    "id": a["id"],
                    "start": self._fmt(a["onset_s"]),
                    "end": self._fmt(a["offset_s"]),
                    "onset_s": round(a["onset_s"], 1),
                    "offset_s": round(a["offset_s"], 1),
                    "dur": round(a["offset_s"] - a["onset_s"], 1),
                    "note": a["note"] or "",
                }
                for a in anns
            ]
        )
        self.table.value = df if len(df) else self._empty_table()
        self._on_row_select()
        bands = dict(
            left=[a["onset_s"] for a in anns], right=[a["offset_s"] for a in anns]
        )
        for mark_src in self.mark_sources:
            mark_src.data = dict(bands)
        tick = self.rec.duration_s / 400 if self.rec else 1.0
        self.ov_marks.data = dict(
            left=[a["onset_s"] for a in anns],
            right=[max(a["offset_s"], a["onset_s"] + tick) for a in anns],
        )

    # ------------------------------------------------------------ annotations
    def _on_drag_select(self, event):
        if not (self.rec and event.final):
            return
        if self.training_mode:
            pn.state.notifications.info(
                "Training sample — marking is disabled here.", duration=3000
            )
            return
        if self.store.status_of(self.clinician_id, self.label) == "reviewed":
            pn.state.notifications.warning(
                "This night is finished. Reopen it to change marks.", duration=4000
            )
            return
        geom = event.geometry or {}
        x0, x1 = geom.get("x0"), geom.get("x1")
        if x0 is None or x1 is None or abs(x1 - x0) < 0.05:
            return
        self._add_mark(x0, x1)

    def _add_mark(self, onset: float, offset: float):
        if not self.rec or self.training_mode:
            return
        self._touch()
        onset = float(np.clip(min(onset, offset), 0, self.rec.duration_s))
        offset = float(np.clip(max(onset, offset), 0, self.rec.duration_s))
        self.store.add_annotation(self.clinician_id, self.label, onset, offset)
        self._refresh_table()
        pn.state.notifications.info(
            f"Marked {self._fmt(onset)} – {self._fmt(offset)}", duration=2500
        )

    def _selected_id(self):
        sel = self.table.selection
        if not sel or self.table.value.empty:
            return None
        return int(self.table.value.iloc[sel[0]]["id"])

    def _on_row_select(self, event=None):
        """Load the selected mark into the boundary inputs."""
        ann_id = self._selected_id()
        enabled = ann_id is not None and not self.training_mode
        self.onset_input.disabled = not enabled
        self.offset_input.disabled = not enabled
        self.apply_btn.disabled = not enabled
        if not enabled:
            return
        row = self.table.value[self.table.value["id"] == ann_id].iloc[0]
        self.onset_input.value = float(row["onset_s"])
        self.offset_input.value = float(row["offset_s"])

    def _apply_edit(self, _):
        ann_id = self._selected_id()
        if ann_id is None:
            return
        self._touch()
        self.store.update_annotation(
            ann_id,
            onset_s=float(self.onset_input.value),
            offset_s=float(self.offset_input.value),
        )
        self._refresh_table()

    def _goto_selected(self, _):
        ann_id = self._selected_id()
        if ann_id is None:
            return
        row = self.table.value[self.table.value["id"] == ann_id].iloc[0]
        self._set_t0(float(row["onset_s"]) - self.window / 4)

    def _delete_selected(self, _):
        ann_id = self._selected_id()
        if ann_id is None:
            pn.state.notifications.warning("Select a mark first.", duration=3000)
            return
        self.store.delete_annotation(ann_id)
        self.table.selection = []
        self._refresh_table()

    def _on_table_edit(self, event):
        if event.column != "note":
            return
        ann_id = int(self.table.value.iloc[event.row]["id"])
        self.store.update_annotation(ann_id, note=event.value)

    # ----------------------------------------------------------------- finish
    def _finish(self, _):
        if not self.rec:
            return
        self._flush_time()
        n = len(self.store.annotations(self.clinician_id, self.label))
        self.store.set_status(self.clinician_id, self.label, "reviewed")
        self._export_now()
        self._update_status_controls()
        self.refresh_worklist()
        msg = (
            f"{self.label} saved with {n} seizure mark(s)."
            if n
            else f"{self.label} saved as reviewed, no seizures found."
        )
        pn.state.notifications.success(msg, duration=6000)

    def _export_now(self) -> bool:
        """Rebuild the analysis CSVs from the database.

        Always a full rebuild covering every reviewer and every recording, so
        the files on disk are current whenever this runs. Patient identity only
        appears if the private blinding map is present on this machine; without
        it the export is still complete, just blinded.
        """
        try:
            from .export import export

            export(self.store.db_path, self.map_path, self.export_dir)
            return True
        except Exception as exc:
            print(f"Export failed: {exc}")
            return False

    def _save_and_exit(self, _=None):
        """Finish the session: save time, refresh the export, close down."""
        self._flush_time()
        exported = self._export_now()
        n_done = sum(
            1 for r in self.store.worklist(self.clinician_id)
            if r["status"] == "reviewed"
        )
        note = (
            "Your work has been saved and the results file has been updated."
            if exported
            else "Your work has been saved."
        )
        self.layout[:] = [
            pn.Column(
                pn.pane.Markdown("## Session finished"),
                pn.pane.Markdown(
                    f"Thank you, {self.clinician_name}. {note}\n\n"
                    f"You have completed **{n_done}** night(s) in total.\n\n"
                    "**You can close this window now.**"
                ),
                pn.pane.HTML(
                    "<script>setTimeout(function(){ try { window.close(); } "
                    "catch (e) {} }, 1500);</script>",
                    height=0, margin=0,
                ),
                width=460,
            )
        ]
        self._shutdown_soon()

    def _shutdown_soon(self, delay: float = 5.0):
        """Stop the server once the goodbye message has had time to render."""
        import os
        import threading

        def stop():
            try:
                pn.state.kill_all_servers()
            except Exception:
                pass
            os._exit(0)

        threading.Timer(delay, stop).start()

    def _reopen(self, _):
        self.store.set_status(self.clinician_id, self.label, "in_progress")
        self._update_status_controls()
        self.refresh_worklist()


def create_app():
    store = Store(config.db_path)
    app = ReviewApp(
        store, config.blinded_dir, config.cache_dir,
        clock_mode=config.clock_mode,
        export_dir=config.export_dir,
        map_path=config.map_path,
        samples_dir=config.samples_dir,
    )
    template = pn.template.FastListTemplate(
        title="Overnight seizure review",
        main=[app.layout],
        theme_toggle=False,
        header_background="#16181d",
        accent_base_color="#3d5a80",
    )
    return template