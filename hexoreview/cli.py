"""Command line for the study coordinator.

  hexoreview scan     assign blind labels to new EDF files
                      (also builds the training-sample set when
                       --samples-dir and --samples-csv are given)
  hexoreview reviewer add / list
  hexoreview precache convert recordings ahead of time so opening is instant
  hexoreview run      start the dashboard
  hexoreview export   write annotations.csv and coverage.csv
  hexoreview status   who has reviewed what
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .blinding import load_map, scan
from .store import Store


def cmd_scan(args):
    # keep the training-sample recordings out of the scored set, even if the
    # samples folder happens to sit inside source_dir
    samples_src = Path(args.samples_dir).resolve() if args.samples_dir else None
    exclude = [samples_src] if samples_src else None

    added = scan(
        Path(args.source_dir), Path(args.blinded_dir), Path(args.map),
        seed=args.seed, copy=args.copy, exclude_dirs=exclude,
    )
    total = len(load_map(Path(args.map)))
    print(f"\n{len(added)} new recording(s) added; {total} in the set.")
    print(f"Blinded files: {args.blinded_dir}")
    print(f"Private map:   {args.map}  <- keep this away from reviewers")

    # Training samples: full recordings in --samples-dir that contain seizures
    # are linked in under neutral labels, with their marks recorded.
    if args.samples_dir and args.samples_csv:
        from .samples import build_samples

        print(f"\nBuilding training samples from {args.samples_dir} …")
        build_samples(
            Path(args.samples_dir),
            Path(args.samples_out),
            Path(args.samples_csv),
            copy=args.copy,
        )
        print(f"Training samples: {args.samples_out}")
    elif args.samples_dir or args.samples_csv:
        print(
            "\nNote: training samples need BOTH --samples-dir and --samples-csv; "
            "skipping that step."
        )


def cmd_reviewer(args):
    store = Store(Path(args.db))
    if args.action == "add":
        role = "coordinator" if args.coordinator else "reviewer"
        store.add_clinician(args.id, args.name or args.id, args.passcode or "", role)
        print(f"{role.capitalize()} '{args.id}' registered.")
    else:
        rows = store.clinicians()
        if not rows:
            print("No reviewers registered yet.")
        for r in rows:
            lock = "passcode set" if r["passcode"] else "no passcode"
            role = r["role"] if "role" in r.keys() else "reviewer"
            print(
                f"  {r['clinician_id']:<12} {r['display_name']:<24} "
                f"{role:<12} ({lock})"
            )


def cmd_precache(args):
    from .data import Recording

    blinded = Path(args.blinded_dir)
    files = sorted(blinded.glob("*.edf"))
    for i, edf in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {edf.name} …", flush=True)
        rec = Recording(edf, Path(args.cache))
        print(f"    {rec.duration_s / 3600:.2f} h at {rec.sfreq:g} Hz")
    print("Cache ready.")


def _port_in_use(port: int) -> bool:
    """True if something is already listening on localhost:port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.6)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def cmd_run(args):
    import webbrowser

    url = f"http://localhost:{args.port}/dashboard"

    # Closing the browser tab leaves the server running. Rather than failing
    # with "port already in use", point the reviewer back at the session that
    # is already there.
    if _port_in_use(args.port):
        print(f"The dashboard is already running. Opening {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    env = dict(os.environ)
    env["HEXOREVIEW_BLINDED_DIR"] = str(Path(args.blinded_dir).resolve())
    env["HEXOREVIEW_DB"] = str(Path(args.db).resolve())
    env["HEXOREVIEW_CACHE"] = str(Path(args.cache).resolve())
    env["HEXOREVIEW_CLOCK"] = "1" if args.clock else "0"
    env["HEXOREVIEW_MAP"] = str(Path(args.map).resolve())
    env["HEXOREVIEW_EXPORT"] = str(Path(args.out_dir).resolve())
    env["HEXOREVIEW_SAMPLES"] = str(Path(args.samples_out).resolve())
    # panel serve runs the target as a plain script, so hand it a launcher that
    # imports the installed package rather than the package file itself
    launcher = Path(tempfile.mkdtemp(prefix="hexoreview-")) / "dashboard.py"
    pkg_parent = Path(__file__).resolve().parent.parent
    launcher.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(pkg_parent)!r})\n"
        "from hexoreview.app import create_app\n"
        "create_app().servable()\n"
    )
    cmd = [sys.executable, "-m", "panel", "serve", str(launcher),
           "--port", str(args.port)]
    if not args.no_browser:
        cmd.append("--show")
    if args.allow_remote:
        cmd += ["--allow-websocket-origin", f"*:{args.port}", "--address", "0.0.0.0"]
    print(f"Starting dashboard on {url}")
    print("Close this window when you are finished reviewing.")
    subprocess.run(cmd, env=env)


def cmd_export(args):
    from .export import export

    paths = export(
        Path(args.db), Path(args.map), Path(args.out or args.export_dir)
    )
    for name, path in paths.items():
        print(f"{name:<12} -> {path}")


def cmd_status(args):
    store = Store(Path(args.db))
    store.register_dir(Path(args.blinded_dir))
    ids = [c["clinician_id"] for c in store.reviewers()]
    for cid in ids:
        rows = store.worklist(cid)
        done = [r for r in rows if r["status"] == "reviewed"]
        with_marks = [r for r in done if r["n_marks"]]
        mins = sum(store.active_seconds(cid, r["blind_label"]) for r in rows) / 60
        print(
            f"{cid:<12} {len(done)}/{len(rows)} nights finished, "
            f"{len(with_marks)} with seizure marks, "
            f"{sum(r['n_marks'] for r in rows)} marks total, "
            f"{mins:.0f} min reviewing"
        )


def main(argv=None):
    p = argparse.ArgumentParser(prog="hexoreview", description=__doc__)
    p.add_argument("--db", default="review/reviews.sqlite")
    p.add_argument("--blinded-dir", default="blinded")
    p.add_argument("--cache", default="review/cache")
    p.add_argument("--map", default="private/blinding_map.csv")
    p.add_argument("--export-dir", default="export")
    p.add_argument(
        "--samples-out", default="review/samples",
        help="where the training-sample set and its manifest are written / read",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="assign blind labels to new EDF files")
    s.add_argument("source_dir")
    s.add_argument("--seed", type=int, default=20260720)
    s.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    s.add_argument(
        "--samples-dir",
        help="folder of full recordings for training samples "
             "(keep it OUTSIDE source_dir)",
    )
    s.add_argument(
        "--samples-csv",
        help="seizure CSV; with --samples-dir, builds the training-sample set",
    )
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("reviewer", help="register or list reviewers")
    s.add_argument("action", choices=["add", "list"])
    s.add_argument("--id")
    s.add_argument("--name")
    s.add_argument("--passcode")
    s.add_argument(
        "--coordinator", action="store_true",
        help="register a coordinator, who can export but cannot review",
    )
    s.set_defaults(func=cmd_reviewer)

    s = sub.add_parser("precache", help="pre-convert recordings")
    s.set_defaults(func=cmd_precache)

    s = sub.add_parser("run", help="start the dashboard")
    s.add_argument("--port", type=int, default=5006)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--allow-remote", action="store_true")
    s.add_argument(
        "--clock", action="store_true",
        help="show time of day on screen instead of elapsed time",
    )
    s.add_argument(
        "--out-dir", default="export",
        help="where the dashboard writes annotations.csv and coverage.csv",
    )
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("export", help="write analysis CSVs")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("status", help="progress per reviewer")
    s.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()