"""Click-to-open animations for ten shape types, built from the refined tier.

The production sweep deletes d3plot after extracting the dynain, because the
dataset only needs the pre-springback state and the animation binaries are the
bulk of the disk cost. So a viewable animation has to be re-run rather than
recovered.

For each shape type this picks the deepest accepted case at the fillet-derived
mesh (1.5-2.5 mm), rebuilds its deck from the stored parameters, runs it keeping
d3plot, and drops an OPEN.bat that launches LS-PrePost on the result.

    python .\\scripts\\build_showcase_refined.py
    python .\\scripts\\build_showcase_refined.py --only circle,rect
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("geometry", "meshing", "shapes"):
    sys.path.insert(0, str(ROOT / sub))

import pipeline as P  # noqa: E402
from cup_draw import CupParams  # noqa: E402
from run_sweep import find_ls_prepost, find_solver  # noqa: E402

OUT = ROOT.parent / "Showcase_Refined"
MATERIAL = json.loads((ROOT / "materials" / "assumed_forming_steel.json")
                      .read_text(encoding="utf-8"))

# Ten types spanning every family, plus the polygon and lobed variants that
# behave differently: odd lobe counts are half-symmetry, the cross is the only
# outline with concave junctions.
WANTED = ["circle", "ellipse", "rect", "trapezoid", "cross",
          "polygon3", "polygon6", "polygon7", "lobed4", "lobed5"]

BAT = """@echo off
title {label}
cd /d "%~dp0run"
echo Opening {label} in LS-PrePost...
echo.
echo   press the play button to animate the draw
echo   colour by strain: Fringe Comp - Strain - Lower surface -
echo                     2nd principal strain - Apply
echo.
start "" "{lspp}" d3plot
"""


def refined_cases() -> dict:
    """Deepest accepted refined case per shape type."""
    best = {}
    for f in glob.glob(str(ROOT / "shapes" / "dataset" / "*" / "parameters.json")):
        p = json.loads(Path(f).read_text(encoding="utf-8"))
        if p.get("blank_element_size", 5.0) >= 4.0:
            continue                      # coarse tier
        d = Path(f).parent
        rep = sorted(d.glob("case_report_*.json"))
        if not rep:
            continue
        ch = json.loads(rep[0].read_text(encoding="utf-8"))["checks"]
        if not ch.get("accepted"):
            continue
        o = p["outline"]
        key = o["family"]
        if key == "polygon":
            key = f"polygon{o['n_sides']}"
        if key == "lobed":
            key = f"lobed{o['n_lobes']}"
        if key not in best or p["height"] > best[key][0]["height"]:
            best[key] = (p, d.name, ch)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--solver", default=None, help="path to the LS-DYNA executable")
    ap.add_argument("--ls-prepost", default=None, help="path to LS-PrePost")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    solver = find_solver(args.solver)
    ls_prepost = find_ls_prepost(args.ls_prepost)
    wanted = ([w.strip() for w in args.only.split(",")] if args.only
              else WANTED)
    found = refined_cases()
    args.out.mkdir(parents=True, exist_ok=True)
    index, t0 = [], time.time()

    for i, key in enumerate(wanted, start=1):
        if key not in found:
            print(f"[{i:02d}] {key:<12} no accepted refined case -- skipped",
                  flush=True)
            continue
        params, case, checks = found[key]
        params = dict(params)
        params.pop("derived", None)
        params.pop("blankholder_force", None)
        p = CupParams(**params)

        dest = args.out / f"{i:02d}_{key}"
        run = dest / "run"
        print(f"[{i:02d}] {key:<12} {case[:44]}", flush=True)
        print(f"     depth {p.height} mm, mesh {p.blank_element_size} mm, "
              f"thinning {100 * checks['max_thinning']:.1f}%", flush=True)

        try:
            P.build_case(p, dest, MATERIAL, **P.DECK_KW)
        except Exception as exc:                              # noqa: BLE001
            print(f"     BUILD FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        run.mkdir(parents=True, exist_ok=True)
        (run / "draw.k").write_bytes((dest / "draw.k").read_bytes())
        t = time.time()
        r = subprocess.run([str(solver), "i=draw.k", "memory=1200m",
                            f"ncpu={args.ncpu}"],
                           cwd=run, capture_output=True, text=True, timeout=7200)
        (run / "run.log").write_text(r.stdout + r.stderr, encoding="utf-8",
                                     errors="replace")
        ok = "N o r m a l" in (r.stdout + r.stderr)
        secs = time.time() - t
        n_d3 = len(list(run.glob("d3plot*")))
        mb = sum(f.stat().st_size for f in run.glob("d3plot*")) / 1e6

        # dump the memory-hungry restart files, keep the animation
        for f in run.glob("d3dump*"):
            try:
                f.unlink()
            except OSError:
                pass

        (dest / "OPEN.bat").write_text(BAT.format(label=key, lspp=ls_prepost),
                                       encoding="utf-8")
        print(f"     {'OK ' if ok else 'FAILED'} {secs:5.0f}s   "
              f"{n_d3} d3plot files, {mb:.0f} MB", flush=True)
        index.append((f"{i:02d}_{key}", p, checks, ok, mb))

    lines = ["# Refined-mesh showcase — ten shape types", "",
             "Every case here ran at the **fillet-derived mesh** (1.5–2.5 mm "
             "blank, 4–6 elements around the tightest fillet), not the 5 mm "
             "mesh of the original dataset.", "",
             "Double-click **OPEN.bat** in any folder to view that part animate "
             "in LS-PrePost.", "",
             "To colour by strain: **Fringe Comp → Strain → Lower surface → "
             "2nd principal strain → Apply** "
             "(2nd principal = *minor*, 1st principal = *major*).", "",
             "To hide the tooling: **Model → Part**, untick punch / die / binder.",
             "",
             "| # | type | depth | mesh | thinning | size |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for folder, p, ch, ok, mb in index:
        lines.append(f"| {folder.split('_')[0]} | **{folder[3:]}** | "
                     f"{p.height:.1f} mm | {p.blank_element_size} mm | "
                     f"{100 * ch['max_thinning']:.1f}% | {mb:.0f} MB |")
    lines += ["", "`run/` holds the animation data and is large. Delete those "
              "folders when finished; the decks stay valid and can be re-run.", ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{len(index)} animations in {args.out}   "
          f"({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
