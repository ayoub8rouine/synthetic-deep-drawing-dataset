"""Build a click-to-open showcase: one geometry per shape type, each pre-solved.

For every distinct shape in the dataset this rebuilds the deck from its stored
parameters, runs it to termination, and drops an OPEN.bat next to the results
that launches LS-PrePost on the d3plot. Double-clicking the .bat is all that is
needed to see the part animate.

    python .\\scripts\\build_showcase.py
    python .\\scripts\\build_showcase.py --only circle,cross     # rebuild a subset
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("geometry", "meshing", "shapes", "u_shape"):
    sys.path.insert(0, str(ROOT / sub))

from run_sweep import find_ls_prepost, find_solver  # noqa: E402

OUT = ROOT.parent / "Showcase_LSDYNA"
MATERIAL = json.loads((ROOT / "materials" / "assumed_forming_steel.json")
                      .read_text(encoding="utf-8"))

# label -> (dataset dir, folder-name regex). Order fixes the numbering.
SHAPES = [
    ("box_rectangular_pan", "dataset", r"^tool_"),
    ("uchannel_hat_section", "u_shape/dataset", r"^uchannel_"),
    ("circle_cylindrical", "shapes/dataset", r"^circle_"),
    ("ellipse_elliptical", "shapes/dataset", r"^ellipse_"),
    ("rect_rectangular", "shapes/dataset", r"^rect_"),
    ("trapezoid", "shapes/dataset", r"^trapezoid_"),
    ("cross_plus", "shapes/dataset", r"^cross_"),
    ("polygon3_triangle", "shapes/dataset", r"^polygon3_"),
    ("polygon4_square", "shapes/dataset", r"^polygon4_"),
    ("polygon5_pentagon", "shapes/dataset", r"^polygon5_"),
    ("polygon6_hexagon", "shapes/dataset", r"^polygon6_"),
    ("polygon7_heptagon", "shapes/dataset", r"^polygon7_"),
    ("polygon8_octagon", "shapes/dataset", r"^polygon8_"),
    ("lobed2_cocoon", "shapes/dataset", r"^lobed2_"),
    ("lobed3_clover", "shapes/dataset", r"^lobed3_"),
    ("lobed4_quatrefoil", "shapes/dataset", r"^lobed4_"),
    ("lobed5_clover5", "shapes/dataset", r"^lobed5_"),
    ("lobed6_rose", "shapes/dataset", r"^lobed6_"),
]

BAT = """@echo off
title {label}
cd /d "%~dp0run"
echo Opening {label} in LS-PrePost...
start "" "{lspp}" d3plot
"""


def _load_pipeline(name: str, path: Path):
    """Import a pipeline.py under a unique module name.

    Both u_shape/ and shapes/ define a module called "pipeline". A plain import
    binds whichever is seen first, so the second family silently gets the first
    family's build_case and fails on a missing attribute.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pick(dataset: str, pattern: str) -> tuple[str, dict] | None:
    """Deepest accepted case of this shape -- deepest draws show the most strain."""
    best = None
    for rep in glob.glob(str(ROOT / dataset / "*" / "case_report_*.json")):
        case = os.path.basename(os.path.dirname(rep))
        if not re.match(pattern, case):
            continue
        c = json.loads(Path(rep).read_text(encoding="utf-8"))
        if not c["checks"]["accepted"]:
            continue
        p = c["parameters"]
        depth = p.get("height") or max(p["section_a"]["height"], p["section_b"]["height"])
        if best is None or depth > best[0]:
            best = (depth, case, c["checks"])
    return (best[1], best[2]) if best else None


def build_deck(dataset: str, case: str, dest: Path) -> str:
    """Rebuild the deck from stored parameters. Returns a one-line description."""
    params = json.loads((ROOT / dataset / case / "parameters.json")
                        .read_text(encoding="utf-8"))
    params.pop("derived", None)

    if dataset == "dataset":                                   # box
        import build_forming_deck as B
        from box_draw import DrawParams
        p = DrawParams(**params)
        B.build_case(p, dest, MATERIAL, peak_velocity=3000.0)
        return f"{2*p.punch_half_length:.0f} x {2*p.punch_half_width:.0f} mm, {p.height:.0f} mm deep"

    if dataset.startswith("u_shape"):                          # u-channel
        UP = _load_pipeline("uchannel_pipeline", ROOT / "u_shape" / "pipeline.py")
        from u_channel import Section, UChannelParams
        params["section_a"] = Section(**params["section_a"])
        params["section_b"] = Section(**params["section_b"])
        p = UChannelParams(**params)
        UP.build_case(p, dest, MATERIAL, **UP.DECK_KW)
        return f"{p.length:.0f} mm long, {p.height:.0f} mm deep, both ends independent"

    CP = _load_pipeline("cup_pipeline", ROOT / "shapes" / "pipeline.py")   # cup
    from cup_draw import CupParams
    params.pop("blankholder_force", None)
    p = CupParams(**params)
    CP.build_case(p, dest, MATERIAL, **CP.DECK_KW)
    o = p.build_outline()
    return f"{o.symmetry} model, {p.height:.0f} mm deep, draw ratio {p.draw_ratio:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated label substrings")
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--solver", default=None, help="path to the LS-DYNA executable")
    ap.add_argument("--ls-prepost", default=None, help="path to LS-PrePost")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    solver = find_solver(args.solver)
    ls_prepost = find_ls_prepost(args.ls_prepost)
    wanted = args.only.split(",") if args.only else None
    args.out.mkdir(parents=True, exist_ok=True)
    index, t_start = [], time.time()

    for i, (label, dataset, pattern) in enumerate(SHAPES, start=1):
        if wanted and not any(w.strip() in label for w in wanted):
            continue
        found = pick(dataset, pattern)
        if not found:
            print(f"[{i:02d}] {label:24s} NO ACCEPTED CASE -- skipped")
            continue
        case, checks = found
        dest = args.out / f"{i:02d}_{label}"
        print(f"[{i:02d}] {label:24s} {case[:52]}", flush=True)

        try:
            desc = build_deck(dataset, case, dest)
        except Exception as exc:                               # noqa: BLE001
            print(f"     BUILD FAILED: {type(exc).__name__}: {exc}")
            continue

        run = dest / "run"
        run.mkdir(exist_ok=True)
        (run / "draw.k").write_bytes((dest / "draw.k").read_bytes())
        t0 = time.time()
        r = subprocess.run([str(solver), "i=draw.k", "memory=200m", f"ncpu={args.ncpu}"],
                           cwd=run, capture_output=True, text=True, timeout=3600)
        (run / "run.log").write_text(r.stdout + r.stderr, encoding="utf-8",
                                     errors="replace")
        ok = "N o r m a l" in (r.stdout + r.stderr)
        secs = time.time() - t0

        (dest / "OPEN.bat").write_text(
            BAT.format(label=label, lspp=ls_prepost), encoding="utf-8"
        )
        thin = 100 * checks["max_thinning"]
        print(f"     {'OK ' if ok else 'FAILED'} {secs:5.0f}s   thinning {thin:.1f}%   "
              f"{len(list(run.glob('d3plot*')))} d3plot files", flush=True)
        index.append((f"{i:02d}_{label}", case, desc, thin, ok))

    # ---- index -----------------------------------------------------------
    lines = ["# LS-DYNA showcase — one geometry per shape type", "",
             "Double-click **OPEN.bat** in any folder to view that part in LS-PrePost.", "",
             "Once it opens: press ▶ to animate the draw. To colour by strain:",
             "**Fringe Comp → Strain → Lower surface → 2nd principal strain → Apply**",
             "(2nd principal = *minor* strain, 1st principal = *major*.)", "",
             "To see the part without the tooling around it: **Model → Part**, untick "
             "punch / die / binder.", "",
             "| # | shape | geometry | max thinning |", "| --- | --- | --- | ---: |"]
    for folder, case, desc, thin, ok in index:
        lines.append(f"| {folder.split('_')[0]} | **{folder[3:]}** | {desc} | "
                     f"{thin:.1f}% |")
    lines += ["", f"Every deck is a self-contained `draw.k`. To re-run one by hand:", "",
              "```", 'cd run', f'"{solver}" i=draw.k memory=200m ncpu=4', "```", "",
              "`run/` folders hold the `d3plot` animation data and are large "
              "(~90–160 MB each). Delete them when finished; the decks stay valid.", ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{len(index)} geometries in {args.out}   "
          f"({(time.time()-t_start)/60:.0f} min)")


if __name__ == "__main__":
    main()
