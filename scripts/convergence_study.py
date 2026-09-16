"""Mesh convergence study: how much does thinning move as the blank is refined?

The dataset runs 4-5 mm blank elements against 5-14 mm die radii, giving a median
of 2.5 elements across the die radius where sheet-forming practice expects 5-8.
That is the single biggest open question about the simulation side, and it is
unanswerable by argument -- it needs the same case run at several densities.

For each case, the blank element size is halved and quartered while every other
parameter is held fixed. What matters is not the absolute thinning but whether it
is still moving: if refining from 4 mm to 2 mm changes peak thinning by a few
percent and 2 mm to 1 mm changes it by less, the coarse mesh is defensible for
the quantity we actually store.

Tool element size is refined alongside the blank, because rigid-tool faceting is
part of the same discretisation error -- a 6 mm tool facet on a 6 mm die radius
imprints on the sheet regardless of how fine the blank is.

    python .\\scripts\\convergence_study.py --out ..\\convergence
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "geometry"))
sys.path.insert(0, str(ROOT / "meshing"))
sys.path.insert(0, str(ROOT / "postprocessing"))

import numpy as np  # noqa: E402
from box_draw import DrawParams  # noqa: E402
import build_forming_deck as D  # noqa: E402
import extract_case as X  # noqa: E402
from run_sweep import find_solver  # noqa: E402

MATERIAL = ROOT / "materials" / "assumed_forming_steel.json"

# Three box cases spanning the die-radius-to-element ratio, all from the
# reference-verified family so the geometry itself is not in question.
CASES = [
    dict(name="h25_r20_d0", height=25.0, draft=0.0, corner_radius=20.0),
    dict(name="h35_r35_d4", height=35.0, draft=4.0, corner_radius=35.0),
    dict(name="h45_r50_d8", height=45.0, draft=8.0, corner_radius=50.0),
]
# Blank element size, mm. 4.0 is the production value.
LEVELS = [4.0, 2.0, 1.0]
BHF = 400000.0


def run_one(p: DrawParams, work: Path, solver: Path, material: dict,
            ncpu: int, memory: str) -> dict:
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    D.build_case(p, work, material, peak_velocity=3000.0)

    t0 = time.time()
    proc = subprocess.run(
        [str(solver), "i=draw.k", f"memory={memory}", f"ncpu={ncpu}"],
        cwd=work, capture_output=True, text=True, timeout=7200)
    wall = time.time() - t0
    log = (work / "run.log")
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")

    dyn = work / "dynain"
    if not dyn.exists():
        return dict(ok=False, wall_s=wall, note="no dynain -- run did not finish")

    d = X.parse_dynain(dyn)
    t = d["thickness"]
    out = dict(ok=True, wall_s=wall,
               elements=int(len(t)),
               min_thickness=float(t.min()),
               max_thinning=float(1.0 - t.min()),
               mean_thickness=float(t.mean()))
    # free up the bulky binaries immediately; the study only needs dynain
    for f in list(work.glob("d3plot*")) + list(work.glob("d3dump*")):
        try:
            f.unlink()
        except OSError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT.parent / "convergence")
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--memory", default="900m")
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()

    solver = find_solver(args.solver)
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    for case in CASES:
        name = case.pop("name")
        results[name] = {}
        for h in LEVELS:
            p = DrawParams(profile_radius=5.0, clearance_ratio=1.1,
                           blankholder_force=BHF,
                           blank_element_size=h,
                           # refine the tools with the blank: rigid-tool faceting
                           # is part of the same discretisation error
                           tool_element_size=max(1.5 * h, 1.5),
                           **case)
            work = args.out / f"{name}_h{h:g}"
            print(f"[{name}] blank {h:g} mm ...", flush=True)
            try:
                r = run_one(p, work, solver, material, args.ncpu, args.memory)
            except Exception as exc:  # noqa: BLE001
                r = dict(ok=False, note=f"{type(exc).__name__}: {exc}")
            results[name][h] = r
            if r.get("ok"):
                print(f"    {r['elements']:6d} elements   thinning "
                      f"{100 * r['max_thinning']:5.2f}%   {r['wall_s']:.0f}s", flush=True)
            else:
                print(f"    FAILED: {r.get('note')}", flush=True)
            (args.out / "convergence.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
        case["name"] = name

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'case':<14}{'h [mm]':>8}{'elements':>10}{'thinning':>11}"
          f"{'d vs prev':>11}{'wall':>8}")
    print("=" * 72)
    for name, byh in results.items():
        prev = None
        for h in LEVELS:
            r = byh.get(h, {})
            if not r.get("ok"):
                print(f"{name:<14}{h:>8g}{'--':>10}{'FAILED':>11}")
                continue
            thin = 100 * r["max_thinning"]
            delta = "" if prev is None else f"{thin - prev:+.2f} pp"
            print(f"{name:<14}{h:>8g}{r['elements']:>10d}{thin:>10.2f}%"
                  f"{delta:>11}{r['wall_s']:>7.0f}s")
            prev = thin
        print("-" * 72)

    # headline: largest change on the finest refinement step
    finals = []
    for name, byh in results.items():
        a, b = byh.get(LEVELS[-2], {}), byh.get(LEVELS[-1], {})
        if a.get("ok") and b.get("ok"):
            finals.append(abs(100 * (b["max_thinning"] - a["max_thinning"])))
    if finals:
        print(f"\nLargest thinning change on the final refinement "
              f"({LEVELS[-2]:g} -> {LEVELS[-1]:g} mm): {max(finals):.2f} "
              f"percentage points")
    print(f"\nwrote {args.out / 'convergence.json'}")


if __name__ == "__main__":
    main()
