"""Mesh convergence study for the cup family -- the mesh that actually shipped.

Every cup in the dataset ran at a 5.0 mm blank element against 6-13 mm die
radii, which puts a median of 2.8 elements around the die shoulder arc where
sheet-forming practice expects 5-8. The existing study in
``convergence_study.py`` cannot answer whether that is defensible: it was run on
the box family and its coarsest level was 4.0 mm, finer than production.

This runs three cups -- one per symmetry and concavity class -- at 5.0, 3.0,
2.0 and 1.5 mm and records two different kinds of quantity:

    shape    max radial draw-in of the blank edge
    field    peak thinning, peak major strain

Shape is expected to converge quickly; the fields are the ones in question. The
tool mesh is refined alongside the blank, because a coarse rigid facet on the
die shoulder imprints on the sheet no matter how fine the blank is.

    python .\\scripts\\cup_convergence.py --out ..\\convergence_cups
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("geometry", "meshing", "postprocessing", "shapes"):
    sys.path.insert(0, str(ROOT / sub))

import numpy as np  # noqa: E402

import extract_case as X  # noqa: E402
import pipeline as P  # noqa: E402
from cup_draw import CupParams  # noqa: E402
from run_sweep import find_solver  # noqa: E402

MATERIAL = ROOT / "materials" / "assumed_forming_steel.json"

# One case per class of difficulty, all inside the sampled production window.
#   circle  quarter symmetry, convex, the classical benchmark
#   rect    quarter symmetry, corners that concentrate strain
#   lobed5  half symmetry, concave junctions -- the tightest offset in the family
CASES = [
    ("circle", dict(
        outline={"family": "circle", "radius": 75.0},
        height=30.0, draft_deg=12.0, punch_radius=6.0, die_radius=8.0,
        draw_ratio=1.70, clearance_ratio=1.20)),
    ("rect", dict(
        outline={"family": "rect", "half_length": 85.0, "half_width": 60.0,
                 "corner_radius": 22.0},
        height=32.0, draft_deg=10.0, punch_radius=6.0, die_radius=9.0,
        draw_ratio=1.65, clearance_ratio=1.20)),
    ("lobed5", dict(
        outline={"family": "lobed", "n_lobes": 5, "mean_radius": 78.0,
                 "amplitude": 9.0},
        height=28.0, draft_deg=12.0, punch_radius=6.0, die_radius=8.0,
        draw_ratio=1.60, clearance_ratio=1.20)),
]

# Blank element size, mm. 5.0 is what every production cup ran at.
LEVELS = [5.0, 3.0, 2.0, 1.5]
PRESSURE = 3.5


def measure(work: Path) -> dict:
    """Shape and field quantities from the pre-springback state."""
    d = X.parse_dynain(work / "dynain")
    t = d["thickness"]
    out = {
        "elements": int(len(t)),
        "nodes": int(len(d["nodes"])),
        "min_thickness": float(t.min()),
        "max_thinning": float(1.0 - t.min()),
        "mean_thickness": float(t.mean()),
    }
    if "major_strain" in d:
        out["max_major_strain"] = float(d["major_strain"].max())
        out["max_plastic_strain"] = float(d["plastic_strain"].max())

    # draw-in: how far the outermost blank edge has pulled toward the axis.
    # Radial extent is mesh-independent in a way that a per-node value is not.
    if "nodes_initial" in d:
        r0 = np.linalg.norm(d["nodes_initial"][:, :2], axis=1)
        r1 = np.linalg.norm(d["nodes"][:, :2], axis=1)
        edge = r0 >= r0.max() - 1e-6
        out["draw_in_mm"] = float(r0[edge].mean() - r1[edge].mean())
        out["flange_extent_mm"] = float(r1.max())
    return out


def run_one(p: CupParams, work: Path, solver: Path, material: dict,
            ncpu: int, memory: str, timeout: float) -> dict:
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    report = P.build_case(p, work, material, **P.DECK_KW)
    t0 = time.time()
    proc = subprocess.run(
        [str(solver), "i=draw.k", f"memory={memory}", f"ncpu={ncpu}"],
        cwd=work, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    (work / "run.log").write_text(proc.stdout + proc.stderr,
                                  encoding="utf-8", errors="replace")

    if not (work / "dynain").exists():
        return dict(ok=False, wall_s=wall, note="no dynain -- run did not finish",
                    total_nodes=report.get("total_nodes"))

    out = dict(ok=True, wall_s=wall,
               total_nodes=report.get("total_nodes"),
               total_elements=report.get("total_elements"))
    out.update(measure(work))
    for f in list(work.glob("d3plot*")) + list(work.glob("d3dump*")):
        try:
            f.unlink()
        except OSError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT.parent / "convergence_cups")
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--memory", default="1200m")
    ap.add_argument("--timeout", type=float, default=10800.0)
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()

    solver = find_solver(args.solver)
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, kw in CASES:
        results[name] = {}
        for h in LEVELS:
            p = CupParams(blank_element_size=h,
                          # rigid-tool faceting is part of the same error
                          tool_element_size=max(1.2 * h, 1.5),
                          blankholder_pressure=PRESSURE, **kw)
            arc = (math.pi / 2 * p.die_radius) / h
            work = args.out / f"{name}_h{h:g}"
            print(f"[{name}] blank {h:g} mm  ({arc:.1f} elem on die arc) ...",
                  flush=True)
            try:
                r = run_one(p, work, solver, material, args.ncpu, args.memory,
                            args.timeout)
            except Exception as exc:  # noqa: BLE001
                r = dict(ok=False, note=f"{type(exc).__name__}: {exc}")
            r["die_arc_elements"] = round(arc, 2)
            results[name][h] = r
            if r.get("ok"):
                print(f"    {r['total_nodes']:6d} nodes  {r['elements']:6d} blank el"
                      f"   thinning {100 * r['max_thinning']:5.2f}%"
                      f"   draw-in {r.get('draw_in_mm', float('nan')):5.2f} mm"
                      f"   {r['wall_s']:.0f}s", flush=True)
            else:
                print(f"    FAILED: {r.get('note')}", flush=True)
            (args.out / "convergence_cups.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'case':<10}{'h':>6}{'arc':>6}{'nodes':>8}{'thinning':>10}{'d%':>8}"
          f"{'major eps':>11}{'draw-in':>9}{'d mm':>7}{'wall':>8}")
    print("=" * 92)
    for name, byh in results.items():
        prev_t = prev_d = None
        for h in LEVELS:
            r = byh.get(h, {})
            if not r.get("ok"):
                print(f"{name:<10}{h:>6.1f}   failed")
                continue
            thin = 100 * r["max_thinning"]
            di = r.get("draw_in_mm", float("nan"))
            dt = "" if prev_t is None else f"{thin - prev_t:+.2f}"
            dd = "" if prev_d is None else f"{di - prev_d:+.2f}"
            print(f"{name:<10}{h:>6.1f}{r['die_arc_elements']:>6.1f}"
                  f"{r['total_nodes']:>8d}{thin:>9.2f}%{dt:>8}"
                  f"{r.get('max_major_strain', float('nan')):>11.4f}"
                  f"{di:>9.2f}{dd:>7}{r['wall_s']:>7.0f}s")
            prev_t, prev_d = thin, di
        print("-" * 92)
    print(f"\nwrote {args.out / 'convergence_cups.json'}")


if __name__ == "__main__":
    main()
