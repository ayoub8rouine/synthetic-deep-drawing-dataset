"""Sample, build and sweep deep-drawn cups across several plan-outline families.

Reuses the deck builder, acceptance checks, dynain reader and dataset writer
unchanged; only the geometry and the sampling are specific to this family set.

Parameter ranges follow published single-draw practice rather than being
invented, so sampled cases land inside the process window instead of tearing:

    drawing ratio      DR = Db/Dp <= 2.0            (limiting draw ratio 2.2)
    reduction          (Db-Dp)/Db <= 0.5
    cup height         H <= base diameter           (H = b/2 (DR^2 - 1) at LDR)
    radial clearance   ~1.3 t is the reported optimum; below t it irons
    holding pressure   0.015 x yield = 3.15 MPa, and 2.76 MPa measured
    die profile radius 5-35 t;  punch profile radius 3-33 t
    wall draft         >= 6 deg, measured here: every U-channel split had a
                       near-vertical wall at one end

    python .\\shapes\\pipeline.py --sample
    python .\\shapes\\pipeline.py --case 0
    python .\\shapes\\pipeline.py --run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("geometry", "meshing", "postprocessing", "scripts"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(HERE))

from build_forming_deck import build_deck  # noqa: E402
from extract_case import write_case  # noqa: E402
from run_sweep import LicenseBusy, check_license, find_solver, run_solver  # noqa: E402
import outlines as OL  # noqa: E402
from cup_draw import (  # noqa: E402
    CupParams, build_all, build_punch, write_pointcloud, write_stl,
)

DATASET = HERE / "dataset"
RUNS = HERE / "simulations"
MATERIAL = ROOT / "materials" / "assumed_forming_steel.json"

# 0.015 x yield (210 MPa) is 3.15 MPa; 2.76 MPa is the measured figure for mild
# steel. Bracketing that band gives a meaningful blankholder sweep.
# 2.0 MPa was measured to be marginal: over 157 runs it held every family at
# >=85% except the trapezoid, which fell to 7/13, and 8 of 14 total rejections
# were binder-tracking failures (the flange slipping between die and binder).
# Both published values sit above it -- 0.015 x yield = 3.15 MPa, and 2.76 MPa
# measured -- so the lower bound is raised to 2.8 MPa rather than kept at 2.0.
PRESSURES = [2.8, 3.5]

DECK_KW = dict(target_dt=2.0e-7, peak_velocity=2000.0)

# Outline families and the ranges each is sampled over. `depth_factor` scales the
# draw depth against the part's smaller half-extent, capped by H <= base diameter.
CATALOGUE = [
    {"family": "circle", "radius": (55.0, 90.0)},
    {"family": "ellipse", "semi_major": (75.0, 105.0), "semi_minor": (45.0, 70.0)},
    {"family": "rect", "half_length": (65.0, 100.0), "half_width": (45.0, 75.0),
     "corner_radius": (15.0, 30.0)},
    {"family": "polygon", "n_sides": (3, 8), "circumradius": (60.0, 95.0),
     "corner_radius": (10.0, 22.0)},
    # n_lobes = 2 is the cocoon / peanut; 3-6 give clover and rose forms
    {"family": "lobed", "n_lobes": (2, 6), "mean_radius": (60.0, 90.0),
     "amplitude": (6.0, 16.0)},
    # the inner junctions of a cross are concave, so inner_radius is what
    # decides whether the die offset survives; kept well above the ~7 mm floor
    {"family": "cross", "arm_half_length": (70.0, 100.0),
     "arm_half_width": (28.0, 48.0), "corner_radius": (10.0, 20.0),
     "inner_radius": (12.0, 24.0)},
    {"family": "trapezoid", "half_length": (70.0, 100.0),
     "half_width_a": (55.0, 80.0), "half_width_b": (35.0, 55.0),
     "corner_radius": (14.0, 24.0)},
]

PROCESS = {
    "depth_factor": (0.25, 0.75),    # x the smaller base half-extent
    "draft_deg": (6.0, 20.0),
    "punch_radius": (4.0, 10.0),     # x thickness
    "die_radius": (6.0, 13.0),       # x thickness
    "draw_ratio": (1.40, 1.90),
    "clearance_ratio": (1.10, 1.35),
}


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def maximin_lhs(n: int, dim: int, seed: int = 0, tries: int = 4000) -> np.ndarray:
    """Latin hypercube improved by swapping to maximise the closest pair."""
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    x = np.column_stack([rng.permutation(rng.uniform(cut[:n], cut[1:]))
                         for _ in range(dim)])

    def closest(a):
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min()

    best = closest(x)
    for _ in range(tries):
        j = rng.integers(dim)
        i1, i2 = rng.choice(n, 2, replace=False)
        x[[i1, i2], j] = x[[i2, i1], j]
        d = closest(x)
        if d > best:
            best = d
        else:
            x[[i1, i2], j] = x[[i2, i1], j]
    return x


def _lerp(rng_pair, u: float):
    lo, hi = rng_pair
    v = lo + u * (hi - lo)
    return int(round(v)) if isinstance(lo, int) else round(float(v), 1)


# ---------------------------------------------------------------------------
# mesh sizing
# ---------------------------------------------------------------------------

# Elements wanted around the 90-degree arc of the tightest fillet. Sheet-forming
# practice asks for 5-8; below about 5 the shoulder stops being a bend and
# becomes a chamfer, and the thinning peak it produces is averaged away. The
# first 635 cups ran at a flat 5.0 mm against 6-13 mm die radii, which gave a
# median of 2.8 -- measured too coarse, so the size is now derived per case.
ELEMENTS_PER_FILLET = 5.0
BLANK_SIZE_LIMITS = (1.5, 2.5)   # mm; the lower bound caps cost, the upper
                                 # bound keeps large gentle parts honest

# Rigid tools are the contact surface, so their faceting is part of the same
# discretisation error: a 6 mm facet on an 8 mm radius stands 0.56 mm off the
# true arc, more than half the sheet thickness. Kept just coarser than the
# blank, which holds the chord error under 10% of thickness.
TOOL_SIZE_RATIO = 1.2


def mesh_sizes(punch_radius: float, die_radius: float) -> tuple[float, float]:
    """Blank and tool element size for a case, set by its tightest fillet."""
    r = min(punch_radius, die_radius)
    lo, hi = BLANK_SIZE_LIMITS
    blank = min(max((math.pi / 2 * r) / ELEMENTS_PER_FILLET, lo), hi)
    return round(blank, 2), round(TOOL_SIZE_RATIO * blank, 2)


def sample_cases(n: int = 30, seed: int = 0, thickness: float = 1.0) -> list:
    """`n` well-separated cups spread across the catalogue families."""
    per_family = int(np.ceil(n / len(CATALOGUE)))
    cases = []
    for fi, entry in enumerate(CATALOGUE):
        keys = [k for k in entry if k != "family"]
        unit = maximin_lhs(per_family * 2, len(keys) + len(PROCESS), seed + fi)
        made = 0
        for row in unit:
            if made >= per_family:
                break
            spec = {"family": entry["family"]}
            spec.update({k: _lerp(entry[k], u) for k, u in zip(keys, row)})
            proc = {k: _lerp(v, u)
                    for (k, v), u in zip(PROCESS.items(), row[len(keys):])}
            try:
                # size the depth from the outline first: CupParams validates the
                # depth against the fillets, so it cannot be constructed with a
                # placeholder height just to measure the footprint
                o = OL.build(spec, 6.0)
                pts = o.points
                base = min(float(np.abs(pts[:, 0]).max()),
                           float(np.abs(pts[:, 1]).max()))
                height = round(proc["depth_factor"] * base, 1)
                rp = proc["punch_radius"] * thickness
                rd = proc["die_radius"] * thickness
                h_blank, h_tool = mesh_sizes(rp, rd)
                cases.append(CupParams(
                    outline=spec, height=height,
                    blank_element_size=h_blank,
                    tool_element_size=h_tool,
                    punch_radius=rp,
                    die_radius=rd,
                    draft_deg=proc["draft_deg"],
                    draw_ratio=proc["draw_ratio"],
                    clearance_ratio=proc["clearance_ratio"],
                    thickness=thickness))
                made += 1
            except ValueError:
                continue          # infeasible geometry or outside the window
    return cases[:n]


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def build_case(p: CupParams, out_dir: Path, material: dict, **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = build_all(p)
    deck, report = build_deck(p, meshes, material, **kwargs)
    (out_dir / "draw.k").write_text(deck, encoding="utf-8")
    (out_dir / "parameters.json").write_text(p.to_json(), encoding="utf-8")

    geom = out_dir / "geometry"
    geom.mkdir(exist_ok=True)
    exported = dict(meshes)
    if p.punch_extension_ratio > 0:
        exported["punch"] = build_punch(replace(p, punch_extension_ratio=0.0))
    for name, mesh in exported.items():
        write_stl(mesh, geom / f"{name}.stl")
        write_pointcloud(mesh, geom / f"{name}.npz")
        report.setdefault("mesh_quality", {})[name] = {
            "free_edges": mesh.free_edges(),
            "non_manifold_edges": mesh.non_manifold_edges(),
            "min_edge_mm": mesh.min_edge_length(),
            "max_aspect_ratio": float(mesh.aspect_ratio().max()),
        }
    report["shape_family"] = p.outline["family"]
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def case_done(p: CupParams) -> bool:
    bhf = int(round(p.blankholder_force))
    return (DATASET / p.case_name() / "Data" / "Bauteil"
            / f"Tiefgezogenes_Bauteil_{bhf}.npz").exists()


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--memory", default="600m")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-d3plot", action="store_true")
    # a short session buys more distinct geometries by running one
    # blankholder setting per case instead of the full pair
    ap.add_argument("--pressures", default=None,
                    help="comma-separated MPa, default %s" % (PRESSURES,))
    args = ap.parse_args()

    cases = sample_cases(args.n, args.seed)

    if args.sample:
        print(f"{len(cases)} cups\n")
        print(f"{'#':>3} {'outline':<26}{'sym':<8}{'h':>6}{'draft':>7}{'DR':>6}"
              f"{'rp':>5}{'rd':>5}{'H/Dbase':>9}{'nodes':>7}")
        for i, p in enumerate(cases):
            o = p.build_outline()
            n = sum(len(m.vertices) for m in build_all(p).values())
            print(f"{i:>3} {o.name[:25]:<26}{o.symmetry:<8}{p.height:>6.1f}"
                  f"{p.draft_deg:>7.1f}{p.draw_ratio:>6.2f}{p.punch_radius:>5.1f}"
                  f"{p.die_radius:>5.1f}"
                  f"{p.height / (2 * min(p.base_half_extents())):>9.2f}{n:>7d}")
        return

    material = json.loads(MATERIAL.read_text(encoding="utf-8"))

    if args.case is not None:
        p = cases[args.case]
        out = RUNS / p.case_name() / f"bhf_{int(p.blankholder_force)}"
        r = build_case(p, out, material, **DECK_KW)
        print(json.dumps({k: r[k] for k in
                          ("total_nodes", "total_elements", "estimated_cycles",
                           "blankholder_pressure_mpa", "stroke_mm")}, indent=2))
        print(f"\nwrote {out / 'draw.k'}")
        return

    if not args.run:
        ap.print_help()
        return

    solver = find_solver()
    pressures = ([float(x) for x in args.pressures.split(",")]
                 if args.pressures else PRESSURES)
    jobs = [(c, q) for c in cases for q in pressures]
    print(f"solver: {solver}\n{len(cases)} cups x {len(PRESSURES)} pressures "
          f"= {len(jobs)} runs\n")

    manifest, started, ran = [], time.time(), 0
    for case, pressure in jobs:
        if args.limit is not None and ran >= args.limit:
            break
        p = replace(case, blankholder_pressure=pressure)
        tag = f"{p.case_name()[:50]}  {pressure:g}MPa"
        if case_done(p):
            print(f"skip      {tag}")
            continue
        run_dir = RUNS / p.case_name() / f"bhf_{int(p.blankholder_force)}"
        entry = {"case": p.case_name(), "pressure_mpa": pressure,
                 "blankholder_force": p.blankholder_force,
                 "family": p.outline["family"], "run_dir": str(run_dir)}
        try:
            build_case(p, run_dir, material, **DECK_KW)
            entry["solver"] = run_solver(solver, run_dir, args.memory, args.ncpu,
                                         args.timeout)
            check_license(run_dir)
            if not (run_dir / "dynain").exists():
                raise RuntimeError("solver produced no dynain")
            summary = write_case(run_dir, DATASET)
            entry["checks"] = summary["checks"]
            entry["status"] = "accepted" if summary["checks"]["accepted"] else "rejected"
        except LicenseBusy as exc:
            print(f"ABORTED: {exc}")
            break
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for pattern in () if args.keep_d3plot else ("d3plot*", "d3dump*"):
                for f in run_dir.glob(pattern):
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass

        ran += 1
        manifest.append(entry)
        c = entry.get("checks")
        detail = entry.get("error") or (
            f"thin={c['max_thinning']:.1%} binder={c['binder_below_die_mm']:+.2f}mm "
            f"{entry['solver']['wall_seconds']:.0f}s" if c else "")
        print(f"{entry['status']:<9} {tag}  {detail}")
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "sweep_manifest.json").write_text(
            json.dumps({"family": "cups", "runs": manifest}, indent=2), encoding="utf-8")

    ok = sum(1 for e in manifest if e["status"] == "accepted")
    print(f"\n{ok}/{len(manifest)} accepted in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
