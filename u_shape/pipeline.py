"""Build, sample and sweep U-channel cases.

Reuses the box family's deck builder, acceptance checks, `dynain` reader and
dataset writer unchanged -- all of that is geometry-agnostic. Only the geometry
and the parameter sampling are specific to this shape.

    python .\\u_shape\\pipeline.py --sample          # show the 30 sampled shapes
    python .\\u_shape\\pipeline.py --case 0          # build one case, no solver
    python .\\u_shape\\pipeline.py --run             # generate the dataset
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from u_channel import (  # noqa: E402
    Section, UChannelParams, build_all, build_punch,
    keyword_blocks, write_pointcloud, write_stl,
)

DATASET = HERE / "dataset"
RUNS = HERE / "simulations"
MATERIAL = ROOT / "materials" / "assumed_forming_steel.json"

# Blankholder forces. The clamped flange here is about 25,000 mm2 for the half
# model, so these span roughly 1.5-4 MPa, the same window the box family needed
# to actually hold the flange.
FORCES = [150000.0, 300000.0]

# A hat section deforms mostly by bending and draw-in rather than the biaxial
# stretch a box corner sees, so its plastic work is about a quarter of the box
# family's. The same punch speed therefore reads as a much higher kinetic to
# internal energy ratio -- 6.9% at 3000 mm/s, over the 5% quasi-static limit.
# Kinetic energy scales with v^2, so 2000 mm/s brings it to about 3%, at the
# cost of a 50% longer run.
DECK_KW = dict(target_dt=2.0e-7, peak_velocity=2000.0)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

# Per-end section ranges. Both ends are sampled independently, so a case can be
# a straight prismatic channel or strongly tapered along its length.
BOUNDS = {
    "floor_half_width": (40.0, 85.0),
    "height": (20.0, 50.0),
    # Every one of the 11 splits in the first U-channel sweep had a near-vertical
    # wall at one end; none of the 49 accepted cases did. That is where the
    # draft >= 6 deg rule came from, and it was applied to every later family but
    # never back to this one. Raising the floor closes that inconsistency.
    "wall_angle_deg": (6.0, 25.0),
    "floor_radius": (5.0, 14.0),
    "flange_radius": (5.0, 14.0),
}
KEYS = list(BOUNDS)


def maximin_lhs(n: int, dim: int, seed: int = 0, tries: int = 4000) -> np.ndarray:
    """Latin hypercube sample, improved by swapping to push points apart.

    A plain LHS can still leave points close together. Maximising the smallest
    pairwise distance spreads the shapes out, which is what makes a small
    dataset informative: 30 well-separated geometries beat 30 clustered ones.
    """
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    x = np.empty((n, dim))
    for j in range(dim):
        x[:, j] = rng.permutation(rng.uniform(cut[:n], cut[1:]))

    def min_dist(a: np.ndarray) -> float:
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min()

    best = min_dist(x)
    for _ in range(tries):
        j = rng.integers(dim)
        i1, i2 = rng.choice(n, 2, replace=False)
        x[[i1, i2], j] = x[[i2, i1], j]
        d = min_dist(x)
        if d > best:
            best = d
        else:
            x[[i1, i2], j] = x[[i2, i1], j]  # revert
    return x


def _feasible_from(unit: np.ndarray) -> list:
    out = []
    for row in unit:
        sections = []
        for half in (row[:len(KEYS)], row[len(KEYS):]):
            vals = {k: BOUNDS[k][0] + v * (BOUNDS[k][1] - BOUNDS[k][0])
                    for k, v in zip(KEYS, half)}
            sections.append(Section(**{k: round(v, 1) for k, v in vals.items()}))
        try:
            out.append(UChannelParams(section_a=sections[0], section_b=sections[1]))
        except ValueError:
            continue  # e.g. fillets that do not fit in a short wall
    return out


def sample_geometries(n: int = 30, seed: int = 0) -> list:
    """`n` well-separated U-channels, both end sections sampled independently.

    Some samples are geometrically infeasible (fillets too large for the wall),
    so the request is grown until `n` survive rather than silently returning
    fewer.
    """
    dim = 2 * len(KEYS)
    request = n
    for _ in range(6):
        out = _feasible_from(maximin_lhs(request, dim, seed))
        if len(out) >= n:
            return out[:n]
        request = int(request * 1.2) + 2
    raise RuntimeError(f"could only sample {len(out)} feasible shapes, wanted {n}")


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def build_case(p: UChannelParams, out_dir: Path, material: dict, **kwargs) -> dict:
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
    report["shape_family"] = "uchannel"
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def case_done(p: UChannelParams) -> bool:
    bhf = int(round(p.blankholder_force))
    return (DATASET / p.case_name() / "Data" / "Bauteil"
            / f"Tiefgezogenes_Bauteil_{bhf}.npz").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="list the sampled shapes")
    ap.add_argument("--case", type=int, default=None, help="build one case, no solver")
    ap.add_argument("--run", action="store_true", help="run the sweep")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--memory", default="500m")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-d3plot", action="store_true",
                    help="keep the full time-history plot files (~150 MB per run); "
                         "the dataset only needs dynain, so they are deleted by default")
    args = ap.parse_args()

    geometries = sample_geometries(args.n, args.seed)

    if args.sample:
        print(f"{len(geometries)} shapes\n")
        print(f"{'#':>3}  {'end A: w  h  beta  rf  rc':<30}{'end B: w  h  beta  rf  rc':<30}"
              f"{'stroke':>8}{'nodes':>8}")
        for i, p in enumerate(geometries):
            a, b = p.section_a, p.section_b
            n = sum(len(m.vertices) for m in build_all(p).values())
            fmt = lambda s: (f"{s.floor_half_width:5.1f}{s.height:6.1f}"  # noqa: E731
                             f"{s.wall_angle_deg:6.1f}{s.floor_radius:5.1f}"
                             f"{s.flange_radius:5.1f}")
            print(f"{i:>3}  {fmt(a):<30}{fmt(b):<30}{p.height:8.1f}{n:8d}")
        return

    material = json.loads(MATERIAL.read_text(encoding="utf-8"))

    if args.case is not None:
        p = geometries[args.case]
        out = RUNS / p.case_name() / f"bhf_{int(p.blankholder_force)}"
        print(json.dumps(build_case(p, out, material, **DECK_KW), indent=2)[:900])
        print(f"\nwrote {out / 'draw.k'}")
        return

    if not args.run:
        ap.print_help()
        return

    solver = find_solver()
    print(f"solver: {solver}")
    jobs = [(p, f) for p in geometries for f in FORCES]
    print(f"{len(geometries)} shapes x {len(FORCES)} forces = {len(jobs)} runs\n")

    manifest, started, ran = [], time.time(), 0
    for geom, force in jobs:
        if args.limit is not None and ran >= args.limit:
            break
        p = replace(geom, blankholder_force=force)
        tag = f"{p.case_name()[:52]}  bhf={int(force)}"
        if case_done(p):
            print(f"skip      {tag}")
            continue
        run_dir = RUNS / p.case_name() / f"bhf_{int(force)}"
        entry = {"case": p.case_name(), "blankholder_force": force,
                 "run_dir": str(run_dir)}
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
            for pattern in () if args.keep_d3plot else ("d3plot*", "d3dump*", "d3full*"):
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
            json.dumps({"family": "uchannel", "runs": manifest}, indent=2),
            encoding="utf-8")

    ok = sum(1 for e in manifest if e["status"] == "accepted")
    print(f"\n{ok}/{len(manifest)} accepted in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
