"""Compare generated drawn parts with the reference dataset's drawn parts.

`geometry/verify_reference.py` checks that the *tools* match. This checks the
other half: that simulating those tools produces a part shaped like the one the
reference dataset ships, for the geometries present in both.

Reported as a symmetric Chamfer distance between the two point clouds, plus the
bounding-box differences that are easier to reason about. For each of our
samples the closest reference blankholder force is used, since the two datasets
do not share a force scale (see the README).

    python .\\postprocessing\\compare_to_reference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT.parent / "Synthetic Deep Drawing Dataset" / "Dataset"


def chamfer(a: np.ndarray, b: np.ndarray) -> tuple:
    """Symmetric mean and 95th-percentile nearest-neighbour distance."""
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    both = np.r_[da, db]
    return float(both.mean()), float(np.percentile(both, 95))


def reference_parts(reference: Path, case: str) -> dict:
    d = reference / case / "Data" / "Bauteil"
    if not d.is_dir():
        return {}
    out = {}
    for f in d.glob("*.npz"):
        digits = "".join(ch for ch in f.stem if ch.isdigit())
        if digits:
            out[int(digits)] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="reference Dataset directory (default: %(default)s)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.dataset.is_dir():
        ap.error(f"generated dataset not found at '{args.dataset}'")
    if not args.reference.is_dir():
        ap.error(
            f"reference dataset not found at '{args.reference}'. "
            "Pass --reference with the path to its Dataset directory."
        )

    rows = []
    for case in sorted(c for c in args.dataset.iterdir() if c.is_dir()):
        refs = reference_parts(args.reference, case.name)
        if not refs:
            continue
        # the reference's highest force is its best-clamped, most comparable case
        ref_force = max(refs)
        ref = np.load(refs[ref_force])["points"]
        for f in sorted((case / "Data" / "Bauteil").glob("*.npz")):
            z = np.load(f)
            mine = z["points"]
            mean, p95 = chamfer(mine, ref)
            # Two independent random samples of the SAME surface are already a
            # finite distance apart, set by point spacing. Resampling our own
            # part gives that floor, so the comparison above can be read against
            # something rather than in absolute terms.
            floor = np.nan
            if "nodes" in z and "quads" in z:
                import sys
                sys.path.insert(0, str(ROOT / "geometry"))
                from box_draw import SurfaceMesh
                m = SurfaceMesh(z["nodes"], z["quads"])
                floor = chamfer(m.sample(len(mine), seed=1)[0],
                                m.sample(len(mine), seed=2)[0])[0]
            force = int("".join(ch for ch in f.stem if ch.isdigit()))
            rows.append({
                "case": case.name,
                "our_force_N": force,
                "reference_force_N": ref_force,
                "chamfer_mean_mm": mean,
                "chamfer_p95_mm": p95,
                "sampling_floor_mm": float(floor),
                "depth_ours_mm": float(mine[:, 2].max() - mine[:, 2].min()),
                "depth_reference_mm": float(ref[:, 2].max() - ref[:, 2].min()),
                "flange_x_ours_mm": float(mine[:, 0].max()),
                "flange_x_reference_mm": float(ref[:, 0].max()),
            })

    if not rows:
        print("no cases overlap with the reference dataset")
        raise SystemExit(0)

    print(f"{'case':<48}{'ours':>8}{'ref':>8}{'chamfer':>10}{'depth o/r':>14}"
          f"{'flangeX o/r':>16}")
    for r in rows:
        print(f"{r['case'][:47]:<48}{r['our_force_N']:>8}{r['reference_force_N']:>8}"
              f"{r['chamfer_mean_mm']:>9.2f}m"
              f"{r['depth_ours_mm']:>7.1f}/{r['depth_reference_mm']:<6.1f}"
              f"{r['flange_x_ours_mm']:>9.1f}/{r['flange_x_reference_mm']:<6.1f}")

    fl = np.array([r["sampling_floor_mm"] for r in rows])
    ch = np.array([r["chamfer_mean_mm"] for r in rows])
    dd = np.array([r["depth_ours_mm"] - r["depth_reference_mm"] for r in rows])
    fd = np.array([r["flange_x_ours_mm"] - r["flange_x_reference_mm"] for r in rows])
    print(f"\n{len(rows)} comparisons")
    print(f"  chamfer mean   {ch.mean():.2f} mm   worst {ch.max():.2f} mm")
    print(f"  sampling floor {np.nanmean(fl):.2f} mm   (two random samples of the "
          f"same surface are already this far apart)")
    print(f"  depth error    {dd.mean():+.2f} mm   worst {np.abs(dd).max():.2f} mm")
    print(f"  flange error   {fd.mean():+.2f} mm   worst {np.abs(fd).max():.2f} mm")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
