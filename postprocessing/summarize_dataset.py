"""Audit a generated dataset: does it load, is it consistent, what is in it.

Loads every sample the way a training script would, so anything that would break
a dataloader shows up here rather than in the middle of an epoch.

    python .\\postprocessing\\summarize_dataset.py
    python .\\postprocessing\\summarize_dataset.py --csv summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TOOL_FILES = {
    "blank": "00_Blank_Viertel.npz",
    "die": "01_Matrize_Viertel.npz",
    "binder": "02_Niederhalter_Viertel.npz",
    "punch": "03_Stempel_Viertel.npz",
}
NAME_RE = re.compile(
    r"tool_radii2_(?P<r2>[\d.]+)_radii1_(?P<r1>[\d.]+)_cr_(?P<cr>[\d.]+)"
    r"_delta_(?P<delta>[\d.]+)_height_(?P<h>[\d.]+)$"
)


def audit_case(case: Path) -> tuple:
    """Return (rows, problems) for one case directory."""
    rows, problems = [], []
    name = NAME_RE.match(case.name)
    if not name:
        return rows, [f"{case.name}: folder name does not parse"]
    height = float(name["h"])

    data = case / "Data"
    tools = {}
    for key, fname in TOOL_FILES.items():
        f = data / fname
        if not f.exists():
            problems.append(f"{case.name}: missing {fname}")
            continue
        pts = np.load(f)["points"]
        if not np.isfinite(pts).all():
            problems.append(f"{case.name}: {fname} has non-finite points")
        tools[key] = pts

    parts = sorted((data / "Bauteil").glob("*.npz")) if (data / "Bauteil").is_dir() else []
    if not parts:
        problems.append(f"{case.name}: no Bauteil samples")

    for pf in parts:
        m = re.search(r"\d+", pf.name)
        if not m:
            problems.append(f"{pf.name}: no force in filename")
            continue
        z = np.load(pf)
        pts, thk = z["points"], z["thickness"]
        if not np.isfinite(pts).all():
            problems.append(f"{pf.name}: non-finite points")
        depth = float(pts[:, 2].max() - pts[:, 2].min())
        # the cup cannot be deeper than the cavity by more than a sheet or two
        if depth > height + 3.0:
            problems.append(f"{pf.name}: drawn depth {depth:.2f} exceeds "
                            f"height {height:.0f} by more than 3 mm")

        report = case / f"case_report_{m.group()}.json"
        checks = json.loads(report.read_text(encoding="utf-8"))["checks"] if report.exists() else {}
        rows.append({
            "case": case.name,
            "corner_radius": float(name["r2"]),
            "profile_radius": float(name["r1"]),
            "clearance_ratio": float(name["cr"]),
            "draft": float(name["delta"]),
            "height": height,
            "blankholder_force_N": int(m.group()),
            "points": len(pts),
            "drawn_depth_mm": depth,
            "depth_error_mm": depth - height,
            "min_thickness_mm": float(thk.min()) if thk.size else np.nan,
            "max_thinning": float(1 - thk.min()) if thk.size else np.nan,
            "flange_x_mm": float(pts[:, 0].max()),
            "flange_y_mm": float(pts[:, 1].max()),
            "binder_below_die_mm": checks.get("binder_below_die_mm", np.nan),
            "peak_ke_over_ie": checks.get("peak_ke_over_ie", np.nan),
            "accepted": checks.get("accepted"),
        })
    return rows, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    cases = sorted(c for c in args.dataset.iterdir() if c.is_dir())
    # A glob-based dataloader treats each visible entry as a case, so only a
    # visible non-directory is a problem. Dotfiles such as .gitkeep are skipped
    # by glob and are harmless.
    stray = [c.name for c in args.dataset.iterdir()
             if not c.is_dir() and not c.name.startswith(".")]

    rows, problems = [], []
    for case in cases:
        r, p = audit_case(case)
        rows += r
        problems += p
    if stray:
        problems.append(f"non-directory entries in the dataset root would break a "
                        f"dataloader that globs it: {stray}")

    if not rows:
        print("no samples found")
        raise SystemExit(1)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"{len(cases)} geometries, {len(df)} samples\n")

    print("coverage")
    for col in ("corner_radius", "height", "draft", "blankholder_force_N"):
        vals = sorted(df[col].unique())
        shown = ", ".join(f"{v:g}" for v in vals[:10]) + (" ..." if len(vals) > 10 else "")
        print(f"  {col:<20} {len(vals):>3} values: {shown}")

    print("\nphysics")
    for col, unit in (("drawn_depth_mm", "mm"), ("depth_error_mm", "mm"),
                      ("max_thinning", ""), ("min_thickness_mm", "mm"),
                      ("binder_below_die_mm", "mm"), ("peak_ke_over_ie", "")):
        s = df[col].dropna()
        if s.empty:
            continue
        print(f"  {col:<22} min {s.min():+8.3f}  mean {s.mean():+8.3f}  "
              f"max {s.max():+8.3f} {unit}")

    print("\nblankholder force vs thinning")
    g = df.groupby("blankholder_force_N")["max_thinning"]
    for force, s in g:
        print(f"  {force:>7} N  n={len(s):>3}  thinning {s.mean():.1%} "
              f"(min {s.min():.1%}, max {s.max():.1%})")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:20]:
            print("  -", p)
    else:
        print("\nno problems found")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
