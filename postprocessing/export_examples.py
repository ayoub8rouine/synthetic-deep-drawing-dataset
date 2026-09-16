"""Pick a spread of geometries from each category and export them as PLY.

One sample per geometry, chosen to be as different from each other as possible
rather than taken in folder order, so the exported set actually shows the range
the dataset covers. Cups are additionally spread across their outline families
so no family is missed.

    python .\\postprocessing\\export_examples.py
    python .\\postprocessing\\export_examples.py --per-category 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_trellis as E  # noqa: E402

ROOT = HERE.parent
EXAMPLES = ROOT.parent / "Examples"

CATEGORIES = {
    "box": ROOT / "dataset",
    "uchannel": ROOT / "u_shape" / "dataset",
    "cup": ROOT / "shapes" / "dataset",
}


def case_features(case_dir: Path) -> tuple:
    """A few numbers describing the shape, for spreading the selection."""
    params = json.loads((case_dir / "parameters.json").read_text(encoding="utf-8"))
    d = params.get("derived", {})
    if "outline" in params:                       # cup
        o = params["outline"]
        family = o["family"]
        feat = [params["height"], params["draft_deg"], params["draw_ratio"],
                d.get("part_max_radius_mm", 0.0),
                float(o.get("n_sides", o.get("n_lobes", 0)))]
    elif "section_a" in params:                   # u-channel
        a, b = params["section_a"], params["section_b"]
        family = "uchannel"
        feat = [a["height"], b["height"], a["floor_half_width"],
                b["floor_half_width"], a["wall_angle_deg"], b["wall_angle_deg"]]
    else:                                         # box
        family = "box"
        feat = [params["height"], params["corner_radius"], params["draft"],
                params["clearance_ratio"]]
    return family, np.asarray(feat, dtype=float)


def spread(features: np.ndarray, k: int) -> list:
    """Greedy farthest-point selection: each pick is the most different so far."""
    if len(features) <= k:
        return list(range(len(features)))
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x = features / scale
    chosen = [int(np.argmax(np.linalg.norm(x - x.mean(0), axis=1)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(x[:, None, :] - x[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def collect(root: Path) -> list:
    out = []
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        samples = sorted((case / "Data" / "Bauteil").glob("*.npz"))
        if not samples or not (case / "parameters.json").exists():
            continue
        try:
            family, feat = case_features(case)
        except Exception:  # noqa: BLE001
            continue
        out.append({"case": case.name, "family": family, "feat": feat,
                    "sample": samples[len(samples) // 2]})
    return out


def select(entries: list, k: int) -> list:
    """Spread within each family, then across families."""
    families = sorted({e["family"] for e in entries})
    per = max(1, k // len(families))
    picked = []
    for fam in families:
        group = [e for e in entries if e["family"] == fam]
        feats = np.array([e["feat"] for e in group])
        picked += [group[i] for i in spread(feats, min(per, len(group)))]
    # top up to k with whatever is most different from what is already picked
    if len(picked) < k:
        rest = [e for e in entries if e not in picked]
        if rest:
            feats = np.array([e["feat"] for e in rest])
            picked += [rest[i] for i in spread(feats, min(k - len(picked), len(rest)))]
    return picked[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--out", type=Path, default=EXAMPLES)
    args = ap.parse_args()

    index = {}
    for name, root in CATEGORIES.items():
        if not root.exists():
            continue
        entries = collect(root)
        picks = select(entries, args.per_category)
        out_dir = args.out / name / "geometries"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        print(f"\n{name}: {len(entries)} geometries available, exporting {len(picks)}")
        for e in picks:
            z = np.load(e["sample"])
            mesh = E.build_asset(z["nodes"], z["quads"], z["thickness"],
                                 normalise=False)
            stem = e["case"][:70]
            mesh.export(out_dir / f"{stem}.ply")
            row = {
                "case": e["case"], "family": e["family"],
                "file": f"{stem}.ply",
                "size_mm": [round(float(v), 1) for v in mesh.extents],
                "vertices": int(len(mesh.vertices)),
                "watertight": bool(mesh.is_watertight),
                "max_thinning": float(1 - z["thickness"].min()),
            }
            rows.append(row)
            print(f"   {e['family']:<10} {stem[:52]:<54}"
                  f"{mesh.extents[0]:6.0f} x{mesh.extents[1]:6.0f} x{mesh.extents[2]:6.1f} mm")
        (args.out / name / "geometries.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        index[name] = rows
    print(f"\nwrote {sum(len(v) for v in index.values())} geometry files")


if __name__ == "__main__":
    main()
