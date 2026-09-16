"""Render any simulated field on a drawn part, and bake it into a coloured PLY.

The part is mirrored back from its symmetry sector to the whole object first, so
what you see is the real component rather than a quarter or half of one.

    python .\\postprocessing\\visualise_field.py --case lobed5 --field major_strain
    python .\\postprocessing\\visualise_field.py --case lobed5 --field minor_strain --fld

Fields available per sample: major_strain, minor_strain, major_strain_membrane,
minor_strain_membrane, thickness_strain, plastic_strain, thickness.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_trellis as E  # noqa: E402

ROOT = HERE.parent
LABELS = {
    "major_strain": "major principal strain",
    "minor_strain": "minor principal strain",
    "major_strain_membrane": "major membrane strain",
    "minor_strain_membrane": "minor membrane strain",
    "thickness_strain": "through-thickness strain",
    "plastic_strain": "effective plastic strain",
    "thickness": "sheet thickness [mm]",
}


def find_sample(pattern: str) -> Path:
    roots = ["dataset", "u_shape/dataset", "shapes/dataset"]
    for r in roots:
        hits = sorted(glob.glob(str(ROOT / r / f"*{pattern}*" / "Data" / "Bauteil" / "*.npz")))
        if hits:
            return Path(hits[0])
    raise SystemExit(f"no sample matching {pattern!r}")


def whole_part(z, field: str) -> tuple:
    """Mirror the sector to the whole part, carrying the field with it."""
    nodes, quads = z["nodes"], z["quads"]
    values = E.node_values(quads, z[field], len(nodes))
    faces = E.quads_to_tris(quads)
    sym_x, sym_y = E.detect_symmetry(nodes)
    if sym_x:
        nodes, faces = E.mirror(nodes, faces, 0)
        values = np.r_[values, values]
    if sym_y:
        nodes, faces = E.mirror(nodes, faces, 1)
        values = np.r_[values, values]
    if sym_x or sym_y:
        nodes, faces, values = E.weld(nodes, faces, values)
    return nodes, faces, values


def make_norm(values: np.ndarray) -> tuple:
    """Diverging about zero when the field changes sign, else sequential.

    Centring a diverging map on zero is what makes tension and compression
    readable at a glance; forcing it on an all-positive field would waste half
    the colour range.
    """
    lo, hi = float(values.min()), float(values.max())
    if lo < -1e-6 and hi > 1e-6:
        return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi), plt.cm.RdBu_r, True
    return Normalize(lo, hi), plt.cm.viridis, False


def render(nodes, faces, values, title, out_png, cbar_label):
    norm, cmap, diverging = make_norm(values)
    fig = plt.figure(figsize=(17, 7.5))
    for i, (elev, azim, label) in enumerate(
            [(34, -60, "isometric"), (88, -90, "top (plan)"), (6, -90, "side")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        tri = nodes[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        light = np.array([0.35, -0.5, 0.79])
        shade = np.clip(np.abs(n @ light), 0.35, 1.0)
        colours = cmap(norm(values[faces].mean(1)))[:, :3] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colours,
                                             linewidths=0, shade=False))
        lim = np.abs(nodes[:, :2]).max() * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim * 0.7, lim * 0.7)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=11)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.62, pad=0.02)
    cb.set_label(cbar_label + ("  (blue = compressive, red = tensile)"
                               if diverging else ""), fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=135, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return norm, cmap


def strain_distribution(z, out_png, title):
    """Major vs minor principal strain: where each element sits in strain space.

    This is deliberately NOT called a forming limit diagram. An FLD is defined by
    its forming limit curve, and no FLC is defined in this project -- acceptance
    is on raw thinning, not on FLD margin. Without a limit curve the plot shows
    the strain state of every element but says nothing about proximity to
    failure, so labelling it an FLD would overstate it.

    The right branch is biaxial stretch and the left is drawing. The grey line is
    the pure-drawing path (major = -minor, constant thickness) and is explicitly
    annotated, because on a plot titled "forming limit diagram" a reference line
    reads as the limit curve.
    """
    major, minor = z["major_strain"], z["minor_strain"]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(minor, major, s=6, c=z["thickness"], cmap="viridis", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.plot([-0.5, 0.0], [0.5, 0.0], color="0.6", lw=1, ls=":")
    ax.annotate("pure drawing\n(major = -minor, constant thickness)\nreference line "
                "- NOT a forming limit curve",
                xy=(-0.34, 0.34), xytext=(-0.47, 0.10), fontsize=7.5, color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.set_xlabel("minor principal strain")
    ax.set_ylabel("major principal strain")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.97, "drawing\n(minor < 0)", transform=ax.transAxes,
            va="top", fontsize=9, color="0.35")
    ax.text(0.72, 0.97, "biaxial\nstretch", transform=ax.transAxes,
            va="top", fontsize=9, color="0.35")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=Normalize(z["thickness"].min(), z["thickness"].max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="thickness [mm]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=135, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="substring of the case folder")
    ap.add_argument("--field", default="major_strain", choices=sorted(LABELS))
    ap.add_argument("--out", type=Path,
                    default=ROOT.parent / "Examples")
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--fld", action="store_true",
                    help="also plot the major vs minor strain distribution")
    args = ap.parse_args()

    path = find_sample(args.case)
    case = os.path.basename(path.parents[2])
    z = np.load(path)
    if args.field not in z.files:
        raise SystemExit(f"{args.field} not stored in {path.name}")

    nodes, faces, values = whole_part(z, args.field)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{args.case}_{args.field}"
    title = (f"{case[:60]}  -  {LABELS[args.field]}\n"
             f"min {values.min():+.3f}   mean {values.mean():+.3f}   "
             f"max {values.max():+.3f}")
    png = args.out / f"{stem}.png"
    norm, cmap = render(nodes, faces, values, title, png, LABELS[args.field])

    # bake the same colours onto the watertight solid
    from scipy.spatial import cKDTree
    mesh = E.build_asset(z["nodes"], z["quads"], z["thickness"], normalise=False)
    _, idx = cKDTree(nodes).query(mesh.vertices)
    mesh.visual.vertex_colors = (np.array(cmap(norm(values[idx]))) * 255).astype(np.uint8)
    ply = args.out / f"{stem}.ply"
    mesh.export(ply)

    print(f"{case}")
    print(f"  {LABELS[args.field]}: min {values.min():+.4f}  mean {values.mean():+.4f}"
          f"  max {values.max():+.4f}")
    print(f"  wrote {png.name} and {ply.name}")

    if args.fld:
        fld = args.out / f"{stem.rsplit('_', 2)[0]}_strain_distribution.png"
        strain_distribution(
            z, fld,
            f"{case[:55]}\nstrain distribution - no forming limit curve defined")
        print(f"  wrote {fld.name}")


if __name__ == "__main__":
    main()
