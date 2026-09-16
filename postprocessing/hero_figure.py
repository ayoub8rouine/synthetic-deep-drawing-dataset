"""Hero figure: one drawn part from each category, thickness-mapped.

Two earlier versions failed the same way, for opposite reasons, so the fixes are
worth stating:

*   **Dead space.** A drawn pan is a flat object, and a matplotlib 3-D axes
    always reserves a cube. Setting a small ``box_aspect`` puts the part in a
    thin slab through the middle of that cube and surrounds it with white. The
    fix is not to fight the axes: render each panel on an identical camera with
    identical limits, then **crop each to its ink** and paste at native pixel
    size. Framing is tight and the shared scale survives, because nothing is
    resized after rendering.
*   **Washed-out colour.** Thickness runs 0.79-1.53 mm, but the tails are a
    handful of elements; normalising to them maps the entire part to the pale
    centre of the map. The range is set from a percentile of |t - nominal| and
    clipped, so the ordinary variation is what gets the colour and the extremes
    saturate.

Colour is diverging about the undeformed 1.0 mm, which is the natural zero of
the process: warm is thinned, cool is thickened. No wireframe -- at these
element sizes it aliases into moire and the mesh is shown elsewhere.

    python .\\postprocessing\\hero_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_trellis as E  # noqa: E402

ROOT = HERE.parent

PANELS = [
    ("Box", "rectangular pan", ROOT / "dataset" /
     "tool_radii2_20_radii1_5_cr_1.1_delta_0_height_25"),
    ("U-channel", "hat section", ROOT / "u_shape" / "dataset" /
     "uchannel_w43.1-64.5_h49.5-32_beta24.4-20.2_rf9-10.7_rc9.3-8.9"),
    ("Cup", "5-lobed clover", ROOT / "shapes" / "dataset" /
     "lobed5_R88.5_A13.7_h49.1_draft18.7_rp8.4_rd11.3_dr1.6"),
]

NOMINAL = 1.0                 # undeformed sheet thickness, mm
ELEV, AZIM = 27.0, -60.0
LIGHT = np.array([0.34, -0.60, 0.72])
INK, MUTED = "#16191d", "#5d666f"

# Saturation reaches the ends quickly so ordinary thinning is legible; the pale
# band around nominal is deliberately narrow.
CMAP = LinearSegmentedColormap.from_list("forming", [
    "#6e1f0c", "#b5561f", "#dc9558", "#f0cdaf", "#f4f2ee",
    "#cfe0e8", "#77aecb", "#2f6f97", "#0d3550",
])


def load(case: Path) -> tuple:
    """Whole part (mirrored back from its symmetry sector) + per-triangle thickness."""
    samples = sorted((case / "Data" / "Bauteil").glob("*.npz"))
    z = np.load(samples[len(samples) // 2])
    nodes, quads, thick = z["nodes"], z["quads"], z["thickness"]

    faces = E.quads_to_tris(quads)
    vals = E.node_values(quads, thick, len(nodes))

    sym_x, sym_y = E.detect_symmetry(nodes)
    for axis, on in ((0, sym_x), (1, sym_y)):
        if on:
            nodes, faces = E.mirror(nodes, faces, axis)
            vals = np.r_[vals, vals]
    tri_t = vals[faces].mean(axis=1)
    raw_min = float(thick.min())   # true element minimum, for the caption
    # No welding: coincident nodes on a mirror plane are invisible in a shaded
    # render. The field is averaged from the triangle's three nodes, which is
    # both smoother than flat per-element colour and immune to the fact that
    # quads_to_tris concatenates its two triangle sets rather than interleaving
    # them -- a per-element repeat here silently colours the wrong elements.
    return nodes - nodes.mean(axis=0), faces, tri_t, raw_min


def render(nodes, faces, tri_t, norm, half_xy: float, half_z: float,
           dpi: int) -> np.ndarray:
    """One panel on a fixed camera, returned as an RGB array cropped to its ink."""
    fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")

    tri = nodes[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    lit = np.abs(n @ (LIGHT / np.linalg.norm(LIGHT)))

    rgba = CMAP(norm(tri_t))
    rgba[:, :3] *= (0.70 + 0.38 * lit)[:, None]
    np.clip(rgba, 0.0, 1.0, out=rgba)

    ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, linewidths=0,
                                         edgecolors="none", shade=False))
    ax.set_xlim(-half_xy, half_xy)
    ax.set_ylim(-half_xy, half_xy)
    ax.set_zlim(-half_z, half_z)
    ax.set_box_aspect((1, 1, half_z / half_xy))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    ink = (img < 250).any(axis=2)
    if not ink.any():
        return img
    ys, xs = np.where(ink)
    pad = 2
    return img[max(ys.min() - pad, 0):ys.max() + pad + 1,
               max(xs.min() - pad, 0):xs.max() + pad + 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT.parent / "Examples" / "images" / "three_categories.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    parts = []
    for label, sub, case in PANELS:
        nodes, faces, tri_t, raw_min = load(case)
        parts.append(dict(label=label, sub=sub, nodes=nodes, faces=faces,
                          t=tri_t, raw_min=raw_min))
        print(f"{label:11s} {np.ptp(nodes[:,0]):.0f} x {np.ptp(nodes[:,1]):.0f} x "
              f"{np.ptp(nodes[:,2]):.0f} mm   t {tri_t.min():.3f}-{tri_t.max():.3f}")

    # One camera and one colour scale for all three panels.
    half_xy = max(np.abs(p["nodes"][:, :2]).max() for p in parts) * 1.02
    half_z = max(np.abs(p["nodes"][:, 2]).max() for p in parts) * 1.10
    dev = np.concatenate([np.abs(p["t"] - NOMINAL) for p in parts])
    r = float(np.percentile(dev, 97))
    norm = Normalize(vmin=NOMINAL - r, vmax=NOMINAL + r, clip=True)
    print(f"colour range {NOMINAL - r:.3f} - {NOMINAL + r:.3f} mm "
          f"(97th pct deviation {r:.3f})")

    tiles = [render(p["nodes"], p["faces"], p["t"], norm, half_xy, half_z, args.dpi)
             for p in parts]

    # ---- compose ---------------------------------------------------------
    gap = int(0.045 * max(t.shape[1] for t in tiles))
    band_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    band_h = max(t.shape[0] for t in tiles)
    cap_h = int(band_h * 0.30)          # captions
    key_h = int(band_h * 0.24)          # legend strip
    margin = int(band_w * 0.022)

    W, H = band_w + 2 * margin, band_h + cap_h + key_h + margin
    fig = plt.figure(figsize=(W / args.dpi, H / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    x = margin
    for tile, p in zip(tiles, parts):
        h, w = tile.shape[:2]
        y = key_h + cap_h + (band_h - h)          # top-align the band
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        ax.imshow(tile, interpolation="antialiased")
        ax.set_axis_off()

        cx = (x + w / 2) / W
        fig.text(cx, (key_h + cap_h * 0.66) / H, f"{p['label']}  ·  {p['sub']}",
                 ha="center", va="center", fontsize=12.5, fontweight="semibold",
                 color=INK)
        thin = 100 * (1 - p["raw_min"] / NOMINAL)
        fig.text(cx, (key_h + cap_h * 0.30) / H,
                 f"{np.ptp(p['nodes'][:,0]):.0f} × {np.ptp(p['nodes'][:,1]):.0f} mm"
                 f"   ·   {np.ptp(p['nodes'][:,2]):.0f} mm deep"
                 f"   ·   {thin:.0f}% max thinning",
                 ha="center", va="center", fontsize=9.6, color=MUTED)
        x += w + gap

    # ---- legend ----------------------------------------------------------
    bar_w, bar_h = 0.30, (key_h * 0.20) / H
    cax = fig.add_axes([0.5 - bar_w / 2, (key_h * 0.34) / H, bar_w, bar_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([NOMINAL - r, NOMINAL, NOMINAL + r])
    cb.set_ticklabels([f"≤ {NOMINAL - r:.2f}", f"{NOMINAL:.2f}", f"≥ {NOMINAL + r:.2f}"])
    cb.ax.tick_params(labelsize=9, colors=MUTED, length=0, pad=4)
    cb.outline.set_edgecolor("#c8cdd4")
    cb.outline.set_linewidth(0.6)

    fig.text(0.5 - bar_w / 2 - 0.012, (key_h * 0.34 + key_h * 0.10) / H, "thinned",
             ha="right", va="center", fontsize=9, color="#b5561f", fontweight="medium")
    fig.text(0.5 + bar_w / 2 + 0.012, (key_h * 0.34 + key_h * 0.10) / H, "thickened",
             ha="left", va="center", fontsize=9, color="#2f6f97", fontweight="medium")
    fig.text(0.5, (key_h * 0.86) / H,
             "Simulated sheet thickness, mm   ·   nominal 1.00 mm   ·   "
             "all three panels at one common scale",
             ha="center", va="center", fontsize=9.4, color=MUTED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}  ({W}x{H} px)")


if __name__ == "__main__":
    main()
