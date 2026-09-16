"""Capture one viewer-style image per geometry.

Matches the look of a 3D-viewer screenshot: white background, dark shaded
surface with the mesh wireframe showing through, a slightly raised viewing
angle, and a small RGB axis triad in the corner.

One image per geometry (not per blankholder force), mirrored from its symmetry
sector to the whole part.

    python .\\postprocessing\\capture_geometries.py --out .\\captures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_trellis as E  # noqa: E402

ROOT = HERE.parent
CATEGORIES = {
    "box": ROOT / "dataset",
    "uchannel": ROOT / "u_shape" / "dataset",
    "cup": ROOT / "shapes" / "dataset",
}

ELEV, AZIM = 44.0, -96.0        # slightly raised and turned, as in the example
LIGHT = np.array([0.25, -0.55, 0.80])


def whole_part(z) -> tuple:
    nodes, quads = z["nodes"], z["quads"]
    faces = E.quads_to_tris(quads)
    dummy = np.ones(len(nodes))
    sym_x, sym_y = E.detect_symmetry(nodes)
    if sym_x:
        nodes, faces = E.mirror(nodes, faces, 0)
        dummy = np.r_[dummy, dummy]
    if sym_y:
        nodes, faces = E.mirror(nodes, faces, 1)
        dummy = np.r_[dummy, dummy]
    if sym_x or sym_y:
        nodes, faces, dummy = E.weld(nodes, faces, dummy)
    return nodes - nodes.mean(axis=0), faces


def capture(nodes: np.ndarray, faces: np.ndarray, out: Path) -> None:
    fig = plt.figure(figsize=(6.14, 4.61), dpi=100)
    ax = fig.add_axes([-0.05, -0.09, 1.10, 1.18], projection="3d")

    tri = nodes[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    light = LIGHT / np.linalg.norm(LIGHT)
    shade = np.clip(np.abs(n @ light), 0.0, 1.0)
    face = (0.06 + 0.30 * shade)[:, None] * np.ones(3)
    ax.add_collection3d(Poly3DCollection(
        tri, facecolors=face, edgecolors=(0, 0, 0, 0.7), linewidths=0.11,
        shade=False))

    span = float(np.abs(nodes[:, :2]).max())
    ax.set_xlim(-span * 0.74, span * 0.74)
    ax.set_ylim(-span * 0.74, span * 0.74)
    ax.set_zlim(-span * 0.66, span * 0.66)
    ax.set_box_aspect((1, 0.82, 0.60))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    gizmo = fig.add_axes([0.885, 0.015, 0.105, 0.145], projection="3d")
    gizmo.set_axis_off()
    for vec, colour in (((1, 0, 0), "red"), ((0, 1, 0), "green"), ((0, 0, 1), "blue")):
        gizmo.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=colour, lw=2)
        gizmo.scatter([vec[0]], [vec[1]], [vec[2]], color=colour, s=20)
    gizmo.set_xlim(-1, 1)
    gizmo.set_ylim(-1, 1)
    gizmo.set_zlim(-1, 1)
    gizmo.view_init(elev=ELEV, azim=AZIM)

    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flat", action="store_true",
                    help="all images in one folder instead of per category")
    args = ap.parse_args()

    total = 0
    for name, root in CATEGORIES.items():
        if not root.exists():
            continue
        out_dir = args.out if args.flat else args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cases = sorted(p for p in root.iterdir() if p.is_dir())
        for case in cases:
            samples = sorted((case / "Data" / "Bauteil").glob("*.npz"))
            if not samples:
                continue
            try:
                z = np.load(samples[len(samples) // 2])
                nodes, faces = whole_part(z)
                capture(nodes, faces, out_dir / f"{case.name[:80]}.png")
                total += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {case.name}: {type(exc).__name__}: {exc}")
        print(f"{name}: {len(cases)} geometries -> {out_dir}")
    print(f"\ncaptured {total} images")


if __name__ == "__main__":
    main()
