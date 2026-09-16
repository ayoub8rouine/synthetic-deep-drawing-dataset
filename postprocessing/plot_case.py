"""Render one dataset case: tools, initial blank, and the drawn part.

Draws the analytic tool profiles rather than a sampled cloud, so the punch nose,
die entry and cavity roof are shown exactly as the generator defines them, with
the simulated part laid over the top.

    python .\\postprocessing\\plot_case.py <dataset case dir> --out case.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))

from box_draw import DrawParams, die_profile, punch_profile  # noqa: E402

TOOL_C = {"die": "#b0413e", "punch": "#2f6f9f", "binder": "#3d8f5b"}


def tool_sections(p: DrawParams) -> dict:
    """Tool outlines in the y = 0 plane as (x, z) polylines."""
    a = p.punch_half_length
    size = p.tool_element_size
    pp, s_top = punch_profile(p, size)
    dp, s_flange, s_roof = die_profile(p, size)

    punch = np.vstack([[[0.0, p.z_blank_bottom]], np.column_stack([a + pp[:, 0], pp[:, 1]])])
    die = np.vstack([
        [[p.tool_half_length, p.z_blank_top]],
        np.column_stack([a + dp[:, 0], dp[:, 1]]),
        [[0.0, p.z_blank_top + p.height]],
    ])
    binder = np.array([[a + p.binder_offset, p.z_blank_bottom],
                       [p.tool_half_length, p.z_blank_bottom]])
    return {"punch": punch, "die": die, "binder": binder}


def section(points: np.ndarray, band: float = 6.0) -> np.ndarray:
    s = points[points[:, 1] < band][:, [0, 2]]
    return s[np.argsort(s[:, 0])]


def plot(case: Path, force: int | None, out: Path) -> Path:
    params = json.loads((case / "parameters.json").read_text(encoding="utf-8"))
    p = DrawParams(**{k: v for k, v in params.items() if k != "derived"})

    parts = sorted((case / "Data" / "Bauteil").glob("*.npz"))
    if force is not None:
        parts = [f for f in parts if str(force) in f.name] or parts
    part = np.load(parts[0])
    bhf = "".join(ch for ch in parts[0].stem if ch.isdigit())
    nodes, thk = part["nodes"], part["thickness"]

    tools = tool_sections(p)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{case.name}   blankholder force {int(bhf):,} N", fontsize=12)

    # The punch extension is tooling only; draw it dashed so it is not mistaken
    # for part of the forming surface.
    z_form = -(p.thickness / 2 + p.height)

    def draw_tools(a, dz_moving: float) -> None:
        for name, xy in tools.items():
            dz = 0.0 if name == "punch" else dz_moving
            if name == "punch":
                keep = xy[:, 1] >= z_form - 1e-9
                a.plot(xy[keep, 0], xy[keep, 1], color=TOOL_C[name], lw=2, label=name)
                a.plot(xy[~keep, 0], xy[~keep, 1], color=TOOL_C[name], lw=1.2, ls=":")
            else:
                a.plot(xy[:, 0], xy[:, 1] + dz, color=TOOL_C[name], lw=2, label=name)

    a0, a1 = ax[0, 0], ax[0, 1]
    draw_tools(a0, 0.0)
    a0.plot([0, p.blank_half_length], [0, 0], color="k", lw=3, label="blank (initial)")
    a0.set_title("initial state, section at y = 0")

    draw_tools(a1, -p.height)
    sec = section(nodes)
    a1.plot(sec[:, 0], sec[:, 1], "k.-", lw=1.2, ms=3, label="drawn part")
    a1.set_title(f"final state, die and binder travelled {p.height:g} mm down")

    lo = min(z_form - p.height, float(nodes[:, 2].min())) - 4
    hi = p.z_blank_top + p.height + 4
    for a in (a0, a1):
        a.set_xlabel("x [mm]")
        a.set_ylabel("z [mm]")
        a.set_ylim(lo, hi)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="lower left")

    # --- C: drawn part in plan, coloured by thickness ------------------------
    a2 = ax[1, 0]
    pts = part["points"]
    quads = part["quads"]
    node_thk = np.zeros(len(nodes))
    count = np.zeros(len(nodes))
    for c in range(4):
        np.add.at(node_thk, quads[:, c], thk)
        np.add.at(count, quads[:, c], 1)
    node_thk /= np.maximum(count, 1)
    sc = a2.scatter(nodes[:, 0], nodes[:, 1], c=node_thk, s=6, cmap="viridis")
    fig.colorbar(sc, ax=a2, label="thickness [mm]")
    a2.set_title("thickness over the drawn part (plan view)")
    a2.set_xlabel("x [mm]")
    a2.set_ylabel("y [mm]")
    a2.set_aspect("equal")

    # --- D: thickness distribution ------------------------------------------
    a3 = ax[1, 1]
    a3.hist(thk, bins=50, color="#2f6f9f", alpha=0.85)
    a3.axvline(p.thickness, color="k", ls="--", lw=1.5, label=f"initial {p.thickness:g} mm")
    a3.axvline(thk.min(), color="#b0413e", ls="-", lw=1.5,
               label=f"min {thk.min():.3f} mm ({1 - thk.min() / p.thickness:.1%} thinning)")
    a3.set_title("element thickness distribution")
    a3.set_xlabel("thickness [mm]")
    a3.set_ylabel("elements")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--force", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("case.png"))
    args = ap.parse_args()
    print(plot(args.case, args.force, args.out))


if __name__ == "__main__":
    main()
