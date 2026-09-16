"""Turn dataset samples into 3D assets a TRELLIS-style pipeline can consume.

TRELLIS encodes an asset by voxelising its surface and rendering it from many
views, then attaching DINOv2 features to the active voxels; it decodes to GLB.
Mesh topology is therefore irrelevant -- but three properties of the raw dataset
are not, and each is fixed here:

1. **Sectors, not objects.** Simulations exploit symmetry, so a part is stored as
   a quarter or half. A generative 3D model needs the whole object, so the sector
   is mirrored back across whichever planes it was cut on. Which planes are in
   play is detected from the geometry rather than trusted from metadata.

2. **Sheets, not solids.** The part is a zero-thickness midsurface with open
   boundary edges. Surface voxelisation of an open sheet is ambiguous and mesh
   decoders assume a closed surface, so the midsurface is offset by +/- half the
   local thickness along its normals and the boundary is stitched, giving a
   watertight solid. The per-element thickness from the simulation is real data,
   so the solid carries the thinning rather than being a uniform shell.

3. **npz, not a mesh file.** Exports GLB (and optionally PLY/OBJ), normalised
   into a unit cube the way TRELLIS expects.

    python .\\postprocessing\\export_trellis.py --all
    python .\\postprocessing\\export_trellis.py --dataset shapes/dataset --format ply

Only the blank (initial and final) is exported. The tools are stored as STL per
run and would need the same mirror-and-thicken treatment if they are wanted as
conditioning assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------


def quads_to_tris(quads: np.ndarray) -> np.ndarray:
    return np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def detect_symmetry(nodes: np.ndarray, tol: float = 1e-4) -> tuple:
    """Which planes the sector was cut on, read off the geometry itself."""
    return (bool(nodes[:, 0].min() > -tol), bool(nodes[:, 1].min() > -tol))


def mirror(nodes: np.ndarray, faces: np.ndarray, axis: int) -> tuple:
    """Reflect a sector through a coordinate plane and join the two halves."""
    flipped = nodes.copy()
    flipped[:, axis] *= -1.0
    both = np.vstack([nodes, flipped])
    # reversing the winding keeps the reflected half facing outward
    faces_both = np.vstack([faces, faces[:, ::-1] + len(nodes)])
    return both, faces_both


def node_values(quads: np.ndarray, per_quad: np.ndarray, n_nodes: int) -> np.ndarray:
    """Average a per-element field onto nodes."""
    total = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    for c in range(quads.shape[1]):
        np.add.at(total, quads[:, c], per_quad)
        np.add.at(count, quads[:, c], 1)
    return total / np.maximum(count, 1)


def node_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = nodes[faces]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out = np.zeros_like(nodes)
    for c in range(3):
        np.add.at(out, faces[:, c], fn)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Directed edges used by exactly one triangle: the open rim."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return e[idx[counts == 1]]


def thicken(nodes: np.ndarray, faces: np.ndarray,
            thickness: np.ndarray) -> trimesh.Trimesh:
    """Offset a midsurface by +/- half its local thickness into a closed solid."""
    n = node_normals(nodes, faces)
    half = (thickness / 2.0)[:, None]
    top = nodes + half * n
    bottom = nodes - half * n
    verts = np.vstack([top, bottom])
    m = len(nodes)
    tris = [faces, faces[:, ::-1] + m]
    rim = boundary_edges(faces)
    if len(rim):
        a, b = rim[:, 0], rim[:, 1]
        # two triangles per rim edge close the side wall
        tris.append(np.column_stack([a, b, b + m]))
        tris.append(np.column_stack([a, b + m, a + m]))
    mesh = trimesh.Trimesh(verts, np.vstack(tris), process=True)
    # trimesh 4 replaced remove_duplicate_faces/remove_degenerate_faces
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def weld(nodes: np.ndarray, faces: np.ndarray, values: np.ndarray,
         tol: float = 1e-6) -> tuple:
    """Merge coincident nodes, carrying a per-node field with them.

    Mirroring leaves two coincident copies of every node on the symmetry plane.
    Until they are merged the seam still *looks* like an open boundary, so the
    rim-stitching below would wall it off and bury a septum through the middle
    of the part. Welding first leaves only the genuine outer rim.
    """
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    idx = remap[inverse]
    out_nodes = nodes[first[order]]
    out_values = np.zeros(len(out_nodes))
    counts = np.zeros(len(out_nodes))
    np.add.at(out_values, idx, values)
    np.add.at(counts, idx, 1)
    new_faces = idx[faces]
    keep = ((new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2]))
    return out_nodes, new_faces[keep], out_values / np.maximum(counts, 1)


def build_asset(nodes: np.ndarray, quads: np.ndarray, thickness: np.ndarray,
                normalise: bool = True) -> trimesh.Trimesh:
    """Sector midsurface -> whole, watertight, unit-cube solid."""
    faces = quads_to_tris(quads)
    thk = node_values(quads, thickness, len(nodes)) if thickness.size else \
        np.full(len(nodes), 1.0)
    sym_x, sym_y = detect_symmetry(nodes)
    if sym_x:
        nodes, faces = mirror(nodes, faces, 0)
        thk = np.r_[thk, thk]
    if sym_y:
        nodes, faces = mirror(nodes, faces, 1)
        thk = np.r_[thk, thk]
    if sym_x or sym_y:
        nodes, faces, thk = weld(nodes, faces, thk)
    mesh = thicken(nodes, faces, thk)
    if normalise:
        mesh.apply_translation(-mesh.bounding_box.centroid)
        scale = float(np.max(mesh.extents))
        if scale > 0:
            mesh.apply_scale(1.0 / scale)
    return mesh


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_sample(npz_path: Path, out_dir: Path, fmt: str = "glb",
                  normalise: bool = True) -> dict:
    z = np.load(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    info = {"sample": stem}
    thickness = z["thickness"] if "thickness" in z.files else np.empty(0)

    for state, key in (("final", "nodes"), ("initial", "nodes_initial")):
        if key not in z.files:
            continue
        mesh = build_asset(z[key], z["quads"], thickness, normalise)
        path = out_dir / f"{stem}_{state}.{fmt}"
        mesh.export(path)
        info[state] = {
            "file": path.name,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
        }
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, action="append",
                    help="dataset root; repeatable. Defaults to all three.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: <dataset>/../trellis_assets)")
    ap.add_argument("--format", default="glb", choices=["glb", "ply", "obj"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-normalise", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    roots = args.dataset or [ROOT / "dataset", ROOT / "u_shape" / "dataset",
                             ROOT / "shapes" / "dataset"]
    summary, failed = [], 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        out_root = args.out or root.parent / "trellis_assets"
        samples = sorted(root.glob("*/Data/Bauteil/*.npz"))
        if args.limit:
            samples = samples[: args.limit]
        for s in samples:
            case = s.parents[2].name
            try:
                info = export_sample(s, out_root / case, args.format,
                                     not args.no_normalise)
                info["case"] = case
                info["dataset"] = root.name
                summary.append(info)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {case}/{s.name}: {type(exc).__name__}: {exc}")
        print(f"{root}: exported {len(samples) - failed} -> {out_root}")

    if summary:
        wt = sum(1 for r in summary if r.get("final", {}).get("watertight"))
        print(f"\n{len(summary)} samples, {wt} watertight finals, {failed} failed")
        out = (args.out or ROOT) / "trellis_export_manifest.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
