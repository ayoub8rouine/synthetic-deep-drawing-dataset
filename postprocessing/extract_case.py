"""Turn one finished LS-DYNA run into a dataset sample and judge whether to keep it.

The final blank state is read from ``dynain``, the plain-text keyword file that
``*INTERFACE_SPRINGBACK_LSDYNA`` writes at normal termination. That avoids
needing a d3plot binary reader and gives deformed nodes plus per-element
thickness in one go, so thinning is available for the accept/reject decision.

Output layout per case, deliberately drop-in compatible with the reference
``dataloader.py`` while adding the initial state it lacks::

    <case>/Data/00_Blank_Viertel.npz          flat blank, INITIAL timestep
    <case>/Data/01_Matrize_Viertel.npz        die, initial position
    <case>/Data/02_Niederhalter_Viertel.npz   binder, initial position
    <case>/Data/03_Stempel_Viertel.npz        punch, initial position
    <case>/Data/Bauteil/Tiefgezogenes_Bauteil_<bhf>.npz   FINAL timestep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))

from box_draw import SurfaceMesh  # noqa: E402

TOOL_FILES = {
    "die": "01_Matrize_Viertel.npz",
    "binder": "02_Niederhalter_Viertel.npz",
    "punch": "03_Stempel_Viertel.npz",
    "blank": "00_Blank_Viertel.npz",
}


# ---------------------------------------------------------------------------
# solver output parsing
# ---------------------------------------------------------------------------


def _floats(line: str, width: int, count: int, start: int) -> list:
    """Read fixed-width numeric fields, falling back to whitespace splitting."""
    chunks = [line[start + i * width : start + (i + 1) * width] for i in range(count)]
    try:
        return [float(c) for c in chunks]
    except ValueError:
        return [float(v) for v in line.split()[1 : count + 1]]


def element_normals(nodes: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Unit normal of each deformed shell element, from its diagonals."""
    p = nodes[quads]
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def as_tensor(strain: np.ndarray) -> np.ndarray:
    """(n, 6) as (exx, eyy, ezz, exy, eyz, ezx) -> (n, 3, 3) symmetric tensors."""
    xx, yy, zz, xy, yz, zx = (strain[:, i] for i in range(6))
    t = np.empty((len(strain), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = xx, yy, zz
    t[:, 0, 1] = t[:, 1, 0] = xy
    t[:, 1, 2] = t[:, 2, 1] = yz
    t[:, 2, 0] = t[:, 0, 2] = zx
    return t


def principal_in_plane(strain: np.ndarray, normal: np.ndarray) -> tuple:
    """Major, minor and through-thickness strain in each element's own frame.

    LS-DYNA writes these strains in the GLOBAL frame. On a channel wall, whose
    normal points along y, the global "zz" component is therefore an in-plane
    strain, not a thickness strain -- taking (xx, yy, xy) as the in-plane block
    is only correct for elements that happen to lie flat. Rotating the tensor
    into a frame built on the element normal fixes that, and makes the
    through-thickness component agree with the measured thinning.
    """
    t = as_tensor(strain)
    # any two unit vectors orthogonal to the normal span the shell's plane
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(normal), 1))
    flat = np.abs(normal[:, 2]) > 0.9
    helper[flat] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, normal)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(normal, e1)

    rot = np.stack([e1, e2, normal], axis=2)          # columns are the new axes
    local = np.einsum("nki,nkl,nlj->nij", rot, t, rot)

    a, b, c = local[:, 0, 0], local[:, 1, 1], local[:, 0, 1]
    mean = 0.5 * (a + b)
    radius = np.sqrt((0.5 * (a - b)) ** 2 + c ** 2)
    return mean + radius, mean - radius, local[:, 2, 2]


def parse_dynain(path: Path) -> dict:
    """Deformed nodes, connectivity, thickness, plastic strain and strains.

    ``*INITIAL_STRESS_SHELL`` carries the stress tensor and the effective
    plastic strain at each through-thickness integration point;
    ``*INITIAL_STRAIN_SHELL`` carries the full strain tensor at the lower and
    upper surface. Both are packed in fixed 10-character fields with no
    separating space, so they must be read by column, not by splitting.
    """
    node_ids, coords = [], []
    ref_ids, ref_coords = [], []
    elem_nodes, thickness = [], []
    stress_eid, stress_rows = [], []
    strain_eid, strain_rows = [], []
    section, pending, want = None, None, 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.startswith("$"):
            continue
        if raw.startswith("*"):
            key = raw.strip().upper()
            if key.startswith("*NODE"):
                section = "node"
            elif key.startswith("*REFERENCE_GEOMETRY"):
                section = "reference"
            elif key.startswith("*ELEMENT_SHELL"):
                section = "shell_thickness" if "THICKNESS" in key else "shell"
            elif key.startswith("*INITIAL_STRESS_SHELL"):
                section = "stress"
            elif key.startswith("*INITIAL_STRAIN_SHELL"):
                section = "strain"
            else:
                section = None
            pending, want = None, 0
            continue

        if section == "node":
            node_ids.append(int(raw[:8]))
            coords.append(_floats(raw, 16, 3, 8))
        elif section == "reference":
            ref_ids.append(int(raw[:8]))
            ref_coords.append(_floats(raw, 16, 3, 8))
        elif section in ("shell", "shell_thickness"):
            if pending is None:
                n = [int(raw[8 * i : 8 * (i + 1)]) for i in range(2, 6)]
                if section == "shell":
                    elem_nodes.append(n)
                else:
                    pending = n
            else:
                elem_nodes.append(pending)
                thickness.append(float(np.mean(_floats(raw, 16, 4, 0))))
                pending = None
        elif section == "stress":
            if want == 0:
                stress_eid.append(int(raw[:10]))
                # EID, NPLANE, NTHICK, ... -> one row per integration point
                want = int(raw[20:30]) or 1
                pending = []
            else:
                pending.append(_floats(raw, 10, 8, 0))
                want -= 1
                if want == 0:
                    stress_rows.append(pending)
        elif section == "strain":
            if want == 0:
                strain_eid.append(int(raw[:10]))
                want, pending = 2, []      # lower and upper surface
            else:
                pending.append(_floats(raw, 10, 6, 0))
                want -= 1
                if want == 0:
                    strain_rows.append(pending)

    node_ids = np.asarray(node_ids, dtype=np.int64)
    coords = np.asarray(coords, dtype=float)
    order = np.argsort(node_ids)
    lookup = np.full(node_ids.max() + 1, -1, dtype=np.int64)
    lookup[node_ids[order]] = np.arange(len(node_ids))
    quads = lookup[np.asarray(elem_nodes, dtype=np.int64)]
    if (quads < 0).any():
        raise ValueError(f"{path}: element references an undefined node")

    out = {
        "nodes": coords[order],
        "quads": quads,
        "thickness": np.asarray(thickness, dtype=float) if thickness else None,
    }

    # *REFERENCE_GEOMETRY is the undeformed blank, written node-for-node with
    # *NODE. Sorting both by node id gives an exact initial-to-final
    # correspondence, which is what a mesh-based model needs and what a sampled
    # point cloud cannot provide.
    if ref_ids:
        ref = np.asarray(ref_coords, dtype=float)
        ref_order = np.argsort(np.asarray(ref_ids, dtype=np.int64))
        out["nodes_initial"] = ref[ref_order]
        if len(ref_order) == len(order):
            out["displacement"] = out["nodes"] - out["nodes_initial"]

    if stress_rows:
        # per integration point: t, 6 stress components, effective plastic strain
        s = np.asarray(stress_rows, dtype=float)
        out["stress"] = s[:, :, 1:7]
        out["plastic_strain_ip"] = s[:, :, 7]
        out["plastic_strain"] = s[:, :, 7].max(axis=1)
    if strain_rows:
        e = np.asarray(strain_rows, dtype=float)
        out["strain_lower"], out["strain_upper"] = e[:, 0, :], e[:, 1, :]
        normal = element_normals(out["nodes"], out["quads"])
        lo_major, lo_minor, _ = principal_in_plane(out["strain_lower"], normal)
        up_major, up_minor, _ = principal_in_plane(out["strain_upper"], normal)
        # the more strained surface is the one that would split
        outer = up_major >= lo_major
        out["major_strain"] = np.where(outer, up_major, lo_major)
        out["minor_strain"] = np.where(outer, up_minor, lo_minor)
        # membrane part: bending removed, so it pairs with the thickness change
        mid = 0.5 * (out["strain_lower"] + out["strain_upper"])
        mm, mn, mz = principal_in_plane(mid, normal)
        out["major_strain_membrane"] = mm
        out["minor_strain_membrane"] = mn
        out["thickness_strain"] = mz
    return out


def parse_glstat(path: Path) -> dict:
    """Last reported global energy state, plus the peak kinetic/internal ratio."""
    keys = {
        "time": "time",
        "kinetic energy": "kinetic_energy",
        "internal energy": "internal_energy",
        "hourglass energy": "hourglass_energy",
        "sliding interface energy": "sliding_energy",
        "external work": "external_work",
        "total energy / initial energy": "energy_ratio",
    }
    states, current = [], {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*([a-z /]+?)\.{2,}\s*(-?[\d.]+E?[+-]?\d*)\s*$", line, re.I)
        if not m:
            continue
        label, value = m.group(1).strip().lower(), float(m.group(2))
        if label == "time" and current:
            states.append(current)
            current = {}
        if label in keys:
            current[keys[label]] = value
    if current:
        states.append(current)
    if not states:
        return {}

    ke = np.array([s.get("kinetic_energy", 0.0) for s in states])
    ie = np.array([s.get("internal_energy", 0.0) for s in states])
    ratio = ke / np.maximum(ie, 1e-9)
    # Judge quasi-staticness only once real forming work is being done. Early in
    # the stroke internal energy is still near zero, so the ratio there says
    # nothing useful about the draw.
    window = ie > 0.25 * ie.max() if ie.max() > 0 else np.zeros(len(ie), bool)
    last = states[-1]
    return {
        **last,
        "peak_ke_over_ie": float(ratio[window].max()) if window.any() else 0.0,
        "final_ke_over_ie": float(ke[-1] / max(ie[-1], 1e-9)),
        "states": len(states),
    }


def parse_rbdout(path: Path) -> dict:
    """Final z displacement of each rigid body, keyed by LS-DYNA's body number."""
    if not path.exists():
        return {}
    body, last = None, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*rigid body\s+(\d+)\s*\Z", line.rstrip())
        if m:
            body = int(m.group(1))
        elif line.strip().startswith("displacements:") and body is not None:
            last[body] = float(line.split()[3])
            body = None
    return last


def binder_tracking(run_dir: Path, die_body: int = 3, binder_body: int = 4) -> float | None:
    """How far the binder ended up below the die, in mm.

    The binder is force-driven and free in z, so it should ride down with the
    die separated only by the sheet. A large negative value means the flange
    stopped being clamped and slid down between the tools.
    """
    disp = parse_rbdout(run_dir / "rbdout")
    if die_body not in disp or binder_body not in disp:
        return None
    return disp[binder_body] - disp[die_body]


def normal_termination(messag: Path) -> bool:
    if not messag.exists():
        return False
    return "N o r m a l" in messag.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------


def evaluate(run_dir: Path, params: dict, thinning_limit: float = 0.30,
             ke_ie_limit: float = 0.05, energy_ratio_band: float = 0.10,
             binder_slip_limit: float = 1.0) -> dict:
    """Physics and numerics acceptance checks for one run."""
    final = parse_dynain(run_dir / "dynain")
    glstat = parse_glstat(run_dir / "glstat")
    t0 = params["thickness"]
    thk = final["thickness"]

    checks = {
        "normal_termination": normal_termination(run_dir / "messag"),
        "max_thinning": float(1.0 - thk.min() / t0) if thk is not None else None,
        "max_thickening": float(thk.max() / t0 - 1.0) if thk is not None else None,
        "peak_ke_over_ie": glstat.get("peak_ke_over_ie"),
        "final_ke_over_ie": glstat.get("final_ke_over_ie"),
        "energy_ratio": glstat.get("energy_ratio"),
        "hourglass_over_internal": (
            glstat.get("hourglass_energy", 0.0) / max(glstat.get("internal_energy", 1e-9), 1e-9)
        ),
        "internal_energy": glstat.get("internal_energy"),
        "binder_below_die_mm": binder_tracking(run_dir),
    }

    reasons = []
    if not checks["normal_termination"]:
        reasons.append("solver did not reach normal termination")
    if checks["max_thinning"] is None:
        reasons.append("dynain carried no thickness data")
    elif checks["max_thinning"] > thinning_limit:
        reasons.append(f"max thinning {checks['max_thinning']:.1%} exceeds "
                       f"{thinning_limit:.0%} (split risk)")
    if checks["peak_ke_over_ie"] is not None and checks["peak_ke_over_ie"] > ke_ie_limit:
        reasons.append(f"kinetic/internal energy peaked at "
                       f"{checks['peak_ke_over_ie']:.1%}, not quasi-static")
    if checks["energy_ratio"] is not None and abs(checks["energy_ratio"] - 1.0) > energy_ratio_band:
        reasons.append(f"energy ratio {checks['energy_ratio']:.3f} outside "
                       f"1 +/- {energy_ratio_band}")
    if checks["hourglass_over_internal"] > 0.10:
        reasons.append(f"hourglass energy {checks['hourglass_over_internal']:.1%} of internal")
    slip = checks["binder_below_die_mm"]
    if slip is not None and abs(slip) > binder_slip_limit:
        reasons.append(f"binder ended {abs(slip):.2f} mm below the die: the flange "
                       f"was not clamped for the whole stroke")

    checks["accepted"] = not reasons
    checks["rejection_reasons"] = reasons
    return checks, final


def write_case(run_dir: Path, dataset_dir: Path, n_points: int = 10000,
               seed: int = 0, **limits) -> dict:
    params = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "build_report.json").read_text(encoding="utf-8"))
    checks, final = evaluate(run_dir, params, **limits)

    case_dir = dataset_dir / params["derived"]["case_name"]
    data_dir = case_dir / "Data"
    (data_dir / "Bauteil").mkdir(parents=True, exist_ok=True)

    # initial state: the tools, and the flat blank as a MESH sharing the final
    # part's node ordering, so node i maps initial -> final directly.
    geom = run_dir / "geometry"
    for tool, fname in TOOL_FILES.items():
        src = geom / f"{tool}.npz"
        if src.exists():
            (data_dir / fname).write_bytes(src.read_bytes())
    if "nodes_initial" in final:
        blank_src = np.load(geom / "blank.npz") if (geom / "blank.npz").exists() else None
        np.savez_compressed(
            data_dir / TOOL_FILES["blank"],
            points=blank_src["points"] if blank_src is not None else final["nodes_initial"],
            idx=blank_src["idx"] if blank_src is not None else np.zeros(0),
            nodes=final["nodes_initial"], quads=final["quads"])

    # final state: the deformed blank
    mesh = SurfaceMesh(final["nodes"], final["quads"])
    points, quad_idx = mesh.sample(n_points, seed)
    bhf = int(round(params["blankholder_force"]))
    fields = {
        "points": points,
        "idx": quad_idx,
        "nodes": final["nodes"],
        "quads": final["quads"],
        "thickness": final["thickness"] if final["thickness"] is not None else np.empty(0),
    }
    # per-element field data, present whenever the solver wrote it
    for key in ("nodes_initial", "displacement",
                "plastic_strain", "plastic_strain_ip", "stress",
                "strain_lower", "strain_upper", "major_strain", "minor_strain",
                "major_strain_membrane", "minor_strain_membrane", "thickness_strain"):
        if key in final:
            fields[key] = final[key]
    np.savez_compressed(data_dir / "Bauteil" / f"Tiefgezogenes_Bauteil_{bhf}.npz", **fields)

    summary = {
        "case_name": params["derived"]["case_name"],
        "blankholder_force_N": params["blankholder_force"],
        "parameters": params,
        "build": report,
        "checks": checks,
        "final_blank": {
            "nodes": int(len(final["nodes"])),
            "elements": int(len(final["quads"])),
            "bbox_min": final["nodes"].min(0).tolist(),
            "bbox_max": final["nodes"].max(0).tolist(),
        },
    }
    # One report per force: a case directory holds several runs that share tools
    # but not results, so a single case_report.json would lose all but the last.
    (case_dir / f"case_report_{bhf}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (case_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).resolve().parents[1] / "dataset")
    ap.add_argument("--points", type=int, default=10000)
    args = ap.parse_args()
    summary = write_case(args.run_dir, args.dataset, n_points=args.points)
    print(json.dumps({k: summary[k] for k in ("case_name", "checks", "final_blank")}, indent=2))
    raise SystemExit(0 if summary["checks"]["accepted"] else 1)


if __name__ == "__main__":
    main()
