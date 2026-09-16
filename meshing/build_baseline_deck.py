"""Assemble the single LS-DYNA baseline from validated STL tool surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "simulations" / "baseline_midpoint"
GEOM = CASE / "geometry"


def fmt(values) -> str:
    return ",".join(str(v) for v in values)


def add_mesh(lines, mesh: trimesh.Trimesh, pid: int, node_start: int, elem_start: int, zshift: float):
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    vertices[:, 2] += zshift
    lines.append("*NODE")
    for i, xyz in enumerate(vertices, node_start):
        lines.append(fmt((i, f"{xyz[0]:.8g}", f"{xyz[1]:.8g}", f"{xyz[2]:.8g}")))
    lines.append("*ELEMENT_SHELL")
    for eid, face in enumerate(np.asarray(mesh.faces), elem_start):
        n = face + node_start
        lines.append(fmt((eid, pid, int(n[0]), int(n[1]), int(n[2]), int(n[2]))))
    return node_start + len(vertices), elem_start + len(mesh.faces)


def add_blank(lines, pid: int, node_start: int, elem_start: int, z: float, spacing: float):
    length, width = 420.0, 260.0
    nx, ny = round(length / spacing), round(width / spacing)
    xs, ys = np.linspace(-length / 2, length / 2, nx + 1), np.linspace(-width / 2, width / 2, ny + 1)
    lines.append("*NODE")
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid = node_start + j * (nx + 1) + i
            lines.append(fmt((nid, f"{x:.8g}", f"{y:.8g}", f"{z:.8g}")))
    lines.append("*ELEMENT_SHELL")
    eid = elem_start
    for j in range(ny):
        for i in range(nx):
            n1 = node_start + j * (nx + 1) + i
            n2, n3, n4 = n1 + 1, n1 + nx + 2, n1 + nx + 1
            lines.append(fmt((eid, pid, n1, n2, n3, n4)))
            eid += 1
    return node_start + (nx + 1) * (ny + 1), eid, (nx + 1) * (ny + 1), nx * ny


def main() -> None:
    cfg = json.loads((CASE / "baseline_config.json").read_text())
    mat = json.loads((ROOT / "materials" / "assumed_forming_steel.json").read_text())
    run = CASE / "run"
    run.mkdir(parents=True, exist_ok=True)
    lines = ["*KEYWORD", "*TITLE", "Single baseline deep-drawing engineering run"]
    lines += [
        "*CONTROL_TERMINATION", "0.5,0,0,0,0",
        "*CONTROL_TIMESTEP", f"0.0,0.9,0,0,{cfg['mass_scaling_target_dt_s']},0,0,0",
        "*CONTROL_ENERGY", "2,2,2,2",
        "*CONTROL_SHELL", "20,2,-1,1,2,2,1",
        "*DATABASE_BINARY_D3PLOT", "0.005",
        "*DATABASE_GLSTAT", "0.001",
        "*DATABASE_MATSUM", "0.001",
        "*DATABASE_RCFORC", "0.001",
        "*DATABASE_SLEOUT", "0.001",
        "*SECTION_SHELL", "1,16,0.833,5,1,0,0,1", "1.6,1.6,1.6,1.6",
        "*SECTION_SHELL", "2,2,0.833,2,1,0,0,0", "1.0,1.0,1.0,1.0",
        "*MAT_PIECEWISE_LINEAR_PLASTICITY",
        fmt((1, mat["density_tonne_per_mm3"], mat["youngs_modulus_mpa"], mat["poisson_ratio"], mat["yield_stress_mpa"], mat["tangent_modulus_mpa"], 0, 0)),
        "0,0,0,0,0,0,0,0",
        "0,0,0,0,0,0,0,0",
        "0,0,0,0,0,0,0,0",
        "*MAT_RIGID", "2,7.85e-9,210000,0.3,0,0,0,0", "1,7,7", "0,0,0,0,0,0,0,0",
        "*MAT_RIGID", "3,7.85e-9,210000,0.3,0,0,0,0", "1,4,7", "0,0,0,0,0,0,0,0",
        "*MAT_RIGID", "4,7.85e-9,210000,0.3,0,0,0,0", "1,4,7", "0,0,0,0,0,0,0,0",
        "*PART", "BLANK", "1,1,1,0,0,0,0,0",
        "*PART", "DIE", "2,2,2,0,0,0,0,0",
        "*PART", "PUNCH", "3,2,3,0,0,0,0,0",
        "*PART", "BINDER", "4,2,4,0,0,0,0,0",
    ]
    nid, eid = 1, 1
    nid, eid, blank_nodes, blank_elements = add_blank(lines, 1, nid, eid, cfg["tool_positions_mm"]["blank_z"], cfg["blank_mesh_mm"])
    counts = {"blank": {"nodes": blank_nodes, "elements": blank_elements}}
    for name, pid in (("die", 2), ("punch", 3), ("binder", 4)):
        mesh = trimesh.load_mesh(GEOM / f"{name}.stl", process=True)
        trimesh.repair.fix_normals(mesh, multibody=True)
        before_n, before_e = nid, eid
        nid, eid = add_mesh(lines, mesh, pid, nid, eid, cfg["tool_positions_mm"][f"{name}_z"])
        counts[name] = {"nodes": nid - before_n, "elements": eid - before_e}
    lines += [
        "*DEFINE_CURVE", "10,0,1,1,0,0", "0.0,0.0", "0.02,1.0", "0.45,1.0", "0.5,0.0",
        "*DEFINE_CURVE", "11,0,1,1,0,0", "0.0,0.0", "0.05,0.0", "0.0501,-250.0", "0.25,-250.0", "0.2501,250.0", "0.45,250.0", "0.4501,0.0", "0.5,0.0",
        "*LOAD_BODY_Z", "10,-9810.0",
        "*LOAD_RIGID_BODY", "4,3,10,-800000.0,0",
        "*BOUNDARY_PRESCRIBED_MOTION_RIGID", "3,3,0,11,1.0,0,1.0e28,0.0",
    ]
    for master in (2, 3, 4):
        lines += [
            "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE",
            f"1,{master},3,3,0,0,0,0",
            "0.12,0.12,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0",
            "2,0.1,0,1.025,2,2,0,1",
        ]
    lines += ["*END"]
    deck = run / "baseline.k"
    deck.write_text("\n".join(lines) + "\n", encoding="ascii")
    summary = {
        "deck": str(deck), "counts": counts, "total_nodes": nid - 1,
        "total_elements": eid - 1,
        "below_student_limit": nid - 1 < 128000 and eid - 1 < 128000,
        "material_status": mat["warning"],
    }
    (CASE / "model_build_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
