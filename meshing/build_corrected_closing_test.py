"""Build a short, contact-focused binder-closing test for the corrected baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from build_baseline_deck import add_blank, add_mesh, fmt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulations" / "baseline_midpoint" / "geometry"
CASE = ROOT / "simulations" / "baseline_corrected"


def active_surface(name: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(SOURCE / f"{name}.stl", process=True)
    trimesh.repair.fix_normals(mesh, multibody=True)
    c, n = mesh.triangles_center, mesh.face_normals
    if name == "binder":
        keep = (n[:, 2] < -0.9) & (c[:, 2] < 4.01)
    elif name == "punch":
        # Bottom, drafted walls and end faces; omit only the rear/top tool surface.
        keep = ~((n[:, 2] > 0.9) & (c[:, 2] > 51.99))
    elif name == "die":
        on_flange = (n[:, 2] > 0.9) & (c[:, 2] > 51.99)
        in_cavity = (np.abs(c[:, 0]) < 170.0) & (np.abs(c[:, 1]) < 105.0) & (c[:, 2] > -1.0)
        keep = on_flange | in_cavity
    else:
        raise ValueError(name)
    selected = mesh.submesh([np.flatnonzero(keep)], append=True, repair=False)
    selected.remove_unreferenced_vertices()
    return selected


def main() -> None:
    run = CASE / "closing_test"
    surf_dir = CASE / "active_surfaces"
    run.mkdir(parents=True, exist_ok=True)
    surf_dir.mkdir(parents=True, exist_ok=True)

    surfaces = {name: active_surface(name) for name in ("die", "punch", "binder")}
    for name, mesh in surfaces.items():
        mesh.export(surf_dir / f"{name}_forming.stl")

    lines = [
        "*KEYWORD", "*TITLE", "Corrected baseline - binder closing test only",
        "*CONTROL_TERMINATION", "0.06,0,0,0,0",
        "*CONTROL_TIMESTEP", "0,0.9,0,0,-5e-7,0,0,0",
        # SLSFAC=0.0095 is a slight conservative reduction from the official
        # example's 0.01; R16.1 then reports contact stability above the natural dt.
        "*CONTROL_CONTACT", "0.0095,0,1,2,0,2,2", "0,0,0,0,4.0,0,0,0",
        "*CONTROL_ENERGY", "2,1,2,0",
        "*CONTROL_OUTPUT", "1,3,1,1,0,0,0",
        "*CONTROL_SHELL", "20,1,-1,1,2,2,1",
        "*DATABASE_BINARY_D3PLOT", "0.001",
        "*DATABASE_GLSTAT", "0.0002",
        "*DATABASE_MATSUM", "0.0002",
        "*DATABASE_RBDOUT", "0.0002",
        "*DATABASE_RCFORC", "0.0002",
        "*DATABASE_SLEOUT", "0.0002",
        "*SECTION_SHELL", "1,2,1.0,5,1,0,0,1", "1.6,1.6,1.6,1.6,0",
        "*SECTION_SHELL", "2,2,1.0,3,1,0,0,0", "0.1,0.1,0.1,0.1,0",
        "*MAT_TRANSVERSELY_ANISOTROPIC_ELASTIC_PLASTIC",
        "1,7.85e-9,210000,0.3,210,1400,1.5,0",
        "*MAT_RIGID", "2,7.85e-9,210000,0.3,0,0,0,0", "1,7,7", "0,0,0,0,0,0,0,0",
        "*MAT_RIGID", "3,7.85e-9,210000,0.3,0,0,0,0", "1,7,7", "0,0,0,0,0,0,0,0",
        "*MAT_RIGID", "4,7.85e-9,210000,0.3,0,0,0,0", "1,4,7", "0,0,0,0,0,0,0,0",
        "*PART", "BLANK", "1,1,1,0,0,0,0,0",
        "*PART", "DIE_FORMING_SURFACE", "2,2,2,0,0,0,0,0",
        "*PART", "PUNCH_FORMING_SURFACE", "3,2,3,0,0,0,0,0",
        "*PART_INERTIA", "BINDER_FORMING_SURFACE",
        "4,2,4,0,0,0,0,0",
        "0,0,54.2,1.0,0,0",
        "5633.333,0,0,14700,0,20333.333",
        "0,0,0,0,0,0",
    ]
    nid, eid = 1, 1
    nid, eid, bn, be = add_blank(lines, 1, nid, eid, 53.2, 5.0)
    counts = {"blank": {"nodes": bn, "elements": be}}
    shifts = {"die": 0.0, "punch": 55.0, "binder": 50.2}  # binder bottom=54.2
    for name, pid in (("die", 2), ("punch", 3), ("binder", 4)):
        n0, e0 = nid, eid
        nid, eid = add_mesh(lines, surfaces[name], pid, nid, eid, shifts[name])
        counts[name] = {"nodes": nid - n0, "elements": eid - e0}
    lines += [
        "*DEFINE_CURVE", "10,0,1,1,0,0", "0,1", "0.06,1",
        "*DEFINE_CURVE", "20,0,1,1,0,0", "0,0", "0.02,0", "0.04,1", "0.06,1",
        # Close the binder to first contact with a half-cosine displacement
        # history.  Its velocity is zero at both endpoints; the constraint dies
        # at 0.02 s, when the independently ramped holding force takes over.
        "*DEFINE_CURVE", "40,0,1,1,0,0",
        "0,0", "0.002,-0.003671", "0.004,-0.014324", "0.006,-0.030916",
        "0.008,-0.051824", "0.010,-0.075", "0.012,-0.098176",
        "0.014,-0.119084", "0.016,-0.135676", "0.018,-0.146329", "0.020,-0.15",
        "*BOUNDARY_PRESCRIBED_MOTION_RIGID", "4,3,2,40,1,0,0.02,0",
        # Restrict gravity to the deformable sheet.  Without this selector,
        # *LOAD_BODY_Z also accelerates the binder's numerical inertia.
        "*SET_PART_LIST", "101,0,0,0,0", "1,0,0,0,0,0,0,0",
        "*LOAD_BODY_PARTS", "101",
        "*LOAD_BODY_Z", "10,-9810",
        "*LOAD_RIGID_BODY", "4,3,20,-800000,0",
    ]
    for master in (2, 3, 4):
        lines += [
            "*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE",
            f"1,{master},3,3,0,0,0,0",
            # VDC=20 follows the official forming example to suppress local
            # contact chatter while preserving the specified Coulomb friction.
            "0.12,0.12,0,0,20,1,0,0",
            "1,1,0.8,1,1,1,1,1",
        ]
    lines += ["*END"]
    deck = run / "closing_test.k"
    deck.write_text("\n".join(lines) + "\n", encoding="ascii")
    report = {
        "status": "not yet validated",
        "purpose": "gravity and smooth binder-force closing only; punch fixed",
        "total_nodes": nid - 1,
        "total_elements": eid - 1,
        "counts": counts,
        "initial_positions_mm": {"die_top": 52.0, "blank_mid": 53.2, "binder_bottom": 54.2, "punch_bottom": 55.0},
        "nominal_surface_gaps_mm": {"die_to_blank_bottom": 0.35, "blank_top_to_binder": 0.15, "blank_top_to_punch": 0.95},
        "binder_force_n": 800000,
        "binder_closing": {"prescribed_displacement_mm": -0.15, "end_time_s": 0.02, "end_velocity_mm_s": 0.0},
        "binder_numerical_inertia": {"translational_mass_tonne": 1.0, "purpose": "documented quasi-static force-control stabilizer; sensitivity check required"},
        "material": "MAT_037 engineering assumption, R=1.5; not StampFormer data",
    }
    (CASE / "closing_test_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
