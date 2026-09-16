"""Build a runnable LS-DYNA deep-drawing deck from `box_draw` geometry.

Modelled as a double-action inverted draw on one symmetry quarter:

    punch   fixed rigid tool (all DOF constrained)
    die     rigid, prescribed DISPLACEMENT downward by exactly `height`
    binder  rigid, force-controlled in z (the blankholder force), free to follow
    blank   deformable shell, symmetry constraints on x=0 and y=0

Everything starts in exact contact -- the die flange sits on z=+t/2, the binder
and punch faces on z=-t/2, and the blank midsurface is z=0 -- so there is no
closing gap to cross and no initial penetration to resolve.

Two deliberate numerical choices, both reported in the build report so they can
be checked rather than trusted:

* The stroke is driven by a cycloidal displacement curve, so velocity is zero at
  both ends of the stroke and the total travel is exactly `height`. Peak velocity
  is raised above the physical press speed (the usual explicit-forming time
  scaling); the resulting kinetic/internal energy ratio is checked afterwards.
* The rigid tools keep the real steel density. Lowering it to keep rigid-body
  kinetic energy out of the GLSTAT balance is tempting, but LS-DYNA derives the
  contact penalty stiffness from the master segments, so a light tool collapses
  the contact-stability time step and forces huge mass scaling. Because the
  binder is force-driven its mass does matter; it follows from the binder plate
  geometry, and the resulting inertial force is reported next to the blankholder
  force so the ratio can be checked. Blank energies are read from MATSUM (per
  material) rather than GLSTAT, which is dominated by the driven die.

Units: mm, s, tonne, N, MPa.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))

from dataclasses import replace  # noqa: E402

from box_draw import (  # noqa: E402
    DrawParams,
    SurfaceMesh,
    build_all,
    build_punch,
    write_pointcloud,
    write_stl,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "materials" / "assumed_forming_steel.json"

# part / section / material ids
PID = {"blank": 1, "punch": 2, "die": 3, "binder": 4}
NODE_OFFSET = {"blank": 0, "punch": 200000, "die": 400000, "binder": 600000}
LC_HARDENING, LC_DIE_MOTION, LC_BINDER_FORCE = 1, 2, 3
NSET_SYM_X, NSET_SYM_Y = 1, 2
PSET_BLANK = 1


# ---------------------------------------------------------------------------
# fixed-format helpers
# ---------------------------------------------------------------------------


def f10(value) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:10d}"
    if value is None or value == "":
        return " " * 10
    s = f"{float(value):.4g}"
    if len(s) > 10:
        s = f"{float(value):.3e}"
    return f"{s:>10}"


def card(*values) -> str:
    return "".join(f10(v) for v in values)


def curve_points(x: np.ndarray, y: np.ndarray) -> str:
    return "\n".join(f"{a:20.10g}{b:20.10g}" for a, b in zip(x, y))


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------


def hardening_curve(material: dict, max_strain: float = 1.0, n: int = 40) -> tuple:
    """Tabulated curve extended past the data with a fitted power law.

    A hardening curve that flat-lines at the last tabulated point drives
    premature strain localisation, so the tail is extrapolated as
    sigma = K (eps0 + eps)^n fitted to the last two supplied points.
    """
    data = np.asarray(material["plastic_strain_stress_curve"], dtype=float)
    eps, sig = data[:, 0], data[:, 1]
    if eps[-1] >= max_strain:
        return eps, sig
    e0 = 0.005
    (e1, s1), (e2, s2) = data[-2], data[-1]
    exponent = math.log(s2 / s1) / math.log((e2 + e0) / (e1 + e0))
    k = s2 / (e2 + e0) ** exponent
    tail = np.linspace(eps[-1], max_strain, n)[1:]
    return np.r_[eps, tail], np.r_[sig, k * (tail + e0) ** exponent]


def material_cards(material: dict, kind: str) -> str:
    ro = material["density_tonne_per_mm3"]
    e = material["youngs_modulus_mpa"]
    pr = material["poisson_ratio"]
    sigy = material["yield_stress_mpa"]
    etan = material["tangent_modulus_mpa"]
    if kind == "mat37":
        # the key was renamed once the anisotropic card was actually installed;
        # accept either so an older material file still loads
        r = material.get("anisotropy_r_values") or \
            material["anisotropy_r_values_reserved_for_hill48_upgrade"]
        r_bar = (r["r0"] + 2 * r["r45"] + r["r90"]) / 4.0
        return (
            "*MAT_TRANSVERSELY_ANISOTROPIC_ELASTIC_PLASTIC\n"
            "$#     mid        ro         e        pr      sigy      etan         r     hlcid\n"
            + card(PID["blank"], ro, e, pr, sigy, etan, r_bar, LC_HARDENING) + "\n"
        )
    if kind == "mat24":
        return (
            "*MAT_PIECEWISE_LINEAR_PLASTICITY\n"
            "$#     mid        ro         e        pr      sigy      etan      fail      tdel\n"
            + card(PID["blank"], ro, e, pr, sigy, etan, 0.0, 0.0) + "\n"
            "$#       c         p      lcss      lcsr        vp\n"
            + card(0.0, 0.0, LC_HARDENING, 0, 0.0) + "\n"
            + card(*([0.0] * 8)) + "\n"
            + card(*([0.0] * 8)) + "\n"
        )
    raise ValueError(f"unknown material kind {kind!r}")


def rigid_card(pid: int, ro: float, e: float, pr: float, con1: int, con2: int) -> str:
    return (
        "*MAT_RIGID\n"
        "$#     mid        ro         e        pr         n    couple         m     alias\n"
        + card(pid, ro, e, pr, 0.0, 0.0, 0.0, 0.0) + "\n"
        "$#     cmo      con1      con2\n"
        + card(1.0, con1, con2) + "\n"
        "$#  lco/a1        a2        a3        v1        v2        v3\n"
        + card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + "\n"
    )


# ---------------------------------------------------------------------------
# deck
# ---------------------------------------------------------------------------


def symmetry_node_sets(mesh: SurfaceMesh, tol: float = 1e-6) -> tuple:
    v = mesh.vertices
    return (
        np.flatnonzero(np.abs(v[:, 0]) < tol) + 1,
        np.flatnonzero(np.abs(v[:, 1]) < tol) + 1,
    )


def node_set_card(sid: int, ids: np.ndarray, title: str) -> str:
    out = ["*SET_NODE_LIST_TITLE", title, card(sid, 0.0, 0.0, 0.0)]
    for i in range(0, len(ids), 8):
        out.append("".join(f"{int(n):10d}" for n in ids[i : i + 8]))
    return "\n".join(out) + "\n"


def forming_contact(cid: int, ssid: int, msid: int, friction: float, title: str) -> str:
    return (
        "*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE_ID\n"
        + f"{cid:10d}{title:<70}\n"
        "$#    ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr\n"
        + card(ssid, msid, 3, 3, 0, 0, 1, 1) + "\n"
        "$#      fs        fd        dc        vc      vdc    penchk        bt        dt\n"
        + card(friction, friction, 0.0, 0.0, 20.0, 0, 0.0, 1.0e20) + "\n"
        "$#     sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf\n"
        + card(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0) + "\n"
    )


def build_deck(p: DrawParams, meshes: dict, material: dict, *,
               peak_velocity: float = 3000.0,
               friction: float = 0.12,
               target_dt: float = 4.0e-7,
               tool_shell_thickness: float = 0.1,
               binder_mass_tonne: float = 4.0e-4,
               material_kind: str = "mat37",
               d3plot_states: int = 12) -> tuple:
    """Return (deck_text, report)."""
    stroke = p.height
    end_time = 2.0 * stroke / peak_velocity

    blank, punch, die, binder = (meshes[k] for k in ("blank", "punch", "die", "binder"))

    # Tool shells are deliberately thin. The contact offsets each surface by half
    # its shell thickness, so 1 mm tools closing on a 1 mm blank leave a 0 mm gap
    # instead of a 1 mm one and pinch the cup bottom with ~100 kN at bottom dead
    # centre, which then drags the force-controlled binder past the die.
    steel_ro = material["density_tonne_per_mm3"]
    binder_area = float(binder.quad_areas().sum())
    # The binder is force-driven, so its mass is set explicitly rather than left
    # to fall out of the (now thin) shell thickness. A denser tool only raises
    # the contact-stability time step limit, so this direction is safe.
    binder_ro = binder_mass_tonne / (binder_area * tool_shell_thickness)
    # A quarter model carries a quarter of the full-tool force, a half model a
    # half. This was hard-coded to 4 and so under-loaded every half model.
    share = 4.0 if len(p.symmetry_planes()) == 2 else 2.0
    bhf_quarter = p.blankholder_force / share
    ramp_time = 0.15 * end_time
    inertial_force = binder_mass_tonne * peak_velocity / ramp_time

    eps, sig = hardening_curve(material)
    t = np.linspace(0.0, end_time, 101)
    disp = -stroke * (t / end_time - np.sin(2 * math.pi * t / end_time) / (2 * math.pi))

    sym_x, sym_y = symmetry_node_sets(blank)

    out = ["*KEYWORD", "*TITLE", f"{p.case_name()} bhf={p.blankholder_force:g}N"]

    out.append("*CONTROL_TERMINATION")
    out.append("$#  endtim    endcyc     dtmin    endeng    endmas     nosol")
    out.append(card(end_time, 0, 0.0, 0.0, 0.0, 0))

    out.append("*CONTROL_TIMESTEP")
    out.append("$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st")
    out.append(card(0.0, 0.9, 0, 0.0, -target_dt, 0, 0, 0))

    out.append("*CONTROL_SHELL")
    out.append("$# wrpang     esort     irnxx    istupd    theory       bwc     miter      proj")
    out.append(card(20.0, 1, -1, 1, 2, 2, 1, 0))

    out.append("*CONTROL_CONTACT")
    out.append("$# slsfac    rwpnal    islchk    shlthk    penopt    thkchg     orien   enmass")
    out.append(card(0.1, 0.0, 2, 1, 1, 1, 1, 0))
    out.append("$# usrstr    usrfrc     nsbcs    interm     xpene     ssthk      ecdt   tiedprj")
    out.append(card(0, 0, 0, 0, 4.0, 1, 0, 0))
    out.append("$#  sfric     dfric       edc       vfc        th     th_sf    pen_sf")
    out.append(card(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    out.append("$# ignore    frceng   skiprwg    outseg   spotstp   spotdel   spothin")
    out.append(card(1, 1, 0, 0, 0, 0, 0.0))

    out.append("*CONTROL_HOURGLASS")
    out.append("$#     ihq        qh")
    out.append(card(4, 0.05))

    out.append("*CONTROL_ENERGY")
    out.append("$#    hgen      rwen    slnten     rylen")
    out.append(card(2, 2, 2, 2))

    out.append("*CONTROL_ACCURACY")
    out.append("$#     osu       inn    pidosu      iacc")
    out.append(card(1, 4, 0, 1))

    out.append("*CONTROL_OUTPUT")
    out.append("$#   npopt    neecho    nrefup    iaccop     opifs    ipnint    ikedit    iflush")
    out.append(card(1, 3, 0, 0, 0.0, 0, 100, 0))

    out.append("*DATABASE_BINARY_D3PLOT")
    out.append(card(end_time / d3plot_states))
    for name in ("GLSTAT", "MATSUM", "RCFORC", "RBDOUT", "SLEOUT"):
        out.append(f"*DATABASE_{name}")
        out.append(card(end_time / 200.0))
    out.append("*DATABASE_EXTENT_BINARY")
    out.append("$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg")
    out.append(card(0, 0, -5, 1, 1, 1, 1, 1))
    out.append("$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat")
    out.append(card(0, 1, 0, 1, 1, 1, 2, 1))

    # ---- parts, sections, materials ----
    for name, pid in PID.items():
        out.append("*PART")
        out.append(f"{name}")
        out.append("$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid")
        out.append(card(pid, pid, pid, 0, 0, 0, 0, 0))

    out.append("*SECTION_SHELL")
    out.append("$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp")
    out.append(card(PID["blank"], 2, 0.8333, 5, 1, 0, 0, 1))
    out.append(card(*([p.thickness] * 4), 0.0, 0.0, 0.0, 0))
    for name in ("punch", "die", "binder"):
        out.append("*SECTION_SHELL")
        out.append(card(PID[name], 2, 0.8333, 2, 1, 0, 0, 1))
        out.append(card(*([tool_shell_thickness] * 4), 0.0, 0.0, 0.0, 0))

    out.append(material_cards(material, material_kind).rstrip())
    e, pr = material["youngs_modulus_mpa"], material["poisson_ratio"]
    # punch: fully fixed. die: z free (prescribed). binder: z free (force driven).
    out.append(rigid_card(PID["punch"], steel_ro, e, pr, 7, 7).rstrip())
    out.append(rigid_card(PID["die"], steel_ro, e, pr, 4, 7).rstrip())
    out.append(rigid_card(PID["binder"], binder_ro, e, pr, 4, 7).rstrip())

    # ---- curves ----
    out.append("*DEFINE_CURVE_TITLE")
    out.append("effective stress vs effective plastic strain")
    out.append(card(LC_HARDENING, 0, 1.0, 1.0, 0.0, 0.0, 0))
    out.append(curve_points(eps, sig))

    out.append("*DEFINE_CURVE_TITLE")
    out.append("die z displacement, cycloidal, zero velocity at both ends")
    out.append(card(LC_DIE_MOTION, 0, 1.0, 1.0, 0.0, 0.0, 0))
    out.append(curve_points(t, disp))

    out.append("*DEFINE_CURVE_TITLE")
    out.append("blankholder force ramp (quarter model)")
    out.append(card(LC_BINDER_FORCE, 0, 1.0, 1.0, 0.0, 0.0, 0))
    out.append(curve_points(
        np.array([0.0, ramp_time, end_time * 1.1]),
        np.array([0.0, bhf_quarter, bhf_quarter]),
    ))

    # ---- mesh ----
    from box_draw import keyword_blocks
    for name in ("blank", "punch", "die", "binder"):
        out.append(f"$ ---- {name} ----")
        out.append(keyword_blocks(meshes[name], PID[name],
                                  NODE_OFFSET[name], NODE_OFFSET[name]).rstrip())

    # ---- sets, constraints, loads ----
    if "x" in p.symmetry_planes():
        out.append(node_set_card(NSET_SYM_X, sym_x, "blank symmetry plane x=0").rstrip())
    if "y" in p.symmetry_planes():
        out.append(node_set_card(NSET_SYM_Y, sym_y, "blank symmetry plane y=0").rstrip())
    out.append("*SET_PART_LIST_TITLE")
    out.append("blank")
    out.append(card(PSET_BLANK, 0.0, 0.0, 0.0))
    out.append(card(PID["blank"], 0, 0, 0, 0, 0, 0, 0))

    planes = p.symmetry_planes()
    out.append("*BOUNDARY_SPC_SET")
    out.append("$#    nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz")
    if "x" in planes:
        out.append(card(NSET_SYM_X, 0, 1, 0, 0, 0, 1, 1))
    if "y" in planes:
        out.append(card(NSET_SYM_Y, 0, 0, 1, 0, 1, 0, 1))

    out.append("*BOUNDARY_PRESCRIBED_MOTION_RIGID_ID")
    out.append(f"{1:10d}{'die stroke':<70}")
    out.append("$#     pid       dof       vad      lcid        sf       vid     death     birth")
    out.append(card(PID["die"], 3, 2, LC_DIE_MOTION, 1.0, 0, 1.0e20, 0.0))

    out.append("*LOAD_RIGID_BODY")
    out.append("$#     pid       dof      lcid        sf       cid        m1        m2        m3")
    out.append(card(PID["binder"], 3, LC_BINDER_FORCE, 1.0, 0, 0, 0, 0))

    # ---- contacts ----
    for cid, tool in enumerate(("punch", "die", "binder"), start=1):
        out.append(forming_contact(cid, PID["blank"], PID[tool], friction,
                                   f"blank to {tool}").rstrip())

    # ---- final blank state, written as a plain-text keyword file ----
    out.append("*INTERFACE_SPRINGBACK_LSDYNA")
    out.append("$#    psid      nshv     ftype    ftensr    nthhsv     rflag   intstrn")
    out.append(card(PSET_BLANK, 0, 0, 0, 0, 0, 1))

    out.append("*END")

    counts = {k: {"nodes": len(m.vertices), "elements": len(m.quads)}
              for k, m in meshes.items()}
    total_nodes = sum(v["nodes"] for v in counts.values())
    total_elements = sum(v["elements"] for v in counts.values())
    report = {
        "case": p.case_name(),
        "blankholder_force_N_full": p.blankholder_force,
        "blankholder_force_N_on_model": bhf_quarter,
        "model_share_of_full_tool": 1.0 / share,
        # The physically meaningful quantity: force per unit clamped flange area.
        # Below roughly 0.5 MPa the flange is not held and slides between the
        # tools; see the `binder_below_die_mm` acceptance check.
        "blankholder_pressure_mpa": bhf_quarter / max(p.clamped_flange_area(), 1e-9),
        "stroke_mm": stroke,
        "end_time_s": end_time,
        "peak_die_velocity_mm_s": peak_velocity,
        "target_timestep_s": target_dt,
        "tool_shell_thickness_mm": tool_shell_thickness,
        "estimated_cycles": int(end_time / target_dt),
        "friction": friction,
        "material_card": material_kind,
        "binder": {
            "mass_tonne_quarter": binder_mass_tonne,
            "density_tonne_per_mm3": binder_ro,
            "ramp_time_s": ramp_time,
            "peak_inertial_force_N": inertial_force,
            "inertia_over_bhf": inertial_force / bhf_quarter,
        },
        "counts": counts,
        "total_nodes": total_nodes,
        "total_elements": total_elements,
        "within_student_limit": total_nodes < 128000 and total_elements < 128000,
        "symmetry_nodes": {"x0": int(len(sym_x)), "y0": int(len(sym_y))},
    }
    return "\n".join(out) + "\n", report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_case(p: DrawParams, out_dir: Path, material: dict, **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = build_all(p)
    deck, report = build_deck(p, meshes, material, **kwargs)
    (out_dir / "draw.k").write_text(deck, encoding="utf-8")
    (out_dir / "parameters.json").write_text(p.to_json(), encoding="utf-8")

    geom = out_dir / "geometry"
    geom.mkdir(exist_ok=True)
    # The punch's prismatic extension is tooling that never touches the sheet,
    # so the exported cloud uses the forming punch only, matching the reference.
    exported = dict(meshes)
    if p.punch_extension_ratio > 0:
        exported["punch"] = build_punch(replace(p, punch_extension_ratio=0.0))

    for name, mesh in exported.items():
        write_stl(mesh, geom / f"{name}.stl")
        write_pointcloud(mesh, geom / f"{name}.npz")
        report.setdefault("mesh_quality", {})[name] = {
            "free_edges": mesh.free_edges(),
            "non_manifold_edges": mesh.non_manifold_edges(),
            "min_edge_mm": mesh.min_edge_length(),
            "max_aspect_ratio": float(mesh.aspect_ratio().max()),
        }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=float, default=25.0)
    ap.add_argument("--draft", type=float, default=0.0)
    ap.add_argument("--corner-radius", type=float, default=20.0)
    ap.add_argument("--profile-radius", type=float, default=5.0)
    ap.add_argument("--clearance-ratio", type=float, default=1.1)
    ap.add_argument("--bhf", type=float, default=25000.0)
    ap.add_argument("--blank-element-size", type=float, default=4.0)
    ap.add_argument("--tool-element-size", type=float, default=6.0)
    ap.add_argument("--peak-velocity", type=float, default=3000.0)
    ap.add_argument("--material", choices=["mat37", "mat24"], default="mat37")
    args = ap.parse_args()

    p = DrawParams(
        height=args.height,
        draft=args.draft,
        corner_radius=args.corner_radius,
        profile_radius=args.profile_radius,
        clearance_ratio=args.clearance_ratio,
        blankholder_force=args.bhf,
        blank_element_size=args.blank_element_size,
        tool_element_size=args.tool_element_size,
    )
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    report = build_case(p, args.out, material,
                        peak_velocity=args.peak_velocity, material_kind=args.material)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
