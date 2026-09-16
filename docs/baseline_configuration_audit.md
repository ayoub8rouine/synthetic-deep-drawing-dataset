# Baseline configuration audit

Status: **the baseline passes.** A 25 mm draw at 200 kN blankholder force reaches
normal termination with the binder 0.03 mm from the die, the part 0.03 mm from
its geometric target, 16.7 percent max thinning, kinetic/internal energy at
1.7 percent, hourglass energy at 0.01 percent of internal, and an energy ratio of
1.0001.

This supersedes the earlier audit, which rejected a run whose kinetic energy was
1.2e10 against an internal energy of 6.5 -- that model had nothing in contact.

## What the geometry actually is

The earlier `geometry/generate_baseline.py` lofted a "crossmember" from nested
rounded rectangles using tuned coefficients. It bore no relation to the reference
dataset and its validity checks (watertight, manifold, non-degenerate) could not
detect that. It is superseded by `geometry/box_draw.py`.

The reference dataset is a rectangular box draw. Decoding its folder names and
measuring its point clouds gives, for all 228 cases:

| token | meaning | evidence |
| --- | --- | --- |
| `radii2` | plan corner radius | recovered to +0.001 mm mean error, 0.407 mm worst, over 228 cases |
| `radii1` | punch nose / die entry / roof fillet | punch top face ends at `a - radii1` |
| `cr` | clearance ratio | die wall minus punch wall = `cr * t` exactly |
| `delta` | draft as total lateral inset over `height` | predicted fillet tangent 165.64 mm vs measured 165.74 mm |
| `height` | draw depth | cavity depth and punch height |

`geometry/verify_reference.py` re-runs this comparison. Straight-region residual
over all 228 cases: punch 0.128 mm mean, die 0.087 mm, binder 0.000 mm. The
residual floor is the reference's own faceting -- its meshes cut ~16 mm chords
across the plan corner arcs and coarse chords across the R5 profile fillets, so
its sampled points sit inside the true surfaces.

## Adopted configuration

### Tools and blank

- Active forming faces only; no rear or side faces of closed solids as masters.
- Quad meshes built analytically and welded by construction. Zero non-manifold
  edges across all 228 generated tool sets; normal orientation is asserted at
  build time, not assumed.
- Blank normals up, die faces down, punch and binder faces up.
- Die cavity offset from the punch by `cr * t` plus the draft correction
  `tan(draft) * t/2`, which is what the reference geometry measures to.
- Blank at 4 mm, shell formulation 2, five through-thickness integration points.
- Tool shells are 0.1 mm. Contact offsets each surface by half its shell
  thickness, so 1 mm tools closing on a 1 mm blank leave a 0 mm gap at bottom
  dead centre and pinch the cup bottom.

### Contact

- `*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE` per blank/tool pair, blank slave.
- `*CONTROL_CONTACT` with `SHLTHK=1`, `THKCHG=1`, `ORIEN=1`, `IGNORE=1`.
- Friction 0.12, static equal to dynamic.
- The tools start in exact contact, so initial penetration is zero by
  construction rather than by tolerance.
- Rigid tools keep the real steel density. Lowering it to suppress rigid-body
  kinetic energy collapses the contact-stability time step: at 1/100 density the
  limit fell to 5.0e-8 s and mass scaling added 2.7x the blank's own mass. At
  steel density the limit is 5.0e-7 s and added mass is zero.

### Kinematics and loading

- Punch fixed. Die on `*BOUNDARY_PRESCRIBED_MOTION_RIGID` with `VAD=2`
  (displacement) and a cycloidal curve: zero velocity at both ends of the stroke
  and total travel exactly `height`.
- Binder on `*LOAD_RIGID_BODY`, free in z, with a smooth ramp over 15 percent of
  the stroke. Its mass is set explicitly; the peak inertial force is reported
  next to the blankholder force and sits near 8 percent of it.
- Peak die velocity 3000 mm/s. This is time scaling, not the physical press
  speed; the kinetic/internal energy ratio is checked afterwards rather than
  assumed.
- The punch continues below the forming depth so the binder always has punch
  wall beside it at bottom dead centre.

### Blankholder force

The reference labels of 15-40 kN give only 0.22 MPa on this flange. The flange is
then not clamped: it slides down between the tools, the binder follows, and the
part is ~4 mm deeper than the cavity. The sweep uses 100-400 kN (0.9-3.5 MPa).
This is recorded as `blankholder_pressure_mpa` in every build report.

### Material and output

- `*MAT_TRANSVERSELY_ANISOTROPIC_ELASTIC_PLASTIC` with a normal anisotropy
  ratio from the documented assumed r-values. The coefficients are an
  engineering assumption and are not StampFormer data.
- The hardening curve is extended past the last tabulated point with a fitted
  power law. Flat-lining at the last point drives premature localisation.
- Five integration points stored; `STRFLG=1`. `GLSTAT`, `MATSUM`, `RBDOUT`,
  `RCFORC` and `SLEOUT` all written.
- `*INTERFACE_SPRINGBACK_LSDYNA` writes the final blank as plain-text `dynain`,
  with deformed nodes and per-element thickness. No d3plot reader is needed.

## Acceptance

`postprocessing/extract_case.py` decides. A case enters the dataset only on
normal termination, max thinning below 30 percent, kinetic/internal energy below
5 percent once real forming work is under way, energy ratio within 1 +/- 0.10,
hourglass below 10 percent of internal, and the binder within 1 mm of the die.
The numbers are written to `case_report.json` whether the case passes or not.

Springback is not yet run. It would be a separate double-precision analysis after
tool removal, seeded from the same `dynain`.

## References

- Ansys LS-PrePost Metal Forming application documentation.
- Official LS-DYNA Deep Drawing with Adaptivity example.
- Official LS-DYNA Rectangular Cup Drawing anisotropic example.
