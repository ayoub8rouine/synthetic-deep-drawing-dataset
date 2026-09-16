# Implementation plan and validation gates

1. **Installation gate** — resolve LS-Run, solver, and LS-PrePost; run the tiny
   keyword deck; record executable versions and termination status.
2. **Geometry gate** — build one undercut-free crossmember-like punch surface,
   offset the die by sheet thickness plus forming clearance, construct binder and
   flat blank, then export STEP/STL. Check OpenCASCADE validity and mesh normals.
3. **Baseline model gate** — mesh the blank with shells and tools as rigid
   surfaces; assemble gravity, binder loading, stamping, and unloading/springback
   phases. Keep each node and element count below 128,000.
4. **Physics gate** — run one case only. Check normal termination, fatal messages,
   contact/penetration, distorted or unstable elements, kinetic/internal and
   hourglass/internal energy ratios, mass scaling, punch travel, strain, thinning,
   and springback completion. Produce machine-readable JSON plus plots/notebook.
5. **Sampling gate** — only after an acceptable baseline, create five Latin
   Hypercube geometry configurations. Generating or running a larger batch requires
   explicit user confirmation.

## Baseline assumptions requiring confirmation before the physics run

The requested process values are fixed initially: 1.6 mm thickness, 800 kN
blank-holder force, 0.12 Coulomb friction, and 250 mm/s punch speed. Geometry will
start at the midpoint of every requested range. The following materially affect
the physics and therefore will be documented and surfaced in configuration rather
than silently embedded: forming clearance convention, blank dimensions, punch
stroke, tool/blank mesh size, steel grade and anisotropic yield coefficients,
stress-strain curve, density, elastic constants, mass-scaling target, binder load
ramp, gravity settling time, and springback method/duration.

Material values used later will be engineering assumptions for a representative
sheet-forming steel, not values attributed to the StampFormer paper.
