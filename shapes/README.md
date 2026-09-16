# Cup shape families

A third dataset alongside the rectangular box (`../dataset`) and the U-channel
(`../u_shape/dataset`), covering many plan outlines rather than one.

The point of this folder is the **registry**: adding a shape family means adding
one function to `outlines.py`. The tool builder, deck builder, contacts,
acceptance checks, `dynain` reader and dataset writer are all outline-agnostic
and are reused unchanged.

## Families

| family | shapes it covers | symmetry |
| --- | --- | --- |
| `circle` | cylindrical cup | quarter |
| `ellipse` | elliptical cup | quarter |
| `rect` | square and rectangular cup | quarter |
| `polygon` | triangle, square, pentagon, hexagon, heptagon, octagon | quarter (even N), half (odd N) |
| `lobed` | cocoon (N=2), clover, rose, star: r = R + A cos(N theta) | quarter (even N), half (odd N) |
| `trapezoid` | four-sided, wider at one end | half |
| `cross` | plus / cross-shaped cup, four arms about a square centre | quarter |

Odd-count shapes keep only the x-axis mirror, so they use a half model. Nothing
needs a full model, which keeps runs at roughly 40-90 s.

`cross` is the only family with **concave** plan corners. Its four inner
junctions shrink under the outward die offset, so `inner_radius` is the
parameter that decides whether the shape can be tooled at all; it is sampled
well above the roughly 7 mm floor. Despite the concavity the shape is still
star-convex about the origin, so the same ray-cast sampling applies.

## How a shape becomes a mesh

There is no CAD step. Parameters go straight to a mesh:

```
outline(theta) at a fixed set of angles  ─┐
                                          ├─►  structured quad grid
profile(offset, z) at fixed stations     ─┘
```

Node `(i, j)` is `outline.points_at(profile_offset[j])[i]` lifted to
`profile_z[j]`. Watertight by construction, node count known in advance.

The only interface a family must provide is `points_at(offset)`, returning the
outline pushed out by a normal distance with a **fixed point count at fixed polar
angles** and the first and last points exactly on the symmetry axes. That is what
makes the swept surface a clean grid, and it makes the die an exact offset of the
punch.

Flat faces -- punch crown, cavity roof, die flange, binder -- are filled by
gmsh's planar mesher with every boundary node pinned transfinite, so they reuse
the swept edge exactly (verified: boundary node error 0.0, pure quads, and it
handles non-star-convex outlines such as a cross).

## Process window

Cases outside the single-draw window are rejected when they are built, not after
a wasted solver run. Published criteria:

| criterion | limit | source |
| --- | --- | --- |
| drawing ratio | `Db/Dp <= 2.0` | limiting draw ratio is 2.2 |
| reduction | `(Db-Dp)/Db <= 0.5` | standard |
| cup height | `H <= base diameter` | from `H = b/2 (DR^2 - 1)` at LDR |
| radial clearance | `~1.3 t` optimum; below `1.0 t` it irons | measured optimum |
| holding pressure | `0.015 x yield` = 3.15 MPa; 2.76 MPa measured | two sources agree |
| die profile radius | 5-35 t | tested range |
| punch profile radius | 3-33 t | tested range |

Plus two measured here, not taken from the literature:

**Wall draft >= 6 degrees.** Every U-channel case that split had a near-vertical
wall at one end; none of the 48 accepted ones did. A vertical wall gives the
material no path to feed in, so it stretches instead of drawing.

**Corner radius >= 0.45 x draw depth**, for faceted outlines. The published
limiting draw ratio is derived for *cylindrical* cups. On a faceted plan the
rounded corner draws like a much smaller cup, so its local draw ratio is far
higher than the nominal one and it splits while the flat sides are still
comfortable. Over 29 faceted cases the boundary was clean:

```
accepted   corner_radius / depth = 0.48 - 1.20
rejected   corner_radius / depth = 0.24 - 0.38
```

Before this rule, polygons passed 4 of 10 while circles, ellipses and clovers
passed 10 of 10. It is a good candidate to check against the polygonal-cup
literature, which should state it directly.

## Blankholder is a pressure, not a force

`blankholder_pressure` (MPa) is the input; the force follows from the clamped
flange area and the model's symmetry share. Force alone is meaningless across
shapes of different flange area -- which is exactly what made the reference
dataset's 15-40 kN look reasonable when it was really 0.22 MPa, an order of
magnitude too low to hold the flange.

## Use

```powershell
python .\shapes\pipeline.py --sample            # list the sampled cups
python .\shapes\pipeline.py --case 0            # build one, no solver
python .\shapes\pipeline.py --run               # generate the dataset
python .\shapes\pipeline.py --run --seed 7      # a further, different batch
```

Output layout and npz fields match the other two families exactly, so a single
dataloader reads all three.
