"""Tool set for a deep-drawn cup of any plan outline.

The geometry family is supplied by `outlines.py`; everything here is
outline-agnostic. Same conventions as `geometry/box_draw.py`, so the deck
builder, acceptance checks and dataset writer are reused unchanged:

    blank midsurface at z = 0, thickness t
    punch   fixed, forming face at z = -t/2, wall down, prismatic tail
    binder  flat ring at z = -t/2 outboard of the die opening
    die     flange face at z = +t/2, cavity rising to z = +t/2 + height
    motion  die and binder travel down by `height`; the punch is fixed

Only the symmetry sector is modelled (a quarter, or a half for odd-count
shapes), with the corresponding symmetry planes declared to the deck builder.

Surfaces are built two ways. Walls and fillets are swept: the outline is
re-evaluated at each profile station's offset, giving a structured quad grid
that welds by construction. The flat faces -- punch crown, cavity roof, die
flange, binder -- are filled with gmsh's planar mesher with every boundary node
pinned, so they reuse the swept edge exactly. That is a safe use of gmsh:
meshing a flat region bounded by a closed polyline is reliable, unlike the 3D
lofting and boolean cuts that made the original generator fail.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "geometry"))
sys.path.insert(0, str(HERE))

from box_draw import (  # noqa: E402
    SurfaceMesh, _grid_quads, keyword_blocks, write_pointcloud, write_stl,
)
import outlines as OL  # noqa: E402


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass
class CupParams:
    """One cup case. Lengths in mm, forces in N, angles in degrees."""

    outline: dict = field(default_factory=lambda: {"family": "circle", "radius": 70.0})

    height: float = 30.0
    draft_deg: float = 8.0
    punch_radius: float = 6.0        # nose fillet, typically 4-8 x thickness
    die_radius: float = 8.0          # entry fillet, typically 6-10 x thickness

    thickness: float = 1.0
    clearance_ratio: float = 1.3     # ~130% of t is the reported optimum

    draw_ratio: float = 1.7          # blank size / part size; LDR is 2.2
    tool_margin: float = 25.0        # tool footprint beyond the blank

    tool_element_size: float = 6.0
    blank_element_size: float = 5.0

    # Blankholder is specified as a PRESSURE, which is what the literature rules
    # are written in and what actually decides whether the flange is held:
    # p = 0.015 x yield strength (3.15 MPa for this 210 MPa steel), and an
    # independent measurement of 400 psi = 2.76 MPa for mild steel. The force
    # follows from the clamped flange area, so it stays right as shapes change.
    blankholder_pressure: float = 3.0
    punch_extension_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.thickness <= 0 or self.clearance_ratio < 1.0:
            raise ValueError("bad thickness or clearance ratio")
        if self.punch_radius <= self.thickness / 2:
            raise ValueError("punch_radius must exceed t/2")
        if self.die_radius <= self.thickness / 2 + self.clearance:
            raise ValueError("die_radius must exceed t/2 + clearance")
        if not 0.0 <= self.draft_deg < 45.0:
            raise ValueError("draft_deg must be in [0, 45)")
        if 2 * self.punch_radius > self.height or 2 * self.die_radius > self.height:
            raise ValueError("fillets do not fit in the draw depth")
        o = self.build_outline()
        if o.min_concave_radius <= self.die_flange_offset:
            raise ValueError(
                f"outline has a concave radius of {o.min_concave_radius:.1f} mm, "
                f"below the {self.die_flange_offset:.1f} mm the die offset needs")
        self.check_drawable()

    def check_drawable(self) -> None:
        """Reject cases outside the published single-draw process window.

        Sampling geometry blindly wastes solver time on parts that tear, and
        worse, puts non-physical results in the dataset. These are the standard
        criteria:

            drawing ratio     DR = Db/Dp <= 2.0   (limiting draw ratio 2.2)
            reduction         r = (Db-Dp)/Db <= 0.5
            cup height        H <= base diameter, from H = b/2 (DR^2 - 1)
        """
        if not 1.05 <= self.draw_ratio <= 2.0:
            raise ValueError(f"draw ratio {self.draw_ratio:.2f} outside 1.05-2.0")
        r = 1.0 - 1.0 / self.draw_ratio
        if r > 0.5:
            raise ValueError(f"reduction {r:.2f} exceeds 0.5")
        base = self.base_half_extents()
        if self.height > 2.0 * min(base):
            raise ValueError(
                f"draw depth {self.height:.0f} mm exceeds the base diameter "
                f"{2 * min(base):.0f} mm: not a single-draw part")

        # Corner criterion, measured here rather than taken from the literature.
        # The published limiting draw ratio is derived for CYLINDRICAL cups. On a
        # faceted plan the rounded corner draws like a much smaller cup, so its
        # local draw ratio is far higher than the nominal one and it splits while
        # the flat sides are still comfortable. Over 29 faceted cases the split
        # boundary sat cleanly at corner_radius / depth = 0.45:
        #     accepted  0.48 - 1.20
        #     rejected  0.24 - 0.38
        corner = self.outline.get("corner_radius")
        if corner is not None and corner < 0.45 * self.height:
            raise ValueError(
                f"corner radius {corner:.1f} mm is below 0.45 x depth "
                f"({0.45 * self.height:.1f} mm): the corner will split first")

    # ---- derived ---------------------------------------------------------

    @property
    def clearance(self) -> float:
        return self.clearance_ratio * self.thickness

    @property
    def tan_draft(self) -> float:
        return math.tan(math.radians(self.draft_deg))

    @property
    def z_blank_top(self) -> float:
        return self.thickness / 2.0

    @property
    def z_blank_bottom(self) -> float:
        return -self.thickness / 2.0

    @property
    def die_offset_at_flange(self) -> float:
        return self.clearance + self.tan_draft * self.thickness / 2.0

    @property
    def die_flange_offset(self) -> float:
        """Offset where the die entry fillet meets the flat flange face."""
        r, tan = self.die_radius, self.tan_draft
        return self.die_offset_at_flange - tan * r + r * math.hypot(1.0, tan)

    def build_outline(self) -> OL.Outline:
        return OL.build(self.outline, self.tool_element_size)

    def part_size(self) -> float:
        return self.build_outline().max_radius()

    def base_half_extents(self) -> tuple:
        """Half-extents of the part footprint in x and y."""
        pts = self.build_outline().points
        return (float(np.abs(pts[:, 0]).max()), float(np.abs(pts[:, 1]).max()))

    def blank_half_dims(self) -> tuple:
        """Rectangular blank, sized so the local draw ratio matches on each axis."""
        rx, ry = self.base_half_extents()
        return self.draw_ratio * rx, self.draw_ratio * ry

    @property
    def blankholder_force(self) -> float:
        """Full-tool force implied by the specified pressure."""
        share = 4.0 if self.build_outline().symmetry == "quarter" else 2.0
        return self.blankholder_pressure * self.clamped_flange_area() * share

    def tool_half_dims(self) -> tuple:
        bx, by = self.blank_half_dims()
        return bx + self.tool_margin, by + self.tool_margin

    def symmetry_planes(self) -> tuple:
        return ("x", "y") if self.build_outline().symmetry == "quarter" else ("y",)

    def clamped_flange_area(self) -> float:
        """Blank area outboard of the die opening: what the binder clamps."""
        o = self.build_outline()
        bx, by = self.blank_half_dims()
        blank = bx * by if o.symmetry == "quarter" else 2 * bx * by
        pts = o.points_at(self.die_flange_offset)
        # polygon area of the sector, doubled/quadrupled back to the same basis
        area = 0.5 * abs(np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]))
        opening = area if o.symmetry == "quarter" else 2 * area
        return max(blank - opening, 1e-9)

    def case_name(self) -> str:
        o = self.build_outline()
        return (f"{o.name}_h{self.height:g}_draft{self.draft_deg:g}"
                f"_rp{self.punch_radius:g}_rd{self.die_radius:g}"
                f"_dr{self.draw_ratio:g}")

    def to_json(self) -> str:
        d = asdict(self)
        # blankholder_force is a property derived from pressure, so asdict() does
        # not emit it; the dataset writer reads it from here by name.
        d["blankholder_force"] = self.blankholder_force
        o = self.build_outline()
        bx, by = self.blank_half_dims()
        d["derived"] = {
            "outline_name": o.name,
            "symmetry": o.symmetry,
            "min_concave_radius_mm": (None if math.isinf(o.min_concave_radius)
                                      else o.min_concave_radius),
            "clearance_mm": self.clearance,
            "stroke_mm": self.height,
            "part_max_radius_mm": o.max_radius(),
            "blank_half_dims_mm": [bx, by],
            "clamped_flange_area_mm2": self.clamped_flange_area(),
            "blankholder_force_N_full_tool": self.blankholder_force,
            "reduction": 1.0 - 1.0 / self.draw_ratio,
            "height_over_base_diameter": self.height / (2 * min(self.base_half_extents())),
            "case_name": self.case_name(),
        }
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------------------
# profiles: (offset, z) generators, identical in form to the box family
# ---------------------------------------------------------------------------


def punch_profile(p: CupParams, size: float) -> tuple:
    """Nose fillet, drafted wall, prismatic tail. Returns (profile, crown offset)."""
    t, r, tan = p.thickness, p.punch_radius, p.tan_draft
    sec = math.hypot(1.0, tan)
    z_top = -t / 2.0
    z_c = z_top - r
    s_c = tan * r - p.height * tan - r * sec
    psi_end = 0.5 * math.pi - math.atan(tan)
    n = max(3, int(round(psi_end * r / size)))
    psi = np.linspace(0.0, psi_end, n + 1)
    fillet = np.column_stack([s_c + r * np.sin(psi), z_c + r * np.cos(psi)])

    z_bot = -(t / 2.0 + p.height)
    m = max(2, int(round(abs(fillet[-1, 1] - z_bot) / size)))
    zw = np.linspace(fillet[-1, 1], z_bot, m + 1)[1:]
    wall = np.column_stack([-tan * (zw + t / 2.0 + p.height), zw])

    ext = p.height * p.punch_extension_ratio
    tail = np.empty((0, 2))
    if ext > 0:
        k = max(2, int(round(ext / size)))
        ze = np.linspace(z_bot, z_bot - ext, k + 1)[1:]
        tail = np.column_stack([np.zeros_like(ze), ze])
    return np.vstack([fillet, wall, tail]), s_c


def die_profile(p: CupParams, size: float) -> tuple:
    """Entry fillet, cavity wall, roof fillet. Returns (profile, flange, roof)."""
    t, tan = p.thickness, p.tan_draft
    rd, rp = p.die_radius, p.punch_radius + p.thickness / 2 + p.clearance
    sec = math.hypot(1.0, tan)
    s0 = p.die_offset_at_flange
    z_flange, z_roof = t / 2.0, t / 2.0 + p.height
    psi_end = 0.5 * math.pi - math.atan(tan)

    n = max(3, int(round(psi_end * rd / size)))
    psi = np.linspace(0.0, psi_end, n + 1)
    s_ce = s0 - tan * rd + rd * sec
    entry = np.column_stack([s_ce - rd * np.sin(psi), z_flange + rd - rd * np.cos(psi)])

    m = max(3, int(round(psi_end * rp / size)))
    psi2 = np.linspace(0.0, psi_end, m + 1)
    z_cr = z_roof - rp
    s_cr = s0 + tan * t / 2.0 - tan * z_cr - rp * sec
    roof = np.column_stack([s_cr + rp * np.sin(psi2), z_cr + rp * np.cos(psi2)])[::-1]

    if roof[0, 1] <= entry[-1, 1]:
        raise ValueError("die fillets overlap: draw depth too small for the radii")
    k = max(2, int(round((roof[0, 1] - entry[-1, 1]) / size)))
    zw = np.linspace(entry[-1, 1], roof[0, 1], k + 1)[1:-1]
    wall = np.column_stack([s0 + tan * t / 2.0 - tan * zw, zw])
    return np.vstack([entry, wall, roof]), s_ce, s_cr


# ---------------------------------------------------------------------------
# meshing
# ---------------------------------------------------------------------------


def sweep(outline: OL.Outline, profile: np.ndarray, flip: bool = False) -> SurfaceMesh:
    """Sweep a (offset, z) profile along the outline into a structured grid."""
    nu, nv = outline.n_points, len(profile)
    verts = np.empty((nu * nv, 3))
    for j, (s, z) in enumerate(profile):
        verts[j::nv, :2] = outline.points_at(s)
        verts[j::nv, 2] = z
    return SurfaceMesh(verts, _grid_quads(nu, nv, flip))


def _outer_ray(half_x: float, half_y: float, theta: np.ndarray) -> np.ndarray:
    """The tool's bounding rectangle sampled at the outline's polar angles."""
    c, s = np.cos(theta), np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.minimum(np.where(np.abs(c) > 1e-12, half_x / np.abs(c), np.inf),
                       np.where(np.abs(s) > 1e-12, half_y / np.abs(s), np.inf))
    return np.column_stack([r * c, r * s])


def ring(outline: OL.Outline, offset: float, half_x: float, half_y: float,
         z: float, size: float, flip: bool = False) -> SurfaceMesh:
    """Flat annulus between the outline and the tool's outer rectangle."""
    theta = OL._angles(outline.symmetry, outline.n_points)
    inner = outline.points_at(offset)
    outer = _outer_ray(half_x, half_y, theta)
    span = float(np.max(np.linalg.norm(outer - inner, axis=1)))
    nv = max(2, int(round(span / size)) + 1)
    w = np.linspace(0.0, 1.0, nv)[None, :, None]
    pts = (1 - w) * inner[:, None, :] + w * outer[:, None, :]
    verts = np.empty((outline.n_points * nv, 3))
    verts[:, :2] = pts.reshape(-1, 2)
    verts[:, 2] = z
    return SurfaceMesh(verts, _grid_quads(outline.n_points, nv, flip))


def fill_sector(boundary: np.ndarray, z: float, size: float,
                flip: bool = False) -> SurfaceMesh:
    """Mesh the sector interior, reusing the boundary nodes exactly.

    The loop is the outline plus the two symmetry edges back through the
    origin. Every boundary segment is made transfinite with two nodes, which
    pins the outline nodes so this face welds to the swept wall without a crack.
    """
    import gmsh

    axis_a = _densify(np.array([[0.0, 0.0], boundary[0]]), size)[:-1]
    axis_b = _densify(np.array([boundary[-1], [0.0, 0.0]]), size)[1:-1]
    loop = np.vstack([axis_a, boundary, axis_b])

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sector")
        tags = [gmsh.model.geo.addPoint(x, y, z, size) for x, y in loop]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)])
                 for i in range(len(tags))]
        for line in lines:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)
        surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.6)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(2)
        _, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        remap = {int(t): i for i, t in enumerate(node_tags)}
        quads = []
        etypes, _, econn = gmsh.model.mesh.getElements(2)
        for et, conn in zip(etypes, econn):
            k = {2: 3, 3: 4}.get(int(et))
            if k is None:
                continue
            block = np.asarray(conn, dtype=np.int64).reshape(-1, k)
            idx = np.vectorize(remap.get)(block)
            quads.append(idx if k == 4 else np.column_stack([idx, idx[:, -1]]))
        quads = np.vstack(quads)
    finally:
        gmsh.finalize()
    coords[:, 2] = z
    return SurfaceMesh(coords, quads[:, ::-1].copy() if flip else quads)


def _densify(points: np.ndarray, size: float) -> np.ndarray:
    out = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        n = max(1, int(round(np.linalg.norm(b - a) / size)))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(points[-1])
    return np.asarray(out)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def _orient(mesh: SurfaceMesh, mode: str, sign: int, what: str) -> SurfaceMesh:
    """Flip a patch so its normals face the sheet, then verify.

    Winding depends on which way an outline happens to be traversed, and that
    differs between families, so the direction is measured rather than assumed.
    `mode` is "z" for flat faces or "radial" for swept walls, where the test is
    whether the normal points away from (+1) or toward (-1) the axis.
    """
    n, a = mesh.quad_normals(), mesh.quad_areas()
    if mode == "z":
        score = float(np.sum(n[:, 2] * a) / a.sum())
    else:
        centre = mesh.vertices[mesh.quads].mean(axis=1)[:, :2]
        radial = centre / np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-9)
        score = float(np.sum(np.sum(n[:, :2] * radial, axis=1) * a) / a.sum())
    if score * sign < 0:
        mesh = SurfaceMesh(mesh.vertices, mesh.quads[:, ::-1].copy(), mesh.groups)
        score = -score
    if abs(score) < 1e-6:
        raise AssertionError(f"{what}: normals are ambiguous ({mode} score {score:+.3g})")
    return mesh


def build_punch(p: CupParams) -> SurfaceMesh:
    o = p.build_outline()
    size = p.tool_element_size
    profile, s_crown = punch_profile(p, size)
    crown = _orient(fill_sector(o.points_at(s_crown), p.z_blank_bottom, size),
                    "z", +1, "punch crown")
    wall = _orient(sweep(o, profile), "radial", +1, "punch wall")
    mesh = (crown + wall).weld()
    mesh.groups = {"punch": np.arange(len(mesh.quads))}
    return mesh


def build_die(p: CupParams) -> SurfaceMesh:
    o = p.build_outline()
    size = p.tool_element_size
    hx, hy = p.tool_half_dims()
    profile, s_flange, s_roof = die_profile(p, size)
    flange = _orient(ring(o, s_flange, hx, hy, p.z_blank_top, size),
                     "z", -1, "die flange")
    cavity = _orient(sweep(o, profile), "radial", -1, "die cavity")
    roof = _orient(fill_sector(o.points_at(s_roof), p.z_blank_top + p.height, size),
                   "z", -1, "die roof")
    mesh = (flange + cavity + roof).weld()
    mesh.groups = {"die": np.arange(len(mesh.quads))}
    return mesh


def build_binder(p: CupParams) -> SurfaceMesh:
    o = p.build_outline()
    hx, hy = p.tool_half_dims()
    mesh = _orient(ring(o, p.die_flange_offset, hx, hy, p.z_blank_bottom,
                        p.tool_element_size), "z", +1, "binder").weld()
    mesh.groups = {"binder": np.arange(len(mesh.quads))}
    return mesh


def build_blank(p: CupParams) -> SurfaceMesh:
    o = p.build_outline()
    bx, by = p.blank_half_dims()
    x0 = 0.0 if o.symmetry == "quarter" else -bx
    nx = max(2, int(round((bx - x0) / p.blank_element_size)))
    ny = max(2, int(round(by / p.blank_element_size)))
    x = np.linspace(x0, bx, nx + 1)
    y = np.linspace(0.0, by, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    mesh = _orient(SurfaceMesh(verts, _grid_quads(nx + 1, ny + 1)), "z", +1, "blank")
    mesh.groups = {"blank": np.arange(len(mesh.quads))}
    return mesh


def build_all(p: CupParams) -> dict:
    return {"punch": build_punch(p), "die": build_die(p),
            "binder": build_binder(p), "blank": build_blank(p)}
