"""Parametric rectangular-box deep-drawing geometry.

This module replaces the loft-based ``generate_baseline.py`` construction. Every
surface here is built analytically from closed-form profiles and swept along a
rounded-rectangle plan outline, so there is no CAD boolean, no ``addThruSections``
and no meshing heuristic that can silently produce an invalid tool.

Convention (verified against the reference dataset in the sibling
``Synthetic Deep Drawing Dataset/Dataset`` folder)
--------------------------------------------------------------------------
Double-action inverted draw, modelled on one symmetry quarter (x >= 0, y >= 0):

  * blank midsurface at z = 0, thickness ``t``   -> blank spans [-t/2, +t/2]
  * punch  : fixed lower male tool, top face at z = -t/2, body extending down
  * binder : lower blankholder plate, top face at z = -t/2, a flat frame
  * die    : upper female tool, flange face at z = +t/2, cavity extending up
  * motion : die + binder travel DOWN by ``height``; the punch stays fixed, so
             the cup forms in the +z direction relative to the flange.

At the end of the stroke the flange therefore sits near z = -height and the cup
bottom near z = 0, which is exactly how the reference ``Bauteil`` clouds are
stored: tools at their initial position, part at its final position.

Plan geometry is a rounded rectangle: half-lengths (a, b) with plan corner
radius ``corner_radius`` (the ``radii2`` parameter of the reference dataset).
Offsetting a plan outline by ``s`` grows the half-lengths to (a+s, b+s) and
keeps the corner radius fixed. This matches the reference tools, where the
punch and the die cavity both use corner radius ``radii2``.

Profile geometry uses a single fillet radius ``profile_radius`` (``radii1``)
for the punch nose, the die entry and the die cavity roof, plus a draft given
as ``draft``: the total lateral inset accumulated over the full draw height
(the ``delta`` parameter of the reference dataset).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass
class DrawParams:
    """One deep-drawing case. Lengths in mm, forces in N."""

    # plan geometry of the punch at its widest section (the base of the wall)
    punch_half_length: float = 173.90  # a
    punch_half_width: float = 98.90    # b
    corner_radius: float = 20.0        # radii2, plan corner radius
    profile_radius: float = 5.0        # radii1, punch nose / die entry / roof

    # process geometry
    height: float = 25.0               # draw depth
    draft: float = 0.0                 # delta, total lateral inset over `height`
    thickness: float = 1.0             # blank thickness t
    clearance_ratio: float = 1.1       # cr, die gap = cr * t

    # tool footprint (flange face of the die, binder plate)
    tool_half_length: float = 280.0
    tool_half_width: float = 230.0

    # blank footprint (quarter)
    blank_half_length: float = 250.0
    blank_half_width: float = 190.0

    # discretisation
    tool_element_size: float = 6.0
    blank_element_size: float = 4.0

    # process
    blankholder_force: float = 25000.0  # N on the full (4x) tool
    punch_velocity: float = 250.0       # mm/s plateau

    # Where the blankholder opening sits.
    #   "die_flange" -- flush with the die flange face, so the sheet is clamped
    #                   on both sides over the whole flange. This is how real
    #                   tooling mates and it is the default.
    #   "cavity"     -- flush with the die cavity wall, as in the reference
    #                   dataset. That leaves an annulus one entry radius wide
    #                   where the binder supports the sheet but the die does
    #                   not, and draw tension then bends the sheet over the
    #                   binder's inner edge and pries the flange off the die.
    binder_opening: str = "die_flange"

    # How far the punch wall continues below the forming depth, as a multiple of
    # `height`. The binder descends by `height`, so a punch that stops exactly at
    # the forming depth leaves the binder level with the punch's bottom edge at
    # bottom dead centre: the sheet spanning punch wall to binder then has no
    # tool beneath it and sags several mm, dragging the whole flange down with
    # it. Real tooling mounts the punch on the bed and the binder slides down
    # around it. The extension is prismatic (no extra draft) so it cannot grow
    # wider than the binder opening. It is tool only -- it never touches the
    # sheet, and it is excluded from the exported punch point cloud.
    punch_extension_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.corner_radius <= 0:
            raise ValueError("corner_radius must be > 0")
        if self.corner_radius > min(self.punch_half_length, self.punch_half_width):
            raise ValueError("corner_radius exceeds the punch half-dimensions")
        if self.profile_radius <= 0 or 2 * self.profile_radius > self.height:
            raise ValueError("profile_radius must be > 0 and <= height/2")
        if self.thickness <= 0:
            raise ValueError("thickness must be > 0")
        if self.clearance_ratio < 1.0:
            raise ValueError("clearance_ratio must be >= 1.0 (die gap >= thickness)")
        if self.blank_half_length <= self.punch_half_length:
            raise ValueError("blank must overhang the punch to form a flange")
        if self.tool_half_length <= self.blank_half_length:
            raise ValueError("tool footprint must contain the blank")
        if self.draft < 0:
            raise ValueError("draft must be >= 0")
        if self.draft >= self.punch_half_width - self.corner_radius:
            raise ValueError("draft too large for the punch cross-section")

    # ---- derived ---------------------------------------------------------

    @property
    def clearance(self) -> float:
        """Lateral die gap on one side."""
        return self.clearance_ratio * self.thickness

    @property
    def tan_draft(self) -> float:
        return self.draft / self.height

    @property
    def draft_angle_deg(self) -> float:
        return math.degrees(math.atan(self.tan_draft))

    @property
    def z_blank_top(self) -> float:
        return self.thickness / 2.0

    @property
    def z_blank_bottom(self) -> float:
        return -self.thickness / 2.0

    @property
    def die_offset_at_flange(self) -> float:
        """Die cavity wall offset (relative to a, b) extrapolated to z = +t/2."""
        return self.clearance + self.tan_draft * self.thickness / 2.0

    @property
    def die_flange_offset(self) -> float:
        """Plan offset where the die entry fillet meets the flat flange face."""
        r, tan = self.profile_radius, self.tan_draft
        return self.die_offset_at_flange - tan * r + r * math.hypot(1.0, tan)

    @property
    def binder_offset(self) -> float:
        """Plan offset of the blankholder opening."""
        if self.binder_opening == "die_flange":
            return self.die_flange_offset
        if self.binder_opening == "cavity":
            return self.die_offset_at_flange
        raise ValueError(f"unknown binder_opening {self.binder_opening!r}")

    def symmetry_planes(self) -> tuple:
        """Which coordinate planes the mesh is cut on, for boundary conditions."""
        return ("x", "y")

    def clamped_flange_area(self) -> float:
        """Blank area outside the die opening, i.e. what the binder clamps.

        The deck builder divides the blankholder force by this to report the
        pressure, which is the quantity that actually decides whether the flange
        is held. Any geometry family the deck builder is given must provide it.
        """
        s = self.binder_offset
        return (self.blank_half_length * self.blank_half_width
                - (self.punch_half_length + s) * (self.punch_half_width + s))

    def case_name(self) -> str:
        return (
            f"tool_radii2_{self.corner_radius:g}_radii1_{self.profile_radius:g}"
            f"_cr_{self.clearance_ratio:g}_delta_{self.draft:g}"
            f"_height_{self.height:g}"
        )

    def to_json(self) -> str:
        data = asdict(self)
        data["derived"] = {
            "clearance_mm": self.clearance,
            "draft_angle_deg": self.draft_angle_deg,
            "die_offset_at_flange_mm": self.die_offset_at_flange,
            "case_name": self.case_name(),
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# plan outline
# ---------------------------------------------------------------------------


@dataclass
class OutlineCounts:
    """Fixed segment sample counts, shared by every offset of one plan outline.

    Keeping the counts fixed while the offset changes is what makes every
    station of a sweep node-for-node compatible, so the surfaces weld exactly.
    """

    n_straight_x: int
    n_arc: int
    n_straight_y: int

    @property
    def n_points(self) -> int:
        return self.n_straight_x + self.n_arc + self.n_straight_y + 1

    @property
    def mid_index(self) -> int:
        """Index of the 45 degree point of the corner arc."""
        return self.n_straight_x + self.n_arc // 2


def outline_counts(a: float, b: float, radius: float, size: float) -> OutlineCounts:
    n1 = max(2, int(round((a - radius) / size)))
    n2 = max(4, int(round((0.5 * math.pi * radius) / size)))
    n2 += n2 % 2  # even, so the 45 degree point is a node
    n3 = max(2, int(round((b - radius) / size)))
    return OutlineCounts(n1, n2, n3)


def outline(a: float, b: float, radius: float, counts: OutlineCounts) -> np.ndarray:
    """Rounded-rectangle quarter outline, from (0, b) round to (a, 0)."""
    if radius > min(a, b):
        raise ValueError(f"corner radius {radius} exceeds half-dimensions ({a}, {b})")
    pts = np.empty((counts.n_points, 2))
    n1, n2, n3 = counts.n_straight_x, counts.n_arc, counts.n_straight_y
    pts[: n1 + 1, 0] = np.linspace(0.0, a - radius, n1 + 1)
    pts[: n1 + 1, 1] = b
    cx, cy = a - radius, b - radius
    ang = np.linspace(0.5 * math.pi, 0.0, n2 + 1)[1:]
    pts[n1 + 1 : n1 + 1 + n2, 0] = cx + radius * np.cos(ang)
    pts[n1 + 1 : n1 + 1 + n2, 1] = cy + radius * np.sin(ang)
    pts[n1 + n2 + 1 :, 0] = a
    pts[n1 + n2 + 1 :, 1] = np.linspace(b - radius, 0.0, n3 + 1)[1:]
    return pts


def outer_path(a: float, b: float, n_points: int) -> np.ndarray:
    """Sharp-cornered rectangle quarter path (0, b) -> (a, b) -> (a, 0).

    Sampled by normalised arc length so it corresponds 1:1 with an inner
    rounded outline of the same point count.
    """
    key = np.array([[0.0, b], [a, b], [a, 0.0]])
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(key, axis=0), axis=1))]
    u = np.linspace(0.0, seg[-1], n_points)
    return np.column_stack([np.interp(u, seg, key[:, 0]), np.interp(u, seg, key[:, 1])])


# ---------------------------------------------------------------------------
# mesh assembly
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMesh:
    """A quad-dominant surface patch.

    Quads are the primary representation: they are what LS-DYNA wants for a
    sheet-forming blank and they halve the element count on the rigid tools.
    Triangles are derived on demand for STL export, area-weighted sampling and
    the normal-orientation checks.
    """

    vertices: np.ndarray                        # (n, 3) float
    quads: np.ndarray                           # (m, 4) int
    groups: dict = field(default_factory=dict)  # name -> quad index array

    @property
    def faces(self) -> np.ndarray:
        """Triangulation, two per quad (quad k -> triangles k and m + k)."""
        q = self.quads
        return np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    def __add__(self, other: "SurfaceMesh") -> "SurfaceMesh":
        offset = len(self.vertices)
        groups = dict(self.groups)
        for name, idx in other.groups.items():
            groups[name] = idx + len(self.quads)
        return SurfaceMesh(
            np.vstack([self.vertices, other.vertices]),
            np.vstack([self.quads, other.quads + offset]),
            groups,
        )

    def weld(self, tol: float = 1e-6) -> "SurfaceMesh":
        """Merge coincident nodes and drop quads that collapse to a line."""
        keys = np.round(self.vertices / tol).astype(np.int64)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(first)
        remap = np.empty(len(first), dtype=np.int64)
        remap[order] = np.arange(len(first))
        verts = self.vertices[first[order]]
        quads = remap[inverse[self.quads]]
        distinct = np.array([len(np.unique(q)) for q in quads])
        keep = distinct >= 3
        renum = np.cumsum(keep) - 1
        groups = {name: renum[idx[keep[idx]]] for name, idx in self.groups.items()}
        return SurfaceMesh(verts, quads[keep], groups)

    # ---- quality ---------------------------------------------------------

    @staticmethod
    def _normals(v: np.ndarray, tri: np.ndarray) -> np.ndarray:
        p = v[tri]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norm < 1e-12, 1.0, norm)

    def face_normals(self) -> np.ndarray:
        return self._normals(self.vertices, self.faces)

    def face_areas(self) -> np.ndarray:
        p = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    def quad_normals(self) -> np.ndarray:
        """Area-weighted mean of a quad's two triangle normals."""
        m = len(self.quads)
        n, a = self.face_normals(), self.face_areas()
        w = (n[:m] * a[:m, None] + n[m:] * a[m:, None])
        norm = np.linalg.norm(w, axis=1, keepdims=True)
        return w / np.where(norm < 1e-12, 1.0, norm)

    def quad_areas(self) -> np.ndarray:
        m = len(self.quads)
        a = self.face_areas()
        return a[:m] + a[m:]

    def _edge_counts(self) -> np.ndarray:
        q = self.quads
        e = np.sort(
            np.vstack([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]]), axis=1
        )
        e = e[e[:, 0] != e[:, 1]]
        _, counts = np.unique(e, axis=0, return_counts=True)
        return counts

    def free_edges(self) -> int:
        """Edges used by exactly one element (the open boundary of the patch)."""
        return int(np.count_nonzero(self._edge_counts() == 1))

    def non_manifold_edges(self) -> int:
        return int(np.count_nonzero(self._edge_counts() > 2))

    def min_edge_length(self) -> float:
        p = self.vertices[self.quads]
        d = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
        return float(d[d > 1e-9].min())

    def aspect_ratio(self) -> np.ndarray:
        p = self.vertices[self.quads]
        d = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
        return d.max(axis=1) / np.maximum(d.min(axis=1), 1e-9)

    def sample(self, n: int, seed: int = 0) -> tuple:
        """Uniform area-weighted surface sampling, as in the reference clouds."""
        rng = np.random.default_rng(seed)
        tri = self.faces
        areas = self.face_areas()
        idx = rng.choice(len(tri), size=n, p=areas / areas.sum())
        u = rng.random((n, 1))
        v = rng.random((n, 1))
        flip = (u + v) > 1
        u[flip] = 1 - u[flip]
        v[flip] = 1 - v[flip]
        t = self.vertices[tri[idx]]
        pts = t[:, 0] + u * (t[:, 1] - t[:, 0]) + v * (t[:, 2] - t[:, 0])
        return pts, idx % len(self.quads)


def _grid_quads(nu: int, nv: int, flip: bool = False) -> np.ndarray:
    """Quads of a structured (nu x nv) point grid indexed as i*nv + j."""
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    p00 = (i * nv + j).ravel()
    p10 = ((i + 1) * nv + j).ravel()
    p11 = ((i + 1) * nv + j + 1).ravel()
    p01 = (i * nv + j + 1).ravel()
    q = np.column_stack([p00, p10, p11, p01])
    return q[:, ::-1].copy() if flip else q


def sweep(profile: np.ndarray, plan: tuple, counts: OutlineCounts,
          flip: bool = False) -> SurfaceMesh:
    """Sweep a (s, z) profile along the rounded-rectangle plan outline.

    ``plan`` is (a, b, radius). Grid index is i*nv + j with i along the outline
    and j along the profile.
    """
    a, b, radius = plan
    nv = len(profile)
    nu = counts.n_points
    verts = np.empty((nu * nv, 3))
    for j, (s, z) in enumerate(profile):
        xy = outline(a + s, b + s, radius, counts)
        verts[j::nv, :2] = xy
        verts[j::nv, 2] = z
    return SurfaceMesh(verts, _grid_quads(nu, nv, flip))


def coons_patch(outline_xy: np.ndarray, counts: OutlineCounts, z: float,
                flip: bool = False) -> SurfaceMesh:
    """Fill the interior of a quarter outline with a transfinite (Coons) grid.

    The patch corners are (0,0), (a,0), the 45 degree outline point, and (0,b).
    The outer boundary reuses the outline nodes exactly, so the patch welds to
    an adjoining sweep without a crack.
    """
    mid = counts.mid_index
    right = outline_xy[mid:][::-1]   # (a, 0) -> 45 degree point
    top = outline_xy[: mid + 1]      # (0, b) -> 45 degree point

    p00 = np.array([0.0, 0.0])
    p10 = outline_xy[-1]             # (a, 0)
    p11 = outline_xy[mid]            # 45 degree point
    p01 = outline_xy[0]              # (0, b)

    nu = len(top)                    # along xi
    nv = len(right)                  # along eta

    bottom = np.column_stack([np.linspace(p00[0], p10[0], nu), np.zeros(nu)])
    left = np.column_stack([np.zeros(nv), np.linspace(p00[1], p01[1], nv)])

    xi = np.linspace(0.0, 1.0, nu)[:, None, None]
    eta = np.linspace(0.0, 1.0, nv)[None, :, None]
    s = (
        (1 - xi) * left[None, :, :]
        + xi * right[None, :, :]
        + (1 - eta) * bottom[:, None, :]
        + eta * top[:, None, :]
        - (
            (1 - xi) * (1 - eta) * p00
            + xi * (1 - eta) * p10
            + (1 - xi) * eta * p01
            + xi * eta * p11
        )
    )
    verts = np.empty((nu * nv, 3))
    verts[:, :2] = s.reshape(-1, 2)
    verts[:, 2] = z
    return SurfaceMesh(verts, _grid_quads(nu, nv, flip))


def ring_patch(inner_xy: np.ndarray, outer_a: float, outer_b: float, z: float,
               size: float, flip: bool = False) -> SurfaceMesh:
    """Flat frame between a rounded inner outline and a sharp outer rectangle."""
    nu = len(inner_xy)
    outer_xy = outer_path(outer_a, outer_b, nu)
    span = float(np.max(np.linalg.norm(outer_xy - inner_xy, axis=1)))
    nv = max(2, int(round(span / size)) + 1)
    w = np.linspace(0.0, 1.0, nv)[None, :, None]
    pts = (1 - w) * inner_xy[:, None, :] + w * outer_xy[:, None, :]
    verts = np.empty((nu * nv, 3))
    verts[:, :2] = pts.reshape(-1, 2)
    verts[:, 2] = z
    return SurfaceMesh(verts, _grid_quads(nu, nv, flip))


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def punch_profile(p: DrawParams, size: float) -> tuple:
    """(s, z) generator of the punch nose fillet + drafted wall.

    Returns (profile, s_top) where ``s_top`` is the plan offset of the flat top
    face boundary, i.e. where the nose fillet starts.
    """
    t, r, tan = p.thickness, p.profile_radius, p.tan_draft
    sec = math.hypot(1.0, tan)
    z_top = -t / 2.0
    z_c = z_top - r
    s_c = tan * r - p.draft - r * sec            # fillet centre offset
    psi_end = 0.5 * math.pi - math.atan(tan)

    n = max(3, int(round(psi_end * r / size)))
    psi = np.linspace(0.0, psi_end, n + 1)
    fillet = np.column_stack([s_c + r * np.sin(psi), z_c + r * np.cos(psi)])

    z_wall_end = fillet[-1, 1]
    z_bottom = -(t / 2.0 + p.height)
    m = max(2, int(round(abs(z_wall_end - z_bottom) / size)))
    zw = np.linspace(z_wall_end, z_bottom, m + 1)[1:]
    sw = -tan * (zw + t / 2.0 + p.height)
    wall = np.column_stack([sw, zw])

    # Prismatic extension so the binder always has punch wall beside it.
    extension = p.height * p.punch_extension_ratio
    if extension > 0:
        k = max(2, int(round(extension / size)))
        ze = np.linspace(z_bottom, z_bottom - extension, k + 1)[1:]
        wall = np.vstack([wall, np.column_stack([np.zeros_like(ze), ze])])
    return np.vstack([fillet, wall]), s_c


def die_profile(p: DrawParams, size: float) -> tuple:
    """(s, z) generator of the die entry fillet + cavity wall + roof fillet.

    Returns (profile, s_flange, s_roof): the flange-face inner offset and the
    roof-face offset bound the two flat faces of the die.
    """
    t, r, tan = p.thickness, p.profile_radius, p.tan_draft
    sec = math.hypot(1.0, tan)
    s0 = p.die_offset_at_flange
    z_flange = t / 2.0
    z_roof = z_flange + p.height
    psi_end = 0.5 * math.pi - math.atan(tan)
    n = max(3, int(round(psi_end * r / size)))
    psi = np.linspace(0.0, psi_end, n + 1)

    # entry fillet: tangent to the flange plane and to the cavity wall
    s_ce = s0 - tan * r + r * sec
    entry = np.column_stack([s_ce - r * np.sin(psi), z_flange + r - r * np.cos(psi)])

    # roof fillet: tangent to the roof plane and to the cavity wall
    z_cr = z_roof - r
    s_cr = s0 + tan * t / 2.0 - tan * z_cr - r * sec
    roof = np.column_stack([s_cr + r * np.sin(psi), z_cr + r * np.cos(psi)])[::-1]

    s_wall0, z_wall0 = entry[-1]
    s_wall1, z_wall1 = roof[0]
    if z_wall1 <= z_wall0:
        raise ValueError("height too small for the entry and roof fillets")
    m = max(2, int(round((z_wall1 - z_wall0) / size)))
    zw = np.linspace(z_wall0, z_wall1, m + 1)[1:-1]
    sw = s0 + tan * t / 2.0 - tan * zw
    wall = np.column_stack([sw, zw])
    return np.vstack([entry, wall, roof]), s_ce, s_cr


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def _assert_outward(mesh: SurfaceMesh, group: str, axis: int, sign: int) -> None:
    idx = mesh.groups[group]
    n = mesh.quad_normals()[idx]
    areas = mesh.quad_areas()[idx]
    mean = float(np.sum(n[:, axis] * areas) / areas.sum())
    if mean * sign <= 0:
        raise AssertionError(
            f"{group}: mean normal component {mean:+.3f} on axis {axis} "
            f"has the wrong sign (expected {sign:+d})"
        )


def build_punch(p: DrawParams) -> SurfaceMesh:
    """Male tool: flat top face + nose fillet + drafted wall. Normals point out."""
    size = p.tool_element_size
    counts = outline_counts(p.punch_half_length, p.punch_half_width, p.corner_radius, size)
    profile, s_top = punch_profile(p, size)

    top_xy = outline(p.punch_half_length + s_top, p.punch_half_width + s_top,
                     p.corner_radius, counts)
    face = coons_patch(top_xy, counts, p.z_blank_bottom)
    wall = sweep(profile, (p.punch_half_length, p.punch_half_width, p.corner_radius), counts)

    face.groups = {"punch_face": np.arange(len(face.quads))}
    wall.groups = {"punch_wall": np.arange(len(wall.quads))}
    mesh = (face + wall).weld()
    _assert_outward(mesh, "punch_face", axis=2, sign=+1)
    return mesh


def build_die(p: DrawParams) -> SurfaceMesh:
    """Female tool: flange + entry fillet + cavity + roof. Normals face the blank."""
    size = p.tool_element_size
    counts = outline_counts(p.punch_half_length, p.punch_half_width, p.corner_radius, size)
    profile, s_flange, s_roof = die_profile(p, size)
    a, b, rc = p.punch_half_length, p.punch_half_width, p.corner_radius

    flange_xy = outline(a + s_flange, b + s_flange, rc, counts)
    flange = ring_patch(flange_xy, p.tool_half_length, p.tool_half_width,
                        p.z_blank_top, size, flip=True)
    cavity = sweep(profile, (a, b, rc), counts)
    roof_xy = outline(a + s_roof, b + s_roof, rc, counts)
    roof = coons_patch(roof_xy, counts, p.z_blank_top + p.height, flip=True)

    flange.groups = {"die_flange": np.arange(len(flange.quads))}
    cavity.groups = {"die_cavity": np.arange(len(cavity.quads))}
    roof.groups = {"die_roof": np.arange(len(roof.quads))}
    mesh = (flange + cavity + roof).weld()
    _assert_outward(mesh, "die_flange", axis=2, sign=-1)
    _assert_outward(mesh, "die_roof", axis=2, sign=-1)
    return mesh


def build_binder(p: DrawParams) -> SurfaceMesh:
    """Blankholder: flat frame at z = -t/2, opening matched to the die cavity."""
    size = p.tool_element_size
    counts = outline_counts(p.punch_half_length, p.punch_half_width, p.corner_radius, size)
    s0 = p.binder_offset
    inner_xy = outline(p.punch_half_length + s0, p.punch_half_width + s0,
                       p.corner_radius, counts)
    mesh = ring_patch(inner_xy, p.tool_half_length, p.tool_half_width,
                      p.z_blank_bottom, size)
    mesh.groups = {"binder": np.arange(len(mesh.quads))}
    mesh = mesh.weld()
    _assert_outward(mesh, "binder", axis=2, sign=+1)
    return mesh


def build_blank(p: DrawParams) -> SurfaceMesh:
    """Flat quarter blank shell at z = 0, structured grid, normals up."""
    nu = max(2, int(round(p.blank_half_length / p.blank_element_size))) + 1
    nv = max(2, int(round(p.blank_half_width / p.blank_element_size))) + 1
    x = np.linspace(0.0, p.blank_half_length, nu)
    y = np.linspace(0.0, p.blank_half_width, nv)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nu * nv)])
    mesh = SurfaceMesh(verts, _grid_quads(nu, nv))
    mesh.groups = {"blank": np.arange(len(mesh.quads))}
    _assert_outward(mesh, "blank", axis=2, sign=+1)
    return mesh


def build_all(p: DrawParams) -> dict:
    return {
        "punch": build_punch(p),
        "die": build_die(p),
        "binder": build_binder(p),
        "blank": build_blank(p),
    }


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def write_stl(mesh: SurfaceMesh, path: Path) -> None:
    dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data = np.zeros(len(mesh.faces), dtype=dtype)
    data["n"] = mesh.face_normals()
    data["v"] = mesh.vertices[mesh.faces]
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(mesh.faces)).tobytes())
        fh.write(data.tobytes())


def write_pointcloud(mesh: SurfaceMesh, path: Path, n: int = 10000, seed: int = 0) -> None:
    pts, idx = mesh.sample(n, seed)
    np.savez_compressed(path, points=pts, idx=idx)


def write_keyword(mesh: SurfaceMesh, path: Path, part_id: int,
                  node_offset: int = 0, element_offset: int = 0) -> None:
    """Standalone *NODE / *ELEMENT_SHELL deck for one part."""
    path.write_text(
        "*KEYWORD\n" + keyword_blocks(mesh, part_id, node_offset, element_offset) + "*END\n",
        encoding="utf-8",
    )


def keyword_blocks(mesh: SurfaceMesh, part_id: int, node_offset: int = 0,
                   element_offset: int = 0) -> str:
    """*NODE and *ELEMENT_SHELL text for one part.

    Quads are written as quads. A quad that welding collapsed to a triangle is
    written as a degenerate quad (n4 == n3), which is how LS-DYNA expects one.
    """
    out = ["*NODE"]
    for i, (x, y, z) in enumerate(mesh.vertices, start=node_offset + 1):
        out.append(f"{i:8d}{x:16.6f}{y:16.6f}{z:16.6f}{0:8d}{0:8d}")
    out.append("*ELEMENT_SHELL")
    for e, q in enumerate(mesh.quads + node_offset + 1, start=element_offset + 1):
        n1, n2, n3, n4 = (int(v) for v in q)
        out.append(f"{e:8d}{part_id:8d}{n1:8d}{n2:8d}{n3:8d}{n4:8d}")
    return "\n".join(out) + "\n"
