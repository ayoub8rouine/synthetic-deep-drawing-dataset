"""Parametric U-channel (hat-section) deep-drawing geometry.

A second shape family alongside `geometry/box_draw.py`, inspired by the
parametric U-channel models of https://github.com/ToLe814/uchannel. That package
is used as the *specification* -- its parameter meanings and sketches -- not as a
runtime dependency, because it produces the finished part as a STEP surface and
we need the tool set. Offsetting a CAD surface into a punch and die risks
self-intersection at every concave fillet; offsetting the 2D cross-section
instead is exact:

    a straight segment shifts by d along its normal
    a fillet radius becomes r - d on a left turn, r + d on a right turn

which is what `filleted_section` below does. Same guarantee as the box family:
no CAD kernel, no booleans, weld-by-construction quad meshes.

Geometry
--------
A hat section lying along x. In the y-z plane the part midsurface runs

    flange -- fillet(rc) -- slant(beta) -- fillet(rf) -- floor -- ... mirrored

with the floor on top and the flanges at z = -height, so the channel is drawn in
the +z direction exactly as the box family draws its cup. The cross-section is
lofted between an end section at x = 0 and another at x = length, so width,
depth, wall angle and radii can all vary along the channel.

Tool convention matches `box_draw` exactly (blank midsurface at z = 0):

    punch   fixed, floor-forming face at z = -t/2, walls down, prismatic tail
    binder  flat strips at z = -t/2 outboard of the die opening
    die     flange face at z = +t/2, cavity rising to z = +t/2 + height
    motion  die and binder travel down by `height`; the punch is fixed

Unlike the box this is modelled full, not on a symmetry quarter: the two end
sections are free to differ, so there is no symmetry plane to exploit.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "geometry"))

from box_draw import (  # noqa: E402
    SurfaceMesh,
    _grid_quads,
    keyword_blocks,
    write_pointcloud,
    write_stl,
)

__all__ = [
    "Section", "UChannelParams", "filleted_section",
    "build_punch", "build_die", "build_binder", "build_blank", "build_all",
    "keyword_blocks", "write_pointcloud", "write_stl",
]


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """One end cross-section of the channel."""

    floor_half_width: float = 60.0     # half width of the flat floor
    height: float = 35.0               # floor height above the flange plane
    wall_angle_deg: float = 12.0       # wall lean from vertical, widening downward
    floor_radius: float = 8.0          # floor-to-wall fillet -> punch nose radius
    flange_radius: float = 8.0         # wall-to-flange fillet -> die entry radius

    def validate(self, thickness: float, clearance: float) -> None:
        if self.floor_half_width <= 0:
            raise ValueError("floor_half_width must be > 0")
        if self.height <= 0:
            raise ValueError("height must be > 0")
        if not 0.0 <= self.wall_angle_deg < 60.0:
            raise ValueError("wall_angle_deg must be in [0, 60)")
        # The punch takes the floor fillet minus half a thickness, the die takes
        # the flange fillet minus half a thickness and the clearance. Both must
        # stay positive or the offset cross-section is not a valid shape.
        if self.floor_radius <= thickness / 2:
            raise ValueError("floor_radius must exceed half the sheet thickness")
        if self.flange_radius <= thickness / 2 + clearance:
            raise ValueError("flange_radius must exceed t/2 + clearance")
        # the two fillets must fit in the wall between floor and flange
        run = self.height * math.tan(math.radians(self.wall_angle_deg))
        wall = math.hypot(self.height, run)
        need = (self.floor_radius + self.flange_radius) * math.tan(
            math.radians(45.0 - self.wall_angle_deg / 2.0))
        if wall <= need * 0.6:
            raise ValueError("wall too short for the floor and flange fillets")

    @property
    def wall_run(self) -> float:
        """Horizontal spread of the wall over its height."""
        return self.height * math.tan(math.radians(self.wall_angle_deg))


@dataclass
class UChannelParams:
    """One U-channel case. Lengths in mm, forces in N."""

    section_a: Section = field(default_factory=Section)
    section_b: Section = field(default_factory=Section)

    length: float = 320.0              # tool length along x
    flange_width: float = 55.0         # flat flange each side of the die opening

    thickness: float = 1.0
    clearance_ratio: float = 1.1

    blank_length: float = 300.0        # along x, shorter than the tools
    blank_width: float = 320.0         # across y

    tool_element_size: float = 6.0
    blank_element_size: float = 5.0

    blankholder_force: float = 300000.0
    punch_extension_ratio: float = 0.6

    name: str = "uchannel"

    def __post_init__(self) -> None:
        for s in (self.section_a, self.section_b):
            s.validate(self.thickness, self.clearance)
        if self.clearance_ratio < 1.0:
            raise ValueError("clearance_ratio must be >= 1.0")
        if self.blank_length >= self.length:
            raise ValueError("tools must be longer than the blank in x")
        if self.blank_width <= 2 * max(self.die_half_width(u)
                                       for u in np.linspace(0, 1, 11)):
            raise ValueError("blank must overhang the die opening to form a flange")
        self._check_buildable()

    def _check_buildable(self) -> None:
        """Try the actual construction at stations along the loft.

        Checking only the two end sections is not enough: the wall length, the
        fillet tangent lengths and the wall angle interact non-linearly, so an
        interpolated station can have overlapping fillets while both ends are
        fine. Building it is the only exact test, and it is cheap.
        """
        counts = SectionCounts()
        y_outer = self.blank_width / 2.0 + 20.0
        for u in np.linspace(0.0, 1.0, 11):
            s = self.section_at(u)
            filleted_section(s, self.thickness / 2 + self.clearance, counts,
                             y_outer, -self.clearance, mode="die")
            filleted_section(s, -self.thickness / 2, counts, 0.0, -self.height,
                             mode="punch", tail=1.0)

    # ---- derived ---------------------------------------------------------

    @property
    def clearance(self) -> float:
        return self.clearance_ratio * self.thickness

    @property
    def height(self) -> float:
        """Stroke. The deeper end sets it so both ends are fully formed."""
        return max(self.section_a.height, self.section_b.height)

    @property
    def z_blank_top(self) -> float:
        return self.thickness / 2.0

    @property
    def z_blank_bottom(self) -> float:
        return -self.thickness / 2.0

    def section_at(self, u: float) -> Section:
        """Linear loft between the two end sections, u in [0, 1]."""
        a, b = self.section_a, self.section_b
        return Section(
            floor_half_width=(1 - u) * a.floor_half_width + u * b.floor_half_width,
            height=(1 - u) * a.height + u * b.height,
            wall_angle_deg=(1 - u) * a.wall_angle_deg + u * b.wall_angle_deg,
            floor_radius=(1 - u) * a.floor_radius + u * b.floor_radius,
            flange_radius=(1 - u) * a.flange_radius + u * b.flange_radius,
        )

    def die_half_width(self, u: float) -> float:
        """Half width of the die opening, i.e. where the flat flange starts."""
        s = self.section_at(u)
        d = self.thickness / 2 + self.clearance
        # the flange fillet of the die, whose radius has shrunk by d
        return s.floor_half_width + s.wall_run + (s.flange_radius - d) * math.tan(
            math.radians(45.0 - s.wall_angle_deg / 2.0)) + d * math.tan(
            math.radians(s.wall_angle_deg))

    def symmetry_planes(self) -> tuple:
        """Only y = 0. The two end sections may differ, so x is not a plane."""
        return ("y",)

    def clamped_flange_area(self) -> float:
        """Blank area outboard of the die opening, i.e. what the binder clamps."""
        us = np.linspace(0.0, 1.0, 21)
        open_w = np.array([2 * self.die_half_width(u) for u in us])
        # only the part of the opening the blank actually covers
        covered = np.minimum(open_w, self.blank_width)
        return float(self.blank_length * (self.blank_width - covered.mean()) / 2.0)

    def case_name(self) -> str:
        a, b = self.section_a, self.section_b
        return (f"uchannel_w{a.floor_half_width:g}-{b.floor_half_width:g}"
                f"_h{a.height:g}-{b.height:g}"
                f"_beta{a.wall_angle_deg:g}-{b.wall_angle_deg:g}"
                f"_rf{a.floor_radius:g}-{b.floor_radius:g}"
                f"_rc{a.flange_radius:g}-{b.flange_radius:g}")

    def to_json(self) -> str:
        data = asdict(self)
        data["derived"] = {
            "clearance_mm": self.clearance,
            "stroke_mm": self.height,
            "die_half_width_a_mm": self.die_half_width(0.0),
            "die_half_width_b_mm": self.die_half_width(1.0),
            "clamped_flange_area_mm2": self.clamped_flange_area(),
            "case_name": self.case_name(),
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# cross-section
# ---------------------------------------------------------------------------


@dataclass
class SectionCounts:
    """Point counts per segment, fixed across every station of the loft.

    As in the box family, holding the counts constant while the section changes
    is what makes consecutive stations node-for-node compatible, so the lofted
    surface is a clean structured grid.
    """

    n_flange: int = 8
    n_flange_arc: int = 6
    n_wall: int = 8
    n_floor_arc: int = 6
    n_floor: int = 14
    n_tail: int = 4


def counts_for(p: "UChannelParams", y_outer: float) -> SectionCounts:
    """Segment counts, sized from the SHORTEST station of each segment.

    One count is shared by every station of the loft, so sizing from the mean
    section would over-refine wherever a segment is shorter than average and
    leave sub-millimetre elements there. LS-DYNA derives the contact penalty
    stiffness from the master segment size, so those tiny elements collapse the
    stable time step -- measured at 2.1e-7 s instead of 5e-7 s before this was
    fixed. Sizing from the shortest station instead keeps the finest elements at
    the target size and merely coarsens the longer stations.
    """
    a, b = p.section_a, p.section_b
    h = p.tool_element_size
    n = lambda length: max(2, int(round(length / h)))  # noqa: E731
    arc = lambda s, r: r * (math.pi / 2 - math.radians(s.wall_angle_deg))  # noqa: E731
    return SectionCounts(
        n_flange=n(min(y_outer - s.floor_half_width - s.wall_run for s in (a, b))),
        n_flange_arc=n(min(arc(s, s.flange_radius) for s in (a, b))),
        n_wall=n(min(math.hypot(s.height, s.wall_run) for s in (a, b))),
        n_floor_arc=n(min(arc(s, s.floor_radius) for s in (a, b))),
        n_floor=n(min(s.floor_half_width for s in (a, b))),
        n_tail=n(p.height * p.punch_extension_ratio),
    )


def _arc(centre: np.ndarray, radius: float, a0: float, a1: float, n: int) -> np.ndarray:
    ang = np.linspace(a0, a1, n + 1)
    return centre + radius * np.column_stack([np.cos(ang), np.sin(ang)])


def filleted_section(sec: Section, offset: float, counts: SectionCounts,
                     y_outer: float, z_flange: float, *,
                     mode: str = "die", tail: float = 0.0) -> np.ndarray:
    """Half of the hat cross-section, running from outboard inward to y = 0.

    ``offset`` is a normal offset of the part midsurface, positive outward (up
    over the floor, away from the channel interior). The die uses +t/2+clearance
    and the punch -t/2. The offset is exact: the section turns one way at the
    flange and the other at the floor, so

        flange fillet  rc -> rc - offset      (concave seen from outside)
        floor  fillet  rf -> rf + offset      (convex seen from outside)

    ``z_flange`` is the datum the section is measured from: the flange plane of
    the *unoffset* midsurface. ``mode`` is "die" (flat flange, entry fillet,
    wall, floor fillet, floor) or "punch" (a prismatic tail, then wall, floor
    fillet, floor -- a punch has no flange and no entry fillet; that radius
    belongs to the die).
    """
    beta = math.radians(sec.wall_angle_deg)
    rc = sec.flange_radius - offset
    rf = sec.floor_radius + offset
    if rc <= 0 or rf <= 0:
        raise ValueError("offset destroys a fillet; check radii against t/2 and clearance")

    z_fl = z_flange + offset
    z_floor = z_flange + sec.height + offset

    # sharp corners the fillets are cut into
    y_corner = sec.floor_half_width + sec.wall_run + offset * math.tan(beta)
    y_top = sec.floor_half_width + offset * math.tan(beta)

    # Going flange -> wall -> floor the direction turns by (90 - beta) at each
    # corner, in opposite senses, so both fillets share the same tangent length.
    half_turn = math.radians(45.0 - sec.wall_angle_deg / 2.0)
    t_c = rc * math.tan(half_turn)
    t_f = rf * math.tan(half_turn)

    wall_dir = np.array([-math.sin(beta), math.cos(beta)])  # up and inward
    p_corner = np.array([y_corner, z_fl])
    p_top = np.array([y_top, z_floor])
    wall_start = p_corner + t_c * wall_dir
    wall_end = p_top - t_f * wall_dir
    floor_start = p_top - np.array([t_f, 0.0])

    if wall_end[1] <= wall_start[1]:
        raise ValueError("fillets overlap: wall too short for these radii")
    if floor_start[0] <= 0:
        raise ValueError("floor fillet is wider than the floor")

    pieces = []
    if mode == "die":
        flange_end = p_corner + np.array([t_c, 0.0])
        pieces.append(np.column_stack([
            np.linspace(y_outer, flange_end[0], counts.n_flange + 1),
            np.full(counts.n_flange + 1, z_fl)]))
        # entry fillet: centre one radius above the flange plane
        c_c = flange_end + np.array([0.0, rc])
        pieces.append(_arc(c_c, rc, -math.pi / 2,
                           -math.pi / 2 - (math.pi / 2 - beta), counts.n_flange_arc)[1:])
    else:
        # the punch has no flange: run straight down from the wall foot
        pieces.append(np.column_stack([
            np.full(counts.n_tail + 1, p_corner[0]),
            np.linspace(z_fl - tail, z_fl, counts.n_tail + 1)]))
        wall_start = p_corner

    pieces.append(np.column_stack([
        np.linspace(wall_start[0], wall_end[0], counts.n_wall + 1),
        np.linspace(wall_start[1], wall_end[1], counts.n_wall + 1)])[1:])
    c_f = floor_start + np.array([0.0, -rf])
    pieces.append(_arc(c_f, rf, beta, math.pi / 2, counts.n_floor_arc)[1:])
    pieces.append(np.column_stack([
        np.linspace(floor_start[0], 0.0, counts.n_floor + 1),
        np.full(counts.n_floor + 1, z_floor)])[1:])
    return np.vstack(pieces)


def inboard_first(half: np.ndarray) -> np.ndarray:
    """Reorder a section to run from y = 0 outward.

    The section is mirrored about y = 0 by construction, so only the y >= 0 half
    is modelled and y = 0 carries a symmetry condition. That halves the model,
    which matters here because the two end sections may differ and there is no
    second symmetry plane to exploit.
    """
    return half[::-1]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def _loft(p: UChannelParams, section_fn, nx: int | None = None,
          flip: bool = False) -> SurfaceMesh:
    """Sweep a cross-section function along x into a structured quad mesh."""
    nx = nx or max(4, int(round(p.length / p.tool_element_size)))
    xs = np.linspace(0.0, p.length, nx + 1) - p.length / 2.0
    us = np.linspace(0.0, 1.0, nx + 1)
    rows = [section_fn(u) for u in us]
    ns = len(rows[0])
    verts = np.empty(((nx + 1) * ns, 3))
    for i, (x, row) in enumerate(zip(xs, rows)):
        verts[i * ns:(i + 1) * ns, 0] = x
        verts[i * ns:(i + 1) * ns, 1:] = row
    return SurfaceMesh(verts, _grid_quads(nx + 1, ns, flip))


def _assert_normal(mesh: SurfaceMesh, axis: int, sign: int, what: str) -> None:
    n, a = mesh.quad_normals(), mesh.quad_areas()
    mean = float(np.sum(n[:, axis] * a) / a.sum())
    if mean * sign <= 0:
        raise AssertionError(f"{what}: mean normal {mean:+.3f} on axis {axis}, "
                             f"expected sign {sign:+d}")


def build_punch(p: UChannelParams) -> SurfaceMesh:
    """Male tool: the channel's inner surface, plus a prismatic tail below.

    The punch is fixed while the flange descends by the full stroke, so its
    datum is the flange's FINAL plane, z = -stroke. A section shallower than the
    stroke therefore has its floor correspondingly lower, which is exactly what
    a channel of varying depth needs.
    """
    counts = counts_for(p, p.blank_width / 2.0 + 20.0)
    d = -p.thickness / 2.0
    tail = p.height * p.punch_extension_ratio

    def sect(u: float) -> np.ndarray:
        return inboard_first(filleted_section(
            p.section_at(u), d, counts, y_outer=0.0, z_flange=-p.height,
            mode="punch", tail=tail))

    mesh = _loft(p, sect).weld()
    mesh.groups = {"punch": np.arange(len(mesh.quads))}
    _assert_normal(mesh, axis=2, sign=+1, what="punch")
    return mesh


def build_die(p: UChannelParams) -> SurfaceMesh:
    """Female tool: flat flange, entry fillet, cavity walls and cavity roof.

    Its datum puts the flat flange face on the blank's top surface, z = +t/2,
    which after offsetting by t/2 + clearance means a datum of -clearance.
    """
    counts = counts_for(p, p.blank_width / 2.0 + 20.0)
    d = p.thickness / 2.0 + p.clearance
    y_outer = p.blank_width / 2.0 + 20.0

    def sect(u: float) -> np.ndarray:
        return inboard_first(filleted_section(
            p.section_at(u), d, counts, y_outer=y_outer, z_flange=-p.clearance,
            mode="die"))

    mesh = _loft(p, sect, flip=True).weld()
    mesh.groups = {"die": np.arange(len(mesh.quads))}
    _assert_normal(mesh, axis=2, sign=-1, what="die")
    return mesh


def build_binder(p: UChannelParams) -> SurfaceMesh:
    """Blankholder: a flat strip each side, outboard of the die opening."""
    nx = max(4, int(round(p.length / p.tool_element_size)))
    xs = np.linspace(0.0, p.length, nx + 1) - p.length / 2.0
    us = np.linspace(0.0, 1.0, nx + 1)
    y_outer = p.blank_width / 2.0 + 20.0
    ny = max(2, int(round((y_outer - p.die_half_width(0.5)) / p.tool_element_size))) + 1

    verts = np.empty(((nx + 1) * ny, 3))
    for i, (x, u) in enumerate(zip(xs, us)):
        ys = np.linspace(p.die_half_width(u), y_outer, ny)
        verts[i * ny:(i + 1) * ny, 0] = x
        verts[i * ny:(i + 1) * ny, 1] = ys
        verts[i * ny:(i + 1) * ny, 2] = p.z_blank_bottom
    mesh = SurfaceMesh(verts, _grid_quads(nx + 1, ny)).weld()
    mesh.groups = {"binder": np.arange(len(mesh.quads))}
    _assert_normal(mesh, axis=2, sign=+1, what="binder")
    return mesh


def build_blank(p: UChannelParams) -> SurfaceMesh:
    """Flat rectangular blank at z = 0."""
    nx = max(2, int(round(p.blank_length / p.blank_element_size)))
    ny = max(2, int(round(p.blank_width / 2 / p.blank_element_size)))
    x = np.linspace(-p.blank_length / 2, p.blank_length / 2, nx + 1)
    y = np.linspace(0.0, p.blank_width / 2, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    verts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    mesh = SurfaceMesh(verts, _grid_quads(nx + 1, ny + 1))
    mesh.groups = {"blank": np.arange(len(mesh.quads))}
    _assert_normal(mesh, axis=2, sign=+1, what="blank")
    return mesh


def build_all(p: UChannelParams) -> dict:
    return {
        "punch": build_punch(p),
        "die": build_die(p),
        "binder": build_binder(p),
        "blank": build_blank(p),
    }
