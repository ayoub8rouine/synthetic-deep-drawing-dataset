"""Plan outlines for deep-drawn cups: the registry the shape catalogue plugs into.

Everything downstream -- profile sweep, tool offsetting, deck, contacts,
acceptance checks, dataset writer -- is outline-agnostic. Adding a shape family
means adding one function here.

An outline is generated over its **symmetry sector** only, so the FE model can be
a quarter or half rather than the full part:

    quarter (theta 0..pi/2)   mirror about both x=0 and y=0
    half    (theta 0..pi)     mirror about y=0 only

Regular polygons and cos-lobed shapes with an even side/lobe count keep both
mirror planes; odd counts keep only the x-axis. Nothing here needs a full model.

The interface is a single method, ``points_at(offset)``, returning the outline
pushed outward by a normal distance -- the punch uses -t/2 and the die
+t/2+clearance. It returns a FIXED number of points at FIXED polar angles
whatever the offset, which is what makes the swept surface a clean structured
grid, and the first and last points sit exactly on the symmetry axes.

Offsetting outward grows a convex radius (always safe) but shrinks a **concave**
one, so an outline is only usable while

    min_concave_radius > thickness/2 + clearance + die_entry_radius

roughly 7 mm for 1 mm sheet. That is tooling physics, not a software limit: a
die cannot have a knife edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Outline:
    """A plan outline over its symmetry sector."""

    points_at: Callable[[float], np.ndarray]
    n_points: int
    min_concave_radius: float   # inf when convex everywhere
    symmetry: str               # "quarter" or "half"
    name: str

    @property
    def points(self) -> np.ndarray:
        return self.points_at(0.0)

    @property
    def sector_angle(self) -> float:
        return math.pi / 2 if self.symmetry == "quarter" else math.pi

    def max_radius(self, offset: float = 0.0) -> float:
        return float(np.linalg.norm(self.points_at(offset), axis=1).max())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _angles(symmetry: str, n: int) -> np.ndarray:
    """Fixed polar angles across the sector, endpoints exactly on the axes."""
    return np.linspace(0.0, math.pi / 2 if symmetry == "quarter" else math.pi, n)


def _ray_cast(dense: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Sample a star-convex closed polyline at given polar angles.

    Works for any outline the origin can "see" every point of, which covers
    every family here. Guarantees the requested angles -- and therefore the
    exact symmetry-axis endpoints -- are hit.
    """
    ang = np.arctan2(dense[:, 1], dense[:, 0]) % (2 * math.pi)
    rad = np.linalg.norm(dense, axis=1)
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    # wrap a period either side so interpolation is continuous at 0 and 2*pi
    ang = np.r_[ang - 2 * math.pi, ang, ang + 2 * math.pi]
    rad = np.r_[rad, rad, rad]
    r = np.interp(theta % (2 * math.pi), ang, rad)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _rounded_polygon(vertices: np.ndarray, radii: np.ndarray,
                     per_arc: int = 14, per_edge: int = 24) -> np.ndarray:
    """Dense closed polyline: a polygon with a fillet at each vertex.

    The straight edges are sampled too, not just the arcs. Ray-casting
    interpolates radius against angle, and across an unsampled flat side that
    interpolation follows the chord rather than the side, which reads a
    rectangle's flat face as bulging outward by several mm.
    """
    n = len(vertices)
    arcs, out = [], []
    for i in range(n):
        prev, cur, nxt = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
        d0 = (prev - cur) / np.linalg.norm(prev - cur)
        d1 = (nxt - cur) / np.linalg.norm(nxt - cur)
        cos_half = np.clip(np.dot(d0, d1), -1.0, 1.0)
        half = math.acos(cos_half) / 2.0
        if half <= 1e-6 or abs(half - math.pi / 2) < 1e-9:
            arcs.append(cur[None, :])
            continue
        tan_len = radii[i] / math.tan(half)
        tan_len = min(tan_len, 0.49 * np.linalg.norm(prev - cur),
                      0.49 * np.linalg.norm(nxt - cur))
        r_eff = tan_len * math.tan(half)
        bis = (d0 + d1)
        bis /= max(np.linalg.norm(bis), 1e-12)
        centre = cur + bis * (r_eff / math.sin(half))
        a0 = math.atan2(*(cur + d0 * tan_len - centre)[::-1])
        a1 = math.atan2(*(cur + d1 * tan_len - centre)[::-1])
        if a1 - a0 > math.pi:
            a1 -= 2 * math.pi
        if a0 - a1 > math.pi:
            a1 += 2 * math.pi
        arc = np.linspace(a0, a1, per_arc)
        arcs.append(centre + r_eff * np.column_stack([np.cos(arc), np.sin(arc)]))

    # stitch: each arc, then the straight run to the next arc's start
    for i, arc in enumerate(arcs):
        nxt = arcs[(i + 1) % n]
        out.append(arc)
        a, b = arc[-1], nxt[0]
        if np.linalg.norm(b - a) > 1e-9:
            t = np.linspace(0.0, 1.0, per_edge)[1:-1, None]
            out.append(a + (b - a) * t)
    return np.vstack(out)


def _smooth_family(symmetry: str, n: int, radial, name: str,
                   min_concave: float) -> Outline:
    """Build an Outline from r(theta), dr, d2r callables via normal offsetting."""
    theta = _angles(symmetry, n)

    def points_at(offset: float) -> np.ndarray:
        r, dr = radial(theta)
        c, s = np.cos(theta), np.sin(theta)
        pts = np.column_stack([r * c, r * s])
        tang = np.column_stack([dr * c - r * s, dr * s + r * c])
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
        out = pts + offset * nrm
        out[0, 1] = 0.0                       # pin to the symmetry axes exactly
        if symmetry == "quarter":
            out[-1, 0] = 0.0
        else:
            out[-1, 1] = 0.0
        return out

    return Outline(points_at, n, min_concave, symmetry, name)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------


def circle(radius: float = 70.0, size: float = 6.0) -> Outline:
    """Cylindrical cup: the classical deep-drawing benchmark."""
    n = max(9, int(round(0.5 * math.pi * radius / size)) + 1)
    return _smooth_family(
        "quarter", n,
        lambda th: (np.full_like(th, radius), np.zeros_like(th)),
        f"circle_r{radius:g}", math.inf)


def ellipse(semi_major: float = 90.0, semi_minor: float = 55.0,
            size: float = 6.0) -> Outline:
    """Elliptical cup. Convex, so any outward offset is safe."""
    a, b = semi_major, semi_minor
    n = max(13, int(round(0.5 * math.pi * max(a, b) / size)) + 1)
    theta = _angles("quarter", n)

    def points_at(offset: float) -> np.ndarray:
        # parametrise by the ellipse angle so the axis endpoints stay exact
        t = theta
        pts = np.column_stack([a * np.cos(t), b * np.sin(t)])
        tang = np.column_stack([-a * np.sin(t), b * np.cos(t)])
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
        out = pts + offset * nrm
        out[0, 1] = 0.0
        out[-1, 0] = 0.0
        return out

    return Outline(points_at, n, math.inf, "quarter", f"ellipse_{a:g}x{b:g}")


def polygon(n_sides: int = 6, circumradius: float = 75.0,
            corner_radius: float = 12.0, size: float = 6.0) -> Outline:
    """Regular N-gon with rounded corners: triangle through octagon.

    Offsetting a rounded regular polygon outward by d gives another one with the
    apothem and every corner radius grown by d, so the offset is exact.
    """
    if n_sides < 3:
        raise ValueError("n_sides must be >= 3")
    symmetry = "quarter" if n_sides % 2 == 0 else "half"
    half = math.pi / n_sides
    apothem = circumradius * math.cos(half)
    r = min(corner_radius, 0.9 * apothem * math.tan(half))
    if r <= 0:
        raise ValueError("corner_radius must be > 0")
    ang = 2 * math.pi * np.arange(n_sides) / n_sides
    n = max(13, int(round(2 * math.pi * circumradius / size / (4 if symmetry == "quarter" else 2))) + 1)
    theta = _angles(symmetry, n)

    def points_at(offset: float) -> np.ndarray:
        rc = (apothem + offset) / math.cos(half)
        verts = rc * np.column_stack([np.cos(ang), np.sin(ang)])
        dense = _rounded_polygon(verts, np.full(n_sides, r + offset))
        out = _ray_cast(dense, theta)
        out[0, 1] = 0.0
        if symmetry == "quarter":
            out[-1, 0] = 0.0
        else:
            out[-1, 1] = 0.0
        return out

    return Outline(points_at, n, math.inf, symmetry,
                   f"polygon{n_sides}_R{circumradius:g}_r{r:g}")


def lobed(n_lobes: int = 4, mean_radius: float = 75.0, amplitude: float = 15.0,
          size: float = 6.0) -> Outline:
    """Clover, rose and star cups: r(theta) = R + A cos(N theta).

    The recesses are concave, so `min_concave_radius` decides whether a given
    amplitude is drawable at all.
    """
    symmetry = "quarter" if n_lobes % 2 == 0 else "half"
    th = np.linspace(0, 2 * math.pi, 720)
    r = mean_radius + amplitude * np.cos(n_lobes * th)
    dr = -amplitude * n_lobes * np.sin(n_lobes * th)
    d2r = -amplitude * n_lobes ** 2 * np.cos(n_lobes * th)
    kappa = (r ** 2 + 2 * dr ** 2 - r * d2r) / np.power(r ** 2 + dr ** 2, 1.5)
    concave = float(1.0 / np.max(-kappa[kappa < 0])) if (kappa < 0).any() else math.inf

    n = max(25, int(round(2 * math.pi * mean_radius / size
                          / (4 if symmetry == "quarter" else 2))) + 1)
    return _smooth_family(
        symmetry, n,
        lambda t: (mean_radius + amplitude * np.cos(n_lobes * t),
                   -amplitude * n_lobes * np.sin(n_lobes * t)),
        f"lobed{n_lobes}_R{mean_radius:g}_A{amplitude:g}", concave)


def rounded_rect(half_length: float = 90.0, half_width: float = 60.0,
                 corner_radius: float = 20.0, size: float = 6.0) -> Outline:
    """Rectangular or square cup, the family the box dataset uses."""
    r = min(corner_radius, 0.9 * min(half_length, half_width))
    n = max(13, int(round((half_length + half_width) / size)) + 1)
    theta = _angles("quarter", n)

    def points_at(offset: float) -> np.ndarray:
        a, b = half_length + offset, half_width + offset
        verts = np.array([[a, b], [-a, b], [-a, -b], [a, -b]])
        dense = _rounded_polygon(verts, np.full(4, r + offset))
        out = _ray_cast(dense, theta)
        out[0, 1] = 0.0
        out[-1, 0] = 0.0
        return out

    return Outline(points_at, n, math.inf, "quarter",
                   f"rect_{half_length:g}x{half_width:g}_r{r:g}")


def trapezoid(half_length: float = 90.0, half_width_a: float = 70.0,
              half_width_b: float = 40.0, corner_radius: float = 18.0,
              size: float = 6.0) -> Outline:
    """Four-sided cup, wider at one end. Symmetric about y = 0 only."""
    n = max(21, int(round(2 * (half_length + half_width_a) / size)) + 1)
    theta = _angles("half", n)
    slope = math.atan2(half_width_a - half_width_b, 2 * half_length)

    def points_at(offset: float) -> np.ndarray:
        a = half_length + offset
        # the slanted sides move out along their own normal
        wa = half_width_a + offset / math.cos(slope)
        wb = half_width_b + offset / math.cos(slope)
        verts = np.array([[a, wb], [-a, wa], [-a, -wa], [a, -wb]])
        dense = _rounded_polygon(verts, np.full(4, corner_radius + offset))
        out = _ray_cast(dense, theta)
        out[0, 1] = 0.0
        out[-1, 1] = 0.0
        return out

    return Outline(points_at, n, math.inf, "half",
                   f"trapezoid_{half_width_a:g}-{half_width_b:g}")


FAMILIES = {
    "circle": circle,
    "ellipse": ellipse,
    "polygon": polygon,
    "lobed": lobed,
    "rect": rounded_rect,
    "trapezoid": trapezoid,
}


def build(spec: dict, size: float = 6.0) -> Outline:
    """Build an outline from a catalogue entry like {'family': 'lobed', ...}."""
    spec = dict(spec)
    fn = FAMILIES[spec.pop("family")]
    return fn(size=size, **spec)


def cross(arm_half_length: float = 85.0, arm_half_width: float = 32.0,
          corner_radius: float = 14.0, inner_radius: float = 16.0,
          size: float = 6.0) -> Outline:
    """Cross / plus-shaped cup: four arms about a square centre.

    The four inner junctions are **concave**, and outward offsetting shrinks
    them, so `inner_radius` is what decides whether the shape is drawable at
    all: it must stay above t/2 + clearance + die entry radius.

    Star-convex about the origin, so the outline is still single valued in
    theta and the usual ray-cast sampling applies.
    """
    a, w = arm_half_length, arm_half_width
    if w >= a:
        raise ValueError("arm_half_width must be smaller than arm_half_length")
    n = max(25, int(round(2 * (a + w) / size)) + 1)
    theta = _angles("quarter", n)
    verts = np.array([[w, -a], [w, -w], [a, -w], [a, w], [w, w], [w, a],
                      [-w, a], [-w, w], [-a, w], [-a, -w], [-w, -w], [-w, -a]])
    # a vertex is concave where the plan turns back into the part
    concave = np.array([False, True, False, False, True, False,
                        False, True, False, False, True, False])

    def points_at(offset: float) -> np.ndarray:
        aa, ww = a + offset, w + offset
        v = np.array([[ww, -aa], [ww, -ww], [aa, -ww], [aa, ww], [ww, ww],
                      [ww, aa], [-ww, aa], [-ww, ww], [-aa, ww], [-aa, -ww],
                      [-ww, -ww], [-ww, -aa]])
        radii = np.where(concave, inner_radius - offset, corner_radius + offset)
        dense = _rounded_polygon(v, np.maximum(radii, 0.05))
        out = _ray_cast(dense, theta)
        out[0, 1] = 0.0
        out[-1, 0] = 0.0
        return out

    return Outline(points_at, n, inner_radius, "quarter",
                   f"cross_{a:g}x{w:g}_r{corner_radius:g}i{inner_radius:g}")


FAMILIES["cross"] = cross
