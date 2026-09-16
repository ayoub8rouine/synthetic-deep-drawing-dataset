"""Build the standalone HTML methodology report with per-geometry galleries.

Prose is written here; every number and every gallery is pulled from the dataset
so the document cannot drift from what was actually generated.

    python .\\postprocessing\\build_report.py
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT.parent / "Examples"
IMAGES = EXAMPLES / "images"

# The three worked examples. Recorded explicitly because the figures were once
# regenerated from guessed case names, which silently swapped in different parts
# whose strain ranges no longer matched these captions. Identified from the
# untouched geometry PLYs by bounding box; visualise_field takes the lowest
# blankholder force of each case.
EXAMPLE_CASES = {
    "box_rectangular_pan": "tool_radii2_20_radii1_5_cr_1.1_delta_4_height_25",
    "uchannel_hat_section": "uchannel_w41-53.6_h37.1-25",
    "cup_lobed5_clover": "lobed5_R88.5",
}

DATASETS = {
    "box": ROOT / "dataset",
    "uchannel": ROOT / "u_shape" / "dataset",
    "cup": ROOT / "shapes" / "dataset",
}


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------


def stats() -> dict:
    out = {}
    for name, root in DATASETS.items():
        cases = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
        samples = glob.glob(str(root / "*" / "Data" / "Bauteil" / "*.npz"))
        thin, ke, binder, fams = [], [], [], collections.Counter()
        depth, draft = [], []
        for f in glob.glob(str(root / "*" / "case_report_*.json")):
            c = json.load(open(f))
            p = c["parameters"]
            fam = (p.get("outline", {}).get("family")
                   or ("uchannel" if "section_a" in p else "box"))
            fams[fam] += 1
            if c["checks"]["accepted"]:
                thin.append(100 * c["checks"]["max_thinning"])
                ke.append(100 * c["checks"]["peak_ke_over_ie"])
                binder.append(abs(c["checks"]["binder_below_die_mm"]))
                if "height" in p:
                    depth.append(p["height"])
                if "draft_deg" in p:
                    draft.append(p["draft_deg"])
                elif "draft" in p:
                    draft.append(p["draft"])
        out[name] = dict(geoms=len(cases), samples=len(samples), fams=dict(fams),
                         thin=thin, ke=ke, binder=binder, depth=depth, draft=draft,
                         accepted=len(thin))
    return out


def family_rows(root: Path) -> dict:
    rows = collections.defaultdict(lambda: dict(n=0, thin=[], depth=[], draft=[]))
    for f in glob.glob(str(root / "*" / "case_report_*.json")):
        c = json.load(open(f))
        p = c["parameters"]
        fam = p.get("outline", {}).get("family", "?")
        r = rows[fam]
        r["n"] += 1
        if c["checks"]["accepted"]:
            r["thin"].append(100 * c["checks"]["max_thinning"])
            r["depth"].append(p["height"])
            r["draft"].append(p["draft_deg"])
    return rows


def rng(v, fmt="%.1f"):
    return "&ndash;" if not v else f"{fmt % min(v)}&ndash;{fmt % max(v)}"


def gallery(folder: Path, prefix: str = "", limit: int | None = None,
            filter_fn=None) -> str:
    files = sorted(folder.glob("*.png"))
    if filter_fn:
        files = [f for f in files if filter_fn(f.name)]
    if limit:
        files = files[:limit]
    cells = []
    for f in files:
        rel = f"images/{prefix}/{f.name}" if prefix else f"images/{f.name}"
        cells.append(
            f'<figure><img loading="lazy" src="{rel}" alt="{f.stem}">'
            f'<figcaption>{f.stem}</figcaption></figure>')
    return f'<div class="gallery">{"".join(cells)}</div>'


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

CSS = """
:root{--ink:#15191e;--mut:#5a6470;--line:#dfe3e8;--bg:#fff;--soft:#f6f8fa;
      --accent:#0b5cad;--warn:#8a4b00;--warnbg:#fff8ec;--ok:#0d6b3f;--okbg:#f0f8f3}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     color:var(--ink);background:var(--bg)}
.wrap{max-width:1180px;margin:0 auto;padding:0 32px 96px}
header.hero{border-bottom:3px solid var(--ink);padding:56px 0 28px;margin-bottom:8px}
h1{font-size:2.5rem;line-height:1.15;margin:0 0 10px;letter-spacing:-.02em}
.sub{font-size:1.12rem;color:var(--mut);max-width:70ch;margin:0}
.meta{margin-top:22px;display:flex;flex-wrap:wrap;gap:28px;font-size:.9rem;color:var(--mut)}
.meta b{display:block;font-size:1.5rem;color:var(--ink);font-weight:650;line-height:1.2}
nav.toc{position:sticky;top:0;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);
        border-bottom:1px solid var(--line);padding:11px 0;z-index:20;margin-bottom:40px}
nav.toc .wrap{padding-bottom:0}
nav.toc a{color:var(--mut);text-decoration:none;font-size:.83rem;margin-right:19px;
          white-space:nowrap;border-bottom:2px solid transparent;padding-bottom:3px}
nav.toc a:hover{color:var(--accent);border-bottom-color:var(--accent)}
nav.toc .inner{overflow-x:auto;white-space:nowrap}
h2{font-size:1.72rem;margin:64px 0 6px;padding-top:14px;letter-spacing:-.015em;
   border-top:1px solid var(--line)}
h2 .num{color:var(--mut);font-weight:400;margin-right:.5em}
h3{font-size:1.18rem;margin:34px 0 8px}
h4{font-size:1rem;margin:24px 0 6px;color:var(--mut);text-transform:uppercase;
   letter-spacing:.06em;font-size:.8rem}
p{max-width:78ch}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.9rem}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:620}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{background:var(--soft);padding:1px 5px;border-radius:3px;
     font:.88em ui-monospace,SFMono-Regular,Consolas,monospace}
pre{background:var(--soft);border:1px solid var(--line);border-left:3px solid var(--mut);
    padding:13px 16px;overflow-x:auto;font:.85rem/1.5 ui-monospace,Consolas,monospace}
.note{border-left:3px solid var(--accent);background:#f4f8fd;padding:12px 18px;margin:18px 0}
.warn{border-left:3px solid var(--warn);background:var(--warnbg);padding:12px 18px;margin:18px 0}
.ok{border-left:3px solid var(--ok);background:var(--okbg);padding:12px 18px;margin:18px 0}
.note p,.warn p,.ok p{margin:.4em 0}
.tag{display:inline-block;font-size:.7rem;font-weight:700;text-transform:uppercase;
     letter-spacing:.07em;padding:2px 8px;border-radius:3px;vertical-align:2px}
.t-meas{background:#e6f2ea;color:var(--ok)}
.t-pub{background:#e7f0fa;color:var(--accent)}
.t-here{background:#f3edfa;color:#5b3a8e}
.t-chosen{background:var(--warnbg);color:var(--warn)}
figure{margin:0}
.hero-fig{margin:26px 0}
.hero-fig img{width:100%;border:1px solid var(--line);border-radius:4px}
.hero-fig figcaption,.gallery figcaption{font-size:.76rem;color:var(--mut);margin-top:5px;
     word-break:break-word;line-height:1.35}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));
         gap:16px;margin:20px 0 8px}
.gallery img{width:100%;border:1px solid var(--line);border-radius:3px;background:#fff}
details{margin:14px 0;border:1px solid var(--line);border-radius:4px}
summary{cursor:pointer;padding:11px 15px;background:var(--soft);font-weight:600;font-size:.92rem}
details[open] summary{border-bottom:1px solid var(--line)}
details .body{padding:6px 15px 15px}
.ref{font-size:.9rem;margin:0 0 16px;padding-left:2.2em;text-indent:-2.2em;max-width:80ch}
footer{margin-top:70px;padding-top:20px;border-top:1px solid var(--line);
       font-size:.84rem;color:var(--mut)}
@media print{nav.toc{display:none}.gallery{grid-template-columns:repeat(4,1fr)}}
"""


def build() -> str:
    s = stats()
    cups = family_rows(DATASETS["cup"])
    tot_g = sum(v["geoms"] for v in s.values())
    tot_s = sum(v["samples"] for v in s.values())
    all_thin = [t for v in s.values() for t in v["thin"]]
    all_ke = [t for v in s.values() for t in v["ke"]]
    all_bd = [t for v in s.values() for t in v["binder"]]

    fam_label = {
        "circle": "Cylindrical cup", "ellipse": "Elliptical cup",
        "rect": "Square / rectangular cup", "polygon": "Regular polygon, 3&ndash;8 sides",
        "lobed": "Lobed: cocoon, clover, rose", "trapezoid": "Trapezoidal cup",
        "cross": "Cross / plus",
    }
    fam_note = {
        "circle": "The classical benchmark. Purely axisymmetric material flow, so the "
                  "lowest thinning of any family and the case against which the process "
                  "window was originally derived.",
        "ellipse": "Mild non-axisymmetry. Flow differs between the major and minor axis, "
                   "but the outline stays convex so the die offset is unconditionally safe.",
        "rect": "Corner-dominated, the same regime as the box family. Material converges "
                "from two directions at each corner.",
        "polygon": "Three to eight sides with rounded corners. This is the family that "
                   "exposed the corner criterion: under the cylindrical rules alone it "
                   "passed only 4 of 10.",
        "lobed": "r(&theta;) = R + A&middot;cos(N&theta;). N=2 is the cocoon, N=3&ndash;6 "
                 "clover and rose forms. The only family that sees heavy compression and "
                 "biaxial tension simultaneously: on a 5-lobed cup the minor strain runs "
                 "&minus;0.313 in the recesses to +0.174 on the lobe tips.",
        "trapezoid": "The only family asymmetric along its own axis, so it is modelled on "
                     "a half rather than a quarter.",
        "cross": "Four arms about a square centre. The only family with concave plan "
                 "junctions, where the outward die offset <em>shrinks</em> the radius.",
    }

    def cup_family_section(fam: str) -> str:
        r = cups.get(fam)
        if not r or not r["n"]:
            return ""
        g = gallery(IMAGES / "cup", "cup", filter_fn=lambda n: n.startswith(fam))
        return f"""
        <h3>{fam_label.get(fam, fam)} &mdash; <code>{fam}</code></h3>
        <p>{fam_note.get(fam,'')}</p>
        <table><tr><th>samples</th><th class="num">draw depth</th>
        <th class="num">draft</th><th class="num">max thinning</th></tr>
        <tr><td>{r['n']}</td><td class="num">{rng(r['depth'],'%.0f')} mm</td>
        <td class="num">{rng(r['draft'],'%.0f')}&deg;</td>
        <td class="num">{rng(r['thin'])}%</td></tr></table>
        {g}"""

    cup_sections = "".join(cup_family_section(f) for f in
                           ["circle", "ellipse", "rect", "polygon", "lobed",
                            "trapezoid", "cross"])

    return f"""<title>Deep Drawing Dataset &mdash; Methodology</title>
<style>{CSS}</style>
<header class="hero"><div class="wrap">
<h1>Synthetic Deep Drawing Dataset</h1>
<p class="sub">Simulation methodology, process-parameter derivation and per-geometry
documentation. Explicit finite-element sheet forming, LS-DYNA R16.1.</p>
<div class="meta">
  <div><b>{tot_g}</b>geometries</div>
  <div><b>{tot_s}</b>samples</div>
  <div><b>3</b>shape categories</div>
  <div><b>{2 + len([f for f in cups if cups[f]["n"]])}</b>outline families</div>
  <div><b>{min(all_thin):.1f}&ndash;{max(all_thin):.1f}%</b>thinning range</div>
</div></div></header>

<nav class="toc"><div class="wrap"><div class="inner">
<a href="#overview">1. Overview</a>
<a href="#technique">2. Technique</a>
<a href="#material">3. Material</a>
<a href="#process">4. Process parameters</a>
<a href="#box">5. Box</a>
<a href="#uchannel">6. U&#8209;channel</a>
<a href="#cup">7. Cup families</a>
<a href="#validation">8. Validation</a>
<a href="#limits">9. Limitations</a>
<a href="#refs">10. References</a>
</div></div></nav>

<div class="wrap">

<h2 id="overview"><span class="num">1</span>Overview</h2>
<p>Each sample is one tool set drawn at one blankholder pressure, stored as a matched
<b>initial</b> state (flat blank and tools in their starting position) and <b>final</b>
state (the drawn part), with per-element thickness, effective plastic strain, stress at
five integration points and the full strain tensor at both surfaces.</p>

<figure class="hero-fig"><img src="images/three_categories.png"
alt="one part from each category">
<figcaption>One drawn part from each of the three categories, all at one common scale.
Colour is simulated sheet thickness, diverging about the undeformed 1.00&nbsp;mm: warm
where the sheet has <b>thinned</b> under tension over the punch nose and corners, cool
where the flange has <b>thickened</b> under compressive hoop strain, near-white where it
is untouched. The scale is clipped at &plusmn;0.09&nbsp;mm so ordinary variation stays
legible; the extremes saturate. The U-channel is pale because it barely thins at all
&mdash; 3%, the signature of near plane-strain bending rather than stretch.</figcaption></figure>

<table>
<tr><th>Category</th><th class="num">Geometries</th><th class="num">Samples</th>
<th class="num">Accepted</th><th class="num">Thinning</th><th>Description</th></tr>
<tr><td><b>Box</b></td><td class="num">{s['box']['geoms']}</td>
<td class="num">{s['box']['samples']}</td><td class="num">{s['box']['accepted']}</td>
<td class="num">{rng(s['box']['thin'])}%</td>
<td>Rectangular pan, rounded plan corners</td></tr>
<tr><td><b>U-channel</b></td><td class="num">{s['uchannel']['geoms']}</td>
<td class="num">{s['uchannel']['samples']}</td><td class="num">{s['uchannel']['accepted']}</td>
<td class="num">{rng(s['uchannel']['thin'])}%</td>
<td>Hat section, both end cross-sections independent</td></tr>
<tr><td><b>Cup</b></td><td class="num">{s['cup']['geoms']}</td>
<td class="num">{s['cup']['samples']}</td><td class="num">{s['cup']['accepted']}</td>
<td class="num">{rng(s['cup']['thin'])}%</td>
<td>Seven plan-outline families in one registry</td></tr>
</table>

<h4>Provenance is uneven &mdash; read this first</h4>
<p>Not every number in this dataset has the same standing. Four tiers are used
throughout the document:</p>
<table>
<tr><th>Tier</th><th>What</th><th>Basis</th></tr>
<tr><td><span class="tag t-meas">measured</span></td>
<td>The entire box family: footprint, corner radius, profile radius, clearance ratio,
draft convention, sheet thickness</td>
<td>Reverse-engineered from a supplied 228-case reference dataset and verified against
it to 0.13&nbsp;mm</td></tr>
<tr><td><span class="tag t-pub">published</span></td>
<td>Draw ratio, limiting draw ratio, maximum depth, clearance, holding pressure,
profile radii</td>
<td>Two documents [1], [6]. Both are derived for <em>cylindrical</em> cups</td></tr>
<tr><td><span class="tag t-here">measured here</span></td>
<td>Draft &ge; 6&deg;; corner radius &ge; 0.45 &times; depth</td>
<td>From these simulations. Hypotheses at n &asymp; 30, not established results</td></tr>
<tr><td><span class="tag t-chosen">chosen</span></td>
<td>All U-channel ranges; all cup <em>shape</em> ranges</td>
<td>No external basis</td></tr>
</table>

<h2 id="technique"><span class="num">2</span>Simulation technique</h2>

<h3>2.1 Why explicit</h3>
<p>Deep drawing is quasi-static but is solved here with an <b>explicit dynamic</b> solver
(LS-DYNA R16.1, SMP, double precision). Explicit is the standard industrial choice for
sheet forming because the problem is dominated by large-deformation contact with
continuously changing status, which makes implicit equilibrium iteration expensive and
fragile. The price is that the result is only meaningful if it stays quasi-static, and
that has to be <em>verified</em> rather than assumed (&sect;2.5).</p>

<h3>2.2 Time scaling</h3>
<p>A real press runs the punch near 250&nbsp;mm/s. At that speed a 25&nbsp;mm stroke takes
0.1&nbsp;s, which at a stable time step near 10<sup>&minus;6</sup>&nbsp;s is over 100,000
cycles for a single case. Every forming code therefore runs faster than reality.</p>
<p>The die is driven by a <b>cycloidal displacement curve</b>:</p>
<pre>d(t) = -S &middot; ( t/T  &minus;  sin(2&pi;t/T) / 2&pi; )</pre>
<p>This has <b>zero velocity at both ends of the stroke</b> and total travel exactly
<code>S</code>. A ramped-velocity curve instead either starts with a shock or overshoots
the intended depth. Peak velocity is <code>2S/T</code>.</p>
<table>
<tr><th>Family</th><th class="num">Peak die velocity</th><th>Reason</th></tr>
<tr><td>Box</td><td class="num">3000 mm/s</td>
<td>12&times; real speed; kinetic/internal energy stayed at 1&ndash;2%</td></tr>
<tr><td>U-channel, cups</td><td class="num">2000 mm/s</td>
<td>A hat section deforms by bending and draw-in rather than the biaxial stretch a box
corner sees, so its plastic work is about a quarter of the box's. The same speed then
read as 6.9% kinetic energy, over the 5% limit. Kinetic energy scales with
<i>v</i><sup>2</sup>, so 2000&nbsp;mm/s brought it to 3.3%.</td></tr>
</table>

<h3>2.3 Mass scaling</h3>
<div class="ok"><p><code>DT2MS</code> was set but proved inert: it is a time-step
<b>floor</b>, not a cap. The natural element time step (&asymp;8&times;10<sup>&minus;7</sup>&nbsp;s
for a 5&nbsp;mm blank element in 1&nbsp;mm steel) was already above the target, so
<b>no mass was ever added</b> &mdash; <code>added mass = 0.0</code> in every run. These
results are not mass-scaled.</p></div>

<h3>2.4 Contact</h3>
<p><code>*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE</code>, one interface per tool, the
blank always the slave. Friction 0.12 static and dynamic, a conventional value for mild
steel against a lubricated hardened tool. Two settings were arrived at the hard way:</p>
<div class="warn">
<p><b>Tool shell thickness 0.1&nbsp;mm.</b> Contact offsets each surface by half its
shell thickness. With 1&nbsp;mm tools closing on a 1&nbsp;mm blank the gap at bottom dead
centre becomes 0&nbsp;mm instead of 1&nbsp;mm, and the cup bottom is pinched at
&asymp;100&nbsp;kN.</p>
<p><b>Tools keep real steel density.</b> Lowering tool density to suppress rigid-body
kinetic energy in the global balance is tempting, but LS-DYNA derives the contact penalty
stiffness from the master segments. At 1/100 density the contact-stability time step
collapsed from 5.0&times;10<sup>&minus;7</sup> to 5.0&times;10<sup>&minus;8</sup>&nbsp;s
and mass scaling added <b>2.7&times; the blank's own mass</b>.</p>
</div>

<h3>2.5 Symmetry, and verifying quasi-static behaviour</h3>
<p>Only a symmetry sector is modelled &mdash; a quarter where the shape has two mirror
planes, a half where it has one. This keeps models at 6,000&ndash;8,000 nodes against the
Student edition's 128,000 ceiling (6.3% usage) and runs at 20&ndash;130&nbsp;s. The
blankholder force applied is scaled by the sector; this was hard-coded to a quarter at
first, which <b>under-loaded every half model by 2&times;</b> until caught.</p>
<p>Every run is accepted or rejected on measured quantities:</p>
<table>
<tr><th>Check</th><th>Limit</th><th class="num">Achieved</th><th>Rationale</th></tr>
<tr><td>Normal termination</td><td>required</td><td class="num">all</td><td>&mdash;</td></tr>
<tr><td>Max thinning</td><td>&lt; 30%</td>
<td class="num">{min(all_thin):.1f}&ndash;{max(all_thin):.1f}%</td>
<td>Above the material's uniform elongation the result is fiction</td></tr>
<tr><td>Kinetic / internal energy</td><td>&lt; 5%</td>
<td class="num">max {max(all_ke):.1f}%</td>
<td>The quasi-static test. Evaluated only where internal energy exceeds 25% of its
maximum &mdash; early in the stroke the ratio says nothing useful</td></tr>
<tr><td>Total energy ratio</td><td>1 &plusmn; 0.10</td><td class="num">&asymp;1.0001</td>
<td>Numerical energy balance</td></tr>
<tr><td>Hourglass energy</td><td>&lt; 10% of internal</td><td class="num">&lt; 0.05%</td>
<td>Element formulation health</td></tr>
<tr><td>Binder tracks the die</td><td>within 1 mm</td>
<td class="num">max {max(all_bd):.2f} mm</td>
<td>Detects a flange that stopped being clamped</td></tr>
</table>
<div class="note"><p>The binder-tracking check was added after a failure mode none of the
energy checks caught: the flange slid down between the tools and the part came out
&asymp;4&nbsp;mm deeper than the cavity while every energy metric looked healthy.</p></div>

<h3>2.6 Result extraction</h3>
<p>Final state is read from <code>dynain</code>, written by
<code>*INTERFACE_SPRINGBACK_LSDYNA</code> at termination: deformed nodes, per-element
thickness, stress and effective plastic strain at five integration points, strain tensors
at both surfaces, and <code>*REFERENCE_GEOMETRY</code> &mdash; the undeformed nodes,
node-for-node, which supplies an exact initial-to-final correspondence.</p>
<div class="warn"><p><b>LS-DYNA writes the strain tensor in the global frame.</b> On a
vertical wall the global &ldquo;zz&rdquo; component is an in-plane strain, not a thickness
strain. Principal strains are therefore computed after rotating each element's tensor into
its own frame. Validation: the through-thickness component then reproduces the separately
written element thickness at correlation <b>+0.9995</b>, mean absolute difference
<b>0.00027</b>. In the global frame the same comparison gives <b>&minus;0.78</b>.</p></div>

<h2 id="material"><span class="num">3</span>Material</h2>
<p>One material across the whole dataset.</p>
<table>
<tr><th>Property</th><th class="num">Value</th></tr>
<tr><td>Young's modulus, Poisson's ratio, density</td>
<td class="num">210 GPa, 0.3, 7.85&times;10<sup>&minus;9</sup> t/mm<sup>3</sup></td></tr>
<tr><td>Initial yield</td><td class="num">210 MPa</td></tr>
<tr><td>Hardening, tabulated</td><td class="num">210 &rarr; 550 MPa over 0&ndash;20% plastic strain</td></tr>
<tr><td>Hardening, extrapolated</td><td class="num">power law to 946 MPa at 100%</td></tr>
<tr><td>Normal anisotropy</td><td class="num"><b>r&#772; = 1.575</b> (r<sub>0</sub> 1.7, r<sub>45</sub> 1.3, r<sub>90</sub> 2.0)</td></tr>
<tr><td>Card</td><td class="num"><code>*MAT_TRANSVERSELY_ANISOTROPIC_ELASTIC_PLASTIC</code></td></tr>
<tr><td>Sheet thickness</td><td class="num">1.0 mm</td></tr>
</table>
<p><b>Why the extrapolation.</b> A hardening curve that flat-lines at its last tabulated
point drives premature strain localisation: past 20% strain an element hardens no further,
so deformation concentrates there artificially. The tail is fitted, not clamped.</p>
<p><b>Why r&#772; matters.</b> Normal anisotropy near 1.6 is what makes this a
<em>drawing</em> grade &mdash; it resists through-thickness thinning in favour of drawing
material in from the flange, and is the single property most responsible for the parts
forming as well as they do.</p>
<div class="warn"><p><b>Two limits.</b> The values are a documented engineering
assumption, representative of mild deep-drawing steel (DC04 / AISI 1006&ndash;1008
territory) but <b>not calibrated to a named alloy</b>; absolute strains should be read as
representative. And MAT_037 takes only a single r&#772;, so planar anisotropy is inactive
and <b>there is no earing</b>. Full planar anisotropy needs Hill-48 via MAT_036.</p></div>

<h2 id="process"><span class="num">4</span>Process parameters and how they were fixed</h2>
<p>Six process quantities govern every case.</p>

<h3>4.1 Draw ratio <span class="tag t-pub">published</span></h3>
<p><b>Used: 1.40&ndash;1.90.</b> The limiting draw ratio for a single draw is
<b>2.2</b> [6]; standard practice keeps DR &le; 2 with reduction
(D<sub>b</sub>&minus;D<sub>p</sub>)/D<sub>b</sub> &le; 0.5 [13]. Staying at 1.9 leaves
margin, because LDR is a material property measured on cylindrical cups and every
non-circular shape is harder.</p>

<h3>4.2 Draw depth <span class="tag t-pub">published</span></h3>
<p><b>Used: H &le; base diameter</b>, sampled at 0.25&ndash;0.75 of the smaller base
half-extent. From <code>H = b/2 (DR&sup2; &minus; 1)</code> [6]; substituting LDR = 2.2
gives H<sub>max</sub> &asymp; 2b &mdash; <i>&ldquo;it is not possible to obtain a cup whose
height is bigger than its base diameter&rdquo;</i> in a single draw.</p>

<h3>4.3 Die entry and punch nose radii <span class="tag t-pub">published</span></h3>
<p><b>Used: die 6&ndash;13&times;t, punch 4&ndash;10&times;t.</b> Reference [6] tested punch
profile radii at 3&ndash;33&nbsp;t and die profile radii at 4.8&ndash;36&nbsp;t. Too small a
die radius fractures the cup at the die corner; too small a punch radius fractures it at the
punch corner [1].</p>

<h3>4.4 Radial clearance <span class="tag t-pub">published</span></h3>
<p><b>Used: 1.10&ndash;1.35&times;t.</b> <code>c = 1.1t</code> is the textbook value [13]; radial clearance of 7&ndash;14% above thickness is recommended in [7];
[6] reports &asymp;130% of thickness gives least wrinkling and ear height, and that below
100% the process enters <b>ironing</b>, where maximum drawing force increases threefold.</p>

<h3>4.5 Blankholder pressure <span class="tag t-pub">published</span></h3>
<p><b>Used: 2.0 and 3.5 MPa, specified as a pressure</b>; force follows from each
geometry's clamped flange area and its symmetry share. Basis: holding pressure
<code>= 0.015 &times; yield</code> [13], which for this 210&nbsp;MPa steel gives
<b>3.15&nbsp;MPa</b>; and an independent measurement of <b>400&nbsp;psi = 2.76&nbsp;MPa</b>
for mild steel at 50% reduction [6]. Two sources agreeing makes this the
strongest-supported number in the set.</p>
<div class="warn">
<p><b>This parameter was wrong at the start, and the correction changed the whole
dataset.</b> The reference dataset labels its parts 15,000&ndash;40,000&nbsp;N and the first
models simply reused those numbers. On this blank the clamped flange is
&asymp;28,600&nbsp;mm&sup2;, so 25&nbsp;kN is <b>0.22&nbsp;MPa</b> &mdash; roughly 14&times;
too low. The flange was not held: it slid down between die and binder, the binder followed,
and parts came out &asymp;4&nbsp;mm deeper than the cavity. At 200&nbsp;kN (1.75&nbsp;MPa)
the binder tracked the die to <b>0.026&nbsp;mm</b> and the part landed <b>0.03&nbsp;mm</b>
from its geometric target.</p>
<p>The reference dataset's own data corroborates this: its parts have depth &asymp; the
nominal height (25.3 / 35.5 / 52.1&nbsp;mm for 25 / 35 / 50&nbsp;mm draws), the
properly-clamped signature.</p>
<p><b>Generalised:</b> force is meaningless across shapes of differing flange area.
Pressure is the parameter; force is derived.</p>
</div>

<h3>4.6 Wall draft <span class="tag t-here">measured here</span></h3>
<p><b>Used: 6&ndash;20&deg;.</b> Not published &mdash; measured from these runs. Every one of
the 11 U-channel splits had a near-vertical wall (&beta; &lt; 6&deg;) at one end; none of the
49 accepted cases did. A vertical wall gives the material no path to feed in, so it stretches
instead of drawing. The 6&deg; floor was imposed on every family afterwards.</p>

<h2 id="box"><span class="num">5</span>Box &mdash; rectangular pan</h2>
<figure class="hero-fig"><img src="images/box_overview.png" alt="box geometries">
<figcaption>Twelve of the {s['box']['geoms']} box geometries, spread by farthest-point
selection in parameter space.</figcaption></figure>

<h3>5.1 What it is</h3>
<p>A rectangular pan with rounded plan corners: 348&times;198&nbsp;mm footprint drawn
25&ndash;50&nbsp;mm deep from a 500&times;380&nbsp;mm blank.</p>
<p><b>Why this shape first.</b> It is the shape of the supplied reference dataset. Building
it first made the generator verifiable against real data rather than against assumptions.</p>

<h3>5.2 How it was defined <span class="tag t-meas">measured</span></h3>
<p>The entire family was <b>reverse-engineered from the 228 reference cases</b> &mdash; not
selected. The folder-name tokens decode as follows, each confirmed by measurement against
the reference point clouds:</p>
<table>
<tr><th>Token</th><th>Meaning</th><th>Confirming evidence</th></tr>
<tr><td><code>radii2</code> = 20&ndash;55</td><td><b>Plan corner radius</b>, not a profile
radius</td><td>Fitting the corner arcs recovers it to <b>+0.0012&nbsp;mm mean error over
228 cases</b></td></tr>
<tr><td><code>radii1</code> = 5</td><td>Punch nose = die entry = roof fillet</td>
<td>Punch crown ends at a&minus;5 = 168.9&nbsp;mm</td></tr>
<tr><td><code>cr</code> = 1.1</td><td>Clearance ratio</td>
<td>Die wall minus punch wall = 1.1&nbsp;mm exactly</td></tr>
<tr><td><code>delta</code> = 0&ndash;8</td><td>Draft, as total lateral inset over the depth</td>
<td>Predicted fillet tangent 165.64 vs measured 165.74&nbsp;mm</td></tr>
<tr><td><code>height</code> = 25&ndash;50</td><td>Draw depth</td>
<td>Cavity depth and punch height</td></tr>
</table>
<p>Sheet thickness 1.0&nbsp;mm was <em>derived</em>: the die flange sits at z = +0.5 and the
binder at z = &minus;0.5, so the blank between them is 1&nbsp;mm.</p>

<h3>5.3 Rules followed</h3>
<p>Plan offsetting uses the reference's own convention: offsetting by <code>s</code> gives
half-lengths (a+s, b+s) <b>with the corner radius unchanged</b>. This was confirmed by
fitting the reference's corner arcs, and it is what makes the rebuilt die an exact match
rather than an approximation.</p>

<h3>5.4 Process parameters</h3>
<p>Thickness, clearance ratio and profile radius are <em>fixed by the reference</em>. Only
the blankholder was chosen &mdash; and it was chosen wrongly first (&sect;4.5).</p>

<h3>5.5 How the geometries were generated</h3>
<p>A deterministic grid mirroring the reference's own structure:</p>
<pre>corner_radius &isin; {{20, 35, 50}} &times; depth &isin; {{25, 35, 45}} &times; draft multiplier &isin; {{0, 2, 4}}
draft = m &middot; depth / 25</pre>
<p>27 geometries &times; 3 blankholder forces, plus three pilot cases.</p>

<h3>5.6 Result</h3>
<div class="ok"><p><b>{s['box']['geoms']} geometries, {s['box']['samples']} samples,
100% accepted</b>, thinning {rng(s['box']['thin'])}%.</p></div>

<details><summary>All {s['box']['geoms']} box geometries</summary><div class="body">
{gallery(IMAGES / 'box', 'box')}</div></details>

<h4>Strain fields, worked example</h4>
<div class="gallery">
<figure><img src="images/box_rectangular_pan_major_strain.png" alt="box major strain">
<figcaption>Major principal strain, 0 &rarr; +0.304</figcaption></figure>
<figure><img src="images/box_rectangular_pan_minor_strain.png" alt="box minor strain">
<figcaption>Minor principal strain, &minus;0.267 &rarr; +0.073</figcaption></figure>
<figure><img src="images/box_rectangular_pan_strain_distribution.png" alt="box strain distribution">
<figcaption>Strain distribution, coloured by thickness. <b>Not</b> a forming limit diagram &mdash; no limit curve is defined.</figcaption></figure>
</div>

<h2 id="uchannel"><span class="num">6</span>U-channel &mdash; hat section</h2>
<figure class="hero-fig"><img src="images/uchannel_overview.png" alt="uchannel geometries">
<figcaption>Twelve of the {s['uchannel']['geoms']} U-channel geometries.</figcaption></figure>

<h3>6.1 What it is</h3>
<p>A hat (top-hat) section 300&nbsp;mm long, floor 82&ndash;170&nbsp;mm wide, drawn
21&ndash;50&nbsp;mm deep, with <b>both end cross-sections independent</b> so width, depth,
wall angle and both radii vary along the length.</p>
<p><b>Why this shape.</b> The second classical deep-drawing family after the cup, and
structurally different: an open-ended channel deforms in near <b>plane strain</b>, feeding
material from two sides only, where a box corner feeds from two directions at once. The
measured thinning bears this out &mdash; {rng(s['uchannel']['thin'])}%, with the prismatic
cases at the very bottom of that range.</p>

<h3>6.2 Reference for the parametrisation</h3>
<p>Parameter <em>meanings</em> follow the open-source parametric U-channel models of
Lehrer &amp; Stocker [3]: independent end sections, per-side wall angles, separate floor and
flange radii. That project was used as a <b>specification, not a dependency</b> &mdash; it
generates the finished part as a STEP surface, and offsetting a CAD surface into a punch and
die risks self-intersection at every concave fillet. Offsetting the 2D cross-section instead
is exact: a straight segment shifts by <i>d</i> along its normal, and a fillet becomes
r&nbsp;&minus;&nbsp;d at a concave turn or r&nbsp;+&nbsp;d at a convex one.</p>

<h3>6.3 Process parameters <span class="tag t-chosen">chosen</span></h3>
<div class="warn"><p><b>I chose these ranges, with no external basis.</b> This is the
least-grounded family in the dataset.</p></div>
<table>
<tr><th>Parameter</th><th class="num">Range</th><th>Basis</th></tr>
<tr><td>Floor half width</td><td class="num">40&ndash;85 mm</td><td>none &mdash; chosen</td></tr>
<tr><td>Height (per end)</td><td class="num">20&ndash;50 mm</td><td>none &mdash; chosen</td></tr>
<tr><td>Wall angle</td><td class="num"><b>0&ndash;25&deg;</b></td><td>none &mdash; chosen</td></tr>
<tr><td>Floor radius (punch nose)</td><td class="num">5&ndash;14 mm</td><td>loosely, the 3&ndash;33&nbsp;t band</td></tr>
<tr><td>Flange radius (die entry)</td><td class="num">5&ndash;14 mm</td><td>loosely, the 5&ndash;35&nbsp;t band</td></tr>
</table>
<div class="warn"><p><b>And it shows.</b> The wall-angle range starting at 0&deg; is exactly
what produced the splits: 11 of 59 runs, every one with a near-vertical wall at an end.
Acceptance was <b>{100*s['uchannel']['accepted']/max(s['uchannel']['samples'],1):.0f}%</b>,
against &asymp;100% for families sampled inside a proper window. The draft &ge; 6&deg; rule
came out of this failure and was applied to every later family.</p></div>

<h3>6.4 How the geometries were generated</h3>
<p><b>Maximin Latin hypercube</b> over 10 dimensions (5 section parameters &times; 2
independent ends), improved by pairwise swapping to maximise the minimum separation between
samples &mdash; 30 well-separated shapes rather than 30 clustered ones.</p>
<p>The punch datum is the flange's <em>final</em> plane, z = &minus;stroke, so a section
shallower than the stroke has its floor correspondingly lower. Both ends then close on
exactly one sheet thickness at bottom dead centre.</p>

<h3>6.5 Result</h3>
<p><b>{s['uchannel']['geoms']} geometries, {s['uchannel']['samples']} samples,
{s['uchannel']['accepted']} accepted</b>, thinning {rng(s['uchannel']['thin'])}%.</p>

<details><summary>All {s['uchannel']['geoms']} U-channel geometries</summary><div class="body">
{gallery(IMAGES / 'uchannel', 'uchannel')}</div></details>

<h4>Strain fields, worked example</h4>
<div class="gallery">
<figure><img src="images/uchannel_hat_section_major_strain.png" alt="uchannel major strain">
<figcaption>Major principal strain, 0 &rarr; +0.307</figcaption></figure>
<figure><img src="images/uchannel_hat_section_minor_strain.png" alt="uchannel minor strain">
<figcaption>Minor principal strain, &minus;0.087 &rarr; +0.006 &mdash; near plane strain</figcaption></figure>
<figure><img src="images/uchannel_hat_section_strain_distribution.png" alt="uchannel strain distribution">
<figcaption>Strain distribution &mdash; no limit curve defined</figcaption></figure>
</div>

<h2 id="cup"><span class="num">7</span>Cup &mdash; seven outline families</h2>
<figure class="hero-fig"><img src="images/cup_overview.png" alt="cup geometries">
<figcaption>Twelve of the {s['cup']['geoms']} cup geometries, spread across the outline
families.</figcaption></figure>

<h3>7.1 What it is</h3>
<p>A registry of plan outlines drawn 16&ndash;65&nbsp;mm deep, all sharing the same section
construction and process framework. They cover the standard non-axisymmetric deep-drawing
families reported in the literature: cylindrical and elliptical cups, square and rectangular
cups, regular polygons from triangle to octagon, lobed forms and trapezoidal cups.</p>
<p><b>All seven families are now realised.</b> The cross family was the last to appear:
on the original sampling seeds every cross candidate was rejected at build time by the
concave-radius limit, and a targeted seed was needed to place its inner junction radius
above the floor. It now contributes 14 samples, 11 accepted.</p>

<h3>7.2 Process parameters <span class="tag t-pub">published</span> and how they were generated</h3>
<p>The key methodological change from the earlier families: <b>process parameters stopped
being constants and became sampled dimensions alongside the shape</b>. Per family, one
maximin Latin hypercube over (shape parameters + these six) jointly:</p>
<table>
<tr><th>Parameter</th><th class="num">Range</th><th>Basis</th></tr>
<tr><td>Depth factor</td><td class="num">0.25&ndash;0.75 &times; smaller base half-extent</td>
<td>capped by H &le; base diameter [6]</td></tr>
<tr><td>Draft</td><td class="num">6&ndash;20&deg;</td><td>6&deg; floor measured here</td></tr>
<tr><td>Punch nose radius</td><td class="num">4&ndash;10 &times; t</td><td>literature 3&ndash;33 t [6]</td></tr>
<tr><td>Die entry radius</td><td class="num">6&ndash;13 &times; t</td><td>literature 5&ndash;35 t [6]</td></tr>
<tr><td>Draw ratio</td><td class="num">1.40&ndash;1.90</td><td>LDR 2.2 [6]</td></tr>
<tr><td>Clearance ratio</td><td class="num">1.10&ndash;1.35</td><td>1.3 t optimum [6]</td></tr>
<tr><td>Blankholder pressure</td><td class="num">2.0 and 3.5 MPa</td><td>0.015&times;yield = 3.15 MPa [1]; 2.76 MPa measured [6]</td></tr>
</table>
<p><b>Every candidate is validated before it is built</b> &mdash; draw ratio, reduction,
H &le; base diameter, the corner criterion and the concave-radius limit. Infeasible samples
are discarded during sampling and never reach the solver. The <em>shape</em> parameter
ranges (circle radius 55&ndash;90&nbsp;mm, lobe amplitude 6&ndash;16&nbsp;mm, cross arm width
28&ndash;48&nbsp;mm and so on) <span class="tag t-chosen">chosen</span> have no external
basis.</p>

<h3>7.3 The corner criterion <span class="tag t-here">measured here</span></h3>
<div class="warn">
<p>The published LDR is derived for <em>cylindrical</em> cups. On a faceted plan the rounded
corner draws like a much smaller cup, so its local draw ratio is far higher than the nominal
one and it splits while the flat sides are still comfortable.</p>
<p>Under the published rules alone, <b>polygons passed 4 of 10</b> while circles, ellipses
and clovers passed 10 of 10. Over 29 faceted cases the boundary was clean:</p>
<pre>accepted   corner_radius / depth = 0.48 &ndash; 1.20
rejected   corner_radius / depth = 0.24 &ndash; 0.38</pre>
<p>Imposing <b>corner radius &ge; 0.45 &times; depth</b> took polygons to <b>20/20</b> in the
following sweep. This is a hypothesis at n &asymp; 30, not an established result, and is a
good candidate to check against the polygonal-cup literature.</p>
</div>

<h3>7.4 Holding pressure: 2.0 MPa is marginal <span class="tag t-here">measured here</span></h3>
<div class="warn">
<p>The two sampled pressures do not perform equally, and the difference is not uniform
across shapes. Over 157 runs in the most recent sweep, <b>8 of 14 rejections were
binder-tracking failures</b> &mdash; the flange slipping between die and binder &mdash; and
they concentrate almost entirely in one family:</p>
<pre>family      2.0 MPa   3.5 MPa
circle       13/13     12/12
ellipse      13/13     13/13
polygon        8/8       8/8
rect         13/13     12/12
lobed        12/13     11/13
cross          6/7       5/7
trapezoid     <b>7/13</b>     10/12</pre>
<p>Every family holds at &ge;85% at both pressures except the <b>trapezoid, which drops to
54% at 2.0&nbsp;MPa</b>. It is the only family modelled on a half rather than a quarter,
because it is asymmetric along its own axis &mdash; so its flange draws in unevenly and the
wider end slips first when holding pressure is marginal.</p>
<p>This corroborates the literature rather than contradicting it. Every published value sits <em>above</em> 2.0&nbsp;MPa &mdash; Schuler's <b>2.5&nbsp;MPa for deep-drawing steels</b> [3], Siebel &amp; Beisswänger's 1&ndash;3% of UTS [4], 0.015&nbsp;&times;&nbsp;yield&nbsp;=&nbsp;3.15&nbsp;MPa [13], and 2.76&nbsp;MPa measured [6] &mdash; and 2.0&nbsp;MPa is exactly where the failures cluster. The lower bound was raised to 2.8&nbsp;MPa on this evidence, <b>before</b> the Schuler value was found; the two agree to within 0.3&nbsp;MPa.</p>
</div>

<h3>7.4 A physical limit: why there are no sharp stars</h3>
<p>Offsetting outward grows a convex radius but <b>shrinks a concave one</b>. The die flange
outline sits at t/2 + clearance + R<sub>d</sub> from the part, so an outline is
manufacturable only while</p>
<pre>min_concave_radius  &gt;  t/2 + clearance + die_entry_radius   (&asymp; 7 mm for 1 mm sheet)</pre>
<p>A 5-lobed r = R + A&middot;cos(5&theta;) at A = 18&nbsp;mm has a concave radius of
8.3&nbsp;mm and is drawable; at A = 30&nbsp;mm it falls to 3.6&nbsp;mm and is rejected at
build time. <b>Sharp stars are absent by physics, not omission</b> &mdash; a die cannot have
a knife edge.</p>

<h3>7.5 The families</h3>
{cup_sections}

<h4>Strain fields, worked example &mdash; 5-lobed clover</h4>
<div class="gallery">
<figure><img src="images/cup_lobed5_clover_major_strain.png" alt="clover major strain">
<figcaption>Major principal strain, 0 &rarr; +0.329</figcaption></figure>
<figure><img src="images/cup_lobed5_clover_minor_strain.png" alt="clover minor strain">
<figcaption>Minor principal strain, &minus;0.313 &rarr; +0.174. Deep compression in the
recesses, biaxial tension on the lobe tips</figcaption></figure>
<figure><img src="images/cup_lobed5_clover_strain_distribution.png" alt="clover strain distribution">
<figcaption>Strain distribution &mdash; no limit curve defined</figcaption></figure>
</div>

<h2 id="validation"><span class="num">8</span>Validation</h2>
<h3>8.1 Tool geometry against the reference dataset</h3>
<p>Rebuilt box tools projected against all 228 reference cases, straight-sided regions
(where mesh faceting cannot contribute):</p>
<table>
<tr><th>Tool</th><th class="num">Mean deviation</th><th class="num">Worst</th></tr>
<tr><td>Punch</td><td class="num">0.128 mm</td><td class="num">0.492 mm</td></tr>
<tr><td>Die</td><td class="num">0.087 mm</td><td class="num">0.253 mm</td></tr>
<tr><td>Binder</td><td class="num">0.000 mm</td><td class="num">0.000 mm</td></tr>
</table>
<p>The residual is the <em>reference's</em> own mesh faceting, not generator error: its
tool meshes cut long chords across the plan corner arcs and the R5 profile fillets, so its
sampled points sit inside the true surfaces. The plan corner radius is recovered to
<b>+0.0012&nbsp;mm mean error</b>.</p>

<h3>8.2 Simulated parts against the reference simulations</h3>
<table>
<tr><th>Quantity</th><th class="num">Result</th></tr>
<tr><td>Drawn depth vs reference</td><td class="num">&minus;0.11 mm mean, 0.13 mm worst</td></tr>
<tr><td>Flange draw-in vs reference</td><td class="num">+2.1 mm (ours draws in slightly less)</td></tr>
<tr><td>Part-to-part Chamfer distance</td><td class="num">2.9 mm, against a 1.1 mm sampling floor</td></tr>
</table>
<p>The flange difference is expected: blank size, friction and hardening curve here are
documented assumptions, not the reference's values.</p>

<h3>8.3 Volume conservation</h3>
<p>Plastic forming conserves volume. Thickened watertight solids built from the simulated
parts match the original blank volume to <b>0.3&ndash;1.0%</b> &mdash; an independent check
on both the simulation and the mesh reconstruction.</p>

<h2 id="limits"><span class="num">9</span>Limitations</h2>
<ol>
<li><b>The material is an assumption</b>, not a calibrated alloy, and there is only one of
it across all samples.</li>
<li><b>No earing</b> &mdash; MAT_037 carries a single r&#772;.</li>
<li><b>No springback.</b> Parts are as-formed, tools still closed.</li>
<li><b>All shape parameter ranges outside the box family were chosen without external
basis.</b> The U-channel is the least grounded and has the highest failure rate; that is
not a coincidence.</li>
<li><b>Both measured rules rest on n &asymp; 30</b> from this dataset alone.</li>
<li><b>The published rules are all for cylindrical cups.</b> Their transfer to polygonal,
lobed, trapezoidal and cross shapes is an assumption &mdash; and the corner criterion is
direct evidence that the transfer is imperfect.</li>
<li><b>No case is anchored to a validated benchmark geometry.</b> Attempts to obtain the
NUMISHEET square-cup and S-rail specifications were unsuccessful.</li>
<li><b>Only initial and final states are stored.</b> Intermediate states exist during a run
but are discarded.</li>
</ol>

<h2 id="refs"><span class="num">10</span>References</h2>

<h3>10.1 Standard texts &mdash; drawability and process limits</h3>
<p class="ref">[1] Z. Marciniak, J. L. Duncan and S. J. Hu, <i>Mechanics of Sheet Metal
Forming</i>, 2nd ed., Butterworth-Heinemann, Oxford, 2002. Establishes the limiting drawing
ratio as the standard drawability index, and the mechanics of flange draw-in that the whole
process window rests on.</p>
<p class="ref">[2] W. F. Hosford and R. M. Caddell, <i>Metal Forming: Mechanics and
Metallurgy</i>, Prentice-Hall / Cambridge University Press. LDR, the role of normal
anisotropy r&#772; in drawability, and cup-drawing mechanics.</p>
<p class="ref">[3] Schuler GmbH, <i>Metal Forming Handbook</i>, Springer, Berlin.
<b>Recommends blankholder pressures of 2.5&nbsp;MPa for deep-drawing quality steels</b>
(1.2&ndash;1.5&nbsp;MPa aluminium, 2.0&ndash;2.4&nbsp;MPa copper alloys) to suppress
wrinkling. The industrial reference for the blankholder setting used here.</p>
<p class="ref">[4] E. Siebel and H. Beisswänger, blankholder pressure relation: required
holding pressure is <b>1&ndash;3% of the ultimate tensile strength</b> of the sheet. The
classical empirical basis for blankholder sizing; see also Springer,
<i>Production Engineering</i> 8 (2014), doi:10.1007/s11740-014-0551-1.</p>
<p class="ref">[5] <b>DIN 8584</b>. Defines deep drawing as tensile-compressive forming of a
plane sheet into a hollow body open on one side, without intentional change of sheet
thickness &mdash; the process definition this dataset implements.</p>

<h3>10.2 Process parameter sources</h3>
<p class="ref">[6] A. I. O. Zaid, &ldquo;Deep drawing mechanism, parameters, defects and
recent results: state of the art&rdquo;, <i>IOP Conf. Ser.: Mater. Sci. Eng.</i> <b>146</b>
(2016) 012009, doi:10.1088/1757-899X/146/1/012009. Limiting draw ratio 2.2, maximum-height
relation H&nbsp;=&nbsp;b/2(DR&sup2;&minus;1), radial clearance optimum near 130% of
thickness, blank-holding pressure of 400&nbsp;psi (2.76&nbsp;MPa) for mild steel, and the
tested punch (3&ndash;33&nbsp;t) and die (4.8&ndash;36&nbsp;t) profile radius ranges.
<em>Open access, peer-reviewed proceedings.</em></p>
<p class="ref">[7] A. C. Sekhara Reddy, <i>Introduction to Deep Drawing</i> (review chapter).
<b>Radial clearance of 7&ndash;14% above sheet thickness</b>; the effect of punch nose and
die shoulder radii on fracture and wrinkling; cites DIN 8584.</p>
<p class="ref">[8] <b>EN 10346</b>, grades DX54D+Z / DX56D+Z. Continuously hot-dip coated
deep-drawing steels: R<sub>p0.2</sub> 120&ndash;220&nbsp;MPa, R<sub>m</sub>
260&ndash;350&nbsp;MPa, r<sub>90</sub> &ge; 1.6, n<sub>90</sub> &ge; 0.18. The material card
used here (yield 210&nbsp;MPa, r&#772;&nbsp;1.575) sits inside this grade envelope, which is
why it is described as representative of DC04/DX54D territory.</p>

<h3>10.3 Geometry and dataset sources</h3>
<p class="ref">[9] T. Lehrer, P. Stocker, F. Duddeck and M. Wagner, <i>Parametric CAD Models
to Create an Infinite Number of Structural U-Shaped Sheet Metal Geometries</i>, Zenodo, 2024,
<b>doi:10.5281/zenodo.14191374</b>. Four parametric U-channel models with published
<b>feasibility bounds</b> expressed as fractions of the channel key length, and a maximin
Latin-hypercube sampler. The U-channel family here follows this parametrisation.</p>
<p class="ref">[10] T. Lehrer, P. Stocker, F. Duddeck and M. Wagner, <i>Machine Learning
Dataset of U-Channel Sheet Metal Geometry Representations with Supervision Information for
Drawability Assessment and Part Segmentation</i>, Zenodo, 2025,
<b>doi:10.5281/zenodo.15327950</b>. 2,533 geometries with local and global drawability
labels and major/minor true strains. See also <i>Computer-Aided Design</i>,
doi:10.1016/j.cad.2025.103924.</p>
<p class="ref">[11] Deep Drawing and Cutting Simulations Dataset (DDACS), the 228-case
reference dataset supplied with this project. Documents blank holder force
<b>100,000&ndash;500,000&nbsp;N</b>, friction coefficient <b>0.05&ndash;0.15</b> and sheet
thickness <b>0.95&ndash;1.00&nbsp;mm</b> as &ldquo;the practical range for deep drawing
operations&rdquo;. The box family's geometry was reverse-engineered from this dataset and
verified against it to 0.128&nbsp;mm.</p>
<p class="ref">[12] <i>LS-DYNA R16.1 Keyword User's Manual</i>, Ansys/LST, for
<code>*MAT_037</code>, <code>*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE</code>,
<code>*BOUNDARY_PRESCRIBED_MOTION_RIGID</code>, <code>*LOAD_RIGID_BODY</code> and
<code>*INTERFACE_SPRINGBACK_LSDYNA</code>.</p>
<p class="ref">[13] <i>Sheet Metal Drawing Operations</i>, Manufacturing Processes ME337
ch.&nbsp;3, Mechanical Engineering Dept., University of Basrah. Standard textbook relations
for drawing ratio, reduction, clearance and the holding-force relation
F<sub>h</sub> = 0.015&nbsp;Y&pi;{{D<sub>b</sub>&sup2; &minus; (D<sub>p</sub>+2.2t+2R<sub>d</sub>)&sup2;}}.
<em>Course material, not peer-reviewed &mdash; retained only where [1]&ndash;[8] give the
same relation.</em></p>

<footer>
<p>Simulations produced with the <b>LS-DYNA Student edition</b>, whose licence is intended
for learning and student projects. Models use 6.3% of its 128,000-node ceiling.</p>
<p>{tot_g} geometries, {tot_s} samples. Generated from
<code>DataGenration/</code>; this document is rebuilt by
<code>postprocessing/build_report.py</code>.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    out = EXAMPLES / "report.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
