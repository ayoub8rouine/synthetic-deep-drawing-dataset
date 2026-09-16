"""Two-page method note: the abstract-level methodology, one block per geometry set.

Deliberately short. It states what was done and on what basis, and stops there --
the full treatment is report.html and the review brief. Numbers are pulled from
the dataset so the note cannot drift from what was actually generated.

    python .\\postprocessing\\build_method_note.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_report import EXAMPLES, rng, stats  # noqa: E402

CSS = """
:root{
  --ground:#fbfbfc; --sheet:#ffffff; --sunk:#f2f4f6;
  --ink:#191c20; --ink-2:#3d454e; --muted:#6d7681; --faint:#98a1ab;
  --rule:#dde1e6; --rule-2:#c8cdd4;
  --accent:#15616d; --accent-soft:#e4f0f2; --accent-ink:#0d454e;
  --flag:#a85d00; --flag-soft:#fbf0e0;
  --good:#2f6b4f; --good-soft:#e6f1eb;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101315; --sheet:#1b1f23; --sunk:#242a2f;
    --ink:#e9ecef; --ink-2:#c2c9d0; --muted:#8d97a1; --faint:#69737d;
    --rule:#2c3238; --rule-2:#3d444b;
    --accent:#5cb3c0; --accent-soft:#163236; --accent-ink:#8ad0da;
    --flag:#d69340; --flag-soft:#33260f;
    --good:#6cb894; --good-soft:#172c23;
  }
}
:root[data-theme="dark"]{
  --ground:#101315; --sheet:#1b1f23; --sunk:#242a2f;
  --ink:#e9ecef; --ink-2:#c2c9d0; --muted:#8d97a1; --faint:#69737d;
  --rule:#2c3238; --rule-2:#3d444b;
  --accent:#5cb3c0; --accent-soft:#163236; --accent-ink:#8ad0da;
  --flag:#d69340; --flag-soft:#33260f;
  --good:#6cb894; --good-soft:#172c23;
}

*{box-sizing:border-box}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font:400 10.2pt/1.45 "IBM Plex Sans",-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.page{
  background:var(--sheet);max-width:210mm;min-height:297mm;margin:22px auto;
  padding:16mm 16mm 14mm;border:1px solid var(--rule);
  box-shadow:0 1px 3px rgba(25,28,32,.07);
}
h1,h2,h3{font-family:Archivo,"IBM Plex Sans",sans-serif;margin:0;text-wrap:balance}
p{margin:0 0 .55em}

/* masthead */
.mast{border-bottom:2px solid var(--ink);padding-bottom:9px;margin-bottom:14px}
.eyebrow{
  font-family:"IBM Plex Mono",monospace;font-size:7.4pt;font-weight:500;
  letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin:0 0 6px;
}
h1{font-size:19pt;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.subtitle{margin:5px 0 0;font-size:10.4pt;color:var(--muted)}
.idline{
  margin-top:9px;display:flex;flex-wrap:wrap;gap:3px 22px;
  font-family:"IBM Plex Mono",monospace;font-size:7.6pt;color:var(--muted);
}
.idline b{color:var(--ink);font-weight:500}

/* section heads */
h2{
  font-size:11.4pt;font-weight:600;letter-spacing:-.01em;margin:16px 0 6px;
  padding-bottom:3px;border-bottom:1px solid var(--rule-2);
  display:flex;align-items:baseline;gap:9px;
}
h2 .n{font-family:"IBM Plex Mono",monospace;font-size:7.8pt;font-weight:500;color:var(--faint)}
.page:first-child h2:first-of-type{margin-top:0}

/* abstract */
.abstract{
  background:var(--sunk);border-left:3px solid var(--accent);
  padding:11px 15px;margin:0 0 4px;
}
.abstract p{font-size:9.9pt;color:var(--ink-2);margin:0}
.abstract p + p{margin-top:.5em}

/* compact step list */
.steps{display:grid;grid-template-columns:1fr 1fr;gap:9px 22px;margin:8px 0 2px}
.stp{display:grid;grid-template-columns:1.5rem 1fr;gap:0 9px}
.stp .k{font-family:"IBM Plex Mono",monospace;font-size:8.6pt;font-weight:600;color:var(--accent)}
.stp b{display:block;font-size:9.6pt;color:var(--ink);margin-bottom:1px}
.stp span{font-size:9pt;color:var(--ink-2);display:block}

/* tables */
table{border-collapse:collapse;width:100%;font-size:9pt;margin:7px 0 3px}
th,td{border-bottom:1px solid var(--rule);padding:4.5px 9px 4.5px 0;text-align:left;vertical-align:top}
th{
  font-family:"IBM Plex Mono",monospace;font-size:7.2pt;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--rule-2);
}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Mono",monospace;font-size:8.6pt;padding-right:0}
tr:last-child td{border-bottom:none}
code{font-family:"IBM Plex Mono",monospace;font-size:.9em;color:var(--accent-ink)}

/* geometry-set blocks */
.set{border:1px solid var(--rule);margin:9px 0 0;break-inside:avoid}
.set-hd{
  background:var(--sunk);border-bottom:1px solid var(--rule);
  padding:6px 12px;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
}
.set-hd h3{font-size:10.8pt;font-weight:700;letter-spacing:-.01em}
.set-hd .meta{
  font-family:"IBM Plex Mono",monospace;font-size:7.8pt;color:var(--muted);margin-left:auto;
}
.set dl{margin:0;padding:2px 12px 8px}
.set .r{display:grid;grid-template-columns:6.6rem 1fr;gap:10px;padding:4.5px 0;
  border-bottom:1px solid var(--rule)}
.set .r:last-child{border-bottom:none}
.set dt{
  font-family:"IBM Plex Mono",monospace;font-size:7.2pt;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--faint);padding-top:1.5px;
}
.set dd{margin:0;font-size:9.1pt;color:var(--ink-2)}
.set dd b{color:var(--ink);font-weight:600}

/* tags */
.tag{
  font-family:"IBM Plex Mono",monospace;font-size:6.8pt;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;padding:1.5px 5px;border-radius:2px;
  white-space:nowrap;
}
.t-meas{background:var(--good-soft);color:var(--good)}
.t-pub{background:var(--accent-soft);color:var(--accent-ink)}
.t-chosen{background:var(--flag-soft);color:var(--flag)}

/* notes */
.note{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:8px 13px;margin:9px 0 2px;font-size:9pt;color:var(--ink-2)}
.note.warn{border-left-color:var(--flag);background:var(--flag-soft)}
.note b{color:var(--ink)}
.note p{margin:0}
.note p + p{margin-top:.45em}

.refs{font-size:8.4pt;color:var(--muted);margin-top:3px}
.refs p{margin:0 0 4px;padding-left:1.7em;text-indent:-1.7em}
.foot{
  margin-top:14px;padding-top:7px;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:7.4pt;color:var(--faint);
  display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
}

@media print{
  body{background:#fff}
  .page{margin:0;border:none;box-shadow:none;min-height:auto;padding:0;
    max-width:none;break-after:page}
  .page:last-child{break-after:auto}
  @page{size:A4;margin:14mm}
}
@media (max-width:640px){
  .page{padding:14px}
  .steps{grid-template-columns:1fr}
  .set .r{grid-template-columns:1fr;gap:1px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def build() -> str:
    s = stats()
    tot_g = sum(v["geoms"] for v in s.values())
    tot_s = sum(v["samples"] for v in s.values())
    b, u, c = s["box"], s["uchannel"], s["cup"]

    def acc(v):
        return f"{v['accepted']}/{v['samples']}"

    return f"""<title>Deep Drawing Method Note</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>{CSS}</style>

<!-- ======================= PAGE 1 ======================= -->
<div class="page">
  <div class="mast">
    <p class="eyebrow">Method note &middot; two pages</p>
    <h1>Synthetic Deep Drawing Dataset</h1>
    <p class="subtitle">Geometry generation and forming simulation methodology,
    summarised per geometry set.</p>
    <div class="idline">
      <span>Geometries <b>{tot_g}</b></span>
      <span>Samples <b>{tot_s}</b></span>
      <span>Categories <b>3</b></span>
      <span>Solver <b>LS-DYNA R16.1 explicit</b></span>
      <span>Sheet <b>1.0 mm, r&#772; 1.575</b></span>
    </div>
  </div>

  <div class="abstract">
    <p>Each sample is one tool set drawn at one blankholder pressure, stored as a matched
    <b>initial</b> state (flat blank, tools at start) and <b>final</b> state (drawn part), with
    per-element thickness, principal strains, effective plastic strain and stress at five
    integration points. Tools are generated <b>analytically</b> &mdash; no CAD kernel, no B-rep,
    no meshing step for formed surfaces &mdash; by sweeping a parametric plan outline against a
    section profile directly into a structured quad mesh, with the die produced as an exact
    normal offset of the punch. Three geometry sets are covered: a rectangular pan, a hat
    section, and a registry of cup plan-outlines. Provenance differs between them and is stated
    per set; only the pan is verified against external reference data.</p>
  </div>

  <h2><span class="n">01</span>Common method</h2>
  <div class="steps">
    <div class="stp"><span class="k">01</span><div><b>Plan outline</b>
      <span>Each family exposes <code>points_at(offset)</code>, returning the outline pushed out
      by a normal distance at a fixed point count and fixed polar angles, endpoints pinned to
      the symmetry axes.</span></div></div>
    <div class="stp"><span class="k">02</span><div><b>Section profile</b>
      <span>Straight segments and circular fillets in r&ndash;z: flange, die entry radius,
      drafted wall, punch nose radius, floor.</span></div></div>
    <div class="stp"><span class="k">03</span><div><b>Sweep</b>
      <span>The outline is re-evaluated at each station's offset; rows stack into a structured
      quad grid that welds by construction &mdash; no merge tolerance, no stitching.</span></div></div>
    <div class="stp"><span class="k">04</span><div><b>Offset</b>
      <span>Punch at &minus;t/2, die at +t/2&nbsp;+&nbsp;clearance. Offsetting straight segments
      and arcs is exact arithmetic, so punch and die cannot fail to mate.</span></div></div>
  </div>

  <div class="note">
    <p><b>Why not a CAD kernel.</b> Offsetting a B-rep surface into a punch and die
    self-intersects at every concave fillet and needs trimming and healing. Offsetting the 2-D
    section first is exact: a straight segment shifts by <code>d</code>, a convex fillet becomes
    <code>r+d</code>, a concave one <code>r&minus;d</code>. The cost is that no STEP or IGES
    exists &mdash; the mesh is the model. The deliverable was never a CAD part.</p>
  </div>

  <h2><span class="n">02</span>Simulation</h2>
  <p>Double-action inverted draw on one symmetry sector (quarter, or half where the shape has a
  single mirror plane). Blank midsurface at z&nbsp;=&nbsp;0; punch fixed; die and binder travel
  down by the draw depth on a <b>cycloidal displacement curve</b> with zero velocity at both
  ends. Blankholding is specified as a <b>pressure</b>, with force derived from each geometry's
  clamped flange area &mdash; force alone is not comparable across shapes of differing flange
  area. Contact is <code>*CONTACT_FORMING_ONE_WAY_SURFACE_TO_SURFACE</code>, friction 0.12,
  blank always slave. Material is <code>*MAT_037</code>, yield 210&nbsp;MPa, r&#772;&nbsp;1.575,
  a documented engineering assumption rather than a calibrated alloy. Final state is read from
  <code>dynain</code>; principal strains are computed after rotating each element's tensor into
  its own frame, which reproduces the separately written thickness at correlation 0.9995.</p>

  <table>
    <tr><th>Acceptance check</th><th class="n">Limit</th><th class="n">Achieved</th></tr>
    <tr><td>Kinetic / internal energy</td><td class="n">&lt; 5%</td>
        <td class="n">max {max(t for v in s.values() for t in v['ke']):.1f}%</td></tr>
    <tr><td>Maximum thinning</td><td class="n">&lt; 30%</td>
        <td class="n">{rng([t for v in s.values() for t in v['thin']])}%</td></tr>
    <tr><td>Hourglass energy</td><td class="n">&lt; 10% of internal</td><td class="n">&lt; 0.05%</td></tr>
    <tr><td>Total energy ratio</td><td class="n">1 &plusmn; 0.10</td><td class="n">&asymp; 1.0001</td></tr>
    <tr><td>Binder tracks the die</td><td class="n">within 1 mm</td>
        <td class="n">max {max(t for v in s.values() for t in v['binder']):.2f} mm</td></tr>
  </table>

  <h2><span class="n">03</span>The three sets at a glance</h2>
  <table>
    <tr><th>Set</th><th>Form</th><th class="n">Geometries</th><th class="n">Accepted</th>
        <th class="n">Thinning</th><th>Design basis</th></tr>
    <tr><td><b>Box</b></td><td>rectangular pan</td><td class="n">{b['geoms']}</td>
        <td class="n">{acc(b)}</td><td class="n">{rng(b['thin'])}%</td>
        <td><span class="tag t-meas">measured</span></td></tr>
    <tr><td><b>U-channel</b></td><td>hat section</td><td class="n">{u['geoms']}</td>
        <td class="n">{acc(u)}</td><td class="n">{rng(u['thin'])}%</td>
        <td><span class="tag t-chosen">chosen</span></td></tr>
    <tr><td><b>Cup</b></td><td>7 plan outlines</td><td class="n">{c['geoms']}</td>
        <td class="n">{acc(c)}</td><td class="n">{rng(c['thin'])}%</td>
        <td><span class="tag t-pub">published</span> <span class="tag t-chosen">chosen</span></td></tr>
  </table>

  <div class="note warn">
    <p><b>Read the provenance, not just the counts.</b> <span class="tag t-meas">measured</span>
    reverse-engineered from external reference data and verified against it.
    <span class="tag t-pub">published</span> taken from literature derived for cylindrical cups.
    <span class="tag t-chosen">chosen</span> set by us with no external basis. The U-channel is
    the least grounded set and has the lowest acceptance rate; that is not a coincidence.</p>
  </div>

  <div class="foot"><span>Method note &middot; page 1 of 2</span>
    <span>Full treatment: report.html &middot; METHODOLOGY.md &middot; GEOMETRY.md</span></div>
</div>

<!-- ======================= PAGE 2 ======================= -->
<div class="page">
  <h2><span class="n">04</span>Method per geometry set</h2>

  <div class="set">
    <div class="set-hd"><h3>Box &mdash; rectangular pan</h3>
      <span class="tag t-meas">measured</span>
      <span class="meta">{b['geoms']} geometries &middot; {b['samples']} samples &middot; {acc(b)} accepted</span></div>
    <dl>
      <div class="r"><dt>Form</dt><dd>348 &times; 198 mm footprint with rounded plan corners,
        drawn 25&ndash;50 mm deep from a 500 &times; 380 mm blank.</dd></div>
      <div class="r"><dt>Origin</dt><dd><b>Reverse-engineered from a supplied 228-case reference
        dataset.</b> Plan corner radius, profile radius, clearance ratio, draft convention and
        sheet thickness were decoded from its folder tokens and confirmed by measurement against
        its point clouds. Nothing here was chosen.</dd></div>
      <div class="r"><dt>Parameters</dt><dd>Deterministic 3&times;3&times;3 grid mirroring the
        reference's own structure: corner radius {{20, 35, 50}} &times; depth {{25, 35, 45}} mm
        &times; draft multiplier {{0, 2, 4}}, with draft = m&nbsp;&middot;&nbsp;depth/25.</dd></div>
      <div class="r"><dt>Verification</dt><dd>Rebuilt tools projected onto all 228 reference
        cases: punch <b>0.128 mm</b> mean deviation, die <b>0.087 mm</b>, binder <b>0.000 mm</b>.
        Plan corner radius recovered to <b>+0.0012 mm</b>. Residual is the reference's own mesh
        faceting, not generator error.</dd></div>
      <div class="r"><dt>Result</dt><dd>{acc(b)} accepted, thinning {rng(b['thin'])}%. Drawn depth
        matches the reference simulations to &minus;0.11 mm mean.</dd></div>
    </dl>
  </div>

  <div class="set">
    <div class="set-hd"><h3>U-channel &mdash; hat section</h3>
      <span class="tag t-chosen">chosen</span>
      <span class="meta">{u['geoms']} geometries &middot; {u['samples']} samples &middot; {acc(u)} accepted</span></div>
    <dl>
      <div class="r"><dt>Form</dt><dd>Hat section 300 mm long, floor 82&ndash;170 mm wide, drawn
        21&ndash;50 mm deep, with <b>both end cross-sections independent</b> &mdash; width, depth,
        wall angle and both radii vary along the length.</dd></div>
      <div class="r"><dt>Origin</dt><dd>Parameter meanings follow the open-source parametric
        U-channel models of Lehrer &amp; Stocker [3], used as a <b>specification, not a
        dependency</b>: that project emits a finished STEP part, and offsetting a CAD surface
        into tooling self-intersects at concave fillets. Dimension ranges were chosen by us.</dd></div>
      <div class="r"><dt>Parameters</dt><dd>Per end: floor half-width 40&ndash;85 mm, height
        20&ndash;50 mm, wall angle 0&ndash;25&deg;, floor radius 5&ndash;14 mm, flange radius
        5&ndash;14 mm.</dd></div>
      <div class="r"><dt>Sampling</dt><dd><b>Maximin Latin hypercube over 10 dimensions</b>
        (5 parameters &times; 2 independent ends), improved by pairwise swaps that maximise the
        minimum separation. Buildability is tested by <b>constructing the section at 11 stations</b>
        along the loft, because an interpolated station can have overlapping fillets while both
        ends are fine.</dd></div>
      <div class="r"><dt>Result</dt><dd>{acc(u)} accepted, thinning {rng(u['thin'])}%. Near
        plane-strain deformation: material feeds from two sides only, so thinning stays low.
        <b>Every failed case had a near-vertical wall</b> at one end &mdash; the source of the
        draft &ge; 6&deg; rule applied to all later sets.</dd></div>
    </dl>
  </div>

  <div class="set">
    <div class="set-hd"><h3>Cup &mdash; plan-outline registry</h3>
      <span class="tag t-pub">published</span> <span class="tag t-chosen">chosen</span>
      <span class="meta">{c['geoms']} geometries &middot; {c['samples']} samples &middot; {acc(c)} accepted</span></div>
    <dl>
      <div class="r"><dt>Form</dt><dd>Seven plan-outline families sharing one section
        construction, drawn 16&ndash;65 mm deep: circle, ellipse, rounded rectangle, polygon
        (3&ndash;8 sides), lobed <code>r = R + A&middot;cos&nbsp;N&theta;</code>, trapezoid,
        cross &mdash; all seven realised in the dataset.</dd></div>
      <div class="r"><dt>Origin</dt><dd>Families are the standard non-axisymmetric deep-drawing
        shapes reported in the literature. <b>Process</b> ranges follow published single-draw
        limits [1][2]; <b>shape</b> ranges were chosen by us.</dd></div>
      <div class="r"><dt>Parameters</dt><dd>Shape and process are sampled <b>jointly</b>, not
        held constant: depth 0.25&ndash;0.75 &times; base half-extent, draft 6&ndash;20&deg;,
        punch nose 4&ndash;10 t, die entry 6&ndash;13 t, draw ratio 1.40&ndash;1.90, clearance
        1.10&ndash;1.35 t, blankholder 2.0 and 3.5 MPa.</dd></div>
      <div class="r"><dt>Rules</dt><dd>Every candidate is validated <b>before</b> construction and
        discarded if infeasible: draw ratio &le; 2.0, reduction &le; 0.5, H &le; base diameter,
        <b>corner radius &ge; 0.45 &times; depth</b>, and a concave-radius floor of
        <code>t/2 + clearance + R_die</code> &asymp; 7 mm &mdash; which is why no sharp stars
        exist. A die cannot have a knife edge.</dd></div>
      <div class="r"><dt>Result</dt><dd>{acc(c)} accepted, thinning {rng(c['thin'])}%. The corner
        rule was derived here: under published rules alone polygons passed 4 of 10; with it
        imposed, 20 of 20.</dd></div>
    </dl>
  </div>

  <h2><span class="n">05</span>Limitations</h2>
  <p>The material card is an engineering assumption, not a calibrated alloy, and is identical
  across all samples; <code>*MAT_037</code> carries a single r&#772;, so there is <b>no earing</b>.
  <b>Springback is not simulated</b> &mdash; parts are as-formed with tools closed. Only initial
  and final states are stored. Blank elements are 4&ndash;5 mm against die radii of
  5&ndash;14 mm, giving a <b>median 2.5 elements across the die radius</b> with no adaptive
  refinement, below the 5&ndash;8 usually expected. A convergence study over three box cases
  at 4 / 2 / 1 mm found <b>draw-in converged to within 0.3 mm</b>, but the <b>thinning field
  not converged</b>: the 99th-percentile thinning falls monotonically with refinement
  (13.5 &rarr; 13.0 &rarr; 10.1 % on one case) and is still moving at 1 mm, so the production
  mesh <b>over-states thinning</b> rather than under-stating it. Peak thinning is
  non-monotonic across refinement (&plusmn;2 pp) because it is a single-element extremum
  &mdash; which is a concern, since acceptance is screened on exactly that quantity.
  <b>No experimental validation has been performed</b>, and no case is anchored to a
  validated benchmark. Rejection is on raw thinning,
  not FLD margin: <b>no forming limit curve is defined</b>. Both empirically derived rules rest
  on n&nbsp;&asymp;&nbsp;30 from this dataset alone, and the published window they supplement is
  derived for cylindrical cups, so its transfer to faceted and lobed plans is an assumption.</p>

  <h2><span class="n">06</span>References</h2>
  <div class="refs">
    <p>[1] <i>Sheet Metal Drawing Operations</i>, Manufacturing Processes ME337 ch. 3, Mechanical
    Engineering Dept., University of Basrah. Drawing ratio, reduction, clearance, and the holding
    force relation F<sub>h</sub> = 0.015&nbsp;Y&pi;{{D<sub>b</sub>&sup2; &minus;
    (D<sub>p</sub>+2.2t+2R<sub>d</sub>)&sup2;}}. Course material, not peer-reviewed.</p>
    <p>[2] A. I. O. Zaid, &ldquo;Deep drawing mechanism, parameters, defects and recent results:
    state of the art&rdquo;, <i>IOP Conf. Ser.: Mater. Sci. Eng.</i> <b>146</b> (2016) 012009.
    Limiting draw ratio 2.2, H = b/2(DR&sup2;&minus;1), clearance optimum near 130% of thickness,
    holding pressure 400 psi for mild steel, punch and die profile radius ranges.</p>
    <p>[3] T. Lehrer and P. Stocker, <i>U-Channel</i>: parametric CAD models for U-shaped sheet
    metals, github.com/ToLe814/uchannel. Used as the U-channel specification, not as a runtime
    dependency.</p>
    <p>[4] Reference dataset supplied with this project, 228 cases. The box family's geometry was
    reverse-engineered from it and verified against it.</p>
  </div>

  <div class="foot"><span>Method note &middot; page 2 of 2</span>
    <span>{tot_g} geometries &middot; {tot_s} samples &middot; detail available on request</span></div>
</div>
"""


if __name__ == "__main__":
    out = EXAMPLES / "method-note.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
