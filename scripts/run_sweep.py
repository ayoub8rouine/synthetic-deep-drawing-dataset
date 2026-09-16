"""Generate the dataset: sweep geometries and blankholder forces through LS-DYNA.

Tool geometry depends only on the shape parameters, so each geometry is built
once and reused by every blankholder force run under it.

The sweep is resumable. A case whose dataset npz already exists is skipped, so
the run can be stopped and restarted, and a failed case can be retried by
deleting its output.

    python .\\scripts\\run_sweep.py --grid pilot
    python .\\scripts\\run_sweep.py --grid reference --ncpu 4
    python .\\scripts\\run_sweep.py --grid reference --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "geometry"))
sys.path.insert(0, str(ROOT / "meshing"))
sys.path.insert(0, str(ROOT / "postprocessing"))

from box_draw import DrawParams  # noqa: E402
from build_forming_deck import build_case  # noqa: E402
from extract_case import write_case  # noqa: E402

DIAGNOSTIC = ROOT / "simulations" / "smoke_test" / "diagnostic.json"


# ---------------------------------------------------------------------------
# parameter grids
# ---------------------------------------------------------------------------


def reference_grid() -> list:
    """The parameter space of the reference dataset.

    draft is delta = m * height / 25 for m in 0..4, i.e. a fixed set of draft
    angles rather than a fixed set of absolute insets.
    """
    cases = []
    for corner, height, m in itertools.product(
        [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        [0, 1, 2, 3, 4],
    ):
        cases.append({"corner_radius": corner, "height": height,
                      "draft": round(m * height / 25.0, 3)})
    return cases


# Blankholder forces, in N on the full (4x) tool.
#
# These are much larger than the reference dataset's 15-40 kN. The flange here is
# about 28,600 mm2 per quarter, so 25 kN total is only 0.22 MPa of blankholder
# pressure -- far too little to clamp the sheet. At that level the flange is not
# held: it slides down between the die and the binder, the binder follows it, and
# the drawn part ends up several mm deeper than the tool geometry allows. Raising
# the force to 200 kN (1.75 MPa) puts the binder within 0.03 mm of the die and the
# part within 0.03 mm of its geometric target. The range below spans roughly
# 0.9-3.5 MPa, which is the normal window for mild forming steel.
FORCES = [100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 400000.0]

GRIDS = {
    # end-to-end proof of the pipeline: a few geometries, two forces each
    "pilot": {
        "geometry": [
            {"corner_radius": 20.0, "height": 25.0, "draft": 0.0},
            {"corner_radius": 35.0, "height": 35.0, "draft": 2.8},
            {"corner_radius": 55.0, "height": 50.0, "draft": 6.0},
        ],
        "blankholder_force": [150000.0, 300000.0],
    },
    # a spread across the whole space, sized to finish in a working day
    "medium": {
        "geometry": [
            {"corner_radius": c, "height": h, "draft": round(m * h / 25.0, 3)}
            for c, h, m in itertools.product([20.0, 35.0, 50.0], [25.0, 35.0, 45.0], [0, 2, 4])
        ],
        "blankholder_force": [150000.0, 250000.0, 400000.0],
    },
    # the full reference parameter space
    "reference": {
        "geometry": reference_grid(),
        "blankholder_force": FORCES,
    },
}


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def find_solver(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if env_path := os.environ.get("LSDYNA_SOLVER"):
        candidates.append(Path(env_path).expanduser())
    if DIAGNOSTIC.exists():
        path = json.loads(DIAGNOSTIC.read_text(encoding="utf-8-sig")).get("solver_path")
        if path:
            candidates.append(Path(path))
    for command in ("ls-dyna", "ls-dyna_smp", "ls-dyna_mpp"):
        if found := shutil.which(command):
            candidates.append(Path(found))

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    for pattern in (
        "LS-DYNA Suite*/lsdyna/ls-dyna*.exe",
        "LSTC/LS-DYNA*/ls-dyna*.exe",
        "ANSYS Inc/*/ansys/bin/winx64/lsdyna*.exe",
    ):
        candidates.extend(sorted(program_files.glob(pattern), reverse=True))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "LS-DYNA solver not found. Pass --solver, set LSDYNA_SOLVER, add the "
        "executable to PATH, or run scripts/diagnose_lsdyna.ps1."
    )


def find_ls_prepost(explicit: str | None = None) -> Path:
    """Find LS-PrePost for generated click-to-open showcase scripts."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if env_path := os.environ.get("LSPREPOST_EXE"):
        candidates.append(Path(env_path).expanduser())
    if found := shutil.which("lsprepost"):
        candidates.append(Path(found))

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.extend(
        sorted(
            program_files.glob("LS-DYNA Suite*/lspp/lsprepost*.exe"),
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "LS-PrePost not found. Pass --ls-prepost, set LSPREPOST_EXE, or add "
        "the executable to PATH."
    )


def run_solver(solver: Path, run_dir: Path, memory: str, ncpu: int,
               timeout: float) -> dict:
    started = time.time()
    with open(run_dir / "solver_stdout.log", "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [str(solver), "i=draw.k", f"memory={memory}", f"ncpu={ncpu}"],
                cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
            code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            code, timed_out = -1, True
    return {"exit_code": code, "timed_out": timed_out,
            "wall_seconds": round(time.time() - started, 1)}


class LicenseBusy(RuntimeError):
    """The Student solver allows one run at a time and something else holds it."""


def check_license(run_dir: Path) -> None:
    """Abort loudly on a licence failure instead of failing every case.

    The Student edition permits a single concurrent solver. If another instance
    is still running -- an orphan from an interrupted sweep, or LS-Run left open
    -- every launch dies in seconds, so a whole sweep can 'finish' having
    produced nothing.
    """
    log = run_dir / "solver_stdout.log"
    if not log.exists():
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    # Error 70022 is the generic licence refusal, not specifically a busy
    # licence: the Student edition raises it both when another solver holds the
    # checkout AND when ncpu exceeds its 4-core cap. Reading the reason back
    # avoids sending someone hunting for a phantom second instance.
    if "restricted to" in text and "cores" in text:
        reason = next((ln.strip() for ln in text.splitlines()
                       if "restricted to" in ln), "core limit exceeded")
        raise LicenseBusy(
            f"{reason} The Student licence caps SMP at 4 cores; re-run with "
            "ncpu=4."
        )
    if "license already in use" in text.lower() or "Error 70022" in text:
        raise LicenseBusy(
            "the LS-DYNA Student licence is already in use. Close LS-Run or any "
            "other solver instance, then re-run this sweep -- it resumes where "
            "it stopped."
        )


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def case_done(dataset: Path, params: DrawParams) -> bool:
    bhf = int(round(params.blankholder_force))
    target = (dataset / params.case_name() / "Data" / "Bauteil"
              / f"Tiefgezogenes_Bauteil_{bhf}.npz")
    return target.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="pilot")
    ap.add_argument("--runs", type=Path, default=ROOT / "simulations" / "sweep")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--memory", default="400m")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N runs")
    ap.add_argument("--keep-d3plot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = GRIDS[args.grid]
    material = json.loads((ROOT / "materials" / "assumed_forming_steel.json")
                          .read_text(encoding="utf-8"))
    jobs = [(g, f) for g in grid["geometry"] for f in grid["blankholder_force"]]

    print(f"grid '{args.grid}': {len(grid['geometry'])} geometries x "
          f"{len(grid['blankholder_force'])} forces = {len(jobs)} runs")
    if args.dry_run:
        for g, f in jobs[:10]:
            print("  ", DrawParams(**g).case_name(), f"bhf={f:g}")
        if len(jobs) > 10:
            print(f"   ... and {len(jobs) - 10} more")
        return

    solver = find_solver(args.solver)
    print(f"solver: {solver}\n")

    manifest, started, ran = [], time.time(), 0
    for geometry, force in jobs:
        if args.limit is not None and ran >= args.limit:
            break
        p = DrawParams(**geometry, blankholder_force=force)
        tag = f"{p.case_name()}  bhf={int(force)}"
        if case_done(args.dataset, p):
            print(f"skip   {tag}")
            continue

        run_dir = args.runs / p.case_name() / f"bhf_{int(force)}"
        entry = {"case": p.case_name(), "blankholder_force": force,
                 "run_dir": str(run_dir)}
        try:
            build_case(p, run_dir, material)
            entry["solver"] = run_solver(solver, run_dir, args.memory, args.ncpu,
                                         args.timeout)
            check_license(run_dir)
            if not (run_dir / "dynain").exists():
                raise RuntimeError("solver produced no dynain (see solver_stdout.log)")
            summary = write_case(run_dir, args.dataset)
            entry["checks"] = summary["checks"]
            entry["status"] = "accepted" if summary["checks"]["accepted"] else "rejected"
        except LicenseBusy as exc:
            print(f"ABORTED: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the sweep
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if not args.keep_d3plot and run_dir.exists():
                for pattern in ("d3plot*", "d3dump*", "d3full*"):
                    for f in run_dir.glob(pattern):
                        # Windows keeps a handle open briefly after the solver
                        # exits. Leaving a plot file behind costs disk, not
                        # correctness, so never let cleanup kill the sweep.
                        try:
                            f.unlink(missing_ok=True)
                        except OSError:
                            pass

        ran += 1
        manifest.append(entry)
        detail = entry.get("error") or (
            f"thinning={entry['checks']['max_thinning']:.1%} "
            f"ke/ie={entry['checks']['peak_ke_over_ie']:.1%} "
            f"{entry['solver']['wall_seconds']:.0f}s"
            if entry.get("checks") else "")
        print(f"{entry['status']:<9} {tag}  {detail}")

        # The manifest lives beside the runs, not in the dataset: the reference
        # dataloader treats every direct child of the dataset directory as a
        # case, so a stray file there would break it.
        args.runs.mkdir(parents=True, exist_ok=True)
        (args.runs / "sweep_manifest.json").write_text(
            json.dumps({"grid": args.grid, "runs": manifest}, indent=2), encoding="utf-8")

    ok = sum(1 for e in manifest if e["status"] == "accepted")
    print(f"\n{ok}/{len(manifest)} accepted in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
