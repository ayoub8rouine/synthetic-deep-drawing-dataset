# Contributing

Thanks for helping improve the synthetic deep-drawing pipeline.

## Development setup

Use Python 3.10 or newer in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

LS-DYNA is optional for geometry work and dry runs, but is required to produce
deformed parts. It must be installed and licensed separately.

## Before opening a pull request

Run the checks that do not require a solver:

```powershell
python -m compileall -q geometry meshing postprocessing scripts shapes u_shape
python .\scripts\run_sweep.py --grid pilot --dry-run
python .\shapes\pipeline.py --sample --n 3
python .\u_shape\pipeline.py --sample --n 3
```

If a change affects an LS-DYNA deck, also run at least one representative case
and record the termination status and acceptance checks in the pull request.

## Repository hygiene

Do not commit generated datasets, solver binaries, `d3plot`/`d3dump` files,
rendered media, or local LS-DYNA logs. The `.gitignore` excludes these outputs.
Keep reusable configuration and small source fixtures under version control.

## Changes to the data format

Keep existing NPZ keys backward-compatible where possible. Document intentional
schema changes in `docs/data-format.md` and include a small migration example.
