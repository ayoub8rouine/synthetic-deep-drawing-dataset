# Dataset format

Each generated geometry is stored in its own case directory. A case can contain
one or more final parts produced with different blankholder forces.

```text
dataset/<case_name>/
|-- Data/
|   |-- 00_Blank_Viertel.npz
|   |-- 01_Matrize_Viertel.npz
|   |-- 02_Niederhalter_Viertel.npz
|   |-- 03_Stempel_Viertel.npz
|   `-- Bauteil/
|       `-- Tiefgezogenes_Bauteil_<force>.npz
|-- parameters.json
`-- case_report_<force>.json
```

The German filenames are retained for compatibility with the reference
StampFormer-style loader.

## Tool and blank archives

The four initial-state NPZ files contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `points` | `(N, 3)` | Sampled surface point cloud in millimetres |
| `idx` | implementation-dependent | Source/sample indices retained for compatibility |

## Final-part archives

Files under `Data/Bauteil/` contain the final state:

| Key | Shape | Meaning |
| --- | --- | --- |
| `points` | `(N, 3)` | Sampled final-part point cloud |
| `idx` | implementation-dependent | Source/sample indices |
| `nodes` | `(V, 3)` | Full deformed shell-node coordinates |
| `quads` | `(E, 4)` | Zero-based quadrilateral connectivity |
| `thickness` | `(E,)` | Final per-element shell thickness |

Coordinates and thicknesses are in millimetres. The filename suffix is the
full-tool blankholder force in newtons.

## Metadata

`parameters.json` stores the geometry, process, discretization, material, and
derived values needed to rebuild the case. `case_report_<force>.json` records
solver termination and acceptance metrics, including thinning, energy ratios,
hourglass energy, and binder closure. Consumers should check the report's
`checks.accepted` field before using a sample for training.

## Symmetry

Most cases are quarter models; shapes without y-axis symmetry use a half model.
Files contain the simulated symmetry sector, not an automatically mirrored full
part. See `postprocessing/export_trellis.py` for the mirroring and welding logic
used by visualization exports.
