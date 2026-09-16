"""Independent CAD/mesh checks for generated baseline geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmsh
import numpy as np
import trimesh


def validate(folder: Path) -> dict:
    report = json.loads((folder / "geometry_report.json").read_text(encoding="utf-8"))
    checks = {}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        for name in ("punch", "die", "binder", "blank"):
            gmsh.clear()
            gmsh.model.add(f"check_{name}")
            entities = gmsh.model.occ.importShapes(str(folder / f"{name}.step"), highestDimOnly=False)
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(3)
            surfaces = gmsh.model.getEntities(2)
            # STL repeats vertices per facet by design; processing welds coincident
            # vertices before topology checks without changing the source file.
            mesh = trimesh.load_mesh(folder / f"{name}.stl", process=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            areas = mesh.area_faces
            unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
            checks[name] = {
                "step_import_entities": len(entities),
                "step_volumes": len(volumes),
                "step_surfaces": len(surfaces),
                "stl_vertices": int(len(mesh.vertices)),
                "stl_faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "degenerate_faces": int(np.count_nonzero(areas < 1e-10)),
                "duplicate_faces": int(len(mesh.faces) - len(unique_faces)),
                "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
            }
            if name == "die":
                # An open cavity introduces inner boundary loops on the die top;
                # detect the opening semantically using upward-facing top facets.
                top_z = float(mesh.bounds[1, 2])
                top_faces = np.flatnonzero(
                    (mesh.triangles_center[:, 2] > top_z - 1e-5)
                    & (mesh.face_normals[:, 2] > 0.9)
                )
                top_area = float(390.0 * 230.0)
                meshed_top_area = float(mesh.area_faces[top_faces].sum())
                checks[name]["cavity_open_at_top"] = meshed_top_area < top_area * 0.95
    finally:
        gmsh.finalize()

    all_valid = all(
        c["step_volumes"] == 1
        and c["watertight"]
        and c["winding_consistent"]
        and c["degenerate_faces"] == 0
        and c["duplicate_faces"] == 0
        and c["finite_vertices"]
        and (name != "die" or c.get("cavity_open_at_top", False))
        for name, c in checks.items()
    )
    result = {
        "all_geometry_checks_pass": all_valid,
        "below_student_limit": report["below_student_limit"],
        "total_nodes": report["total_nodes"],
        "total_surface_elements": report["total_surface_elements"],
        "checks": checks,
    }
    (folder / "validation_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = validate(args.folder)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_geometry_checks_pass"] and result["below_student_limit"] else 1)


if __name__ == "__main__":
    main()
