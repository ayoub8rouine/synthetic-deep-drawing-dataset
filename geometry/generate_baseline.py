"""Generate one draft-safe, StampFormer-inspired baseline tool set.

Outputs are intentionally limited to one case. Geometry is built with Gmsh's
OpenCASCADE kernel and exported as STEP, inspection STL, and LS-DYNA keyword
surface meshes. Dimensions are millimetres.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gmsh
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def rounded_rectangle_wire(x: float, width: float, height: float, radius: float) -> int:
    """Create a closed yz-plane rounded rectangle wire at x."""
    r = min(radius, width * 0.24, height * 0.24)
    y0, y1 = -width / 2, width / 2
    z0, z1 = 0.0, height
    pts = [
        gmsh.model.occ.addPoint(x, y0 + r, z0),
        gmsh.model.occ.addPoint(x, y1 - r, z0),
        gmsh.model.occ.addPoint(x, y1, z0 + r),
        gmsh.model.occ.addPoint(x, y1, z1 - r),
        gmsh.model.occ.addPoint(x, y1 - r, z1),
        gmsh.model.occ.addPoint(x, y0 + r, z1),
        gmsh.model.occ.addPoint(x, y0, z1 - r),
        gmsh.model.occ.addPoint(x, y0, z0 + r),
    ]
    centers = [
        gmsh.model.occ.addPoint(x, y1 - r, z0 + r),
        gmsh.model.occ.addPoint(x, y1 - r, z1 - r),
        gmsh.model.occ.addPoint(x, y0 + r, z1 - r),
        gmsh.model.occ.addPoint(x, y0 + r, z0 + r),
    ]
    curves = [
        gmsh.model.occ.addLine(pts[0], pts[1]),
        gmsh.model.occ.addCircleArc(pts[1], centers[0], pts[2]),
        gmsh.model.occ.addLine(pts[2], pts[3]),
        gmsh.model.occ.addCircleArc(pts[3], centers[1], pts[4]),
        gmsh.model.occ.addLine(pts[4], pts[5]),
        gmsh.model.occ.addCircleArc(pts[5], centers[2], pts[6]),
        gmsh.model.occ.addLine(pts[6], pts[7]),
        gmsh.model.occ.addCircleArc(pts[7], centers[3], pts[0]),
    ]
    return gmsh.model.occ.addWire(curves)


def add_loft(params: dict, clearance: float = 0.0, die_profile: bool = False) -> int:
    """Create a smooth, closed crossmember-like solid from nested sections."""
    length = params["tool_length"]
    height = params["punch_height"] + clearance
    draft = math.radians(params["draft_angle_deg"])
    base_width = 118.0 + 0.30 * (params["D2_bead_position"] - params["D1_bead_position"])
    top_inset = height / max(math.tan(draft), 0.25)
    # D1/D2 explicitly locate the crossmember's two shoulder transitions.
    d1 = min(params["D1_bead_position"], 0.28 * length)
    d2 = min(params["D2_bead_position"], 0.44 * length)
    xs = [-length / 2, -d2, -d1, 0.0, d1, d2, length / 2]
    bulges = [0.0, 0.32, 0.72, 1.0, 0.72, 0.32, 0.0]
    wires = []
    for i, (x, bulge) in enumerate(zip(xs, bulges)):
        # R4 controls the broad centre crown; R5 varies smoothly end-to-end.
        r5 = params["R5_start_radius"] + (params["R5_end_radius"] - params["R5_start_radius"]) * i / (len(xs) - 1)
        width = base_width + 0.42 * params["R4_fillet_radius"] * bulge + 2 * clearance
        # Nested draft-safe profile: rounded rectangle corners are bounded by R2/R5.
        primary_radius = params["R1_die_radius"] if die_profile else params["R2_punch_radius"]
        # R3 controls the end fillets; variable R5 blends toward the centre.
        end_weight = abs(x) / (length / 2)
        radius = (
            primary_radius
            + 0.20 * r5
            + 0.15 * params["R3_fillet_radius"] * end_weight
            + clearance
        )
        wires.append(rounded_rectangle_wire(x, width, height, radius))
    loft = gmsh.model.occ.addThruSections(wires, makeSolid=True, makeRuled=False)
    if isinstance(loft, list):
        return next(tag for dim, tag in loft if dim == 3)
    return loft


def export_entity(name: str, builder, out_dir: Path, mesh_size: float) -> dict:
    gmsh.clear()
    gmsh.model.add(name)
    volume = builder()
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [volume], 1, name)
    boundary_tags = [tag for dim, tag in gmsh.model.getBoundary([(3, volume)], combined=False, oriented=False) if dim == 2]
    gmsh.model.addPhysicalGroup(2, sorted(set(boundary_tags)), 2, f"{name}_surface")
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.write(str(out_dir / f"{name}.step"))
    gmsh.model.mesh.generate(2)
    stl_path = out_dir / f"{name}.stl"
    gmsh.write(str(stl_path))
    # Boolean cavity faces can carry independent OCC parametrization directions.
    # Weld facets and orient each closed shell consistently for inspection/export.
    inspection_mesh = trimesh.load_mesh(stl_path, process=True)
    trimesh.repair.fix_normals(inspection_mesh, multibody=True)
    inspection_mesh.export(stl_path)
    gmsh.write(str(out_dir / f"{name}.key"))
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, _ = gmsh.model.mesh.getElements(2)
    count = sum(len(tags) for tags in element_tags)
    boundary = gmsh.model.getBoundary([(3, volume)], combined=False, oriented=False)
    bbox = gmsh.model.getBoundingBox(3, volume)
    return {
        "volume_tag": volume,
        "nodes": len(node_tags),
        "surface_elements": count,
        "surface_count": len(boundary),
        "bbox": list(bbox),
        "element_types": [int(value) for value in element_types],
    }


def generate(params: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    clearance = params["sheet_thickness"] * params["clearance_ratio"]
    results = {}
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    try:
        results["punch"] = export_entity(
            "punch", lambda: add_loft(params), output, params["mesh_size_tools"]
        )

        def die_builder() -> int:
            length = params["blank_length"] - 30
            width = params["blank_width"] - 30
            # Top is at z=52 mm. The clearance loft reaches 53.76 mm, ensuring
            # that the subtraction creates an open cavity instead of a trapped void.
            block = gmsh.model.occ.addBox(-length / 2, -width / 2, -18.0, length, width, 70.0)
            cavity = add_loft(params, clearance, die_profile=True)
            cut, _ = gmsh.model.occ.cut([(3, block)], [(3, cavity)], removeObject=True, removeTool=True)
            return next(tag for dim, tag in cut if dim == 3)

        results["die"] = export_entity("die", die_builder, output, params["mesh_size_tools"])

        def binder_builder() -> int:
            length, width = params["blank_length"], params["blank_width"]
            outer = gmsh.model.occ.addBox(-length / 2, -width / 2, 4.0, length, width, 12.0)
            opening_l = params["tool_length"] + 22.0
            opening_w = 118.0 + 0.30 * (params["D2_bead_position"] - params["D1_bead_position"]) + 32.0
            inner = gmsh.model.occ.addBox(-opening_l / 2, -opening_w / 2, 3.0, opening_l, opening_w, 14.0)
            cut, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
            return next(tag for dim, tag in cut if dim == 3)

        results["binder"] = export_entity("binder", binder_builder, output, params["mesh_size_tools"])

        def blank_builder() -> int:
            return gmsh.model.occ.addBox(
                -params["blank_length"] / 2,
                -params["blank_width"] / 2,
                0.0,
                params["blank_length"],
                params["blank_width"],
                params["sheet_thickness"],
            )

        results["blank"] = export_entity("blank", blank_builder, output, params["mesh_size_blank"])
    finally:
        gmsh.finalize()

    total_nodes = sum(item["nodes"] for item in results.values())
    total_elements = sum(item["surface_elements"] for item in results.values())
    report = {
        "case_id": params["case_id"],
        "kernel": "Gmsh OpenCASCADE",
        "clearance_mm": clearance,
        "student_limits": {"nodes": 128000, "elements": 128000},
        "counts_are_individual_surface_mesh_sums": True,
        "total_nodes": total_nodes,
        "total_surface_elements": total_elements,
        "below_student_limit": total_nodes < 128000 and total_elements < 128000,
        "geometry": results,
        "parameters": params,
    }
    (output / "geometry_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("baseline_parameters.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "simulations" / "baseline_midpoint" / "geometry")
    args = parser.parse_args()
    params = json.loads(args.config.read_text(encoding="utf-8"))
    report = generate(params, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
