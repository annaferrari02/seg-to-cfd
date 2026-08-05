import sys
import os
from collections import defaultdict

import vtk
import numpy as np


"""
average flow rate and cross sectional area from
Cheng et al: Abdominal aortic hemodynamic conditions in healthy subjects aged
50-70 at rest and during lower limb exercise: in vivo quantification
using MRI, Atherosclerosis 168 (2003) 323/331
"""

# ---- you already have these somewhere ----
# def triangle_area(poly, cid): ...
# -----------------------------------------
def triangle_area(polydata: vtk.vtkPolyData, cell_id: int) -> float:
    """Compute area of a single triangle cell by its cell_id."""
    cell = polydata.GetCell(cell_id)
    npts = cell.GetNumberOfPoints()
    if npts != 3:
        return 0.0

    pts = cell.GetPoints()
    p0 = pts.GetPoint(0)
    p1 = pts.GetPoint(1)
    p2 = pts.GetPoint(2)

    # vtkTriangle::TriangleArea expects 3 points
    return vtk.vtkTriangle.TriangleArea(p0, p1, p2)

def parse_id_list(s: str):
    """Parse '1,2,5' -> {1,2,5}. Empty/None -> empty set."""
    if s is None or str(s).strip() == "":
        return set()
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def scale_inflow_waveform(
    inflow_path: str,
    q_mean_target: float,
    out_dir: str,
    out_name: str = "inflow_3d_scaled.flow",
) -> str:
    """
    Read inflow_3d.flow (t, Q), compute its cycle-average mean flow,
    scale Q(t) so mean becomes q_mean_target, write scaled file to out_dir.
    Returns output file path.
    """
    if not os.path.isfile(inflow_path):
        raise RuntimeError(f"Inflow waveform file not found: {inflow_path}")

    # Load 2 columns: time, flow
    data = np.loadtxt(inflow_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Expected at least 2 columns (time, flow) in {inflow_path}")

    t = data[:, 0].astype(float)
    q = data[:, 1].astype(float)

    if len(t) < 2:
        raise RuntimeError(f"Need at least 2 samples in {inflow_path}")

    # Ensure increasing time
    if np.any(np.diff(t) <= 0):
        # sort by time (robust)
        idx = np.argsort(t)
        t = t[idx]
        q = q[idx]
        if np.any(np.diff(t) <= 0):
            raise RuntimeError("Time column must be strictly increasing (or sortable to strictly increasing).")

    # Cycle-average mean flow over the provided time window using trapezoidal integral
    duration = t[-1] - t[0]
    if duration <= 0:
        raise RuntimeError("Waveform duration <= 0 (check time column).")

    q_mean_old = np.trapz(q, t) / duration
    if q_mean_old == 0:
        raise RuntimeError("Old waveform mean flow is zero; cannot scale.")

    scale = q_mean_target / q_mean_old
    q_scaled = q * scale

    t_scale = 1 / t[-1] 

    t *= t_scale

    out_path = os.path.join(out_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("400 100\n")
        np.savetxt(f, np.column_stack([t, q_scaled]), fmt="%.10g")
    print(f"\nScaled inflow waveform:")
    print(f"  Inflow file: {inflow_path}")
    print(f"  Duration: {duration:.6g} s")
    print(f"  Mean flow (old): {q_mean_old:.6g} mL/s")
    print(f"  Mean flow (target): {q_mean_target:.6g} mL/s")
    print(f"  Scale factor: {scale:.12g}")
    print(f"  Wrote scaled waveform: {out_path}")
    return out_path

def main(
    vtp_path: str,
    out_path: str,
    face_id_array_name: str = "ModelFaceID",
    inlet_id: int = None,
    wall_ids=None,
    map_val: float = None,
    tot_comp: float = None,
    area_to_cm2: float = 1.0,
    res1_split: float = 0.15,
    inflow_file: str = "inflow_3d.flow",
):
    if inlet_id is None:
        raise RuntimeError("inlet_id is required (your file has no face-name array).")
    wall_ids = set(wall_ids) if wall_ids is not None else set()

    # Reference scaling constants (given by you)
    Aref_cm2 = 3.6
    Qref_ml_s = 38.3333

    # Read VTP
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfCells() == 0:
        raise RuntimeError(f"No cells found in: {vtp_path}")

    # Face ID array from CELL data
    face_id_arr = poly.GetCellData().GetArray(face_id_array_name)
    if face_id_arr is None:
        raise RuntimeError(
            f"Cell data array '{face_id_array_name}' not found.\n"
            f"Available cell arrays: {[poly.GetCellData().GetArrayName(i) for i in range(poly.GetCellData().GetNumberOfArrays())]}"
        )

    areas = defaultdict(float)
    radii = defaultdict(float)
    non_tri = 0
    tri_count = 0

    n_cells = poly.GetNumberOfCells()
    for cid in range(n_cells):
        cell = poly.GetCell(cid)

        # Only accumulate triangles
        if cell.GetCellType() == vtk.VTK_TRIANGLE and cell.GetNumberOfPoints() == 3:
            tri_count += 1
            face_id = int(face_id_arr.GetTuple1(cid))

            a = triangle_area(poly, cid)
            areas[face_id] += a
            radii[face_id] = (areas[face_id] / np.pi) ** 0.5
        else:
            if cell.GetNumberOfPoints() > 0:
                non_tri += 1

    # Basic checks
    if inlet_id not in areas:
        raise RuntimeError(f"inlet_id={inlet_id} not found among face IDs: {sorted(areas.keys())}")
    missing_walls = [wid for wid in wall_ids if wid not in areas]
    if missing_walls:
        print(f"Warning: wall_ids not found in mesh (ignored): {missing_walls}")

    # Write output (areas)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# face_id area\n")
        for face_id in sorted(areas.keys()):
            f.write(f"{face_id} {areas[face_id]:.12g}\n")
            print(f"Face ID: {face_id}, Area: {areas[face_id]:.12g}, Radius: {radii[face_id]:.6g}")

        total = sum(areas.values())
        f.write(f"# triangles_used {tri_count}\n")
        f.write(f"# non_triangle_cells_skipped {non_tri}\n")
        f.write(f"# total_area {total:.12g}\n")

    print(f"Wrote: {out_path}")
    print(f"Triangles used: {tri_count}")
    print(f"Non-triangle cells skipped: {non_tri}")
    print(f"Total area: {sum(areas.values())}")

    # AREA is inlet area (convert to cm^2 if needed)
    AREA_cm2 = areas[inlet_id] * float(area_to_cm2)

    # Q_to_distribute = Qref * AREA / Aref
    Q_to_distribute = Qref_ml_s * (AREA_cm2 / Aref_cm2)

    # scale also compliance
    tot_comp = tot_comp * (AREA_cm2 / Aref_cm2)

    print(f"\nInlet face_id: {inlet_id}")
    print(f"Inlet AREA (cm^2): {AREA_cm2:.6g}  (area_to_cm2={area_to_cm2})")
    print(f"Q_to_distribute (mL/s): {Q_to_distribute:.6g}  (Qref={Qref_ml_s}, Aref={Aref_cm2})")

    # Outlets: all faces except inlet and wall(s)
    outlets_excluded = {inlet_id} | (wall_ids & set(areas.keys()))
    outlet_face_ids = [fid for fid in areas.keys() if fid not in outlets_excluded]

    if not outlet_face_ids:
        raise RuntimeError("No outlet faces found after excluding inlet and wall ids.")

    if map_val is None or tot_comp is None:
        raise RuntimeError("map_val and tot_comp must be provided to compute resistances/compliances.")

    # ---- distribute flow using Murray exponent 3 ----
    sum_radii_cubed = 0.0
    for fid in outlet_face_ids:
        r = radii[fid]
        if r <= 0:
            raise RuntimeError(f"Zero/negative radius for face {fid} (area={areas[fid]}).")
        sum_radii_cubed += r ** 3

    if sum_radii_cubed <= 0:
        raise RuntimeError("sum_radii_cubed <= 0, cannot distribute flow.")

    if not (0.0 <= res1_split <= 1.0):
        raise ValueError("res1_split must be in [0, 1].")
    res2_split = 1.0 - res1_split

    flow_distribution = {}
    resistances = {}
    comp_distribution = {}

    for fid in outlet_face_ids:
        r = radii[fid]
        q = Q_to_distribute * ((r ** 3) / sum_radii_cubed)
        flow_distribution[fid] = q
        resistances[fid] = float(map_val) / q
        comp_distribution[fid] = float(tot_comp) * q / Q_to_distribute

    print("\nFlow distribution among outlets (all faces except inlet and wall ids):")
    print(f"{'Face ID':>10} {'Flow(mL/s)':>15} {'R1':>15} {'R2':>15} {'Compliance':>15}")
    for fid in sorted(flow_distribution.keys()):
        R = resistances[fid]
        print(
            f"{fid:>10} {flow_distribution[fid]:>15.6g} "
            f"{R*res1_split:>15.6g} {R*res2_split:>15.6g} {comp_distribution[fid]:>15.6g}"
        )
    # ---- Scale inflow waveform to Qnew mean and save next to out_path ----
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    inflow_path = inflow_file
    # If inflow_file is relative, interpret it relative to the output folder (common workflow)
    if not os.path.isabs(inflow_path):
        candidate = os.path.join(out_dir, inflow_file)
        if os.path.isfile(candidate):
            inflow_path = candidate

    scale_inflow_waveform(
        inflow_path=inflow_path,
        q_mean_target=-1. * Q_to_distribute,
        out_dir=out_dir,
        out_name="inflow.reference.scaled",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# face_id area R1 C R2\n")
        for face_id in sorted(areas.keys()):
            area = areas[face_id]

            if face_id in flow_distribution:
                R = resistances[face_id]
                R1 = R * res1_split
                R2 = R * res2_split
                C = comp_distribution[face_id]
            else:
                # inlet + wall(s) (or any non-outlet face)
                R1 = 0.0
                R2 = 0.0
                C = 0.0

            f.write(f"{face_id} {area:.12g} {R1:.12g} {C:.12g} {R2:.12g}\n")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage:\n"
            "  python3 setup_bcs.py input.vtp output.txt MAP TOT_COMP INLET_ID [WALL_IDS] [FaceIDArray] [area_to_cm2] [res1_split] [inflow_file]\n\n"
            "Where:\n"
            "  INLET_ID    = integer face id for inlet\n"
            "  WALL_IDS    = optional comma-separated ids for wall faces (e.g. 1 or 1,2,3)\n"
            "  inflow_file = optional, default 'inflow.reference'\n\n"
            "Defaults:\n"
            "  FaceIDArray = ModelFaceID\n"
            "  area_to_cm2 = 1.0  (set to 0.01 if your mesh units are mm, since mm^2 -> cm^2)\n"
            "  res1_split  = 0.15\n\n"
            "Example:\n"
            "  python3 setup_bcs.py test.vtp output.txt 119989.8 0.00097508288 14 1 ModelFaceID 1. 0.15"
        )
        sys.exit(1)

    vtp = sys.argv[1]
    out = sys.argv[2]
    map_val = float(sys.argv[3])
    tot_comp = float(sys.argv[4])
    inlet_id = int(sys.argv[5])

    wall_ids = parse_id_list(sys.argv[6]) if len(sys.argv) > 6 else set()
    face_id_arr = sys.argv[7] if len(sys.argv) > 7 else "ModelFaceID"
    area_to_cm2 = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    res1_split = float(sys.argv[9]) if len(sys.argv) > 9 else 0.15

    main(vtp, out, face_id_arr, inlet_id, wall_ids, map_val, tot_comp, area_to_cm2, res1_split)